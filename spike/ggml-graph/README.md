# One traced training step as one ggml graph on the iGPU

**This is a spike, not a package.** Nothing here is wired into `tensor/`'s
`Backend` enum. It answers one question: can a full training step (forward,
backward, optimizer update) that `Trace` records be lowered to **one** ggml
compute graph, built once, and run on the Vega iGPU through ggml's Vulkan
backend, and how fast is it at large batches?

**Answer: yes.** The graph `Trace.gradients` already builds for an MLP
classifier lowers op by op to ggml with no missing op on Vulkan; the graph is
built and allocated once and reused every step; parameters and Adam moments
stay in a backend buffer and are written back inside the same graph. One step
matches the reference backend to 1e-7 on ggml CPU and on the iGPU (fp32
`mul_mat`), 4e-4 with ggml's default fp16 `mul_mat`. From batch 256 up, the
iGPU trains this MLP **1.6–3.7× faster than plain C++ libtorch** on the CPU,
and 6.6–8.5× faster than what `Train.fit` does on libtorch today.

It adds `spike/ggml-graph/` and one line to `../ggml-vulkan/tools/build-ggml.sh`
(`JOBS` sets the parallel jobs). It reads `tensor/` and `train/` in place
(`path` dependencies), so it lowers today's `Trace`, not a copy.

## What is real and what is faked

| Thing | State |
|---|---|
| The graph that is lowered | **real**: `Step.loss` is `Tensor.linear`, `Tensor.relu` and `Loss.cross_entropy`; the backward is `Trace.gradients`; SGD is `Optim.step` of `train/` called on traced tensors |
| One ggml graph per step, built once | **real**: `Raw.compile` builds the cgraph, allocates it once with `ggml_gallocr`, and every `Raw.run` computes that same graph |
| Parameters resident across steps | **real**: states live in one backend buffer; the graph ends with a `ggml_cpy` of each new value into its state |
| Inputs uploaded per batch | **real**: `Raw.run` copies the step's batch from host memory into the input leaves, then computes |
| The batch data | **synthetic**: made in C (`Raw.batches`, an LCG) so a benchmark step crosses no `List`; the correctness tests upload Agentica lists |
| The iGPU really ran | **asserted**: `backend_vulkan` fails unless the backend is Vulkan and the device is not a CPU; `ggml_backend_supports_op` is checked for every node at compile |
| `Bool` and integer tensors | **F32 on the ggml side**: masks are 0/1, labels and `arange` are exact integers below 2^24 |
| Production shape (`Backend`, `Handle`, cache, `Train.fit`) | **not built**: the last section proposes it |

## Run it

```sh
git submodule update --init third_party/ggml
JOBS=2 sh ../ggml-vulkan/tools/build-ggml.sh          # CPU + Vulkan trees (README of ggml-vulkan)
eval "$(sh ../ggml-vulkan/tools/build-ggml.sh | grep '^export ')"
eval "$(sh ../../tensor/tools/build-torch.sh)"         # tensor/'s torch module, and CC=g++
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH:$CPLUS_INCLUDE_PATH"   # g++ ignores C_INCLUDE_PATH
agentica install && agentica check && agentica fmt --check && agentica test
agentica check.agc                                     # the correctness gaps, both backends
agentica bench.agc vulkan sgd 64 256 1024 4096         # also: cpu, cpu4, torch, reference; sgd, two, adam
```

The libtorch ceiling is a plain C++ program:

```sh
g++ -O2 -std=c++17 tools/bench-torch-step.cpp -o "$TMPDIR/bench-torch-step" \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" -ltorch -ltorch_cpu -lc10
"$TMPDIR/bench-torch-step" 0 sgd                       # 0: libtorch's default (4 threads here)
```

The machine is the one of `../ggml-vulkan/README.md` (Ryzen 5 3550H, Vega 8
iGPU through RADV, `uma: 1`, 12 GiB), shared and at load 1–5 while these ran.
The iGPU numbers move by ~10 % between runs, the CPU ones by up to 2×; the
tables give both runs where there were two.

## How a step is lowered

`step.agc` builds a `Lower.Step`: the parameters, the batch and Adam's moments
enter as `Trace.constant`, the loss is traced, `Trace.gradients` derives the
backward, and the optimizer runs on the traced tensors. A `Step` names which
constants are **states** (kept across steps), which are **inputs** (uploaded
per step), and which nodes **write** a state.

`lower.agc` walks every node the outputs and the writes reach, in id order
(§10.2: a topological order), and sends each to `native/ggmlgraph.c`, which
makes the ggml tensor at once. A Vulkan graph needs no ops beyond these:

| Traced op | ggml |
|---|---|
| `constant` | a leaf in the persistent buffer: state, input, or fixed |
| `full` | a one-element leaf, broadcast where it is used ("lazy") |
| `add`, `sub`, `mul`, `div` | the same op; a lazy right operand is ggml's own broadcast |
| `neg`, `exp`, `log`, `relu`, `sqrt`, `sigmoid`, `tanh` | the same unary |
| `matmul(a, b)` | `mul_mat(cont(bᵀ), a)`; for `b = transpose(w)` (the backward's `ct · wᵀ`), `mul_mat(w, a)`, no copy |
| `transpose` | `cont(transpose)` |
| `reshape`, `squeeze`, `unsqueeze` | `reshape_4d` |
| `broadcast_to` | lazy; `repeat_4d` only where an op needs it dense |
| `sum`, `mean` over a leading or trailing block of axes | `sum_rows` after a reshape (or a transpose), `scale` for the mean |
| `max` over such a block | `pool_1d` MAX with the row as the kernel |
| `stop_grad`, `copy`, `contiguous`, `to_dtype` (mask or int to float), `to_device(Cpu)` | identity |
| `arange` | `arange` (F32) |
| `eq(a, b)`, `lt(a, b)` | `1 - step(abs(a - b))`, `step(b - a)` (`scale_bias`, `step`, `abs`) |
| `where(c, x, y)` | `c·x + (1-c)·y`; a zero branch drops out (relu's and cross-entropy's rules) |
| a write | `ggml_cpy(value, state)`, after every update is computed |

The SGD step is 96 traced nodes and 104 ggml nodes; Adam is 212 and 182 (the
lazy fills and broadcasts disappear, the transposes and reshapes add some).

Not lowered, each an error naming the op (`test/`): `pow`, `clamp`,
`gelu_tanh_approx`, `argmax`, `batch_matmul`, `slice`, `concat`, `stack`,
`gather`, `conv1d`, `avg_pool1d`, and a reduction over axes that are not a
leading or trailing block. ggml has an op for most (`clamp`, `gelu`, 4-D
`mul_mat`, views, `concat`, `get_rows`, `im2col`, `pool_1d`, `permute`); none
was needed for this step, so none was written. Two points need care when they
are: ggml's CPU `gelu` reads an fp16 table, and `get_rows` only gathers rank-1
indices along the first axis.

**Missing on Vulkan: nothing this step uses.** Every node of the SGD and Adam
graphs passes `ggml_backend_supports_op` on the RADV device. The emulations
are dtype ones (no bool or integer arithmetic: comparisons through `step`,
`where` through `mul`), not missing kernels. `out_prod` exists on Vulkan and
would avoid two transposes per weight gradient, but its shader is slow here
(`GGMLGRAPH_OUT_PROD=1`: 225 ms per step at batch 4096 against 51 ms).

## Correctness

`agentica check.agc`, and `test/ggml_graph_test.agc` with `agentica test`. A
small MLP (8 → 16 → 16 → 4, a batch of 6, cross-entropy) takes one step on
ggml, and the step `Train.fit` runs takes it on the reference backend. A gap is
`max |ggml - reference| / max |reference|` per tensor, the worst tensor shown:

| backend | step | loss | gradients | new parameters |
|---|---|---|---|---|
| ggml CPU | SGD, one graph | 5.5e-8 | 1.6e-7 | 7.4e-8 |
| ggml CPU | SGD, two graphs | 5.5e-8 | 1.6e-7 | 7.4e-8 |
| ggml CPU | Adam, two steps, one graph | 2.8e-8 | 2.3e-7 | 9.6e-8 |
| ggml Vulkan (fp16 `mul_mat`, the default) | SGD, one graph | 5.5e-8 | 4.1e-4 | 6.8e-6 |
| ggml Vulkan (fp16 `mul_mat`) | Adam, two steps | 5.9e-8 | 3.5e-4 | 7.9e-5 |
| ggml Vulkan, `GGML_VK_DISABLE_F16=1` | SGD, one graph | 5.5e-8 | 1.6e-7 | 7.4e-8 |
| ggml Vulkan, `GGML_VK_DISABLE_F16=1` | Adam, two steps | 5.9e-8 | 1.2e-7 | 9.6e-8 |

The tests assert 1e-5 on the CPU and 3e-3 on the iGPU (the fp16 path), state
"no Vulkan device" instead of passing silently, check that twenty runs of one
compiled program lower the loss, and that an op with no lowering is an error
naming it:

```console
$ agentica test
PASS .../ggml_graph_test.agc / sgd one graph matches the reference on the cpu
PASS .../ggml_graph_test.agc / sgd two graphs match the reference on the cpu
PASS .../ggml_graph_test.agc / two adam steps match the reference on the cpu
PASS .../ggml_graph_test.agc / the igpu step matches the reference
PASS .../ggml_graph_test.agc / the graph is built once and reused
PASS .../ggml_graph_test.agc / an op with no lowering is an error naming it
6 passed, 0 failed, 0 skipped, 0 not executed, 0 file errors
```

## Benchmark

MLP 784 → 512 → 512 → 10, relu, softmax cross-entropy, synthetic data, F32.
A step is about `3.2 MFLOP × batch` (13 GFLOP at 4096). Milliseconds per step,
median of 5–200 steps after warm-up, wall clock in Agentica (the FFI and the VM
are inside); samples/s from the median.

### SGD (lr 0.01)

| batch | reference (`Train.fit`) | libtorch via `tensor/` (`Train.fit`) | libtorch C++ (ceiling) | ggml CPU, 8 threads | ggml Vulkan, fp16 `mul_mat` | ggml Vulkan, fp32 |
|---|---|---|---|---|---|---|
| 64 | 3 999 | 41.6 | 3.1–3.9 | 6.5 (4 thr: 7.3) | 4.7–4.9 | 4.5 |
| 256 | 16 524 | 52.0 | 10.7–11.0 | 16.9 (21.1) | 5.6–6.7 | 7.3 |
| 1024 | 68 574 | 104.5 | 41.4–48.0 | 70.6 (69.6) | 14.1–15.9 | 18.9 |
| 4096 | ~275 000 (not run) | **halts** (below) | 144–170 | 351 (335) | **45.4–51.1** | 68.3 |

Samples per second:

| batch | reference | libtorch via `tensor/` | libtorch C++ | ggml CPU | ggml Vulkan fp16 |
|---|---|---|---|---|---|
| 64 | 16 | 1 540 | 16 300–20 600 | 9 800 | 13 000–13 700 |
| 256 | 15 | 4 930 | 23 300–23 900 | 15 100 | 38 100–45 700 |
| 1024 | 15 | 9 800 | 21 300–24 800 | 14 500 | 64 600–72 800 |
| 4096 | — | — | 24 100–28 400 | 11 700 | **80 200–90 300** |

- **The iGPU wins from batch 256 up**: 1.6–2.0× libtorch C++ at 256,
  2.6–3.4× at 1024, 2.8–3.7× at 4096 (260–290 GFLOPS sustained against
  80–90). At 64 (0.2 GFLOP per step) libtorch is 1.2–1.6× faster. That is the
  "≥ ~1 GFLOP per graph" of `../ggml-vulkan/`, now per step instead of per op.
- Against what `Train.fit` does today on libtorch, the compiled iGPU step is
  8.5× faster at 64 and 6.6–7.4× at 1024.
- ggml's own CPU backend is 1.5–2.4× slower than libtorch here (its `mul_mat`,
  as `../ggml-vulkan/` measured). The CPU half of a ggml backend is for
  correctness and CI, not speed.
- fp16 `mul_mat` is worth 1.3× at 4096, 1.2× at 1024, and nothing below.

### Adam (lr 0.001) and SGD as two graphs

| batch | libtorch via `tensor/` Adam | libtorch C++ Adam | ggml CPU Adam | ggml Vulkan Adam | ggml Vulkan SGD, one graph | ggml Vulkan SGD, two graphs |
|---|---|---|---|---|---|---|
| 64 | 150.5 | 8.8 | 11.5 | 8.4 | 4.9 | 7.8 |
| 256 | 160.0 | 18.1 | 21.7 | 9.6 | 6.7 | 8.4 |
| 1024 | 217.0 | 59.2 | 74.0 | 17.7 | 15.9 | 15.9 |
| 4096 | halts | 170.0 | 353.6 | 48.9 | 51.1 | 49.2 |

- Adam in the graph costs 3.5 ms more than SGD at 64 (182 nodes against 104)
  and nothing measurable at 4096.
- The update as a second graph costs one more dispatch round trip: +2.9 ms at
  64, +1.7 ms at 256, lost in the noise from 1024. One graph is the right
  shape; two only matters where the step is small.

### What the VM costs: lowering against running

`bench.agc` prints, per batch, what happens once and what happens per step
(ggml Vulkan, SGD; Adam in parentheses):

| once per shape | ms |
|---|---|
| trace the loss, `Trace.gradients`, trace the optimizer | 0.6–2.0 (0.6–1.6) |
| lower: walk the graph, one `Raw.node` FFI call per node | 1.2–1.9 (2.4–4.1) |
| compile: build the cgraph, allocate, check every op (C) | 0.9–8.9 (1.8–15.5) |
| upload the parameters as `List[Float]` (670 k values) | 16–74 |
| first `run` (Vulkan pipelines, buffers) | 9–137 |

| per step | batch 64 | batch 4096 |
|---|---|---|
| upload the batch (C, host → device) | 0.03 ms | 1.6 ms |
| compute and wait (C) | 4.8–5.2 ms | 45–50 ms |
| everything else: VM, FFI, the loss readback | within noise | within noise |

"Within noise": the C side's mean per step is within 0.3 ms of the
wall-clock median, the FFI call and the VM included. So the VM is out of the
loop: tracing, differentiating and lowering a step costs 2–6 ms once, and a
step afterwards is one FFI call. For contrast,
`Train.fit` on libtorch traces and differentiates again every step (0.3–0.6 ms,
cheap) but then evaluates ~100 ops one FFI call each (20–98 ms) and runs the
optimizer eagerly (17–22 ms SGD, 118–136 ms Adam, see below).

### Where the iGPU's time goes

`GGML_VK_PERF_LOGGER=1 agentica bench.agc vulkan sgd 4096`, one step: 42.8 ms
of GPU time, of which `mul_mat` is 25.9 ms (450–600 GFLOPS on the large
ones), the twelve `cont` of the transposes 6.7 ms, and the elementwise ops
(`add`, `sub`, `mul`, `relu`, `step`) 9 ms. At batch 64 the GPU reports
4.9 ms of work per step, 2.1 ms of it `mul_mat`: a hundred small kernels, not
the 2 ms round trip, are what a small step costs. Headroom left: the transposes of the
weight gradients (16 %) and a fused `relu` backward (`step` + `mul`, 6 %).

## Found on the way

- **`tensor/`'s torch `full` halts above 1 048 576 elements.** `Torch.full`
  builds the values as `List.repeat(x, n)` and uploads them; at 4096 × 512 the
  list is over `List.repeat`'s cap and the run stops with
  `RUNTIME-INVALID-COUNT` instead of a `Result` (it is why the 4096 libtorch
  rows say "halts"). It is also why the eager optimizer is slow: every
  `scale(t, c)` of `train/optim.agc` materializes a full tensor of the
  parameter's shape through the FFI, every step.
- **A per-step value baked into the graph breaks the cache.** `Optim.adam`
  traces its bias correction as `full` constants that change every step; the
  spike's `Step.adam_step` makes them two scalar inputs instead. A production
  compiler can do this for every `full`: the lowering already makes each a
  one-element leaf, so the cache key can ignore fill values and upload them
  per step.
- **Inputs need a shape, not values.** A traced input is `Trace.constant` of a
  tensor; the spike makes shape-only placeholder tensors (`Step.placeholder`)
  so a 4096 × 784 batch never exists as an Agentica list. A
  `Trace.input(shape, dtype)` would say this directly.
- **One C compiler for every native module.** `tensor/tools/build-torch.sh`
  sets `CC=g++`, so this spike's C file is compiled as C++ too (it casts every
  `malloc`), and g++ ignores `C_INCLUDE_PATH`: ggml's headers need
  `CPLUS_INCLUDE_PATH`.

## Recommended production design

What the numbers say: compile the step, never evaluate op by op on Vulkan.

1. **`Device.Vulkan(Int)` and `Backend.Ggml(Device)`** in `tensor/`, with
   `Handle.Ggml(...)` naming the resource of an `extern module "ggml"` inside
   `tensor/`, `optional = true`, as the torch module is (decision (b′)).
   [agentica#1918](https://github.com/agentica-lang/agentica/issues/1918) is
   what later lets the module move out to its own package with an opaque
   handle; nothing below depends on it, and two native backends are the
   point where note.md §3 item 11 (option A, a table of `Fn` per backend)
   starts to pay. Eager `Tensor.*` on `Ggml` stays possible (a one-node graph
   each, fine on the CPU, 2 ms each on Vulkan) so conformance and readback
   work; it is not the training path.
2. **`Trace.evaluate` on a `Ggml` backend compiles.** Instead of applying
   node by node, it lowers the whole graph it is given (this spike's
   `lower.agc` and C layer) and runs it once. A cache keyed by the graph's
   structure (op sequence, arguments, shapes, dtypes, not fill values or
   constant contents) makes the next step's retrace a hit: tracing again
   costs 1–3 ms, which a step of 5–50 ms absorbs, and it keeps `jit` out of
   the API. Constants and fills become per-step inputs; parameters are
   constants whose tensors already live in the backend, so nothing is
   uploaded for them.
3. **`Train.fit` runs the optimizer in the same graph.** When the
   parameters are on `Ggml`, `fit` traces loss, gradients and `Optim.step`
   together, and the compiled program writes the new parameters and moments
   back into their buffers. Only the loss is read back per step.
4. **Data.** A batch enters as bytes or a native dataset resource, not as
   `List[Float]` (the parameter upload here costs 25–100 ns per value).
5. **Numerics.** fp16 `mul_mat` is ggml-Vulkan's default on this device and
   worth 1.3× at batch 4096; conformance needs a per-device tolerance (3e-3)
   or `GGML_VK_DISABLE_F16`. The design has to pick one, explicitly.

Rough PR plan:

1. `tensor/`: `Device.Vulkan`, `Backend.Ggml`, `Handle.Ggml`, the `ggml`
   native module (creation, `from_list`, `to_list`, `full`, readback), the
   submodule and its build script; CPU-only in CI (31 s cold, 2 s with
   ccache), Vulkan local. DESIGN §2, §7 and §10 updated.
2. `tensor/`: the graph compiler (`lower.agc` and `ggmlgraph.c` grown up),
   `evaluate` on `Ggml` through it, the structural cache, fills as per-step
   inputs; the trace parity suite run on `Ggml` CPU, the iGPU run local.
3. `train/`: `fit` on `Ggml` with the update and the moments inside the
   graph, the loss as the only readback.
4. Coverage: `batch_matmul`, `gelu_tanh_approx`, `clamp`, `pow`, `slice`,
   `concat`, `gather` (`get_rows`), `conv1d` (`im2col`), `argmax`, reductions
   over any axes (`permute`): what `nn/`'s transformer layers need on top of
   the MLP.
5. Speed: the weight-gradient transposes, a fused `relu` backward, the fp16
   decision.
6. Outside this plan, now: the `Torch.full` halt and its cost (an issue for
   `tensor/`).

## What this spike does not prove

- **One model.** An MLP with a cross-entropy head; no attention, no conv, no
  layer norm. The op table says which of those ggml has, not that they
  lower.
- **One machine, shared and loaded.** The iGPU numbers moved ~10 % between
  the two runs, the 8-thread CPU ones up to 2×; libtorch ran at its default 4
  threads (8 threads: 3.1 ms at 64, 144 ms at 4096, slower in between).
- **No cache.** The graph is built once per process by hand; the structural
  cache of the design above is not written, so its hit cost is the traced
  1–3 ms measured here, not a measured lookup.
- **Synthetic batches from C.** A real data pipeline into the input buffers
  is not measured; the upload itself (1.6 ms at 4096 × 784) is.
- **The reference at 4096 is extrapolated** from 64, 256 and 1024 (linear in
  the batch, 67 ms per sample); it was not run.

## CI

Nothing was added. `spike/ggml-graph/agentica.toml` sits two levels below what
the workflow scans (`*/agentica.toml`), like `spike/m0` and
`spike/ggml-vulkan`, and no hosted runner has a Vulkan device.
