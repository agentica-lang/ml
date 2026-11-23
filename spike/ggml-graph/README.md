# One traced training step as one ggml graph on the iGPU

**This is a spike, not a package.** Nothing here is wired into `tensor/`'s
`Backend` enum. It answers one question: can a full training step (forward,
backward, optimizer update) that `Trace` records be lowered to **one** ggml
compute graph, built once, and run on the Vega iGPU through ggml's Vulkan
backend, and how fast is it at large batches?

**Answer: yes.** The graph `Trace.gradients` already builds for an MLP
classifier lowers op by op to ggml with no missing op on Vulkan; the graph is
built and allocated once and reused every step; parameters and Adam moments
stay in a backend buffer and are written back inside the same graph. Steps
match the reference backend to 2e-7 on ggml CPU, and the iGPU matches ggml CPU
to 7e-7 at the benchmark's shapes, with F32 `mul_mat`.

The speedup depends on the precision and on how loaded the machine is:

- **F32 `mul_mat` (the default):** against the best plain C++ libtorch run
  on the CPU, the iGPU is 1.2–1.5× faster at batch 256, 1.5–2.2× at 1024 and
  1.8–2.1× at 4096, and 1.4–1.6× slower at 64.
- **fp16 `mul_mat` (opt-in):** 1.5–1.9×, 2.1–2.9× and 2.7–3.2× at the same
  batches. It costs gradient accuracy (below).
- **Against `Train.fit` on libtorch today:** the compiled step is 5.5–9×
  faster.

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
| The batch data | **synthetic**: made in C (`Raw.batches`, an LCG) so a benchmark step crosses no `List`; the small correctness tests upload Agentica lists |
| `mul_mat` precision | **F32 by default** (`backend_vulkan(false)` sets `GGML_VK_DISABLE_F16` before ggml opens the device); fp16 is opt-in and fixed per process (below) |
| The iGPU really ran | **asserted**: `backend_vulkan` fails unless the backend is Vulkan and the device is not a CPU; `ggml_backend_supports_op` is checked for every node at compile |
| `Bool` and integer tensors | **F32 on the ggml side**: masks are 0/1, labels and `arange` are exact integers below 2^24 |
| Production shape (`Backend`, `Handle`, cache, `Train.fit`) | **not built**: the last section proposes it |

## Run it

From `spike/ggml-graph/`:

```sh
git -C ../.. submodule update --init third_party/ggml
JOBS=2 sh ../ggml-vulkan/tools/build-ggml.sh > "$TMPDIR/ggml-build.log"   # CPU + Vulkan trees
eval "$(grep '^export ' "$TMPDIR/ggml-build.log")"     # ggml's include and library folders
eval "$(sh ../../tensor/tools/build-torch.sh)"         # tensor/'s torch module, and CC=g++
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH:$CPLUS_INCLUDE_PATH"   # g++ ignores C_INCLUDE_PATH
agentica install && agentica check && agentica fmt --check && agentica test
agentica check.agc                  # the gaps against the reference (~1 min); `fp16`, `small`
agentica bench.agc vulkan sgd 64 256 1024 4096   # vulkan16 (fp16), cpu, cpuN, torch, reference; two, adam
```

The build script runs cmake every time (fast with a warm cache) and uses every
CPU unless `JOBS` says otherwise.

The libtorch ceiling is a plain C++ program:

```sh
g++ -O2 -std=c++17 tools/bench-torch-step.cpp -o "$TMPDIR/bench-torch-step" \
  -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" -ltorch -ltorch_cpu -lc10
"$TMPDIR/bench-torch-step" 0 sgd                       # 0: libtorch's default (4 threads here)
```

The machine is the one of `../ggml-vulkan/README.md`: Ryzen 5 3550H, Vega 8
iGPU through RADV, `uma: 1`, 12 GiB. The tables come from two runs:

- **Run A:** the machine was shared and at load 1–5. That slows libtorch on
  the CPU much more than the iGPU.
- **Run B:** the machine was idle (load < 1.6).

The iGPU numbers agree between the runs within ~10 %. The CPU numbers differ
by up to 1.6×, which is why every speedup below is given per run.

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
| a write | `ggml_cpy(value, state)`, after every update is computed; a write whose value is another written state (a swap) is refused |

The SGD step is 96 traced nodes and 104 ggml nodes; Adam is 212 and 182 (the
lazy fills and broadcasts disappear, the transposes and reshapes add some).

The emulations hold where a training step uses them, and have limits:

- **`where`** is exact only when both branches are finite: `0 · ±inf` is NaN,
  so a masked `-inf` (an attention mask) poisons the result where the
  reference does not.
- **`eq`** is `1 - step(|a - b|)`, so `eq(NaN, NaN)` gives 1 where IEEE gives
  false. It is exact only for integers below 2^24.
- **`max`** over a row that is all `-inf` gives `-inf`, and the softmax after
  it gives NaN. The reference backend does the same thing.
- **Dtypes:** the lowering refuses `I64`, `F64` and `F16` nodes, a `to_dtype`
  from an integer to `Bool` or between integer widths, and an integer `div`,
  instead of computing them in F32. The labels of the spike are `I32`.

The comparisons fail on any NaN or infinity (`Compare.rel`, `Raw.gap`), so a
step that hits one of these limits fails instead of passing.

Not lowered: the lowering refuses each with an error naming the op (`test/`
checks it for `gelu_tanh_approx`): `pow`, `clamp`,
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
`where` through `mul`), not missing kernels.

Two dead ends, measured once and removed:

- **`out_prod`** exists on Vulkan and would avoid two transposes per weight
  gradient, but its shader is slow here: 225 ms per step at batch 4096
  against 51 ms with fp16 `mul_mat`.
- **ggml's per-op precision** does not give F32 `mul_mat` on Vulkan.
  `ggml_prec_set_acc(t, GGML_PREC_F32)` changes nothing: the inputs are still
  converted to fp16, and the gaps were identical.
  `ggml_prec_set_src(t, GGML_PREC_F32, 1)` makes the backend refuse the op.

The only switch is `GGML_VK_DISABLE_F16`, which ggml reads when it first opens
the device. So the precision is a property of the process: the first
`backend_vulkan(fp16)` sets it, and a later call that asks for the other
precision is an error. The choice overrides any `GGML_VK_DISABLE_F16` the
user set, and it is fixed for the whole process.

## Correctness

`agentica check.agc` (and `check.agc fp16`) prints every gap below, and
`test/ggml_graph_test.agc` asserts them. A gap is
`max |ggml - other| / max |other|` per tensor, the worst tensor shown. A NaN
or infinity on either side is an error, not a gap.

The small model is an MLP of 8 → 16 → 16 → 4 with a batch of 6. ggml takes
the step, and the step `Train.fit` runs takes it on the reference backend:

| backend | step | loss | gradients | new parameters |
|---|---|---|---|---|
| ggml CPU | SGD, one graph | 5.5e-8 | 1.6e-7 | 7.4e-8 |
| ggml CPU | SGD, two graphs | 5.5e-8 | 1.6e-7 | 7.4e-8 |
| ggml CPU | Adam, two steps, one graph | 5.5e-8 | 2.3e-7 | 9.6e-8 |
| ggml Vulkan, F32 `mul_mat` | SGD, one graph | 5.5e-8 | 1.6e-7 | 7.4e-8 |
| ggml Vulkan, F32 `mul_mat` | Adam, two steps | 5.8e-8 | 1.2e-7 | 9.6e-8 |
| ggml Vulkan, fp16 `mul_mat` | SGD, one graph | 5.5e-8 | 4.1e-4 | 6.8e-6 |
| ggml Vulkan, fp16 `mul_mat` | Adam, two steps | 5.8e-8 | 3.5e-4 | 7.9e-5 |

The benchmark's model is 784 → 512 → 512 → 10 with a batch of 256, and takes
two SGD steps:

| backend | against | loss | gradients | new parameters |
|---|---|---|---|---|
| ggml CPU | reference backend | 4.2e-8 | 2.0e-7 | 1.5e-7 |
| ggml Vulkan, F32 `mul_mat` | reference backend | 4.2e-8 | 6.0e-7 | 1.5e-7 |
| ggml Vulkan, fp16 `mul_mat` | reference backend | 8.4e-7 | **1.1e-1** | 4.5e-5 |
| ggml Vulkan, F32 `mul_mat` | ggml CPU, data made in C | 0 | 6.9e-7 | 1.0e-7 |
| ggml Vulkan, fp16 `mul_mat` | ggml CPU, data made in C | 9.3e-7 | 9.6e-2 | 5.7e-5 |

**fp16 `mul_mat` gets one gradient tensor wrong by 10 %** at these shapes.
The parameters still move by the right amount to 5e-5, because that tensor's
values are small next to the others the update applies. This is why F32 is
the default and fp16 is opt-in.

The reference comparison at the benchmark's shapes is too slow for a test.
One reference step there takes 16 s and about 10⁹ VM steps, and the test
runner allows 30 s and a step budget per test. So `check.agc` makes it. The
test compares the iGPU with ggml CPU instead: it makes the parameters and the
batch in C (`Raw.fill`, `Raw.batches`) and compares the two in C (`Raw.gap`).
ggml CPU is the backend `check.agc` ties to the reference at 2e-7.

The tests assert:

- 1e-5 on the CPU, and 1.5e-6 on the iGPU with F32 (about twice the largest
  gap measured);
- twenty runs of one compiled program lower the loss;
- a swap of two states is refused, and a NaN or ±inf is an error;
- an op with no lowering is an error naming it.

Without the module or a Vulkan device, a test passes and says why in its
assertion message. The runner prints only PASS (note.md §5), so that reason
is in the source, not in the output.

```console
$ agentica test
PASS .../ggml_graph_test.agc / sgd one graph matches the reference on the cpu
PASS .../ggml_graph_test.agc / sgd two graphs match the reference on the cpu
PASS .../ggml_graph_test.agc / two adam steps match the reference on the cpu
PASS .../ggml_graph_test.agc / the igpu step matches the reference
PASS .../ggml_graph_test.agc / two steps of the benchmark model agree between cpu and igpu
PASS .../ggml_graph_test.agc / a write that reads a replaced state is refused
PASS .../ggml_graph_test.agc / a nan or an infinity is an error not a gap
PASS .../ggml_graph_test.agc / the c gap refuses a nan and a size mismatch
PASS .../ggml_graph_test.agc / the graph is built once and reused
PASS .../ggml_graph_test.agc / an op with no lowering is an error naming it
10 passed, 0 failed, 0 skipped, 0 not executed, 0 file errors
```

`bench.agc` prints the loss of the first and the last step of every row, so
a fast step that computes garbage shows. At batch 4096 the SGD loss goes from
2.3066 to 2.3051 on random labels; with Adam on four batches it goes from
2.3066 to 0.70 (memorized). At batch 64 the iGPU's last SGD loss is
2.188410 with F32, 2.188390 with fp16 and 2.188411 on ggml CPU.

## Benchmark

MLP 784 → 512 → 512 → 10, relu, softmax cross-entropy, synthetic data, F32.
A step is about `3.2 MFLOP × batch` (13 GFLOP at 4096). Milliseconds per step,
median of 5–200 steps after warm-up, wall clock in Agentica (the FFI and the VM
are inside); samples/s from the median.

### SGD (lr 0.01)

Run B (idle machine), ms per step:

| batch | libtorch C++, 4 threads (default) | libtorch C++, 8 threads | ggml CPU, 8 threads | **ggml Vulkan, F32** | ggml Vulkan, fp16 | F32 against the best libtorch | fp16 against the best libtorch |
|---|---|---|---|---|---|---|---|
| 64 | 2.96 | 3.10 | 6.46 | 4.60 | 4.32 | 0.64× (loses) | 0.69× (loses) |
| 256 | 9.22 | 8.43 | — | 7.19 | 5.50 | 1.17× | 1.53× |
| 1024 | 30.44 | 28.77 | — | 18.89 | 13.46 | 1.52× | 2.14× |
| 4096 | 125.9 | 119.4 | 288.0 | 67.92 | 44.78 | 1.76× | 2.67× |

Run A (machine at load 1–5), ms per step. libtorch at 4 threads ran twice in
this run:

| batch | reference (`Train.fit`) | libtorch via `tensor/` (`Train.fit`, 4 threads) | libtorch C++, 4 threads | libtorch C++, 8 threads | ggml CPU, 8 threads | **ggml Vulkan, F32** | ggml Vulkan, fp16 | F32 against the best libtorch | fp16 against the best libtorch |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 3 999 | 41.6 | 3.88–3.92 | 3.11 | 6.5 | 4.46 | 4.6–4.9 | 0.70× (loses) | 0.63–0.68× (loses) |
| 256 | 16 524 | 52.0 | 10.7–11.0 | 13.4 | 16.9 | 7.28 | 5.6–6.7 | 1.47× | 1.6–1.9× |
| 1024 | 68 574 | 104.5 | 41.4–48.0 | 69.6 | 70.6 | 18.92 | 14.1–15.9 | 2.19× | 2.6–2.9× |
| 4096 | ~275 000 (not run) | **halts** (below) | 149–170 | 144.1 | 351 | 68.33 | 45.4–51.1 | 2.11× | 2.8–3.2× |

"The best libtorch" is the fastest libtorch C++ row of that run at that
batch, so the comparison never takes the slow side of the noise. Samples per
second on the iGPU with F32: 14k, 35–36k, 53–54k and 59–60k at 64, 256, 1024
and 4096. With fp16: 15k, 38–46k, 65–76k and 80–91k. libtorch C++ does
21–34k idle and 16–28k loaded.

- **F32 (the default):** the iGPU wins from batch 256. It is 1.2–1.5× faster
  at 256, 1.5–2.2× at 1024 and 1.8–2.1× at 4096 against the best libtorch of
  each run (up to 2.5× against libtorch's default 4 threads in run A). It
  loses 1.4–1.6× at 64 (0.2 GFLOP per step). That is the "≥ ~1 GFLOP per
  graph" of `../ggml-vulkan/`, now per step instead of per op.
- **fp16 (opt-in):** 1.5–1.9×, 2.1–2.9× and 2.7–3.2× at the same batches, at
  the gradient error above. Against F32 it is worth 1.3–1.5× at 4096,
  1.2–1.4× at 1024 and 1.1–1.3× at 256. At 64 it is within the noise (one run
  faster, one slower).
- **Against `Train.fit` on libtorch today**, the compiled iGPU step with F32
  is 8.5–9× faster at 64 and 5.5× at 1024.
- **ggml's own CPU backend** is 1.5–2.4× slower than libtorch here (its
  `mul_mat`, as `../ggml-vulkan/` measured). The CPU half of a ggml backend is
  for correctness and CI, not speed.

### Adam (lr 0.001) and SGD as two graphs

Run B (idle), ms per step, ggml Vulkan with F32:

| batch | libtorch C++ Adam, 4 threads | ggml Vulkan Adam | Adam: F32 against libtorch | ggml Vulkan SGD, one graph | ggml Vulkan SGD, two graphs |
|---|---|---|---|---|---|
| 64 | 6.63 | 8.41 | 0.79× | 4.60 | 7.48 |
| 256 | 11.82 | 11.21 | 1.05× | 7.19 | 8.65 |
| 1024 | 34.80 | 22.98 | 1.51× | 18.89 | 20.32 |
| 4096 | 132.1 | 71.59 | 1.84× | 67.92 | 69.26 |

In run A, libtorch through `tensor/` ran Adam at 150.5, 160.0 and 217.0 ms
(64, 256, 1024) and halted at 4096. ggml CPU ran Adam at 11.5, 21.7, 74.0
and 353.6 ms.

- Adam in the graph costs 3.8 ms more than SGD at 64 (182 nodes against 104),
  and 4 ms at 1024–4096.
- The update as a second graph costs one more dispatch round trip: +2.9 ms
  at 64, +1.5 ms at 256, +1.3–1.4 ms from 1024 on. One graph is the right
  shape.

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

| per step (fp16 run A; F32 is slower in compute only) | batch 64 | batch 4096 |
|---|---|---|
| upload the batch (C, host → device) | 0.03 ms | 1.6 ms |
| compute and wait (C) | 4.8–5.2 ms | 45–50 ms |
| everything else: VM, FFI, the loss readback | within noise | within noise |

"Within noise": the C side's mean per step is within 0.3 ms of the
wall-clock median, the FFI call and the VM included. So the VM is out of the
loop: tracing, differentiating and lowering a step costs 2–6 ms once, and a
step afterwards is one FFI call.

For contrast, `Train.fit` on libtorch traces and differentiates again every
step:

- tracing the loss and `Trace.gradients`: 0.3–0.6 ms, cheap;
- evaluating ~100 ops, one FFI call each: 20–98 ms;
- the eager optimizer: 17–22 ms for SGD, 118–136 ms for Adam (see below).

### Where the iGPU's time goes

`GGML_VK_PERF_LOGGER=1 agentica bench.agc vulkan16 sgd 4096` (fp16), one step: 42.8 ms
of GPU time, of which `mul_mat` is 25.9 ms (450–600 GFLOPS on the large
ones), the twelve `cont` of the transposes 6.7 ms, and the elementwise ops
(`add`, `sub`, `mul`, `relu`, `step`) 9 ms. At batch 64 the GPU reports
4.9 ms of work per step, 2.1 ms of it `mul_mat`: a hundred small kernels, not
the 2 ms round trip, are what a small step costs. Headroom left: the
transposes of the weight gradients (16 %) and a fused `relu` backward
(`step` + `mul`, 6 %).

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
   handle. Nothing below depends on it, and two native backends are the point
   where note.md §3 item 11 (option A, a table of `Fn` per backend) starts to
   pay.

   Eager `Tensor.*` on `Ggml` stays possible, so conformance and readback
   work. Each op is then a one-node graph: fine on the CPU, 2 ms on Vulkan.
   It is not the training path.
2. **A compiled step, behind `fit`.** `Train.fit` traces the loss, its
   gradients and `Optim.step` together, lowers the whole graph once (this
   spike's `lower.agc` and C layer), and runs it once per batch. A cache keyed
   by the graph's structure makes the next step's retrace a hit. The key covers
   the ops, arguments, shapes and dtypes, but not fill values or constant
   contents.

   Retracing the loss and its gradients costs 0.3–0.6 ms, and 0.6–2.0 ms with
   the optimizer traced too. The walk that makes the key costs about what
   `lower`'s walk does without its FFI calls, and was not measured. A step of
   5–70 ms absorbs this, and it keeps `jit` out of the API. Fills become
   per-step inputs, so `Optim.adam`'s changing bias correction needs no
   special form.
3. **State semantics: `fit` owns a train state, and nobody donates.**
   - **The train state:** `fit` uploads the parameters once into a train
     state it creates and never returns: parameter and moment buffers on the
     backend. The compiled graph writes into these buffers; nothing else can
     see them.
   - **The result:** at the end `fit` returns fresh, immutable tensors read
     from that state.
   - **How it fits DESIGN §2/§8/§10:**
     - No tensor a user holds is ever mutated.
     - `Optim.step` keeps returning new tensors in its traced form.
     - "The new value of state *i* replaces state *i*" is a fact about
       buffers `fit` owns, not about `Tensor` values.
   - **Telling a state from an input:** `fit` says which traced constant is
     a state, which is an input, and which value writes which state, as
     `Lower.Step` does here. `Trace.evaluate` is not asked to guess: on
     `Ggml` it may compile, but it stays pure, with every constant an input
     and no writes.
   - **Donation (rejected):** passing parameters in and getting them back
     while the old handles go dead would make a `Tensor` the user still holds
     stale (`RUNTIME-NATIVE-CLOSED`). That breaks §2's "Tensor values are
     immutable".
   - **The contract change, named:** DESIGN §10.3's "`evaluate` runs a graph
     through the public `Tensor.*` calls" gets an exception for a backend
     that compiles: the values must agree, but the path does not go through
     `Tensor.*`. That sentence and §7's dispatch both change in PR 2.
4. **One program, not shared pointers.** Production keeps one program per
   compiled step, or shares buffers through views of one context. This
   spike's `share()` instead patches another program's data pointers into a
   leaf, and needs `Lower.Built` to keep that program open. That is a
   spike-only shortcut.
5. **Data.** A batch enters as bytes or a native dataset resource, not as
   `List[Float]`: the parameter upload here costs 25–100 ns per value. Shapes
   are static, so a ragged last batch needs a policy: `drop_last`, or padding
   with a mask, or a second compiled graph for the tail shape. The first two
   keep one graph.
6. **Numerics.** F32 `mul_mat` by default; fp16 opt-in, per process, for
   1.1–1.5× and a 10 % gradient error on one tensor here. Conformance then
   needs one tolerance for F32 (1e-5) and a documented exception for fp16.

Rough PR plan:

1. **`tensor/` types and module:** `Device.Vulkan`, `Backend.Ggml`,
   `Handle.Ggml`, and the `ggml` native module (creation, `from_list`,
   `to_list`, `full`, readback), with the submodule and its build script.
   - CI is CPU-only (31 s cold, 2 s with ccache), Vulkan local. That needs
     the module's Vulkan code behind `#ifdef GGML_USE_VULKAN` and a libs list
     without `ggml-vulkan` and `vulkan` for the CPU build. The `[native]`
     table has one `libs`, so either two module names or a build script that
     writes the flags.
   - DESIGN §2 and §7 updated.
2. **`tensor/` compiler:** the graph compiler (`lower.agc` and `ggmlgraph.c`
   grown up), the structural cache, and fills as per-step inputs.
   - The trace parity suite runs on `Ggml` CPU; the iGPU run stays local.
   - DESIGN §10.3 changes as described in item 3 above.
3. **`train/`:** `fit` on `Ggml` with the train state of item 3 above. The
   update and the moments run inside the graph, the loss is the only
   readback, and the last-batch policy is decided.
4. **Coverage:** `batch_matmul`, `gelu_tanh_approx`, `clamp`, `pow`,
   `slice`, `concat`, `gather` (`get_rows`), `conv1d` (`im2col`), `argmax`,
   and reductions over any axes (`permute`): what `nn/`'s transformer layers
   need on top of the MLP. Each comes with `where`'s non-finite rule (a select,
   not a multiply) once masks can hold `-inf`.
5. **Speed:** the weight-gradient transposes and a fused `relu` backward.
6. **Outside this plan, now:** the `Torch.full` halt and its cost (an issue
   for `tensor/`).

## What this spike does not prove

- **One model.** An MLP with a cross-entropy head; no attention, no conv, no
  layer norm. The op table says which of those ggml has, not that they
  lower.
- **One machine, two runs.** Run A was loaded, run B idle. The iGPU numbers
  agree within ~10 %, the CPU ones by up to 1.6×, so every speedup is given
  per run.
- **No cache.** The graph is built once per process by hand. The structural
  cache of the design above is not written: its hit cost is the 0.6–2.0 ms
  retrace plus an unmeasured key walk, not a measured lookup.
- **Synthetic batches from C.** A real data pipeline into the input buffers
  is not measured; the upload itself (1.6 ms at 4096 × 784) is.
- **The reference at 4096 is extrapolated** from 64, 256 and 1024 (linear in
  the batch, 67 ms per sample); it was not run.

## CI

Nothing was added. `spike/ggml-graph/agentica.toml` sits two levels below what
the workflow scans (`*/agentica.toml`), like `spike/m0` and
`spike/ggml-vulkan`, and no hosted runner has a Vulkan device.
