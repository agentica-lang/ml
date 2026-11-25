# tensor — design

The contract of `tensor/`: the types, the backend interface, the traced graph
and its gradients. Section numbers are cited from the code, so they keep their
gaps.

References: [ml#1](https://github.com/agentica-lang/ml/issues/1) (goals and
the grad-in-Agentica decision),
[agentica#1091](https://github.com/agentica-lang/agentica/issues/1091)
(native arenas; see §8), and
[agentica#1918](https://github.com/agentica-lang/agentica/issues/1918)
(opaque native resources that can live in data and cross modules; see §2 and §7).

## 1. Goals and scope

- **Training and inference.** `tensor/` and `nn/` (forward) run over
  `reference` (the pure Agentica oracle) and libtorch (eager); `train/`
  (losses, SGD/Adam, a training loop) and the traced
  `grad`/`value_and_grad` it needs sit on the same primitives (§10).
- Every backend is **forward-only**. `grad` lives in Agentica over a traced
  graph (ml#1 key decision, as Nx does), so per-backend autograd is out.
- The backend interface is small, and one shared conformance suite checks
  every backend against the reference (§11).
- **`tensor/` declares the native module of every backend it dispatches to**
  (`torch`, and `ggml` for `Backend.Ggml`). A `Tensor` carries the backend's
  handle, and a native handle is a
  `resource`: resources are nominal to the `extern module` block that declares
  them and cannot cross into another module, so `Handle` (§2) can only name a
  handle type declared in this package, and the `Torch` arm of §7 can only
  call a module declared here. The module is `optional` in `agentica.toml`, so
  a machine without libtorch still checks, formats and tests with the arm
  answering `error("backend not available")`; `Tensor.available/1` answers the
  same question before a call does. The cost is that `tensor/` ships the thin
  layer of libtorch and its build table.
  [agentica#1918](https://github.com/agentica-lang/agentica/issues/1918) is the
  change that lets a handle be opaque, at which point the backend code moves
  back out into its own package and this bullet goes away.

## 2. Types

```text
Dtype   = F32 | F16 | F64 | I8 | I16 | I32 | I64 | U8 | Bool | Q8_0   # Q8_0: a quantized weight of ggml only (§7)
Device  = Cpu | Vulkan(Int)                                    # enum; Vulkan(i): ggml's device i
Shape   = List[Int]                                            # static ints, row-major C-order
Values  = Float(Float) | Int(Int) | Bool(Bool)                 # enum, dtype-tagged
Backend = Reference | Torch(Device) | Trace | Ggml(Device)     # enum, see §7, §10
Handle  = Native(TensorHandle) | Plain(Data) | Traced(Node)
        | Ggml(ggml TensorHandle)                              # enum, see §2, §8, §10
Tensor  = { shape: Shape, dtype: Dtype, backend: Backend, handle: Handle }
```

- `Device` names where a backend computes: `Cpu`, or `Vulkan(i)`, the Vulkan
  device `i` in ggml's numbering. `Backend.Torch(Device)`, `Backend.Ggml(Device)`
  and every backend function take the device explicitly (§4). `Torch` and
  `Reference` serve `Cpu` only and answer `error` for `Vulkan`; `Ggml` serves
  both.
- `Tensor` values are **immutable**; every op returns a new `Tensor`.
  A training step returns new parameters.
- `handle` is `Native(TensorHandle)` on the libtorch backend (a
  `struct NAME_state *` + thread-safe `NAME_FREE`, C data only, never terms;
  valid only during the call, never stored), `Ggml(TensorHandle)` on the ggml
  backend (the resource of `extern module "ggml"` in `src/ggml_native.agc`: a
  ggml context and the backend buffer of one tensor on its device), and
  `Plain(Data)` on `reference`, where `Data` is plain Agentica data. `TensorHandle` is the resource the
  `extern module "torch"` block of `src/native.agc` declares: a resource is
  nominal to its block, so the handle type has to be the concrete one and it
  has to live in this package (§7). A backend that only reads a handle it did
  not make answers `error`, never a stopped task. This follows
  the repo's FFI rules (`extern module` is the only source of signatures;
  native-gen regenerates the header; blocking work is `compute def` on the
  compute pool; `List`/`Map` cross via cursors/builders).
- **Dtype rule:** numeric compute is `F32`; `F16`
  and `F64` weights are converted to it on load (§6, lossy for `F64`). Two
  kinds of non-`F32` compute tensors are required by the op set itself: `Bool`
  masks (`eq`, `lt`, the condition of `where`) and integer index tensors
  (`gather` indices, `argmax` output). The other table entries are checkpoint
  dtypes only (§6), kept as indices or masks at runtime; the table grows only
  when a consumer requirement (§5) or conformance demands it.
- No operator overloading: `Tensor.add(a, b)` plus pipes
  (`t |> Tensor.add(other) |> Tensor.relu()`).

## 3. Primitives vs derived ops (the split)

**Primitives** are the only functions a backend implements. Everything else
is **derived**: pure Agentica code over primitives, shared by all backends
and therefore tested once.

### 3.1 Primitives (the backend contract, 49 ops)

Creation: `zeros, ones, empty, full, like, arange, scalar_to_tensor`.
Elementwise: `add, sub, mul, div, neg, exp, log, sqrt, pow, relu, sigmoid,
tanh, gelu_tanh_approx, eq, lt, where, clamp`.
Every transcendental here refuses `NaN` and infinite input, except that `exp(-inf)` is `0`
(a masked score).
Reductions: `sum, mean, max, argmax` (each with `axes: Optional[List[Int]]`,
`keep_dims: Bool`).
Linear algebra: `matmul, transpose, batch_matmul` (plain `dot` is derived).
Attention: `permute, attention, rope`. `permute(t, axes)` is the general axis
permutation (`transpose` reverses every axis). `attention(q, k, v, mask?,
scale?)` is `softmax(scale q kᵀ + mask) v` over `[batch, heads, q_len, dim]`,
with grouped-query attention when the kv heads divide the heads; the mask is
additive (finite or `-inf`; `q`, `k`, `v` and the mask are never `NaN`, and a query whose
every key is `-inf` is an error on every backend), and `causal: true` is the
wrapper's, which builds the causal mask from `arange` and `lt` (it needs no more
queries than keys). `rope(x, positions, base, dims, neox, inverse)` is the
rotary embedding of `x [..., seq, width]`. They are primitives so that ggml and
libtorch can use their fused kernels, and so that the graph has one node where
the per-head loop had a hundred; each has a backward rule over the primitives
(§10.3).
Shape: `reshape, squeeze, unsqueeze, slice, concat, stack, broadcast_to,
gather, scatter_add` (embedding lookup and beam gather need `gather`, and its
backward rule needs `scatter_add(indices, updates, rows)`, which adds the rows
of `updates` into `rows` zero rows at `indices`; everything fancier is
derived).
`set_rows(cache, rows, positions)` is the one op that writes: a copy of `cache`
with its rows at the (distinct) `positions` replaced, on every backend but
ggml. In a function run on ggml (`Trace.run`) it writes the cache where it
lies, which makes the cache an input that no other node reads, the result an
output, and the output the very tensor given: the tensor is consumed, even by
a run that fails (a position out of range is found after the write, and is
clamped into the cache), and it is not checked for finite values. The cache
must be `F32`. `Trace.eval` writes a copy and leaves the cache alone. The
check is at lowering, naming the nodes, so a KV cache is not copied at every
step of a generation. It has no gradient.
`top_ids(x, k)` (the ids of the k largest of a float vector, I32 `[k]`) is a
node `ggml_top_k` runs on the device, so a decoder can read back k logits and
their ids and not the whole vocabulary; the ids are in no order on ggml, and
the derived `top_k` stays on the host. It has no gradient.
Normalization support: `rsqrt, mean_var` are **not** primitives. `layer_norm`
and `rms_norm` are derived (§3.2) on every backend that computes, but a trace
records each as one node, `layer_norm` or `rms_norm`, so that ggml can run it
as one norm (a trace of a transformer step holds a dozen of them, each 10
nodes composed); the gradient has a rule of its own, checked against a central
difference.
The same holds for the update of Adam: `adam_step(p, g, m, v, hyper)` is derived
(`adam_composed`) and a trace of `F32` tensors of rank 3 or less records two
kinds of node, `AdamStep` (whose recorded shape is the three new tensors
stacked, which only the trace and the reference hold) and one `AdamPart` per
new tensor. ggml runs `AdamStep` as `ggml_opt_step_adamw` (`hyper` is that op's 7
numbers), which updates its arguments in place and asserts that its weight is
a parameter: the lowering works on a copy of the weight and marks it by hand (a
copy is not a leaf, which `ggml_set_param` needs; nothing else reads the flag,
no backward pass is built). The weight is always a copy, since the loss and the
gradients read it. The moments are copies too, each a tensor of its own (the
Vulkan backend binds each as a buffer and aborts on a misaligned offset, so they
are not slices of one), except in a function compiled by `Trace.compile` when
both are inputs that no other node reads, not outputs, and the two parts for
them are outputs (`adam_donors`): then they are updated where they lie, and
their results are the input tensors (`Trace.donated`), consumed like the cache
of a `set_rows`, even by a run that fails. The first moment is not checked for
finite values (a non-finite one reaches the new weight through the update,
which is checked); the second is, since an infinite one leaves the weight finite
and frozen. `Trace.eval` and `eval_many` never donate. It has no gradient (a
gradient through the Adam update of `Optim.step` is an error). A StableHLO
module has no node for it: `Trace.stablehlo` writes the composition of ops that
`adam_step` is on the other backends.
Convolution/pooling: `conv1d, avg_pool1d` (stride and padding; no dilation or
groups).
Random: none (dropout is identity at inference).
Movement: `to_dtype, to_device, copy, contiguous`.

Deliberately missing: `softmax`, attention, `silu`, beam search — all
derived (§3.2). `layer_norm` and `rms_norm` are derived too, and a trace
keeps each as one node. The exact (`erf`) GELU is `gelu_erf`, outside this
set; `Trace` records it as one node and differentiates it.

### 3.2 Derived ops (pure Agentica, backend-agnostic)

Built only from §3.1, in `tensor/` (generic) or `nn/` (model-shaped):

- `softmax(x, axis) = div(exp(sub(x, max)), sum(...))` with max-subtraction.
- `layer_norm(x)`, `rms_norm(x)` from `mean`, `pow`, `sqrt` and `div`
  compositions over §3.1 (no `var`/`rsqrt` primitive is needed).
- `gelu` is the user-facing name of `gelu_tanh_approx` (§3.1); the `erf` form
  cannot be expressed over §3.1, so `gelu_erf` is an op of its own: the
  reference sums a series for `erf` (to about 1e-14), libtorch calls
  `torch::gelu`. `silu` derives from `sigmoid` and `mul`.
- `dot(a, b)` as `reshape` + `matmul`; `linear(x, w, b?, out_in)`, `embedding(ids,
  table)` via `matmul`/`gather`. With `out_in`, `w` is `[out, in]` and the
  product is `x wᵀ`: libtorch's `linear` reads `w` as it is, and the trace
  records a transpose that ggml's lowering folds into `mul_mat`.
- `multi_head_attention(q, k, v, heads, mask?)` over `[batch, seq, width]`: a
  wrapper over `attention`. The width splits into heads by `reshape` +
  `permute`, and the answer joins back the same way. A mask that is not
  `[q_len, k_len]` is widened to `[batch, heads, q_len, k_len]`.
- `greedy_step(logits, temperature?)`: `div` by temperature, `softmax`,
  `argmax`. `beam_search` is an Agentica loop over `gather` + `top_k`, and
  `top_k` is a derived sort/select loop.
- `dropout(x, p)` at inference is identity (kept for `nn/` API symmetry).

An op that is neither in §3.1 nor expressible above at acceptable cost enters
§3.1 only through §5, the consumer checklist.

## 4. API shape

```text
Tensor.zeros(shape, dtype, backend) -> Result[Tensor]
Tensor.from_list(data: List[Values], shape, dtype, backend)
                                                  -> Result[Tensor]
Tensor.from_file(path, offset, shape, format, backend, dtype)
                                                  -> Result[Tensor]   # F32/F16/BF16/Q8_0 bytes of a file, read in C
Tensor.to_list(t) -> Result[List[Values]]         # host readback, tests + decode loop
Tensor.append_file(t, path) -> Result[Void]       # the F32 values appended to a file, written in C
Tensor.shape(t) -> Shape                          # no native call
Tensor.dtype(t) -> Dtype                          # no native call
Tensor.backend(t) -> Backend                      # no native call
```

- `convert_file` is how weights are written (`hub/` writes GGUF with it):
  the same chunked, threaded read as `from_file` (`ff_each`), each chunk
  rounded to the target format and written with `pwrite` at its place in an
  existing file, so a checkpoint converts with no tensor and no list in
  memory. Same format on both sides is a byte copy (nothing is rounded, a NaN
  keeps its payload). It lives in the ggml module (`native/ggml.c`), the one
  that compiles `float_file.h` for conversion; F16 refuses a value beyond
  65504 as the half load does.
- `append_file` is how a tensor in memory is written (`hub/` writes
  safetensors with it): the backend copies the `F32` values to host bytes and
  appends them to a file that exists (`ff_append`), so no value crosses as a
  list. Appending needs no offset, and it is the one positional write the
  reference can do in Agentica (`File.open(path, "append")`).
- `from_file` is how weights load: the backend reads the byte range
  itself (`native/float_file.h`: `pread` into the tensor's own memory, F16
  and BF16 widened in chunks) and no value crosses as a list. The tensor is
  `F32`, the compute dtype (§2), or with `dtype: F16` a half weight or with
  `dtype: Q8_0` a quantized weight (ggml only, §7). A `format: Q8_0` range
  (ggml's blocks of 32 values: a binary16 scale and 32 signed bytes, 34
  bytes) loads as `F32` on every backend, each value the scale times the
  byte, as ggml dequantizes it; `convert_file` does not take it; a short file, a bad offset or a missing file
  is an `error`, checked in C against the size of the file. `Reference` reads
  the bytes in Agentica (`File.seek` and `File.read_bytes`, `src/file_range.agc`)
  and decodes them there, which is slow on a large file. ggml takes it through
  one host copy on Vulkan.
- The ggml CPU backend runs on the physical cores (`native/ggml.c`: the distinct
  `thread_siblings_list` of the CPUs the process may run on on Linux,
  `hw.physicalcpu` on macOS, else the CPUs the process may run on or the online
  ones; at most ggml's 512 threads). ggml's threads spin while
  they wait, so a second thread on a core only adds stalls when the machine is
  busy. `TENSOR_THREADS=n` (a positive integer) sets it; `Lower.cpu_threads()`
  says what a process got.
- `TENSOR_LOAD=agentica` (an environment variable) makes the native
  backends take the Agentica read too: Agentica reads the range and passes
  the `Bytes` to the native `from_bytes`, which reads them in place
  (`an_get_bytes`, no copy on entry) and widens them into the backend buffer
  with the same code as `pread`'s. It is for small models, where the read in
  Agentica costs little: it is slower and peaks higher than the native `pread`
  (whisper-tiny on libtorch: 0.5-0.6 s and 0.54 GB, against 0.33-0.36 s and
  0.42-0.49 GB). The default stays the native `pread`.
- Every fallible op returns `Result`; shape/dtype mismatches are `error`,
  never a halt. Introspection is free (fields, no FFI round-trip).
- Device is explicit, never a global: dispatch is **by value** (`t.backend`),
  not by global config, so `reference` (oracle) and `torch` can check each
  other in the same process. A backend value carries its device
  (`Backend.Torch(Device.Cpu)`);
  `Backend.Reference` has none. Batch lives in the shape, not in a global
  either. The one process-wide setting is `Tensor.vulkan_fp16` (on by default),
  because ggml fixes it per process (§7).
- Backend-module functions take the device as an explicit first argument
  (`Torch.matmul(device, a, b)`): dispatch unpacks it from the backend value
  (§7) and the tensor keeps no second copy of it.

## 5. Requirements the API must satisfy (consumer checklist)

The clients of `tensor/` (`asr/`, `hub/`-loaded checkpoints) bound what
§3.1–§4 must support. A pre-trained transformer needs:

- Positional information via `arange + gather` or `full + add` (no dedicated
  primitive).
- N blocks of: `layer_norm` → multi-head self-attention (§3.2) → `layer_norm`
  → MLP (`linear`, `gelu`, `linear`).
- A 1-D convolution stem, which is why `conv1d`/`avg_pool1d` are primitives
  (§5 is the gate, not a minimum for every client).
- Autoregressive decode: masked self-attention and cross-attention, output
  head `linear` → `softmax` → `argmax` (greedy) with optional temperature;
  beam search is the Agentica loop of §3.2. The decode loop runs in Agentica;
  the backend never holds it.
- Weights come from a checkpoint (§6). Training (`train/`, §10) runs on the
  same primitives.

A client proves these are met by reproducing, on `reference`, the output its
reference implementation gives for the same input.

## 6. Checkpoint requirements (weights in, no code execution)

- **safetensors** is the checkpoint format (8-byte LE length, UTF-8 JSON
  header with `dtype`, `shape`, `data_offsets` plus optional `__metadata__`,
  contiguous little-endian C-order buffer, no strides). No code runs at load;
  that property is the reason for choosing it.
- Numeric tensors are converted to `F32` on load (lossy for `F64`), while
  integer index tensors and `Bool` masks keep their dtype because the op set
  needs them (§2).
- The reader and the sharded index are `hub/` (`hub/DESIGN.md`). `gguf`
  checkpoints stay with the whisper.cpp binding.

## 7. Backend dispatch

A `Backend` **enum value is carried by each `Tensor`**, with dispatch
by `case` to one backend module per variant, each sharing the §3.1 signature
set:

```text
r: Result[Tensor] = case a.backend
  Backend.Reference -> Reference.matmul(a, b)              # src/reference.agc
  Backend.Torch(device) -> Torch.matmul(device, a, b)      # src/torch.agc
  Backend.Trace -> ok(Graph.record(Op.Matmul, [a, b], shape, a.dtype))   # §10
  Backend.Ggml(_) -> Ggml.compiled_only("matmul")          # src/ggml.agc
end
```

- The `case` is checked and exhaustive over all four variants, which is the
  property this shape buys: two compute backends (`Reference`, `Torch`), the
  `Trace` recorder (it records a node instead of computing, §10.1), and `Ggml`.
- The `Torch` arm calls `src/torch.agc`, which calls the `compute
  def`s of `extern module "torch"` (`src/native.agc`). Both files live in this
  package, and they have to: the arm needs the module's signatures, and the
  arm's *argument* is a `Tensor` whose `Handle.Native` names the module's
  resource type, which is nominal to that block. The alternatives are ruled
  out by that same property — a table of `Fn` values injected by a backend
  package still has to name the handle type in its signature, and a third
  package owning the dispatch still has to carry both backends' handles in one
  type. An untyped handle (an `Int` into a registry) would break §8, where
  ownership, arena closing and `RUNTIME-NATIVE-CLOSED` all key on the resource
  value. [agentica#1918](https://github.com/agentica-lang/agentica/issues/1918)
  is the change that makes the handle opaque; when it lands the module and its
  wrapper move back out into `torch/`, and neither this `case` nor `Tensor`
  changes shape.
- The module is `optional = true` and built from `native/torch.c` on first
  use, with libtorch found by the environment (`tools/build-torch.sh`). Where
  it is not built, `Backend.Torch(device)` answers
  `error("backend not available")` and
  `Tensor.available(Backend.Torch(device))` is `false`, so a caller can choose a
  backend without catching that error.
- Why this shape: Agentica is static and already checks before running
  (no `defn` macros as in `Nx.Defn`); an enum + `case` keeps dispatch
  visible to the checker and lets `reference` and `torch` coexist by value
  in one process for oracle comparison. The considered alternative — a
  struct of functions per backend (object-style interface) — cannot carry a
  native handle across the package boundary while a resource is nominal to its
  block; [agentica#1918](https://github.com/agentica-lang/agentica/issues/1918)
  would unblock it, and with it the move of this module out of the package.
- The libtorch backend is one `extern module` with a thin C layer (C++ as
  `tch-rs` does), declared as `[native.torch]`. Heavy third parties (ML
  engines, models) enter only as pinned git submodules under `third_party/`,
  never vendored.
- **`Ggml` has no eager ops.** A Vulkan dispatch costs about 2 ms, so ggml is
  fast only when a whole computation is one compiled graph
  ([the ggml-graph spike](https://github.com/agentica-lang/ml/blob/bb056e75b3004706cffb7c851fa94b5e8a3b9253/spike/ggml-graph/README.md)). Its arms create tensors (`zeros`, `ones`,
  `empty`, `full`, `like`, `arange`, `scalar_to_tensor`, `from_list`), read them
  back (`to_list`), copy them between devices (`to_device`) and pass `stop_grad`
  through; every other primitive answers `Ggml.compiled_only(op)`, an `error`
  that names the op and points at `Trace`. Running each op as a one-node graph
  would work, but it is a slow path nobody should train on, so it is not
  offered.
- **A trace on `Ggml` is one compiled graph** (`src/lower.agc`,
  `native/ggml_graph.c`). `Trace.evaluate` sends every node the outputs reach
  to the C layer in id order and runs the result as one ggml graph; it never
  calls `Tensor.*` (§10.3). Inputs and lifted constants are leaves bound on
  every call, and a `Full` is a one-element leaf whose value is set on every
  call. So a compiled graph depends on the ops, shapes, dtypes, integer
  parameters and on which fills are zero (a `where` with a zero branch lowers
  without it), and that string is the key of a process-wide cache: a retrace
  of the same function with new data, constants or fill values compiles
  nothing. The cache holds 32 graphs and evicts the least recently used one
  nobody runs. A graph that will not run again (`eval_many` with `keep`
  false) takes no slot and is dropped after its run.
  `Trace.compile` does the tracing and the lowering (`Lower.program`: the
  nodes, the key, where each leaf's tensor comes from, the fills) once, and
  `Trace.run` is `Lower.execute` over it: it acquires the graph by the key it
  already has, sends the nodes only when the cache lost the graph or another
  run holds it, binds the leaves (an input by its position, a constant to the
  tensor it was traced with) and runs. `Trace.evaluate` is the same two steps
  for one call.
- **Kept constants.** A run of a compiled function leaves its constants bound
  to its pinned graph (`ggml_keep`), so the next run binds only its inputs
  (`ggml_resume`). The graph holds a count on the memory of every tensor it
  points at, so no run reads freed memory. Only a constant that is read in
  place is kept: an input, the cache a `set_rows` writes and an `I32` constant
  (a converted copy) are bound by every run. The kept leaves of a graph are
  one set, the constants of the program of one pin: for any other program
  `ggml_resume` ends the set and that program binds every leaf, so two
  functions that share a graph never read each other's weights. The set ends
  when its pin ends, when one of its tensors is closed (the memory then goes
  as before, and the next run is the error that names the leaf), when a kept
  leaf is bound again and when the graph is destroyed; a graph without a pin
  keeps nothing. The comment of `struct port` in `native/ggml_graph.c` has
  the rules and the locks.
- **No copies in or out.** For one run only, an `F32` leaf points at the
  caller's tensor and an `F32` output at a fresh tensor the run writes (or at
  one that exists, `Trace.run`'s `into`); after the run the graph points at
  nothing again but its kept constants, so the cache keeps no other caller
  tensor alive. Only an `I32` leaf or output is copied, converted through a
  buffer of the graph's own when the run starts (`bind` touches no device); a
  small `I32` tensor made by `from_list` keeps its values on the host for that,
  until a run writes it. An `F32` output that is a contiguous result of an op
  is that op's own memory, bound to the caller's tensor, with no copy (an
  `adam_step`'s weight and moments are views of the copy it updates, which is
  bound); a leaf, a view at an offset or a reshaped view, a half and a node
  returned twice are copied. A cached graph owns its fills and its finiteness
  checks; the intermediates live in one allocator per device that every graph
  shares, planned again only when another graph ran last: the graph that ran
  last keeps its allocation, and only the views of its leaves and outputs
  follow what the run bound. That allocator keeps the largest buffer any
  graph needed until the process ends.
- **Tensor memory.** It is a refcounted `tg_storage`: the handle holds one
  count, and a bind takes another. The graph lets go of its counts when the
  run ends, or at the end of the lease if the run never happens (a failed
  bind, say); the count of a kept constant goes when its set ends. So closing
  a handle in the middle of a run frees the memory after the run, never under
  it. An `I32` output in `into` keeps the tensor it
  copies back to by the same count. `output_into` refuses a tensor that is
  another port of the same run (an input, a constant or another output).
- **Pins.** `Trace.compile` pins its graph (`ggml_pin`, a `Pin` resource):
  `free_slot` skips an entry with `pins > 0`, so eviction never takes it. The
  count is under `cache_lock` alone, and the free hook of a `Pin` takes that
  lock and never waits for the device lock. A graph that has no slot (all 32
  are in use or pinned) gets a pin that holds nothing and runs as an ordinary
  graph, so a compile never fails for want of a slot. The pin ends with the
  task that made it, or at `Trace.close`.
- **Locks.** Builds, runs and Vulkan buffer operations hold one device lock,
  so CPU and Vulkan graphs run one at a time (enough for one iGPU; a lock per
  device can come later). The cache's own lock is held briefly, and a free
  hook (a lease, a tensor) takes only that one: what it must free under the
  device lock is queued for the lock's next holder, so no task's free waits
  for a graph to finish.
- **Half weights.** `Tensor.from_file(..., dtype: F16)` on ggml keeps a weight
  in 16 bits (`GGML_TYPE_F16`): the load converts a chunk at a time, with one
  thread per run of whole chunks (the cores the process may use, at most 8;
  each thread reads its run with `pread` and writes its own part of the
  tensor; a run stops once an earlier one has failed), and fails on a value
  beyond 65504, naming the first such element in file order, never clips it;
  a value too small for F16, from a BF16 file for instance, becomes a
  subnormal or zero. A trace takes a half weight as an input or a
  constant, and the lowering accepts an `F16` node only as that leaf or its
  transpose, as the second operand of `matmul` (so `linear`), or as the first
  of `gather` (so `embedding`). The rows of that `gather` stay `F16` in the
  trace, as on every backend, and ggml has widened them to `F32` already: only
  `to_dtype` to `F32` may take them. `ggml_mul_mat` reads the weight with the
  `F32` activations and answers `F32`. Every other use of a half weight is an
  error naming the op, before anything compiles, except the KV cache of
  `transformers`: `to_dtype(x, F16)` makes an `F16` node (`OP_CAST`, which may
  be an output), `set_rows` writes `F32` rows into an `F16` input in place
  (ggml rounds them), `attention` takes an `F16` key and value (flash attention
  reads them as they are, the products read them in `mul_mat`), and
  `to_dtype(input, F32)` widens an `F16` input. A `slice` of the cache goes
  into an `attention` and nowhere else: when it starts at the first element,
  flash attention reads it as a view of the cache (the first rows of every
  head, no copy), which is how a decode step reads only the rows in use.
  A half weight is for inference: `Trace.value_and_grad` and `grad` refuse
  one (so does `gradients` for a `Constant` one; an `F16` tensor of another
  backend keeps its gradient). Other paths are not guarded: a gradient graph
  that needs an `F16` fill, the zeros of a weight the output does not reach,
  ends in the lowering error, and `gradients` of an `Input` half weight that
  only feeds `linear` or `matmul` returns an `F32` gradient. Use
  `linear(..., out_in: true)` with a checkpoint's `[out, in]` weight: without
  it the lowering transposes (copies) the half weight on every run.
- **Quantized weights.** `Tensor.from_file(..., format: Q8_0, dtype: Q8_0)` on
  ggml keeps the blocks of a Q8_0 file as they are (`GGML_TYPE_Q8_0`: 34 bytes
  for 32 values, about a quarter of `F32`; the range is copied, nothing is
  converted). It is a rank-2 weight whose rows hold whole blocks, and a trace
  uses it where it uses a half weight (a leaf, the weight of `matmul` and
  `linear`, the table of `gather` and `embedding`, no gradient, no output),
  with one more rule: a product reads it only as `linear(..., out_in: true)`
  does, in place (a quantized transpose would need blocks along the other
  axis, so `matmul` by the weight itself and any other op are an error naming
  the op). ggml reads the weight with `F32` activations and answers `F32`;
  its CPU kernels round the activations of the product to Q8_0 blocks (the
  same rounding llama.cpp does) and Vulkan's to Q8_1 or F16, so the result is
  good to the half step of that rounding and of its scale's rounding to F16,
  `amax * (1 / 254 + 2^-11)` of a block of 32 activations per product, not to `F32`'s 1e-7. `gather` of the table is
  exact (ggml widens the rows). `to_list` of a quantized weight dequantizes it.
  A Q8_0 file of another backend or of `weights: F32` is read as `F32`, which is
  exact (a binary16 scale times a byte has 19 significant bits).
- **What a half weight costs.** The weight rounds once, at load. The
  activations of the product stay `F32` on ggml's CPU backend and on Vulkan at
  the F32 choice (measured against a double-precision sum of the same weights:
  7e-8 and 1e-7 apart). With fp16, the Vulkan default, Vulkan computes the
  product in fp16, as it does for an `F32` weight: about 2e-3 apart on outputs
  of 0.4, and an activation beyond 65504 overflows. A kernel that rounded the
  activations to F16 would do the same, so a half-weight result is good to
  about 2^-11 relative per product on any backend. `F32` is the exact path
  and the default.
- The lowering computes in `F32`. `Bool` (0/1) and integers up to 32 bits are
  exact there below 2^24; a node of `F64` or `I64`, an `F16` node that is not
  a half weight (below), an integer `div`, a
  `to_dtype` that is not exact in `F32`, an integer fill or `arange` beyond
  2^24, an `arange` that counts down, a `transpose` above rank 2, a tensor with
  no elements and an op without a lowering are errors naming the node, given
  before anything compiles; an `I32` value beyond 2^24 is an error when it is
  copied in or out. Lowered: `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`,
  `relu`, `sqrt`, `sigmoid`, `tanh`, `matmul`, `transpose`, `reshape`,
  `squeeze`, `unsqueeze`, `broadcast_to`, `sum`, `mean` and `max` over a leading
  or trailing block of axes, `stop_grad`, `copy`, `contiguous`, `arange`,
  `slice`, `concat` and `stack` on any axis, `conv1d` (an `F32` im2col and one
  matmul; ggml's own `conv_1d` makes the im2col `F16`), `gelu_erf`, `gather`
  (ggml's `get_rows` over the source as rows, its indices clamped to them; an
  index outside the rows is an error after the run, counted in the graph; a
  source of more than 2^24 rows is an error), `scatter_add` (ggml's
  `get_rows_back`, with the same index check and the same 2^24 limit; its cost
  is `O(k width)` on the CPU, but the Vulkan shader runs one thread per cell
  of the answer and loops over all `k` indices, so there it is `O(rows k
  width)`. On Vulkan in `F32` mode (the fp16 mode rounds the updates in the
  matmul for a chunk of more than 8 columns, the most that ggml-Vulkan's
  always-float mat-vec kernel takes: 1e5 becomes infinity), with `k` below the
  rows, up to 32768 and with room in the graph, the lowering instead gives each
  update row the sum of the rows that name its index, by one matmul with the
  `k x k` mask of equal indices (`O(k^2 width)`, exact: it adds only the rows
  of one index), and writes those sums into zero rows with `set_rows` (the
  rows of one index write the same sum). The mask is made `kc` columns at a
  time, at most 2^24 cells each, so at most 64 chunks. Anything else keeps
  `get_rows_back`. `TENSOR_SCATTER_ADD=get_rows_back` keeps ggml's kernel, for
  the benchmark (`tools/bench-scatter.agc`). In F32 on the Vega iGPU the
  gradient of a lookup over 1000 x 64 with 256 indices takes 6.3 ms (6.7 ms
  with `get_rows_back`), 10000 x 64 with 1024 indices 8.0 ms (57 ms), 50000 x
  64 with 4096 indices 48 ms (959 ms) and 10000 x 64 with 8192 indices, near
  `k = rows`, 162 ms (389 ms); the CPU takes 0.7, 2.1, 7.6 and 4.4 ms, and a
  round trip of the updates and the answer through the CPU would take 1.3, 3.7,
  11 and 6.0 ms, so the masked matmul is a stopgap, not the floor: with the
  sched executor of #85, placing `scatter_add` on the CPU measured 11 ms
  against 45 ms at 50000 x 64 here; revisit then. The long-term fix is an
  upstream Vulkan kernel for `GGML_OP_GET_ROWS_BACK` with one thread per
  (index, column), which would make this lowering unneeded),
  and
  `permute` (a `ggml_permute` and a copy), `batch_matmul` (`mul_mat` over the
  batch axes) and `rope` (`ggml_rope_ext`, with the sequence axis swapped to
  ggml's third; its positions are `I32`, as every integer here, and a rank above
  4 is an error), `attention` (one `flash_attn_ext` that reads q, k and v as the
  views they are, with the scale and an `F16` mask, in `F32` accumulation: the
  pinned ggml takes an unpadded mask and `F32` k and v on CPU and Vulkan. The
  lowering sends it a node when the mask is absent or is built from `full`s of 0
  and of a value from F32's lowest to -65504 (the causal and the padded masks),
  which `F16` holds exactly, the large ones clamped to -65504; a mask of any
  other values, a width that Vulkan has no kernel for (not a multiple of 8), a
  backend without the op and, on Vulkan with the fp16 products (the default), a
  query of more than one row (the fp16 products beat the F32 kernel over many
  queries) use two `mul_mat` and one `soft_max_ext` as before: `mul_mat` repeats
  the kv heads over a group of query heads, and `soft_max_ext` takes the scale
  and the mask. A query with every key hidden by those fills gives the softmax
  of its scores on the flash path, not their mean). A scale and a base cross as
  the bits of a double and must fit an `F32`. `eq`, `lt` and `where`
  emulated in `F32` arithmetic (`eq` as `1 - step(|a - b|)`, exact for integers
  below 2^24 and true when an operand is NaN; `where` as `c·x + (1 - c)·y`,
  exact only when both branches are finite): enough for an MLP and softmax
  cross-entropy with their gradients, for a training step and its batch, and for
  Whisper's forward (`asr/`). An output that holds an infinity or a NaN is an
  error, and so is a finite one whose sum overflows `F32` (the check is the
  sum of its rows; the CPU adds in double and Vulkan in `F32`, so the edge
  differs: `exp(80)` over 10 000 values fires on both). A graph has at most
  512 outputs.
- Vulkan computes `mul_mat` in fp16 by default (the owner's decision, ml#148):
  ggml reads `GGML_VK_DISABLE_F16` once per process, when it first enumerates
  its devices, and the C layer sets it (only when exact `F32` was asked for)
  before its first registry call (any ggml call, a CPU one, or
  `Tensor.available` of a Vulkan device, included; a `GGML_VK_DISABLE_F16` the
  user exported and no `vulkan_fp16` call keep F32). `Tensor.vulkan_fp16(false)`
  before that opts out to exact `F32` `mul_mat`, process-wide (it is ggml's
  choice per process, not per tensor, the one exception to §4's "never a
  global"); a different choice after it is an error, and on a build without
  Vulkan or without the module it changes nothing. fp16 is 1.7× faster for the products of Qwen3 on
  the Vega iGPU (1.55 ms against 2.7 ms for `m=3072 n=128 k=1024`), 1.1–1.5×
  from batch 256 to 4096 in the spike, and changes gradients by up to about
  10 % at MLP sizes ([the spike's run B](https://github.com/agentica-lang/ml/blob/bb056e75b3004706cffb7c851fa94b5e8a3b9253/spike/ggml-graph/README.md)). It does
  not keep the ids of Hugging Face's greedy `generate`, so every check against
  Hugging Face or whisper.cpp chooses `F32` first: the tools through
  `Load.vulkan_fp16_from_env()` (`VULKAN_FP16=1` runs a check in fp16), the
  tests of the packages through their `support.agc`.
- The ggml module is `[native.ggml]`, `optional = true`, built from
  `native/ggml.c` and `native/ggml_graph.c` against the pinned
  `third_party/ggml`, which `tools/build-engine.sh` builds CPU-only (CI) or with
  Vulkan (local), into a cache outside the checkout. The build table names one library, a linker script that script
  writes, so one table links either entry; the C layer finds Vulkan devices
  through ggml's backend registry at run time.
  `Tensor.available(Backend.Ggml(device))` is `true` when the module is built
  and, for `Vulkan(i)`, ggml sees device `i`; it opens no device. ggml stores
  `F32` and `I32`; `Bool`, `I8`, `I16` and `U8` are stored as `F32` (exact for
  them), `F64`, `I64` and a rank above 4 are refused, and `F16` and `Q8_0` are
  stored only as a half or quantized weight (§7).

## 8. Tensor lifetime

`agentica#1091` (native arenas) semantics, which this section builds on:
`Native.arena do ... end` takes **no block parameter**, runs the block, returns
its value, and at block end closes **newest-first** every native resource and
owned buffer the task opened inside — except what the return value reaches
(struct fields, enum payloads, `Result`/`List`/`Map` items; lambda captures do
**not** keep). A kept view keeps its buffer; kept resources keep their `up`
links. Inner arenas hand kept resources to the enclosing arena; `try` failure
ends the block normally; panic/cancel ends the task, which already closes
everything newest-first.
In-flight `compute def` arguments are held until the call returns (same path
as `Native.close`); use-after-close is `RUNTIME-NATIVE-CLOSED`, never UB;
`Task.spawn_owning` transfers ownership, anything else crossing tasks is
documented-not-refused.

Usage for tensors:

- Intermediates die at a known point instead of waiting for the GC (a 1 GB
  native tensor looks like any other `resource` to the GC; a step makes
  thousands of them). Each op of `src/torch.agc` opens an arena of its own and
  returns the tensor it made as the block's value, so the arena keeps that
  tensor. The libtorch ops are `AN_FRESH` (Agentica 0.37.0, `native/torch.c`):
  a handle owns its `torch::Tensor`, and a view holds its base by libtorch's
  refcount, never through another handle's state, so the arena keeps the
  result alone and the intermediates it came from still close. A caller that
  runs a step nests those calls in one arena of its own, and can return a
  tensor out of it (`test/torch_lifetime_test.agc`). `Handle.Ggml` is not
  fresh: a traced node refers to its arguments.
- Only the returned `Tensor`(s) leave the arena; views keep their buffers
  through the rule above, so there is no explicit `keep` call.
- `Handle.Ggml` follows the same rules: each creation call of `src/ggml.agc`
  opens its own arena, and the resource owns one ggml context and one backend
  buffer, freed together by its free hook. The device itself is opened once
  per process and never closed: a buffer only needs its device's buffer type,
  which lives as long as the process. Every Vulkan buffer operation
  (allocate, write, read, free) runs under one lock in the C layer, so tasks
  on one device never overlap a transfer; a free hook never waits for it and
  leaves the free to the next holder (§7). `fit`'s training step is one
  evaluation (`Trace.eval_many`): it reads the parameters and the optimizer
  state in place and writes the new ones into fresh `Handle.Ggml` tensors (no
  donation).
- `compute def` kernels only push data or return a resource; the loop lives
  in Agentica and the native side never calls back (existing FFI invariant).

## 9. Shape/dtype validation

- Validate **eagerly at call time in the Agentica wrapper**, before dispatch:
  ranks, dimension sizes, dtype legality, and broadcast compatibility return
  `error` without touching native code. Backends may re-check cheaply but
  must not be the only check (the reference oracle defines correctness,
  including the error cases).
- Trace time reuses the same predicates (§10.1).
- Conformance covers values **plus** shapes, dtypes, and error cases on every
  backend, bit-for-bit against `reference` where the dtype is exact.

## 10. Training: tracing and grad

- Functions trace to a graph of §3.1 primitives; `grad`/`value_and_grad` are
  computed **in Agentica over that graph** (ml#1 key decision). Backends stay
  forward-only: no per-backend autograd.
- Nothing in §3.1–§4 assumes training, but a differentiable primitive
  declares its backward rule in Agentica (the end of §10.3 lists the ones still
  without): the trace records the ops, and the
  rule lives in `tensor/`, not in a backend. `nn/` (forward-only layers over
  `tensor/`) and `train/` (losses, SGD/Adam, a training loop) build on them.
- Op admission rule: an op that a backward rule needs enters §3.1; adding one
  is a breaking change to the backend interface. `stop_grad`/detach is not
  one: it is a wrapper-level op with no backend primitive (§10.3).
- The backward rule of a differentiable primitive — an `F32`-output
  arithmetic, reduction, linalg, shape, movement, conv or pooling op — is
  expressible over §3.1 itself; an op whose rule needs more admits that op
  (§3.1). Creation, comparison and index ops cut the trace instead: `zeros`,
  `ones`, `empty`, `full`, `arange` and `scalar_to_tensor` build a value from
  shape/dtype arguments; `like(t)` reads only `t`'s metadata (shape, dtype,
  backend), ignoring its value, so it needs no rule either; `eq`, `lt` and
  `argmax` give `Bool`/integer values. `stop_grad` cuts the trace too.

### 10.1 How a function is traced (decided)

Agentica has no macros (§7), so nothing rewrites a function the way `defn`
does. A function is traced by **calling it with tracer values through the same
API**, as Nx does with its `Expr` backend: `Backend.Trace` is a fifth arm of
every dispatch `case`, and a tensor on it holds `Handle.Traced(Node)` instead
of storage. `Trace.value_and_grad(params, &loss)` calls `loss` with `Trace`
tensors; each `Tensor.*` call runs the wrapper's checks (§9) as it always does
and then records a node, so trace time and eager time share every shape and
dtype check by construction.

Why this and not the alternatives:

- **One API.** `nn/` layers and the derived ops of §3.2 (`softmax`,
  `layer_norm`, attention) are written once, over `Tensor.*`, and run eagerly
  for inference (no node per op, no trace) or traced for training.
- **A separate traced API** (`Trace.add` over a type of its own) was the first
  version and is rejected: every derived op and every `nn/` layer would be
  written twice, and its shape checks would be a second copy of §9.
- **A layer generic over an ops record** (`Ops[T]` of `Fn` fields, run on
  `Tensor` or on a node type) also writes a layer once, but every call becomes
  `o.matmul.call(x, w)` and the derived ops would be rewritten in that style;
  the tracer arm gets the same for free.

The price:

- An op on a traced tensor that has no backward rule cannot be refused by the
  checker. Every op of §3.1 records a node, so any traced function runs
  forward; a gradient that has to pass through an op with no rule makes
  `gradients` answer `error("<op> has no backward rule yet")`, at run time.
- An eager tensor and a traced one never mix (`same_backend`): a tensor from
  outside the function enters with `Trace.constant(t)`, explicitly.
- `Backend` and `Handle` carry a variant that no compute backend handles; a
  backend that reads a handle answers `error` on `Traced`.
- A traced tensor has no values, so host readback and control flow that
  depends on data cannot be traced: `to_list`, `top_k`, `beam_search`, a
  decode loop, `if x > 0`. They stay outside the traced function.
- A new op costs five places: its `Trace` arm, its `Op` variant and its
  `op_name`, `apply` and `vjp` arms. The compiler enforces all of them
  (exhaustive `case`).
- The shape a `Trace` arm records is a second copy of the backend's shape
  inference; `evaluate` checks every computed shape and dtype against the
  recorded one and answers an error naming the op and node.
- A `Constant` holds its tensor, so a graph with a native constant is tied to
  the task that owns that handle (§8).
- `f` is traced again on every call (each training step).

### 10.2 The graph (decided)

- A node is `Node { id, op, args }` behind a traced tensor, whose own `shape`
  and `dtype` are the node's. Its arguments are traced tensors. The graph of a
  function is what its output reaches; there is no graph object and no
  mutable tape.
- `id` is a `Ref` (`Ref.new()` in `src/graph.agc`), the role `make_ref` plays in
  Nx: two uses of a value are one node, and since an argument is always made
  before its user and `Ref`s sort by creation, sorting by `id` is a topological
  order.
- Leaves are `Input` (an argument of the traced function), `Constant(t)` (a
  lifted tensor: data, labels) and `Full(value)` (`zeros`, `ones`, `full`,
  `like`, `scalar_to_tensor` on `Trace`).
- Every argument of `value_and_grad`/`grad` is differentiated; what is not a
  parameter enters as a `Constant`.

### 10.3 Gradients (decided)

- `gradients(out, wrt)` is a **graph transformation**: it returns traced
  tensors built with `Tensor.*`, so each backward rule is a composition of §3.1
  primitives (`relu` uses `lt` and `where`; `broadcast_to` uses `sum` with
  `keep_dims` and `reshape`). The gradient graph can be evaluated, inspected or
  compiled like any other, and `wrt` may name any node, not only an input.
- Cotangents flow only into arguments on a path to `wrt`; a `wrt` node the
  output does not reach gets zeros. A node that is not float (`eq`, `lt`,
  `argmax`, `arange`, an integer `to_dtype`) carries no gradient, so
  comparisons and index ops cut it, and so does the condition of `where`.
  `relu`'s derivative at `0` is `0`.
- `evaluate` runs a graph on the inputs' backend, computing each node once; a
  lifted constant must live on that backend. A backend that computes op by op
  (`Reference`, `Torch`) runs it through the public `Tensor.*` calls. A backend
  that compiles (`Ggml`) runs the whole graph as one graph of its own and does
  not go through `Tensor.*` (§7), so it supports the ops its lowering knows and
  answers an error naming any other. `value_and_grad` evaluates the output and
  its gradients in one pass, so the backward reads the forward values it shares;
  `eval_many` evaluates every tensor a traced function answers in one pass (a
  training step: the loss, the new parameters and the optimizer state).
  Backends stay forward-only.
- First rules: `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `relu`,
  `matmul`, `transpose`, `sum`, `mean`, `broadcast_to`, `reshape`, `where`,
  and the identity of `copy`, `contiguous` and `to_device` to `Cpu` (every
  backend that computes today is on `Cpu`; a move across devices answers an
  error). Each is tested against a central difference of the traced
  function on the reference backend. On libtorch, the forward ops of a
  2-layer MLP's gradient graph are run and compared with the reference.
- A parity suite runs every op of §3.1 traced (`Trace.eval`) and eagerly over
  good and bad inputs: both fail, or both give the same shape, dtype and
  values, and the shape and dtype a trace records are the eager ones.
- `stop_grad(t)` is a wrapper-level op with no backend primitive: the
  wrapper answers `t` itself on every backend that computes and, on `Trace`,
  records a node whose rule is zero. `softmax` subtracts its maximum through
  it: the shift cancels out of the value, so it needs no gradient.
- More rules: `sqrt` (`ct / 2 sqrt(x)`, infinite at `0`), `sigmoid`, `tanh`,
  `gelu_tanh_approx` (the derivative of the tanh form), `gelu_erf`
  (`Phi(x) + x phi(x)`, with `Phi(x)` read back as `gelu_erf(x) / x`, `0.5`
  at `0`), `squeeze` and
  `unsqueeze` (a `reshape` back), `slice` (`ct` padded with zeros by
  `concat`), `concat` (each argument's window of `ct`), `stack` (entry `i`
  along the axis), `gather` (`scatter_add` of `ct` at the indices: a
  repeated index sums its rows), `scatter_add` (for its updates, `gather` of
  `ct` at the indices; the indices have no gradient) and `max`
  (the cotangent goes to the positions that hold the maximum and is split
  evenly between ties, as `jax.numpy.max` and `torch.amax` do). With them,
  `softmax`, `layer_norm`, `rms_norm`, `silu`, `gelu`, `embedding` and
  `multi_head_attention` are differentiable; each rule and the last two are
  tested against a central difference.
- `permute` (the inverse permutation), `rope` (the same rotation turned the
  other way, `inverse`; the positions have no gradient) and `attention`. The
  backward of `attention` recomputes the probabilities from `q`, `k` and the
  mask, once for each of the three arguments it feeds (the graph does not share
  the scores between them), folds the query heads of a group into the rows of
  the scores (`[B, H, Sq, D]` as `[B, Hkv, H/Hkv * Sq, D]`) so every product is
  a rank-4 `batch_matmul` and the kv heads are never repeated, and gives the
  mask `dscores` summed over the batch and head axes it broadcast. Each is checked against a
  central difference, with grouped heads, a mask per head and the causal flag,
  and on ggml against the reference node by node (`test/ggml_attention_test.agc`).
- Still open: rules for `pow`, `clamp`, `batch_matmul`, `to_dtype` between
  float dtypes (a precision cast), `conv1d` and `avg_pool1d`; the rule of
  `where`'s condition for gradients of gradients; and how an
  evaluation's intermediate tensors map onto the arenas of §8 (today they
  live until `evaluate` returns).

## 11. Conformance

- `reference` is the oracle. One shared suite runs on every backend:
  values, shapes, dtypes, and error paths.
- Every backend is checked against `reference` through that one suite, so a
  backend is correct by construction or not at all.

## 12. StableHLO

- `Trace.stablehlo(specs, f)` traces `f` once, as `Trace.compile` does, and
  writes the traced outputs as one StableHLO module (`src/stablehlo.agc`), a
  second lowering of the graph beside the ggml one (§7). The text knows no
  backend and no runtime, so XLA through PJRT and IREE can read it; a later
  backend for either reuses it.
- The module is one `func.func public @main`. Its arguments are the inputs, in
  the order of `specs`, then the tensors lifted with `Trace.constant`, in the
  order of their nodes; `Module.constants` holds those tensors, so a run gives
  them as arguments like any other. It returns the outputs `f` answered. A
  constant is an argument, not a literal, so a batch is not in the text.
- Shapes are static. The `Tensor.*` wrappers take exact shapes, so an
  elementwise op needs no implicit broadcast: `broadcast_to` is
  `stablehlo.broadcast_in_dim`, whose operand axes align with the trailing axes
  of the result, and an op that has a scalar (`gelu`, the mean's count) holds
  it as a splat constant of the shape.
- The ops covered are the ones of an MLP forward and of the `fit` step: `add`,
  `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `tanh`, `relu`,
  `gelu_tanh_approx`, `matmul` (`dot_general`, contracting `[1]` with `[0]`),
  `transpose`, `reshape`, `broadcast_to`, `sum`, `mean`, `max` (a `reduce` from
  `0` or `-inf`, then a `reshape` for `keep_dims`), `eq`, `lt`, `where`
  (`select`), `arange` (`iota`), `slice`, `full` and `stop_grad`. Any other op is an
  error naming it. `Q8_0` has no MLIR element type; `Bool` is `i1`, `U8` is
  `ui8`.
- `Train.stablehlo(loss, params, batch, optimizer)` is the step of `fit`, as
  `fit` traces it: the arguments are the parameters, the optimizer's moments,
  then the inputs of the step (`Optim.step_inputs`: SGD's learning rate as a
  rank-0 tensor, Adam's 7 numbers as a rank-1 tensor, AdamW's two rows of
  them), whose Adam update is the composition of `Tensor.adam_step` (no node),
  then the batch. One program is every step: the host computes the numbers
  from the step count and the schedule, as `fit` passes them on ggml, and
  feeds the outputs (the loss, the new parameters and moments) back in.
- **Goldens.** `train/test/fixtures/stablehlo/NAME.mlir` is the emitted text of
  each case (`train/tools/stablehlo_cases.agc`): CI compares it, which finds a
  change and says nothing of correctness. `NAME.golden.json` holds the
  arguments and the outputs IREE (CPU) gave for that text, three chained runs
  for the fit step; CI checks that the reference backend, and `fit` itself,
  reach the same values within `F32` drift. IREE is local tooling only:
  `make iree-venv` installs the pinned official wheels in `.venv-iree/`, and
  `make iree-goldens` rewrites the snapshots and the goldens. `iree-compile`
  also rejects a text that is not valid StableHLO, which is a failure of the
  emitter.
