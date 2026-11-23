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
Dtype   = F32 | F16 | F64 | I8 | I16 | I32 | I64 | U8 | Bool
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

### 3.1 Primitives (the backend contract, 45 ops)

Creation: `zeros, ones, empty, full, like, arange, scalar_to_tensor`.
Elementwise: `add, sub, mul, div, neg, exp, log, sqrt, pow, relu, sigmoid,
tanh, gelu_tanh_approx, eq, lt, where, clamp`.
Reductions: `sum, mean, max, argmax` (each with `axes: Optional[List[Int]]`,
`keep_dims: Bool`).
Linear algebra: `matmul, transpose, batch_matmul` (`multi_head_attention`
uses rank-2 `matmul` per head, §3.2; plain `dot` is derived).
Shape: `reshape, squeeze, unsqueeze, slice, concat, stack, broadcast_to,
gather` (embedding lookup and beam gather need `gather`; everything fancier
is derived).
Normalization support: `rsqrt, mean_var` are **not** primitives —
`layer_norm` is derived (§3.2); the backend only needs the pieces above plus
`broadcast_to`.
Convolution/pooling: `conv1d, avg_pool1d` (stride and padding; no dilation or
groups).
Random: none (dropout is identity at inference).
Movement: `to_dtype, to_device, copy, contiguous`.

Deliberately missing: `softmax`, `layer_norm`, attention, the exact (`erf`)
`gelu`, `silu`, beam search — all derived (§3.2).

### 3.2 Derived ops (pure Agentica, backend-agnostic)

Built only from §3.1, in `tensor/` (generic) or `nn/` (model-shaped):

- `softmax(x, axis) = div(exp(sub(x, max)), sum(...))` with max-subtraction.
- `layer_norm(x)`, `rms_norm(x)` from `mean`, `pow`, `sqrt` and `div`
  compositions over §3.1 (no `var`/`rsqrt` primitive is needed).
- `gelu` is the user-facing name of `gelu_tanh_approx` (§3.1); the `erf` form
  cannot be expressed over §3.1 and is not offered. `silu` derives from
  `sigmoid` and `mul`.
- `dot(a, b)` as `reshape` + `matmul`; `linear(x, w, b?)`, `embedding(ids,
  table)` via `matmul`/`gather`.
- `multi_head_attention(q, k, v, heads, mask?)`: for each `(batch, head)` pair,
  slice its part of the width, scores by rank-2 `matmul`, scale by `mul`,
  `softmax`, context by `matmul`, and join the pieces with `concat`/`stack`.
  No fused kernel, and no `batch_matmul`.
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
Tensor.to_list(t) -> Result[List[Values]]         # host readback, tests + decode loop
Tensor.shape(t) -> Shape                          # no native call
Tensor.dtype(t) -> Dtype                          # no native call
Tensor.backend(t) -> Backend                      # no native call
```

- Every fallible op returns `Result`; shape/dtype mismatches are `error`,
  never a halt. Introspection is free (fields, no FFI round-trip).
- Device is explicit, never a global: dispatch is **by value** (`t.backend`),
  not by global config, so `reference` (oracle) and `torch` can check each
  other in the same process. A backend value carries its device
  (`Backend.Torch(Device.Cpu)`);
  `Backend.Reference` has none. Batch lives in the shape, not in a global
  either. The one process-wide setting is `Tensor.vulkan_fp16`, because ggml
  fixes it per process (§7).
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
  (`spike/ggml-graph/README.md`). Its arms create tensors (`zeros`, `ones`,
  `empty`, `full`, `like`, `arange`, `scalar_to_tensor`, `from_list`), read them
  back (`to_list`), copy them between devices (`to_device`) and pass `stop_grad`
  through; every other primitive answers `Ggml.compiled_only(op)`, an `error`
  that names the op and points at `Trace`. Running each op as a one-node graph
  would work, but it is a slow path nobody should train on, so it is not
  offered.
- **A trace on `Ggml` is one compiled graph** (`src/lower.agc`,
  `native/ggml_graph.c`). `Trace.evaluate` sends every node the outputs reach
  to the C layer in id order and runs the result as one ggml graph; it never
  calls `Tensor.*` (§10.3). Inputs and lifted constants are leaves copied in
  on every call, and a `Full` is a one-element leaf whose value is set on
  every call. So a compiled graph depends on the ops, shapes, dtypes, integer
  parameters and on which fills are zero (a `where` with a zero branch lowers
  without it), and that string is the key of a process-wide cache: a retrace
  of the same function with new data, constants or fill values compiles
  nothing. A graph keeps its leaves and a copy of each output in a buffer of
  its own; the intermediates live in one allocator per device that every
  graph shares, re-planned when another graph ran last. The cache holds 32
  graphs and evicts the least recently used one nobody runs.
- The lowering computes in `F32`. `Bool` (0/1) and integers up to 32 bits are
  exact there below 2^24; a node of `F16`, `F64` or `I64`, an integer `div`,
  a `to_dtype` that is not exact in `F32` and an op without a lowering are
  errors naming the node, given before anything compiles. Lowered: `add`,
  `sub`, `mul`, `div`, `neg`, `exp`, `log`, `relu`, `sqrt`, `sigmoid`,
  `tanh`, `matmul`, `transpose`, `reshape`, `squeeze`, `unsqueeze`,
  `broadcast_to`, `sum`, `mean` and `max` over a leading or trailing block of
  axes, `stop_grad`, `copy`, `contiguous`, `arange`, and `eq`, `lt` and
  `where` emulated in `F32` arithmetic: enough for an MLP and softmax
  cross-entropy with their gradients. An output that holds an infinity or a
  NaN is an error.
- Vulkan computes `mul_mat` in `F32` by default: ggml reads
  `GGML_VK_DISABLE_F16` when the first Vulkan device opens, and the C layer
  sets it then. `Tensor.vulkan_fp16(true)` before that opts in to fp16
  `mul_mat`, process-wide (it is ggml's choice per process, not per tensor,
  the one exception to §4's "never a global"); a different choice after a
  device opened is an error. fp16 is 1.1–1.5× faster and changes gradients
  by up to about 10 % at MLP sizes.
- The ggml module is `[native.ggml]`, `optional = true`, built from
  `native/ggml.c` and `native/ggml_graph.c` against the pinned
  `third_party/ggml`, which `tools/build-engine.sh` builds CPU-only (CI) or with
  Vulkan (local). The build table names one library, a linker script that script
  writes, so one table links either tree; the C layer finds Vulkan devices
  through ggml's backend registry at run time.
  `Tensor.available(Backend.Ggml(device))` is `true` when the module is built
  and, for `Vulkan(i)`, ggml sees device `i`; it opens no device. ggml stores
  `F32` and `I32`; `Bool`, `I8`, `I16` and `U8` are stored as `F32` (exact for
  them), and `F16`, `F64`, `I64` and a rank above 4 are refused.

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
  returns the tensor it made as the block's value, so the arena keeps exactly
  what the value reaches; a caller that runs a step nests those calls in one
  arena of its own (`test/torch_lifetime_test.agc`).
- Only the returned `Tensor`(s) leave the arena; views keep their buffers
  through the rule above, so there is no explicit `keep` call.
- `Handle.Ggml` follows the same rules: each creation call of `src/ggml.agc`
  opens its own arena, and the resource owns one ggml context and one backend
  buffer, freed together by its free hook. The device itself is opened once
  per process and never closed: a buffer only needs its device's buffer type,
  which lives as long as the process. Every Vulkan buffer operation
  (allocate, write, read, free, the free hook included) runs under one lock in
  the C layer, so tasks on one device never overlap a transfer. A compiled step (later) will own its graph and the train state
  in a resource of its own, which `fit` creates and closes; the tensors it
  returns are fresh `Handle.Ggml` values (no donation).
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
- `id` comes from one run-wide counter (an `ATS` table in `src/graph.agc`), the
  role `make_ref` plays in Nx: two uses of a value are one node, and since an
  argument is always made before its user, sorting by `id` is a topological
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
  its gradients in one pass, so the backward reads the forward values it shares.
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
  `gelu_tanh_approx` (the derivative of the tanh form), `squeeze` and
  `unsqueeze` (a `reshape` back), `slice` (`ct` padded with zeros by
  `concat`), `concat` (each argument's window of `ct`), `stack` (entry `i`
  along the axis), `gather` (scatter-add as `onehot . ct`, with
  `onehot[r, k] = indices[k] == r` made from `arange`, `eq` and `to_dtype`:
  a repeated index sums its rows) and `max`
  (the cotangent goes to the positions that hold the maximum and is split
  evenly between ties, as `jax.numpy.max` and `torch.amax` do). With them,
  `softmax`, `layer_norm`, `rms_norm`, `silu`, `gelu`, `embedding` and
  `multi_head_attention` are differentiable; each rule and the last two are
  tested against a central difference.
- `gather`'s rule scales with the table: it makes four `[rows, k]`
  intermediates (a 50k vocabulary over 4096 tokens at `F32` is about 800 MB
  each) and its `matmul` costs `O(rows k width)`. A scatter-add primitive
  would avoid both (ml#25).
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
