# tensor — design proposal (inference-first)

Status: **proposal, not a contract.** The four open questions of ml#1 are
answered here as working proposals; §12 is the list of what stays open.

References: [ml#1](https://github.com/agentica-lang/ml/issues/1) (goals, grad-in-Agentica
decision, backend table, open questions),
[agentica#1091](https://github.com/agentica-lang/agentica/issues/1091)
(native arenas; see §8), and
[agentica#1918](https://github.com/agentica-lang/agentica/issues/1918)
(opaque native resources that can live in data and cross modules; see §2 and §7).

## 1. Goals and scope

Goals:

- **Training and inference.** `tensor/` + `nn/` (forward) run over
  `reference` (pure Agentica oracle) and real eager backends; `train/`
  (losses, SGD/Adam, an `Axon.Loop`-style loop) and the traced
  `grad`/`value_and_grad` it needs sit on the same primitives (§10). The
  design must let either land without changing the other. Clients (bindings,
  applications, `hub/`-loaded checkpoints) state their requirements
  elsewhere; they bound what §3–§6 must support without this document
  knowing them.
- Keep every backend **forward-only**. `grad`, `vmap`, `jit` live in
  Agentica over a traced graph (ml#1 key decision, as Nx does); XLA has no
  autograd of its own, so per-backend autograd is out.
- Keep the backend interface small enough to implement four times
  (`reference`, `torch`, `xla`, `candle`) with one shared conformance
  suite where the reference is the oracle.
- **`tensor/` declares the native module of every backend it dispatches to.**
  A `Tensor` carries the backend's handle, and a native handle is a
  `resource`: resources are nominal to the `extern module` block that declares
  them and cannot cross into another module, so `Handle` (§2) can only name a
  handle type declared in this package, and the `Torch` arm of §7 can only
  call a module declared here. Each such module is `optional` in
  `agentica.toml`, so a machine without the library still checks, formats and
  tests with the arm answering `error("backend not available")`;
  `Tensor.available/1` answers the same question before a call does. The cost
  is that `tensor/` is no longer "nothing native": it ships the thin layer of
  each real backend and its build table. [agentica#1918](https://github.com/agentica-lang/agentica/issues/1918)
  is the change that lets a handle be opaque, at which point the backend code
  moves back out into its own package and this bullet goes away.

Out of scope here (each gets its own design):

- The JIT compiler (`xla/` via PJRT) and the model hub (`hub/` download and
  sharding UX). §10 fixes how tracing/grad attach without changing the
  primitives.
- A ggml quantized backend: quantized checkpoints are consumed by
  specialized bindings, not by this stack.
- Final signatures (§4).

## 2. Types (proposal)

```text
Dtype   = F32 | F16 | F64 | I8 | I16 | I32 | I64 | U8 | Bool   # proposal
Device  = Cpu | Cuda(Int)                                      # enum, proposal
Shape   = List[Int]                                            # static ints, row-major C-order
Values  = Float(Float) | Int(Int) | Bool(Bool)                 # enum, dtype-tagged
Backend = Reference | Torch(Device) | Xla(Device) | Candle(Device) | Trace   # enum, see §7, §10
Handle  = Native(TensorHandle) | Plain(Data) | Traced(Node)    # enum, see §2, §10
Tensor  = { shape: Shape, dtype: Dtype, backend: Backend, handle: Handle }
```

- `Tensor` values are **immutable**; every op returns a new `Tensor`.
  A training step returning new parameters falls out of this later.
- `handle` is `Native(TensorHandle)` on the libtorch backend (a
  `struct NAME_state *` + thread-safe `NAME_FREE`, C data only, never terms;
  valid only during the call, never stored) and `Plain(Data)` on `reference`,
  where `Data` is plain Agentica data. `TensorHandle` is the resource the
  `extern module "torch"` block of `src/native.agc` declares: a resource is
  nominal to its block, so the handle type has to be the concrete one and it
  has to live in this package (§7). A second native backend adds a second arm
  (`Native(CandleHandle)`, …) when it exists; a backend that only reads a
  handle it did not make answers `error`, never a stopped task. This follows
  the repo's FFI rules (`extern module` is the only source of signatures;
  native-gen regenerates the header; blocking work is `compute def` on the
  compute pool; `List`/`Map` cross via cursors/builders).
- **Inference-first dtype rule (proposal):** numeric compute is `F32`; `F16`
  and `F64` weights are converted to it on load (§6, lossy for `F64`). Two
  kinds of non-`F32` compute tensors are required by the op set itself: `Bool`
  masks (`eq`, `lt`, the condition of `where`) and integer index tensors
  (`gather` indices, `argmax` output). The other table entries are checkpoint
  dtypes only (§6), kept as indices or masks at runtime; the table grows only
  when a consumer requirement (§5) or conformance demands it.
- No operator overloading (ml#1 question 2): `Tensor.add(a, b)` plus pipes
  (`t |> Tensor.add(other) |> Tensor.relu()`). If the language ever allows
  `a + b` on user structs, it is sugar over this.

## 3. Primitives vs derived ops (the split)

**Primitives** are the only functions a backend implements. Everything else
is **derived**: pure Agentica code over primitives, shared by all backends
and therefore tested once.

### 3.1 Primitives (proposal: the backend contract, 45 ops inference subset)

Creation: `zeros, ones, empty, full, like, arange, scalar_to_tensor`.
Elementwise: `add, sub, mul, div, neg, exp, log, sqrt, pow, relu, sigmoid,
tanh, gelu_tanh_approx, eq, lt, where, clamp`.
Reductions: `sum, mean, max, argmax` (each with `axes: Optional[List[Int]]`,
`keep_dims: Bool`).
Linear algebra: `matmul, transpose, batch_matmul` (batched `matmul` covers
attention; plain `dot` is derived).
Shape: `reshape, squeeze, unsqueeze, slice, concat, stack, broadcast_to,
gather` (embedding lookup and beam gather need `gather`; everything fancier
is derived).
Normalization support: `rsqrt, mean_var` are **not** primitives —
`layer_norm` is derived (§3.2); the backend only needs the pieces above plus
`broadcast_to`.
Convolution/pooling: `conv1d, avg_pool1d` (a transformer frontend needs
`conv1d`; anything beyond stride/padding-1D is decided by the spike).
Random: **out** for inference v1 (dropout is identity at inference; stochastic
sampling stays behind an explicit key in the later training set).
Movement: `to_dtype, to_device, copy, contiguous`.

Deliberately missing: `softmax`, `layer_norm`, attention, the exact (`erf`)
`gelu`, `silu`, beam search, STFT/mel — all derived or host-side (§3.2, §5).

The names above are 45. Keyed random and the training-only ops of §10 are
outside this inference subset but inside the frozen contract.

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
- `multi_head_attention(q, k, v, mask?)`: split heads by `reshape/transpose`,
  scores by `batch_matmul`, scale by `mul`, `softmax`, context by
  `batch_matmul`, merge by `transpose/reshape`. No fused kernel in v1.
- `greedy_step(logits, temperature?)`: `div` by temperature, `softmax`,
  `argmax`. Beam search is an Agentica loop over `gather` + `top_k` (`top_k`
  is a derived sort/select loop in v1; a native `top_k` primitive enters only
  if the spike shows the loop is unusably slow).
- `dropout(x, p)` at inference is identity (kept for `nn/` API symmetry).

If a consumer needs an op that is neither in §3.1 nor expressible above at
acceptable cost, it enters §3.1 when the contract is frozen: §5 is the only
gate for adding ops.

## 4. Inference-first API shape (proposal)

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
  (`Backend.Torch(Device.Cpu)`, `Backend.Torch(Device.Cuda(0))`);
  `Backend.Reference` has none. Batch lives in the shape, not in a global
  either.
- Backend-module functions take the device as an explicit first argument
  (`Torch.matmul(device, a, b)`): dispatch unpacks it from the backend value
  (§7) and the tensor keeps no second copy of it.
- The `.agc` signatures plus the generated native header are the contract
  once written; until then this document is the only source, and it is still
  a proposal.

## 5. Requirements the API must satisfy (consumer checklist)

The packages have clients: bindings, applications and `hub/`-loaded
checkpoints. Their requirements bound what §3.1–§4 must support; the clients
state their own specifics (shapes, frontends, sampling policy) outside this
document. The generic requirements a pre-trained transformer inference client
imposes are below.

Encoder/decoder (parametric in size, no model-specific constant here):

- Positional information via `arange + gather` or `full + add` (no dedicated
  primitive).
- N blocks of: `layer_norm` (derived) → multi-head self-attention (derived,
  §3.2) → `layer_norm` → MLP (`linear`, `gelu`, `linear`).
- `conv1d`/`avg_pool1d` are primitives because a consumer needs a 1-D
  convolution stem; a text-only client does not, and that is fine (§5 is the
  gate, not a minimum for every client).
- A feature frontend (windowed transform plus a filterbank) is a `compute def`
  candidate: implement in pure Agentica against §3.1 where viable, drop to
  native only if a spike proves the pure version unusable. Either way the
  **signature stays Agentica-side** (`features(pcm: List[Float]) -> Tensor`;
  the exact shape belongs to the client).
- Autoregressive decode: masked self-attention and cross-attention (derived
  MHA), output head `linear` → `softmax` → `argmax` (greedy) with optional
  temperature scaling; beam search is the Agentica loop from §3.2.
- The decode loop runs in Agentica (one `Native.arena` per step, §8); the
  backend never holds the loop.
- Weights come from a checkpoint (§6). Training (`train/`, §10) runs on the
  same primitives.

A client proves the requirements above are met by reproducing, on
`reference`/`torch`, the output its native binding gives for the same input.

## 6. Checkpoint requirements (weights in, no code execution)

- Format-first: **safetensors** is the first supported checkpoint format
  (8-byte LE length, UTF-8 JSON header with `dtype`, `shape`,
  `data_offsets` plus optional `__metadata__`, contiguous little-endian
  C-order buffer, no strides). No code runs at load; that property is the
  reason for choosing it.
- Covered dtypes are the safetensors set (`F32/F16/F64`, `I8/I16/I32/I64`,
  `U8`, `Bool`); numeric tensors are converted to `F32` on load (lossy for
  `F64`), while integer index tensors and `Bool` masks keep their dtype
  because the op set needs them (§2).
- Loader shape (proposal): pure-Agentica header parse (`std/binary` +
  JSON) yielding `{ dtype, shape, offset }` entries, then per-tensor
  `load_slice(path, entry, backend) -> Tensor`. Multi-file sharding and its
  JSON index belong to `hub/`: this document fixes only the single-file
  format and the loader signature that sharding reuses.
- `mmap` where the backend supports it, plain read where it does not;
  the choice is per-backend, hidden behind the same loader signature.
- `gguf`/ggml checkpoints stay with the specialized native bindings; the
  pure stack reads safetensors only in v1.

## 7. Backend dispatch (proposal for ml#1 question 1)

Proposal: a `Backend` **enum value carried by each `Tensor`**, with dispatch
by `case` to one backend module per variant, each sharing the §3.1 signature
set:

```text
r: Result[Tensor] = case a.backend
  Backend.Reference -> Reference.matmul(a, b)              # src/reference.agc
  Backend.Torch(device) -> Torch.matmul(device, a, b)      # src/torch.agc
  Backend.Xla(_) -> error("backend not available")         # module not written yet
  Backend.Candle(_) -> error("backend not available")      # module not written yet
end
```

- The `case` is checked and exhaustive over all five variants (`Trace`, §10.1,
  records a node instead of computing), which is the
  property this shape buys; `xla` and `candle` return `error` until their
  modules exist, so the arm set does not change when they land.
- The `Torch` arm calls `src/torch.agc`, which calls the thirteen `compute
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
  `error("backend not available")` — the same answer the two unwritten
  backends give — and `Tensor.available(Backend.Torch(device))` is `false`, so
  a caller can choose a backend without catching that error.
- Why this shape: Agentica is static and already checks before running
  (no `defn` macros as in `Nx.Defn`); an enum + `case` keeps dispatch
  visible to the checker and lets `reference` and `torch` coexist by value
  in one process for oracle comparison. The considered alternative — a
  struct of functions per backend (object-style interface) — cannot carry a
  native handle across the package boundary while a resource is nominal to its
  block (§12.1); [agentica#1918](https://github.com/agentica-lang/agentica/issues/1918)
  would unblock it, and with it the move of this module out of the package.
- Each native backend is one `extern module` with a thin C layer (libtorch:
  C++ as `tch-rs` does; candle later: `extern "C"` over Rust), shipped via
  `[native.NAME]` + per-platform prebuilt (CPU first; sizes documented).
  Heavy third parties (ML engines, models) enter only as pinned git
  submodules under `third_party/`, never vendored.

## 8. Tensor lifetime (proposal for ml#1 question 3)

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

Proposed usage for tensors (needs sign-off):

- One arena per **inference call**, and one nested arena per **decode step**:
  intermediates die at a known point instead of waiting for the GC (a 1 GB
  native tensor looks like any other `resource` to the GC; a step makes
  thousands of them). Each op of `src/torch.agc` opens an arena of its own and
  returns the tensor it made as the block's value, so the arena keeps exactly
  what the value reaches; a caller that runs a step nests those calls in one
  arena of its own.
- Only the returned `Tensor`(s) leave the arena; views keep their buffers
  through the rule above, so no explicit `keep` call exists in this proposal.
- `compute def` kernels only push data or return a resource; the loop lives
  in Agentica and the native side never calls back (existing FFI invariant).

## 9. Shape/dtype validation (proposal for ml#1 question 4)

- Validate **eagerly at call time in the Agentica wrapper**, before dispatch:
  ranks, dimension sizes, dtype legality, and broadcast compatibility return
  `error` without touching native code. Backends may re-check cheaply but
  must not be the only check (the reference oracle defines correctness,
  including the error cases).
- How much the static checker carries vs runtime stays open: this proposal
  asks only that trace/compile time reuse the same predicates, and that
  `xla/` later requires static shapes to compile the whole graph.
- Conformance covers values **plus** shapes, dtypes, and error cases on every
  backend, bit-for-bit against `reference` where the dtype is exact.

## 10. Training: tracing and grad

- Functions trace to a graph of §3.1 primitives; `grad`/`value_and_grad` are
  computed **in Agentica over that graph** (ml#1 key decision). Backends stay
  forward-only: no per-backend autograd.
- The graph must be compilable (static shapes) so `xla/` can later compile
  it whole via the PJRT C API (`pjrt_c_api.h`, `dlopen` plugin, no own layer).
  `vmap`/`jit` also live in Agentica.
- Nothing in §3.1–§4 assumes training, but every differentiable primitive
  declares its backward rule in Agentica: the trace records the ops, and the
  rule lives in `tensor/`, not in a backend. `nn/` (forward-only layers over
  `tensor/`) and `train/` (losses, SGD/Adam, `Axon.Loop`-style loop with
  `std/logger`) build on them.
- Op admission rule: §3.1 is frozen once, when the contract is written. An op
  that a backward rule needs enters §3.1 then, and so does an op `train/`
  needs with no backward rule of its own — keyed random (`randn`/`randint`/
  `split`) and `stop_grad`/detach. Adding either later would be a breaking
  change to the frozen backend interface.
- Every differentiable primitive — an `F32`-output arithmetic, reduction,
  linalg, shape, movement, conv or pooling op — has a backward rule
  expressible over §3.1 itself; an op whose rule needs more admits that op
  (§3.1). Creation, comparison and index ops cut the trace instead: `zeros`,
  `ones`, `empty`, `full`, `arange` and `scalar_to_tensor` build a value from
  shape/dtype arguments; `like(t)` reads only `t`'s metadata (shape, dtype,
  backend), ignoring its value, so it needs no rule either; `eq`, `lt` and
  `argmax` give `Bool`/integer values. That is why `stop_grad`, which cuts the
  trace too, enters §3.1 with them.

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
  checker. Today such an op answers `error("<op> cannot be traced yet")` at
  trace time; an op recorded without a rule makes `gradients` answer an error
  that names the op.
- An eager tensor and a traced one never mix (`same_backend`): a tensor from
  outside the function enters with `Trace.constant(t)`, explicitly.
- `Backend` and `Handle` carry a variant that no compute backend handles; a
  backend that reads a handle answers `error` on `Traced`.

### 10.2 The graph (decided)

- A node is `Node { id, op, args }` behind a traced tensor, whose own `shape`
  and `dtype` are the node's. Its arguments are traced tensors. The graph of a
  function is what its output reaches; there is no graph object and no
  mutable tape.
- `id` comes from one run-wide counter (an `ATS` table in `src/graph.agc`), the
  role `make_ref` plays in Nx: two uses of a value are one node, and since an
  argument is always made before its user, sorting by `id` is a topological
  order.
- Shapes and dtypes are static, so a traced graph is the static-shape graph
  `xla/` needs.
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
  output does not reach gets zeros. Comparisons and the condition of `where`
  cut the gradient. `relu`'s derivative at `0` is `0`.
- `evaluate` runs a graph through the public `Tensor.*` calls on the inputs'
  backend, computing each node once; a lifted constant must live on that
  backend. `value_and_grad` evaluates the output and its gradients in one
  pass, so the backward reads the forward values it shares. Backends stay
  forward-only.
- First rules: `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `relu`,
  `matmul`, `transpose`, `sum`, `mean`, `broadcast_to`, `reshape` and `where`.
  Each is tested against a central difference of the traced function on the
  reference backend. On libtorch, the forward ops of a 2-layer MLP's gradient
  graph are run and compared with the reference.
- Still open: recording the rest of §3.1 (with or without a rule),
  `stop_grad`, the rule of `where`'s condition for gradients of gradients,
  `vmap`/`jit`, and how an evaluation's intermediate tensors map onto the
  arenas of §8 (today they live until `evaluate` returns).

## 11. Conformance (proposal)

- `reference` is the oracle. One shared suite runs on every backend:
  values, shapes, dtypes, and error paths.
- Every backend is checked against `reference` through that one suite, so a
  backend is correct by construction or not at all.

## 12. Open decisions (ml#1 questions 1–4)

1. Dispatch: **decided — enum + `case`** (§7). The struct-of-functions
   alternative is blocked, not rejected, while a native handle is a `resource`
   nominal to its `extern module`: an injected table still has to name the
   handle type in its signature.
   [agentica#1918](https://github.com/agentica-lang/agentica/issues/1918) is
   the exit.
2. Surface: `Tensor.*` + pipes (§2–§4) until/unless operator overloading on
   user structs exists.
3. Lifetime: arena-per-call/step over #1091 semantics (§8) vs graph-owned
   buffers or explicit scopes.
4. Checking: eager wrapper validation now; the static-vs-runtime split (§9)
   and how much the checker carries are not fixed here.
