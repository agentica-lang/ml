# tensor — design proposal (inference-first)

Status: **proposal, not a contract.** The four open questions of ml#1 are
answered here as working proposals; §12 is the list of what stays open.

References: [ml#1](https://github.com/agentica-lang/ml/issues/1) (goals, grad-in-Agentica
decision, backend table, open questions), and
[agentica#1091](https://github.com/agentica-lang/agentica/issues/1091)
(native arenas; see §8).

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

Out of scope here (each gets its own design):

- The JIT compiler (`xla/` via PJRT) and the model hub (`hub/` download and
  sharding UX). §10 fixes how tracing/grad attach without changing the
  primitives.
- A ggml quantized backend: quantized checkpoints are consumed by
  specialized bindings, not by this stack.
- Final signatures: the `.agc` and native-gen headers are the contract once
  written, and the spike that writes them freezes the minimum.

## 2. Types (proposal)

```text
Dtype   = F32 | F16 | F64 | I8 | I16 | I32 | I64 | U8 | Bool   # proposal
Shape   = List[Int]                                            # static ints, row-major C-order
Backend = reference | torch | xla | candle                     # proposal, see §7
Tensor  = { shape: Shape, dtype: Dtype, backend: Backend, handle: Resource | Data }
```

- `Tensor` values are **immutable**; every op returns a new `Tensor`.
  A training step returning new parameters falls out of this later.
- `handle` is a native **resource** on native backends (libtorch/candle:
  `struct NAME_state *` + thread-safe `NAME_FREE`, C data only, never terms;
  valid only during the call, never stored) and plain Agentica data on
  `reference`. This follows the repo's FFI rules (`extern module` is the only
  source of signatures; native-gen regenerates the header; blocking work is
  `compute def` on the compute pool; `List`/`Map` cross via cursors/builders).
- **Inference-first dtype rule (proposal):** compute requires only `F32`;
  `F16` is load-and-cast (weights may arrive as `F16`, compute upcasts);
  the rest of the table exists for checkpoint loading (§6) and grows only
  when a consumer requirement (§5) or conformance demands it.
- No operator overloading (ml#1 question 2): `Tensor.add(a, b)` plus pipes
  (`t |> Tensor.add(other) |> Tensor.relu()`). If the language ever allows
  `a + b` on user structs, it is sugar over this.

## 3. Primitives vs derived ops (the split)

**Primitives** are the only functions a backend implements. Everything else
is **derived**: pure Agentica code over primitives, shared by all backends
and therefore tested once.

### 3.1 Primitives (proposal: the backend contract, ~45 ops inference subset)

Creation: `zeros, ones, empty, full, like, arange, scalar_to_tensor`.
Elementwise: `add, sub, mul, div, neg, exp, log, sqrt, pow, relu, sigmoid,
tanh, gelu_tanh_approx?, eq, lt, where, clamp`.
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
Convolution/pooling: `conv1d, avg_pool1d` (a transformer frontend needs `conv1d`;
 anything beyond stride/padding-1D is decided by the spike).
Random: **out** for inference v1 (dropout is identity at inference;
 stochastic sampling stays behind an explicit key in the later training set).
Movement: `to_dtype, to_device, copy, contiguous`.

Deliberately missing: `softmax`, `layer_norm`, attention, `gelu` exact,
`silu`, beam search, STFT/mel — all derived or host-side (§3.2, §5).

The names above are 45 (44 without the optional `gelu_tanh_approx?`). Keyed
random and the training-only ops of §10 are outside this inference subset but
inside the frozen contract.

### 3.2 Derived ops (pure Agentica, backend-agnostic)

Built only from §3.1, in `tensor/` (generic) or `nn/` (model-shaped):

- `softmax(x, axis) = div(exp(sub(x, max)), sum(...))` with max-subtraction.
- `layer_norm(x)`, `rms_norm(x)` from `mean`, `pow`, `sqrt` and `div`
  compositions over §3.1 (no `var`/`rsqrt` primitive is needed).
- `gelu_exact`, `silu` from `erf`-free formulations (`tanh` approx first;
  exact formula only if conformance needs it).
- `linear(x, w, b?)`, `embedding(ids, table)` via `matmul`/`gather`.
- `multi_head_attention(q, k, v, mask?)`: split heads by `reshape/transpose`,
  scores by `batch_matmul`, scale by `mul`, `softmax`, context by
  `batch_matmul`, merge by `transpose/reshape`. No fused kernel in v1.
- `greedy_step(logits, temperature?)`: `div` by temperature, `softmax`,
  `argmax`. Beam search is an Agentica loop over `gather` + `top_k_derived`
  (`top_k` itself is a derived sort/select loop in v1; a native `top_k`
  primitive enters only if the spike shows the loop is unusably slow).
- `dropout(x, p)` at inference is identity (kept for `nn/` API symmetry).

If a consumer needs an op that is neither in §3.1 nor expressible above at
acceptable cost, it enters §3.1 when the contract is frozen: §5 is the only
gate for adding ops.

## 4. Inference-first API shape (proposal)

```text
Tensor.zeros(shape, dtype, backend) -> Tensor | error
Tensor.from_list(data: List[Float], shape, dtype, backend) -> Tensor | error
Tensor.to_list(t) -> Result[List[Float] | List[Int] | List[Bool]]
                                                  # host readback, tests + decode loop
Tensor.shape(t) -> Shape                          # no native call
Tensor.dtype(t) -> Dtype                          # no native call
Tensor.backend(t) -> Backend                      # no native call
```

- Every fallible op returns `Result`; shape/dtype mismatches are `error`,
  never a halt. Introspection is free (fields, no FFI round-trip).
- Batch and device are explicit arguments, never globals: dispatch is **by
  value** (`t.backend`), not by global config, so `reference` (oracle) and
  `torch` can check each other in the same process. A backend value names its
  device (`torch:cpu`, `torch:cuda:0`); `reference` has none.
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
  §3.2) → `layer_norm` → MLP (`linear`, `gelu_tanh_approx`, `linear`).
- A frontend that needs a 1-D convolution stem: `conv1d` × 2 is the reason
  `conv1d` is primitive. A feature frontend (windowed transform plus a
  filterbank) is the first `compute def` candidate: implement in pure Agentica
  against §3.1 where viable, drop to native only if the spike proves the pure
  version unusable. Either way the **signature stays Agentica-side**
  (`features(pcm: List[Float]) -> Tensor`; exact shape is the client's).
- Autoregressive decode: masked self-attention and cross-attention (derived
  MHA), output head `linear` → `softmax` → `argmax` (greedy) with optional
  temperature scaling; beam search is the Agentica loop from §3.2.
- The decode loop runs in Agentica (one `Native.arena` per step, §8); the
  backend never holds the loop.
- Weights come from a checkpoint (§6). Training (`train/`, §10) runs on the
  same primitives.

Acceptance for this design: a client reproduces, on `reference`/torch CPU,
the output it gets from its native binding for the same input.

## 6. Checkpoint requirements (weights in, no code execution)

- Format-first: **safetensors** is the first supported checkpoint format
  (8-byte LE length, UTF-8 JSON header with `dtype`, `shape`,
  `data_offsets` plus optional `__metadata__`, contiguous little-endian
  C-order buffer, no strides). No code runs at load; that property is the
  reason for choosing it.
- Covered dtypes are the safetensors set (`F32/F16/F64`, `I8/I16/I32/I64`,
  `U8`, `Bool`); compute in v1 still means §2's rule (upcast to `F32`).
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
by `case` to one backend module per variant sharing the §3.1 signature set:

```text
case t.backend
  when reference then Reference.matmul(a, b)
  when torch then Torch.matmul(a, b)
  # xla / candle join later behind the same case
end
```

- Why this shape: Agentica is static and already checks before running
  (no `defn` macros as in `Nx.Defn`); an enum + `case` keeps dispatch
  visible to the checker and lets `reference` and `torch` coexist by value
  in one process for oracle comparison. The considered alternative — a
  struct of functions per backend (object-style interface) — stays open
  until the owner settles question 1; this document does not prejudge it.
- Each native backend is one `extern module` with a thin C layer (libtorch:
  C++ as `tch-rs` does; candle later: `extern "C"` over Rust), shipped via
  `[native.NAME]` + per-platform prebuilt (CPU first; sizes documented).
  Heavy third parties (ML engines, models) enter only as pinned git
  submodules under `third_party/`, never vendored.

## 8. Tensor lifetime (proposal for ml#1 question 3)

`agentica#1091` (native arenas) semantics, which this section builds on: `Native.arena do ... end` takes **no
block parameter**, runs the block, returns its value, and at block end closes
**newest-first** every native resource and owned buffer the task opened
inside — except what the return value reaches (struct fields, enum payloads,
`Result`/`List`/`Map` items; lambda captures do **not** keep). A kept view
keeps its buffer; kept resources keep their `up` links. Inner arenas hand
kept resources to the enclosing arena; `try` failure ends the block normally;
panic/cancel ends the task, which already closes everything newest-first.
In-flight `compute def` arguments are held until the call returns (same path
as `Native.close`); use-after-close is `RUNTIME-NATIVE-CLOSED`, never UB;
`Task.spawn_owning` transfers ownership, anything else crossing tasks is
documented-not-refused.

Proposed usage for tensors (needs sign-off):

- One arena per **inference call**, and one nested arena per **decode step**:
  intermediates die at a known point instead of waiting for the GC (a 1 GB
  native tensor looks like any other `resource` to the GC; a step makes
  thousands of them).
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
- Nothing in §3.1–§4 assumes training, but every primitive must be
  differentiable in Agentica: the trace records the ops, and each op
  declares its backward rule in `tensor/` (not in a backend). `nn/`
  (forward-only layers over `tensor/`) and `train/` (losses, SGD/Adam,
  `Axon.Loop`-style loop with `std/logger`) build on them.
- Op admission rule: §3.1 is frozen once, when the contract is written. An op
  that a backward rule needs enters §3.1 then, and so does an op `train/`
  needs with no backward rule of its own — keyed random (`randn`/`randint`/
  `split`) and `stop_grad`/detach. Adding either later would be a breaking
  change to the frozen backend interface.
- Acceptance: a small MLP trains end-to-end over `torch` with `reference` as
  the oracle.

## 11. Conformance and testing (proposal)

- `reference` is the oracle. One shared suite runs on every backend:
  values, shapes, dtypes, and error paths.
- Pure Agentica gets unit tests (`*_test.agc`); inference paths get
  `.feature` + steps and runnable doc examples; native headers stay generated
  and `--check`ed in CI so nothing goes stale.
- Benchmarks measure compiled `.agcb` (`run`), never the frontend.

## 12. Open decisions (ml#1 questions 1–4)

1. Dispatch: enum + `case` (§7) vs struct of functions.
2. Surface: `Tensor.*` + pipes (§2–§4) until/unless operator overloading on
   user structs exists.
3. Lifetime: arena-per-call/step over #1091 semantics (§8) vs graph-owned
   buffers or explicit scopes.
4. Checking: eager wrapper validation now; the static-vs-runtime split (§9)
   and how much the checker carries are not fixed here.
