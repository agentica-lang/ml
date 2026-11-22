# tensor — design proposal (inference-first)

Status: **proposal, not a contract.** Implementation is in progress (the
`whisper/` scaffold is merged; this repo has no external blocking gate).
Nothing below finalizes the four open questions of ml#1; each affected
section states its working proposal so the owner can accept, change, or
reject it before the detailed per-function design.

References: `roadmap.md` (execution plan, Fase W Whisper tractor),
[ml#1](https://github.com/agentica-lang/ml/issues/1) (goals, grad-in-Agentica
decision, backend table, open questions), and
[agentica#1091](https://github.com/agentica-lang/agentica/issues/1091)
(native arenas — **verified CLOSED 2026-11-15** with phase-1 design
decisions; see §8).

## 1. Goals, and what is deferred

Goals:

- Run **training and inference**, in that order of arrival but both in scope.
  Inference first: `tensor/` + `nn/` over `reference` (pure Agentica oracle)
  and one real eager backend, proven by **Whisper-B** (the same reference wav
  as the `whisper/` binding transcribes, on `reference`/torch CPU, slower is
  acceptable).
- **Training is a goal, not a non-goal** (owner decision 2026-11-22):
  `train/` (losses, SGD/Adam, an `Axon.Loop`-style loop) and the traced
  `grad`/`value_and_grad` that it needs are part of this stack. This proposal
  only *sequences* them after inference — it does not exclude them. The
  primitives below are chosen so that tracing and grad attach **without
  changing them** (see §10), and any op that training needs (e.g. `stop_grad`,
  `sum` over all axes with a keep-dims option) is admitted from the start
  rather than bolted on later.
- Keep every backend **forward-only**. `grad`, `vmap`, `jit` live in
  Agentica over a traced graph (ml#1 key decision, as Nx does); XLA has no
  autograd of its own, so per-backend autograd is out.
- Keep the backend interface small enough to implement four times
  (`reference`, `torch`, later `xla`, `candle`) with one shared conformance
  suite where the reference is the oracle.

Deferred in this proposal (sequenced later, still goals):

- The JIT compiler (`xla/` via PJRT) and the model hub (`hub/` download and
  sharding UX). §10 sketches how tracing/grad attach without changing the
  primitives.
- A ggml quantized backend (phase 8; the `whisper/` binding already uses
  ggml through whisper.cpp and is out of scope here).
- Final signatures (`.agc` + native-gen headers come in Fase 0/M1, after the
  spike freezes the minimum).

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
  when the Whisper checklist (§5) or conformance demands it.
- No operator overloading (ml#1 question 2, still open). Plan A, as today:
  `Tensor.add(a, b)` plus pipes (`t |> Tensor.add(other) |> Tensor.relu()`).
  If the language ever allows `a + b` on user structs, it is sugar over this.

## 3. Primitives vs derived ops (the split)

**Primitives** are the only functions a backend implements. Everything else
is **derived**: pure Agentica code over primitives, shared by all backends
and therefore tested once.

### 3.1 Primitives (proposal: the backend contract, ~35 ops inference subset)

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
Convolution/pooling: `conv1d, avg_pool1d` (Whisper frontend needs `conv1d`;
 anything beyond stride/padding-1D is deferred to the spike).
Random: **out** for inference v1 (dropout is identity at inference;
 stochastic sampling stays behind an explicit key in the later training set).
Movement: `to_dtype, to_device, copy, contiguous`.

Deliberately missing: `softmax`, `layer_norm`, attention, `gelu` exact,
`silu`, beam search, STFT/mel — all derived or host-side (§4–§5).

### 3.2 Derived ops (pure Agentica, backend-agnostic)

Built only from §3.1, in `tensor/` (generic) or `nn/` (model-shaped):

- `softmax(x, axis) = div(exp(sub(x, max)), sum(...))` with max-subtraction.
- `layer_norm(x)`, `rms_norm(x)` from `mean/var/rsqrt` compositions.
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

If Whisper-B needs an op that is neither in §3.1 nor expressible above at
acceptable cost, it enters §3.1 through the Fase 0 spike — the checklist in
§5 is the gate, not an excuse to grow the set upfront.

## 4. Inference-first API shape (proposal)

```text
Tensor.zeros(shape, dtype, backend) -> Tensor | error
Tensor.from_list(data: List[Float], shape, dtype, backend) -> Tensor | error
Tensor.to_list(t) -> Result[List[Float]]          # host readback, tests + decode loop
Tensor.shape(t) -> Shape                          # no native call
Tensor.dtype(t) -> Dtype                          # no native call
Tensor.backend(t) -> Backend                      # no native call
```

- Every fallible op returns `Result`; shape/dtype mismatches are `error`,
  never a halt. Introspection is free (fields, no FFI round-trip).
- Batch and device are explicit arguments, never globals: dispatch is **by
  value** (`t.backend`), not by global config, so `reference` (oracle) and
  `torch` can check each other in the same process.
- The `.agc` signatures plus the generated native header are the contract
  once written; until then this document is the only source, and it is still
  a proposal.

## 5. Whisper forward requirements (checklist that gates §3.1)

Whisper-B transcribes the same fixture wav as Via A (16 kHz mono PCM floats
from `Wav.decode`; file first, microphone later per roadmap Fase W).
Frontend and model below are **requirements on** the API, not designs of it.

Frontend (log-mel, 80 bins, 30 s windows):

- STFT + mel filterbank + log is the first `compute def` candidate: implement
  in pure Agentica against §3.1 where viable; drop to native only if the
  spike proves the pure version unusable. Either way the **signature stays
  Agentica-side** (`log_mel(pcm: List[Float]) -> Tensor[80, 3000]` shape
  proposal — exact frames fixed at detailed design).
- `conv1d` × 2 (frontend encoder stem) is the reason `conv1d` is primitive.

Encoder (transformer, sizes differ small/medium — shapes stay parametric):

- Learned/sinusoidal positional embeddings via `arange + gather` or
  `full + add` (no dedicated primitive).
- N blocks of: `layer_norm` (derived) → multi-head self-attention
  (derived, §3.2) → `layer_norm` → MLP (`linear`, `gelu`, `linear`).
- All weights come from checkpoint (§6); no training path is needed for the
  Whisper-B milestone. Training arrives in `train/` (§10) on the same
  primitives.

Decoder (autoregressive, greedy first):

- Masked self-attention + cross-attention (both derived MHA) + MLP, same
  blocks as the encoder plus the cross-attention leg.
- Output head: `linear` to vocabulary, `softmax`, `argmax` (greedy) with
  optional temperature scaling; beam search is the Agentica loop from §3.2.
- The decode loop runs in Agentica (one `Native.arena` per step, §8);
  the backend never holds the loop.

Exit criterion (unchanged from roadmap): the Via A fixture wav transcribes
to the same text under the pure stack on `reference`/torch CPU.

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
  `load_slice(path, entry, backend) -> Tensor`; multi-file sharding plus
  its JSON index is the second load case after single-file Whisper weights.
- `mmap` where the backend supports it, plain read where it does not;
  the choice is per-backend, hidden behind the same loader signature.
- `gguf`/ggml checkpoints stay with the `whisper/` binding (Via A);
  the pure stack does not read them in v1.

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
  `[native.NAME]` + per-platform prebuilt (CPU first; sizes documented, as
  with `whisper/`). Heavy third parties (`whisper.cpp`, `ggml`, libtorch,
  models) enter only as pinned git submodules under `third_party/`, never
  vendored — per the 2026-11-22 repo decision.

## 8. Tensor lifetime (proposal for ml#1 question 3 — #1091 verified)

`agentica#1091` (native arenas) is **CLOSED since 2026-11-15** with phase-1
decisions. Verified semantics used here: `Native.arena do ... end` takes **no
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

Proposed usage for tensors (pending owner sign-off):

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
- How much the static checker carries vs runtime stays an **owner decision**:
  this proposal asks only that trace/compile time reuse the same predicates,
  and that `xla/` later requires static shapes to compile the whole graph.
- Conformance covers values **plus** shapes, dtypes, and error cases on every
  backend, bit-for-bit against `reference` where the dtype is exact.

## 10. Training: tracing and grad (sequenced after inference, in scope)

- Functions trace to a graph of §3.1 primitives; `grad`/`value_and_grad` are
  computed **in Agentica over that graph** (ml#1 key decision). Backends stay
  forward-only: no per-backend autograd.
- The graph must be compilable (static shapes) so `xla/` can later compile
  it whole via the PJRT C API (`pjrt_c_api.h`, `dlopen` plugin, no own layer).
  `vmap`/`jit` also live in Agentica.
- Nothing in §3.1–§4 assumes training, but every primitive must be
  differentiable in Agentica: the trace records the ops, and each op
  declares its backward rule in `tensor/` (not in a backend). `nn/`
  (forward-only layers over `tensor/`) comes before `train/` (losses,
  SGD/Adam, `Axon.Loop`-style loop with `std/logger`), per roadmap phases 3–4.
- Op admission rule: if an op cannot express a backward rule (or its rule
  needs an op missing here), the missing op is added to §3.1 then, not
  deferred — otherwise `train/` would force a breaking change to the frozen
  backend interface.
- Exit criterion for this stack (roadmap M4): a small MLP trains end-to-end
  over `torch` with `reference` as the oracle.

## 11. Conformance and testing (proposal)

- `reference` is the oracle. One shared suite runs on every backend:
  values, shapes, dtypes, and error paths.
- Pure Agentica gets unit tests (`*_test.agc`); inference paths get
  `.feature` + steps and runnable doc examples; native headers stay generated
  and `--check`ed in CI so nothing goes stale.
- Benchmarks measure compiled `.agcb` (`run`), never the frontend.

## 12. What needs the owner (maps to ml#1 questions 1–4)

1. Dispatch: enum + `case` (§7) vs struct of functions — proposal, not settled.
2. Surface: `Tensor.*` + pipes (§2–§4) until/unless operator overloading on
   user structs exists — plan A stands.
3. Lifetime: arena-per-call/step over verified #1091 semantics (§8) vs
   graph-owned buffers or explicit scopes — proposal, not settled.
4. Checking: eager wrapper validation now, static-vs-runtime split later
   (§9) — load on the checker TBD.
