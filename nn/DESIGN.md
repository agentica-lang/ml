# nn — design

Functional layers over `tensor/`, forward only (inference). Depends only on
`tensor/` (a member of the same workspace, read from `../tensor`). The
reference and libtorch run a layer eagerly; ggml runs it traced only: the
layer is called on `Trace` tensors and the graph goes to `Trace.eval` or
`Trace.compile` (`tensor/DESIGN.md` §10).

## API

```text
struct Dense        { w: [in, out] ([out, in] with out_in), b: Optional [out], out_in: Bool = false }
struct LayerNorm    { gamma: [width], beta: [width], eps: Float, rms: Bool = false }
struct Embedding    { table: [rows, width] }
struct Conv1d       { weight: [out, in, kernel], bias: Optional [out], stride, padding: Int }
enum   Activation   = Relu | Gelu | GeluErf | Silu
struct Rope         { base: Float = 10000.0, neox: Bool = true, dims: Optional[Int] = none }
struct Attention    { query, key, value, output: Dense, heads: Int, kv_heads: Optional[Int] = none,
                      rope: Optional[Rope] = none, query_norm, key_norm: Optional[LayerNorm] = none }
struct Mlp          { up, down: Dense, activation: Activation, gate: Optional[Dense] = none }
struct EncoderBlock { norm1: LayerNorm, attention: Attention, norm2: LayerNorm, mlp: Mlp,
                      norm_first: Bool = true }
struct DecoderBlock { norm1, self_attention: Attention, norm2, cross_attention: Attention,
                      norm3: LayerNorm, mlp: Mlp, norm_first: Bool = true }
struct EncoderStack { blocks: List[EncoderBlock], final_norm: Optional[LayerNorm] = none }
struct DecoderStack { blocks: List[DecoderBlock], final_norm: Optional[LayerNorm] = none }
struct BlockOptions { kv_heads, rope, qk_norm, bias, rms, gated, norm_first, eps }   (for init_*)

Nn.dense(p, x)                     x [..., in] -> [..., out]
Nn.cosine_similarity(a, b)         [batch, width] twice -> [batch], the cosine of each pair of rows
Nn.layer_norm(p, x)                last axis, then * gamma + beta
Nn.embedding(p, ids)               ids rank-1 integer -> [ids, width]
Nn.conv1d(p, x)                    x [in, L] or [batch, in, L] -> Tensor.conv1d + bias per channel
Nn.positional(p, x)                x [seq, width] or [batch, seq, width] + table rows 0...seq
Nn.activate(a, x)
Nn.attention(p, queries, memory, mask?, causal?, positions?, memory_positions?)
                                   [batch, q, width] over [batch, k, width]: project, then attend
Nn.project(p, queries, memory, positions?, memory_positions?) -> Heads { query, key, value }
                                   split by head, q/k norms, rope at `positions` (default 0...seq)
Nn.attend(p, query, key, value, mask?, causal?)   Tensor.attention over heads or a cache, merged, output dense
Nn.self_attention(p, x, mask?, causal?)
Nn.mlp(p, x)                       down(act(up x)), or down(act(gate x) * up x) with a gate
Nn.encoder_block(p, x, mask?, causal?)
                                   pre-norm: h = x + attn(norm1(x)); h + mlp(norm2(h))
                                   post-norm: h = norm1(x + attn(x)); norm2(h + mlp(h))
Nn.causal_mask(size, dtype, backend)   [size, size]: 0.0 on and below the diagonal, MASKED above
Nn.decoder_block(p, x, memory)     x [batch, t, width] over memory [batch, s, width]:
                                   h = x + self_attn(norm1(x), causal);
                                   h = h + attn(cross, norm2(h), memory); h + mlp(norm3(h))
Nn.encoder_stack(p, x, mask?, causal?) / decoder_stack(p, x, memory)
                                   the blocks in order, then final_norm

Nn.init_dense / init_layer_norm / init_embedding / init_attention /
  init_mlp / init_encoder_block / init_decoder_block / init_encoder_stack /
  init_decoder_stack(..., seed, backend, options?) -> Result[layer]
Init.uniform(shape, limit, seed, backend), Init.glorot_uniform(shape, seed, backend)
```

Every function returns `Result`; the checks are `tensor/`'s, plus the few
that are the layer's own (rank of `positional`, table length, heads).

## Decisions

- **A layer is a struct of parameters and a function `layer(params, x)`.**
  No classes, no state, no model object: a model is a struct of layer
  structs, and a training step will return new structs. Structs over maps
  because the fields are fixed and the checker then knows them.
- **No math of its own.** Every layer composes `tensor/`'s public calls,
  mostly its derived ops (`linear`, `layer_norm`, `rms_norm`, `embedding`,
  `gelu`, `silu`) and its attention primitives (`attention`, `permute`,
  `rope`). nn/ adds what is model-shaped: the
  parameters, the affine part of the norm, the attention projections, the
  residual block, and folding leading axes so `dense` takes any rank.
- **Differentiable for free.** Since a layer is `Tensor.*` calls, it records a
  graph on `Trace` tensors and `Trace.value_and_grad` differentiates it
  (`tensor/DESIGN.md` §10). The tests check `dense`, a relu and a silu
  `mlp`, `embedding` with `positional`, a gelu `encoder_block` (the
  gradient of its input and of one gamma) and a silu `decoder_block` (its
  input and its encoder output) against a central difference.
- **The cosine has no clamp.** PyTorch divides by each norm clamped to
  `1e-8`; `clamp` has no backward rule, so `cosine_similarity` divides by
  `sqrt(|a|^2 |b|^2 + 1e-16)`: the same values to `F32` rounding for a row
  that is not zero, and a cosine of 0 with a finite gradient for one that is.
  In `F16` the term would round to 0; training in `F16` is not a goal.
- **Options are fields with defaults.** A model changes a detail of a block,
  so the detail is a field: `kv_heads`, `rope` and the query and key norms of
  `Attention`, `gate` of `Mlp`, `rms` of `LayerNorm`, `norm_first` of the
  blocks. Each default is the old behaviour, so a struct built field by field
  (as `hub/` does) keeps working. `BlockOptions` carries the same choices to
  `init_*`; it is not a layer. The cross-attention of a decoder block takes
  only its `bias`. A loader that rebuilds a layer (`hub/`, ml#78) must use
  `.with(...)` on the struct it has, not a constructor call, or it drops the
  options added later. The head width is the width of `query` over `heads`,
  so loaded weights may have a head width other than `width / heads`
  (Qwen3); `init_attention` makes only `width / heads`.
- **`project` and `attend` split attention for a cache.** `nn/` has no KV cache,
  but a decoder that keeps one must rotate the new token's key at its own
  position and read the cached keys as they are: `project` takes the integer
  `positions` (queries) and `memory_positions` (keys; the queries' when only
  those are given: only `positions` means self-attention, pass
  `memory_positions` for cross-attention), and `attend` takes the heads or a
  cache of them.
  `attention` is their composition, with the same results as before.
- **Attention is `Tensor.attention`.** `attention` splits the projections into
  heads with `reshape` and `permute`, applies the query and key norms, then
  `rope` (positions `0...seq` of each input, so a cross-attention gets rope
  only if its `Attention` has one), runs `Tensor.attention` and merges the
  heads. Grouped-query attention is `kv_heads` fewer than `heads` (it must
  divide them); the mask is `Tensor.attention`'s (`[q, k]` or
  `[batch|1, heads|1, q, k]`).
- **Pre-norm and post-norm.** `norm_first: true` is the pre-norm of Whisper,
  Qwen and Llama; `false` is BERT's post-norm. A pre-norm stack ends in
  `final_norm`; a post-norm one has none. `causal: true` on `encoder_block`
  makes it a decoder-only block (Qwen). `RMSNorm` is `LayerNorm` with `rms`:
  `beta` is kept and ignored.
- **The decoder is Whisper's, pre-norm by default.** Self-attention is causal
  (`causal: true` in `Tensor.attention`, which adds `-1e9` where a query must
  not see a key: finite, and far enough down that its softmax weight is `0`;
  `causal_mask` builds the same bias for a caller that wants it) and
  cross-attention reuses `attention(p, queries, memory)`: queries come from
  the decoder and keys and values from the encoder output, so `t` and `s`
  differ freely. With fewer queries than keys the mask ends on the last key
  (the end-aligned mask of a decode step); more queries than keys is an
  error, and on `Reference` `q * k` of the mask is at most 1,048,576
  (`Tensor.full`). It assumes an `F32` or `F64` compute dtype. There is no KV
  cache and no decode loop: each call recomputes every position.
- **Weights are `[in, out]`** (`x w`, as `Tensor.linear` takes them), or
  `[out, in]` with `Dense.out_in`. Hugging Face and PyTorch checkpoints store
  `[out, in]`, and `hub/` keeps that layout: `dense` multiplies by its
  transpose, which ggml reads with no copy (a weight `[in, out]` is copied
  transposed on every run: 171 of the 296 ms of kernels of a whisper-medium
  step on Vulkan). `Tensor.linear(..., out_in)` reads it as it is on ggml and
  libtorch.
- **Attention costs what `tensor/` costs.** `Tensor.attention` is one op on
  every backend (a fused kernel on ggml and libtorch), so a layer is a few
  nodes. `Tensor.multi_head_attention` remains in `tensor/` as a wrapper over
  it for callers with `[batch, seq, width]` and no `nn/` layer (`asr/` until it
  moves); `nn/` does not call it.
- **Seeded initializers in Agentica.** `tensor/` has no random op at
  inference (`tensor/DESIGN.md` §3.1), so `init.agc` draws from a
  Park-Miller generator and crosses with `from_list`: one seed gives the same
  weights on every backend, which is what lets the tests compare a model on
  the reference with the same model on libtorch. Sub-layers draw from
  `Init.derive(seed, k)`. Only uniform and Glorot uniform: real weights come
  from checkpoints (`hub/`).
- **`F32` only** for `init_*`, the compute dtype of `tensor/DESIGN.md` §2.
  Layers themselves take any float dtype (the gradient test uses `F64`).
  The gradient tests need `agentica test --max-steps 10000000`: a bare
  `agentica test` runs out of its 1M steps (per file, times its tests + 1).
- **Two GELUs**: `Gelu` is the tanh approximation (`Tensor.gelu`);
  `GeluErf` is the exact `erf` form Whisper and BERT use
  (`Tensor.gelu_erf`). `Trace` differentiates both. The libtorch comparisons
  use `relu` and `silu`.
