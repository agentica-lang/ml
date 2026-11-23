# nn — design

Functional layers over `tensor/`, forward only (inference). Depends only on
`tensor/` (a member of the same workspace, read from `../tensor`), so it runs
on every backend `tensor/` has: the reference and libtorch today.

## API

```text
struct Dense        { w: [in, out], b: Optional [out] }
struct LayerNorm    { gamma: [width], beta: [width], eps: Float }
struct Embedding    { table: [rows, width] }
struct Conv1d       { weight: [out, in, kernel], bias: Optional [out], stride, padding: Int }
enum   Activation   = Relu | Gelu | Silu
struct Attention    { query, key, value, output: Dense, heads: Int }
struct Mlp          { up, down: Dense, activation: Activation }
struct EncoderBlock { norm1: LayerNorm, attention: Attention, norm2: LayerNorm, mlp: Mlp }
struct DecoderBlock { norm1, self_attention: Attention, norm2, cross_attention: Attention,
                      norm3: LayerNorm, mlp: Mlp }

Nn.dense(p, x)                     x [..., in] -> [..., out]
Nn.layer_norm(p, x)                last axis, then * gamma + beta
Nn.embedding(p, ids)               ids rank-1 integer -> [ids, width]
Nn.conv1d(p, x)                    x [in, L] or [batch, in, L] -> Tensor.conv1d + bias per channel
Nn.positional(p, x)                x [seq, width] or [batch, seq, width] + table rows 0...seq
Nn.activate(a, x)
Nn.attention(p, queries, memory, mask?)   [batch, q, width] over [batch, k, width]
Nn.self_attention(p, x, mask?)
Nn.mlp(p, x)
Nn.encoder_block(p, x, mask?)      pre-norm: h = x + attn(norm1(x)); h + mlp(norm2(h))
Nn.causal_mask(size, dtype, backend)   [size, size]: 0.0 on and below the diagonal, MASKED above
Nn.decoder_block(p, x, memory)     x [batch, t, width] over memory [batch, s, width]:
                                   h = x + self_attn(norm1(x), causal_mask(t));
                                   h = h + attn(cross, norm2(h), memory); h + mlp(norm3(h))

Nn.init_dense / init_layer_norm / init_embedding / init_attention /
  init_mlp / init_encoder_block / init_decoder_block(..., seed, backend) -> Result[layer]
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
  mostly its derived ops (`linear`, `layer_norm`, `embedding`, `gelu`,
  `silu`, `multi_head_attention`). nn/ adds what is model-shaped: the
  parameters, the affine part of the norm, the attention projections, the
  residual block, and folding leading axes so `dense` takes any rank.
- **Differentiable for free.** Since a layer is `Tensor.*` calls, it records a
  graph on `Trace` tensors and `Trace.value_and_grad` differentiates it
  (`tensor/DESIGN.md` §10). The tests check `dense`, a relu and a silu
  `mlp`, `embedding` with `positional`, a gelu `encoder_block` (the
  gradient of its input and of one gamma) and a silu `decoder_block` (its
  input and its encoder output) against a central difference.
- **The decoder is Whisper's, pre-norm.** Self-attention is causal and
  cross-attention reuses `attention(p, queries, memory)`: queries come from
  the decoder and keys and values from the encoder output, so `t` and `s`
  differ freely (`multi_head_attention` already allows it). The causal mask is
  the additive bias that `mask` already takes, built from `arange`, `lt` and
  `where` on the input's backend, so it costs no new op and, on `Trace`, is a
  constant, never differentiated. `MASKED` is `-1e9`, not `-inf`: finite, and
  far enough down that its softmax weight is `0`. It assumes an `F32` or `F64`
  compute dtype and a row with at least one key it may see (the causal mask
  always leaves the diagonal). There is no KV cache and no decode loop: each
  call recomputes every position.
- **Weights are `[in, out]`** (`x w`, as `Tensor.linear` takes them).
  Hugging Face and PyTorch checkpoints store `[out, in]`, so `hub/` transposes
  at load time.
- **Attention costs what `tensor/` costs.** `Tensor.multi_head_attention`
  loops over batch, heads and rows in Agentica; a fused version belongs in
  `tensor/`, not here.
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
- **`gelu` does not run on libtorch** (`gelu_tanh_approx` is one of its three
  missing ops), so the libtorch comparisons use `relu` and `silu`. `gelu` is
  the tanh approximation, while Whisper uses the `erf` form: expect drift near
  1e-3 on real checkpoints.
