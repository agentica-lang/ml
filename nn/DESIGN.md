# nn — design

Functional layers over `tensor/`, forward only (inference). Depends only on
`tensor/` (`path = "../tensor"`), so it runs on every backend `tensor/` has:
the reference and libtorch today.

## API

```text
struct Dense        { w: [in, out], b: Optional [out] }
struct LayerNorm    { gamma: [width], beta: [width], eps: Float }
struct Embedding    { table: [rows, width] }
enum   Activation   = Relu | Gelu | Silu
struct Attention    { query, key, value, output: Dense, heads: Int }
struct Mlp          { up, down: Dense, activation: Activation }
struct EncoderBlock { norm1: LayerNorm, attention: Attention, norm2: LayerNorm, mlp: Mlp }

Nn.dense(p, x)                     x [..., in] -> [..., out]
Nn.layer_norm(p, x)                last axis, then * gamma + beta
Nn.embedding(p, ids)               ids rank-1 integer -> [ids, width]
Nn.positional(p, x)                x [seq, width] or [batch, seq, width] + table rows 0...seq
Nn.activate(a, x)
Nn.attention(p, queries, memory, mask?)   [batch, q, width] over [batch, k, width]
Nn.self_attention(p, x, mask?)
Nn.mlp(p, x)
Nn.encoder_block(p, x, mask?)      pre-norm: h = x + attn(norm1(x)); h + mlp(norm2(h))

Nn.init_dense / init_layer_norm / init_embedding / init_attention /
  init_mlp / init_encoder_block(..., seed, backend) -> Result[layer]
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
  (`tensor/DESIGN.md` §10). The gradient reaches every op that has a backward
  rule today: `dense` and `mlp` with `relu` do; `layer_norm` (`sqrt`) and
  attention (`softmax` through `max`) wait for their rules in `tensor/`.
- **Seeded initializers in Agentica.** `tensor/` has no random op at
  inference (`tensor/DESIGN.md` §3.1), so `init.agc` draws from a
  Park-Miller generator and crosses with `from_list`: one seed gives the same
  weights on every backend, which is what lets the tests compare a model on
  the reference with the same model on libtorch. Sub-layers draw from
  `Init.derive(seed, k)`. Only uniform and Glorot uniform: real weights come
  from checkpoints (`hub/`).
- **`F32` only** for `init_*`, the compute dtype of `tensor/DESIGN.md` §2.
  Layers themselves take any float dtype (the gradient test uses `F64`).
- **Left out until a client needs them:** dropout (identity at inference,
  `Tensor.dropout` is there), a causal mask helper, cross-attention blocks
  (the decoder), conv layers, `rms_norm`. `gelu` does not run on libtorch yet
  (`gelu_tanh_approx` is one of its three missing ops), so the libtorch
  comparisons use `relu` and `silu`.
