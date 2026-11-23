---
title: nn
description: Functional layers over tensor (dense, norms, embeddings, convolution, attention, MLPs and transformer blocks), with seeded initializers.
weight: 2
---

`nn` has the layers that the models of this repository are made of. A layer is a struct of parameters and a function `layer(params, x)`. Nothing holds state or changes in place: a model is a struct of layer structs, and a training step returns new ones.

Every layer is made of `tensor` calls, so it runs on every backend that `tensor` has. On `Trace` tensors it records a graph, which `Trace.value_and_grad` differentiates. `nn` has no math of its own.

## Example

A dense layer and a transformer encoder block on the reference backend. The shapes are the ones the layers take: `dense` takes any rank, and attention needs rank 3.

```agentica
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  ref = Tensor.Backend.Reference
  layer = Nn.init_dense(3, 2, 7, ref).unwrap!
  x = Tensor.ones([4, 3]).unwrap!
  puts(Tensor.shape(Nn.dense(layer, x).unwrap!))
  block = Nn.init_encoder_block(4, 2, 8, Nn.Activation.Gelu, 3, ref).unwrap!
  h = Tensor.ones([1, 5, 4]).unwrap!
  puts(Tensor.shape(Nn.encoder_block(block, h).unwrap!))
end
```

```output
[4, 2]
[1, 5, 4]
```

## Layers

| Struct | Holds | Function |
|---|---|---|
| `Dense` | `w` as `[in, out]`, `b` optional `[out]` | `dense(p, x)`: `x w + b` on the last axis |
| `LayerNorm` | `gamma`, `beta`, `eps` | `layer_norm(p, x)` |
| `Embedding` | `table` as `[rows, width]` | `embedding(p, ids)`; `positional(p, x)` adds the first rows of the table |
| `Conv1d` | `weight` as `[out, in, kernel]`, `bias`, `stride`, `padding` | `conv1d(p, x)` |
| `Attention` | four `Dense` projections and `heads` | `attention(p, queries, memory, mask?)`, `self_attention(p, x, mask?)` |
| `Mlp` | `up`, `down`, `activation` | `mlp(p, x)`: `dense`, activation, `dense` |
| `EncoderBlock` | two norms, attention, MLP | `encoder_block(p, x, mask?)`, pre-norm |
| `DecoderBlock` | three norms, self and cross attention, MLP | `decoder_block(p, x, memory)`, causal self attention, then cross attention |

`Activation` is one of these:

- `Relu` and `Silu`, the plain ones.
- `Gelu`, the tanh approximation, which `Trace` differentiates.
- `GeluErf`, the exact form that Whisper uses; `Trace` refuses it.

`causal_mask(size, dtype, backend)` builds the mask that `decoder_block` uses.

## Init

Each layer has an `init_*` that takes a seed and a backend and returns a `Result`:

- `init_dense(inputs, outputs, seed, backend)`
- `init_layer_norm(width, backend)`
- `init_embedding(rows, width, seed, backend)`
- `init_attention(width, heads, seed, backend)`
- `init_mlp(width, hidden, activation, seed, backend)`
- `init_encoder_block(width, heads, hidden, activation, seed, backend)`
- `init_decoder_block(width, heads, hidden, activation, seed, backend)`

- A dense weight starts Glorot-uniform, and a bias starts at zero. A layer norm starts as the plain normalization.
- `import Init from "pkg/nn/init"` gives `Init.glorot_uniform(shape, seed, backend)` for a rank-2 shape, and `Init.uniform(shape, limit, seed, backend)`.
- One seed gives the same `F32` values on every backend. The sub-layers of one block take their seeds from `Init.derive(seed, k)`.
- The generator is for starting weights only. Real weights come from a checkpoint, with `hub`.

## Limits

- `init_*` make `F32` tensors. The layers themselves take any float dtype.
- `attention` runs `tensor`'s `multi_head_attention`, which loops over batch, heads and rows in Agentica. There is no fused kernel.
- `decoder_block` has no KV cache: each call recomputes every position. `asr` has its own cache.
- `causal_mask` uses `-1e9` for a masked position, which assumes an `F32` or `F64` compute dtype.
- `init_attention` needs a width that is divisible by the head count. `positional` refuses more positions than its table has.
- `GeluErf` has no gradient through `Trace`.
