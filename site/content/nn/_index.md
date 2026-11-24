---
title: nn
description: Functional layers over tensor (dense, norms, embeddings, convolution, attention, MLPs and transformer blocks), with seeded initializers.
weight: 2
---

`nn` has the layers that models are built from: dense, layer norm, embeddings, convolution, attention, MLPs and transformer blocks. A layer is a struct of weights and a function that applies it to an input.

## Example

Make a dense layer with 3 inputs and 2 outputs, and run it on 4 rows:

```agentica
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  layer = Nn.init_dense(3, 2, 7, Tensor.Backend.Reference).unwrap!
  x = Tensor.ones([4, 3]).unwrap!
  y = Nn.dense(layer, x).unwrap!
  puts(Tensor.shape(y))
end
```

```output
[4, 2]
```

The shapes are the ones the layers take. `dense` takes any rank. Attention needs rank 3.

## Layers

| Struct | Holds | Function |
|---|---|---|
| `Dense` | `w` as `[in, out]` (or `[out, in]` with `out_in`), `b` optional `[out]` | `dense(p, x)`: `x w + b` on the last axis |
| `LayerNorm` | `gamma`, `beta`, `eps`, `rms` | `layer_norm(p, x)`; with `rms` it is an RMS norm and `beta` is ignored |
| `Embedding` | `table` as `[rows, width]` | `embedding(p, ids)`; `positional(p, x)` adds the first rows of the table |
| `Conv1d` | `weight` as `[out, in, kernel]`, `bias`, `stride`, `padding` | `conv1d(p, x)` |
| `Attention` | four `Dense` projections, `heads`, and optional `kv_heads`, `rope`, `query_norm`, `key_norm` | `attention(p, queries, memory, mask?, causal?, positions?, memory_positions?)`, `self_attention(p, x, mask?, causal?)`; `project` and `attend`, the two halves of `attention` |
| `Mlp` | `up`, `down`, `activation`, optional `gate` | `mlp(p, x)`: `dense`, activation, `dense`; with a `gate`, `down(activation(gate(x)) * up(x))` |
| `EncoderBlock` | two norms, attention, MLP, `norm_first` | `encoder_block(p, x, mask?, causal?)` |
| `DecoderBlock` | three norms, self and cross attention, MLP, `norm_first` | `decoder_block(p, x, memory)`, causal self attention, then cross attention |
| `EncoderStack`, `DecoderStack` | `blocks`, optional `final_norm` | `encoder_stack(p, x, mask?, causal?)`, `decoder_stack(p, x, memory)`: the blocks in order, then the final norm |

An activation is a function applied to each value, such as `relu`. The `Activation` type is one of these:

- `Relu` and `Silu`.
- `Gelu`, a smooth activation in its tanh approximation. `Trace` can take its gradient.
- `GeluErf`, the exact form that Whisper uses. `Trace` refuses it.

A causal mask stops each position from seeing the positions after it. `causal: true` applies it inside `attention`, and `causal_mask(size, dtype, backend)` builds the same additive mask for a caller that wants it.

The module `pkg/nn/kv` holds what a decoder that steps one token at a time shares: `capacity(needed, limit)` (8, 16, 32, ... rows, so a compiled step changes shape only when the capacity grows), `Slots`, `padded_mask(slots, dtype)`, `spread_of(slots, dtype)` and `written(old, fresh, spread, axis)`, which puts the new rows of `fresh` at the positions of the slots in a cache padded with zeros (`axis` is the rows axis of both). On `Ggml` a model can write the rows in place with `Tensor.set_rows` instead.

## Transformer block

An encoder block on the reference backend, run on an input of shape `[1, 5, 4]`:

```agentica
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  ref = Tensor.Backend.Reference
  block = Nn.init_encoder_block(4, 2, 8, Nn.Activation.Gelu, 3, ref).unwrap!
  h = Tensor.ones([1, 5, 4]).unwrap!
  puts(Tensor.shape(Nn.encoder_block(block, h).unwrap!))
end
```

```output
[1, 5, 4]
```

## Model layouts

A model changes a detail of the block, so each detail is an option. The default is the layout of Whisper's encoder.

- `norm_first: false` makes the block post-norm, as BERT: `norm(x + attention(x))` in place of `x + attention(norm(x))`. A pre-norm stack ends in a final norm and a post-norm one does not.
- `kv_heads` fewer than `heads` is grouped-query attention: each key and value head serves several query heads. It must divide `heads`.
- `rope` rotates the queries and keys by the position `0...seq` of each input, or by the integer vector `positions` (queries) and `memory_positions` (keys) when `attention` or `project` gets them. Only `positions` means self-attention (the keys use them too); pass `memory_positions` for cross-attention. `Rope` has `base`, `neox` (the pairing of the columns) and `dims`.
- `query_norm` and `key_norm` normalize each head over its width before `rope` (Qwen3).
- `LayerNorm` with `rms: true` is an RMS norm. An `Mlp` with a `gate` is a gated one (SwiGLU with `Silu`).
- `encoder_block` with `causal: true` is a decoder-only block, as Qwen's. There is no separate decoder-only function.

`init_*` take these choices in a `BlockOptions` as their last argument (`eps` sets the norm epsilon, `0.000001` for Qwen3). A Qwen-like stack of two layers over width 16, with no biases, run with `causal: true` as a decoder-only model:

```agentica
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  ref = Tensor.Backend.Reference
  options = Nn.BlockOptions(
    kv_heads: some(2),
    rope: some(Nn.Rope()),
    qk_norm: true,
    bias: false,
    rms: true,
    gated: true,
  )

  stack = Nn.init_encoder_stack(2, 16, 4, 24, Nn.Activation.Silu, 3, ref, options).unwrap!
  h = Tensor.ones([1, 5, 16]).unwrap!
  puts(Tensor.shape(Nn.encoder_stack(stack, h, none, true).unwrap!))
end
```

```output
[1, 5, 16]
```

## Start values

Each layer has an `init_*` function. It takes a seed (a number that makes the random start the same on every run) and a backend, and returns a `Result`:

- `init_dense(inputs, outputs, seed, backend)`
- `init_layer_norm(width, backend)`
- `init_embedding(rows, width, seed, backend)`
- `init_attention(width, heads, seed, backend, options?)`
- `init_mlp(width, hidden, activation, seed, backend, options?)`
- `init_encoder_block(width, heads, hidden, activation, seed, backend, options?)`
- `init_decoder_block(width, heads, hidden, activation, seed, backend, options?)`
- `init_encoder_stack(layers, width, heads, hidden, activation, seed, backend, options?)` and `init_decoder_stack`, with the same arguments

- A dense weight starts with Glorot-uniform values (random, on a common scale), and a bias starts at zero. A layer norm starts as the plain normalization.
- `import Init from "pkg/nn/init"` gives `Init.glorot_uniform(shape, seed, backend)` for a rank-2 shape, and `Init.uniform(shape, limit, seed, backend)`.
- One seed gives the same `F32` values on every backend. The sub-layers of one block take their seeds from `Init.derive(seed, k)`.
- The random generator only makes starting weights. Real weights come from a checkpoint, loaded with [hub](../hub/_index.md).

## Limits

- `init_*` make `F32` tensors. The layers themselves take any float dtype.
- `attention` runs `tensor`'s `attention`, one op on every backend. `multi_head_attention` stays in `tensor` as a wrapper over it, and `nn` does not call it.
- A layer runs eagerly on `Reference` and libtorch. On ggml it runs traced only: call it on `Trace` tensors and evaluate or compile the graph.
- `causal` with fewer queries than keys ends on the last key (the end-aligned mask of a decode step), and with more queries than keys it is an error. On `Reference`, `q * k` of the causal mask stops at 1,048,576.
- `init_attention` makes heads of `width / heads`; weights loaded from a checkpoint may have another head width (Qwen3), and the layers run them.
- Rotary positions of a layer start at 0 for each input unless `positions` says otherwise. There is no KV cache in `nn`: a decoder that has one calls `project` on the new token alone with its position (`positions` of `[n]` for position `n`), appends the key, already rotated, to its cache, and gives `attend` the cache. One position vector serves every batch row and head.
- `decoder_block` has no KV cache (a store of past keys and values): each call recomputes every position. `asr` has its own cache.
- `causal` and `causal_mask` use `-1e9` for a masked position, which assumes an `F32` or `F64` compute dtype. `causal` needs no more queries than keys.
- `init_attention` needs a width that is divisible by the head count, and `kv_heads` that divides it. `positional` refuses more positions than its table has.
- `GeluErf` has no gradient through `Trace`.

## How it works

A layer has no state of its own, and no layer changes in place. A model is a struct of layer structs, and a training step returns new ones.

Every layer is made of `tensor` calls, so it runs on every backend that `tensor` has. On `Trace` tensors it records a graph, which `Trace.value_and_grad` differentiates. `nn` has no math of its own.
