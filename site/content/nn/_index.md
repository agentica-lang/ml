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
| `Conv2d` | `weight` as `[out, in / groups, kh, kw]`, `bias`, `stride`, `padding`, `groups` | `conv2d(p, x)` on `[batch, in, h, w]`; `global_avg_pool2d(x)` is the mean of each channel over `h` and `w`, `[batch, channels]`; `fold_batch_norm(conv, gamma, beta, mean, var, eps)` folds the inference batch norm that follows a convolution into its weight and bias |
| `Attention` | four `Dense` projections, `heads`, and optional `kv_heads`, `rope`, `query_norm`, `key_norm`, `gate` (Qwen3.5, see below), `head_gate` (LFM3: a `Dense` `[heads, width]`, one gate per head) and `scale` (the factor of `q k^T`, `1 / sqrt(head width)` when none) | `attention(p, queries, memory, mask?, causal?, positions?, memory_positions?)`, `self_attention(p, x, mask?, causal?)`; `project` and `attend`, the two halves of `attention` |
| `Heads` | `query`, `key`, `value` and, for a gated attention, `gate` (shaped as `query`): the heads that `project` returns; `query` is `[batch, heads, q, head]`, and `key` and `value` are `[batch, kv_heads, k, head]` (`kv_heads` is `heads` when none) | the input of `attend`, which takes the `gate` too |
| `Mlp` | `up`, `down`, `activation`, optional `gate` | `mlp(p, x)`: `dense`, activation, `dense`; with a `gate`, `down(activation(gate(x)) * up(x))` |
| `Delta.GatedDeltaNet` (`pkg/nn/delta`) | the linear attention layer of Qwen3.5: five `Dense` projections, a depthwise `conv`, `dt_bias`, `a_log`, `norm`, and the head counts and width | `Delta.gated_delta_net(p, x, state)` on one sequence `[tokens, width]`, after the `DeltaState` of the tokens before it (`delta_start(p, backend)` for the first call; a `Delta.padding` makes the rows after the real ones leave the state alone, for a padded prompt); inference only, no gradient |
| `ShortConv.ShortConv` (`pkg/nn/short_conv`) | the gated short convolution of LFM3: an `input` `Dense` `[3 * width, width]` (`out_in`; the parts `b`, `c`, `x`), a depthwise `conv` `[width, taps]` and an `output` `Dense` | `ShortConv.short_conv(p, x, state, padding?)` on one sequence `[tokens, width]`: `output(c * conv(b * x))`, and the state `[taps - 1, width]` for the next call (`short_start(p, backend)` for the first; `Delta.padding` as for `Delta.gated_delta_net`); no bias, inference only |
| `Moe.Moe` (`pkg/nn/moe`) | the mixture-of-experts block of Granite MoE: a `router` `Dense` `[experts, width]` (`out_in`), the stacked expert weights `gate_up` `[experts, 2 * inner, width]` (gate, then value) and `down` `[experts, width, inner]`, `top_k`, `activation` and `normalize` | `Moe.moe(p, x)` on `[..., width]`: each token goes to its `top_k` experts and their gated MLPs are added by the router's gates (`normalize`: the softmax over the picked scores; otherwise the picked entries of the softmax over all of them); inference only, no gradient |
| `EncoderBlock` | two norms, attention, MLP, `norm_first` | `encoder_block(p, x, mask?, causal?)` |
| `DecoderBlock` | three norms, self and cross attention, MLP, `norm_first` | `decoder_block(p, x, memory)`, causal self attention, then cross attention |
| `EncoderStack`, `DecoderStack` | `blocks`, optional `final_norm` | `encoder_stack(p, x, mask?, causal?)`, `decoder_stack(p, x, memory)`: the blocks in order, then the final norm |

An activation is a function applied to each value, such as `relu`. The `Activation` type is one of these:

- `Relu` and `Silu`.
- `Gelu`, a smooth activation in its tanh approximation.
- `GeluErf`, the exact form that Whisper and BERT use.

`Trace` differentiates all of them.

`activate(activation, x)` applies one of them to a tensor. `mlp` uses it.

`cosine_similarity(a, b)` gives the cosine of each pair of rows of two `[batch, width]` tensors, as `[batch]`, the values of PyTorch's `cosine_similarity`. A row of zeros has a cosine of 0, and `Trace` takes its gradient.

With `gate: true` (the full attention of Qwen3.5) `query` projects to `2 * heads` heads, per head its query rows and then its gate rows, as the checkpoint stores `q_proj`. `project` returns the gate in `Heads.gate`, and `attend(..., gate)` multiplies the merged heads by its sigmoid before `output`; `attend` is an error for a gated attention without the gate and for any other attention with one. The norms of Qwen3.5 scale by `1 + weight`: a loader adds 1 to the weight. With `head_gate` (LFM3) the gate is one number per head, `sigmoid(head_gate(x))` over the input of `query`: `project` repeats it over the head width in `Heads.gate`, and an attention has `gate` or `head_gate`, not both.

A causal mask stops each position from seeing the positions after it. `causal: true` applies it inside `attention`, and `causal_mask(size, dtype, backend)` builds the same additive mask for a caller that wants it. `causal_mask(size, dtype, backend, window)` with a `window` is the mask of a sliding-window layer: query `i` sees the keys `i - window + 1` to `i`, itself included. `padding_bias(mask)` turns the `[batch, seq]` mask of a padded batch (1.0 for a token, 0.0 for padding) into the additive mask `[batch, 1, seq, seq]` that keeps every query off the padding.

The module `pkg/nn/kv` holds what a decoder that steps one token at a time shares: `capacity(needed, limit)` (8, 16, 32, ... rows, so a compiled step changes shape only when the capacity grows), `reach(needed, capacity)` (the rows of the cache an attention reads when `needed` are in use: rounded up to 256 rows, or to an eighth of the capacity when that is more) with `reached(t, rows)` (those first rows of a cache tensor), `Slots` (`positions`, the rows the new tokens take, and `capacity`), `padded_mask(slots, dtype, window?)` (the keys a `window` or more rows before a position are masked too: a sliding-window layer), `spread_of(slots, dtype)` and `written(old, fresh, spread, axis)`, which puts the new rows of `fresh` at the positions of the slots in a cache padded with zeros (`axis` is the rows axis of both). On `Ggml` a model can write the rows in place with `Tensor.set_rows` instead.

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
- `rope` rotates the queries and keys by the position `0...seq` of each input, or by the integer vector `positions` (queries) and `memory_positions` (keys) when `attention` or `project` gets them. Only `positions` means self-attention (the keys use them too); pass `memory_positions` for cross-attention. `Rope` has `base`, `neox` (the pairing of the columns), `dims` and `zero_tail` (LFM3: the rotated columns of a head whose other pairs have a frequency of 0; the heads come out with their columns reordered, the same for queries and keys, so the scores are unchanged).
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

- `init_dense(inputs, outputs, seed, backend, bias?)`
- `init_layer_norm(width, backend, rms?, eps?)`
- `init_embedding(rows, width, seed, backend)`
- `init_attention(width, heads, seed, backend, options?)`
- `init_mlp(width, hidden, activation, seed, backend, options?)`
- `init_encoder_block(width, heads, hidden, activation, seed, backend, options?)`
- `init_decoder_block(width, heads, hidden, activation, seed, backend, options?)`
- `init_encoder_stack(layers, width, heads, hidden, activation, seed, backend, options?)` and `init_decoder_stack`, with the same arguments

- A dense weight starts with Glorot-uniform values (random, on a common scale), and a bias starts at zero. A layer norm starts as the plain normalization.
- `bias: false` leaves out a dense bias. `rms` makes the layer norm an RMS norm, and `eps` sets its epsilon.
- `import Init from "pkg/nn/init"` gives `Init.glorot_uniform(shape, seed, backend)` for a rank-2 shape, and `Init.uniform(shape, limit, seed, backend)`. `Init.uniform_values(count, limit, seed)` gives the same draws as a plain list of floats.
- One seed gives the same `F32` values on every backend. The sub-layers of one block take their seeds from `Init.derive(seed, k)`.
- The random generator only makes starting weights. Real weights come from a checkpoint, loaded with [hub](../hub/_index.md).

## Limits

- `init_*` make `F32` tensors. The layers themselves take any float dtype.
- `attention` runs `tensor`'s `attention`, one op on every backend.
- A layer runs eagerly on `Reference` and libtorch. On ggml and Xla it runs traced only: call it on `Trace` tensors and evaluate or compile the graph.
- `causal` with fewer queries than keys ends on the last key (the end-aligned mask of a decode step), and with more queries than keys it is an error. On `Reference`, `q * k` of the causal mask stops at 1,048,576.
- `init_attention` makes heads of `width / heads`; weights loaded from a checkpoint may have another head width (Qwen3), and the layers run them.
- Rotary positions of a layer start at 0 for each input unless `positions` says otherwise. There is no KV cache in `nn`: a decoder that has one calls `project` on the new token alone with its position (`positions` of `[n]` for position `n`), appends the key, already rotated, to its cache, and gives `attend` the cache. One position vector serves every batch row and head.
- `decoder_block` has no KV cache (a store of past keys and values): each call recomputes every position. `asr` has its own cache.
- `causal` and `causal_mask` use `-1e9` for a masked position, which assumes an `F32` or `F64` compute dtype.
- `init_attention` needs a width that is divisible by the head count, and `kv_heads` that divides it. `positional` refuses more positions than its table has.

## How it works

A layer has no state of its own, and no layer changes in place. A model is a struct of layer structs, and a training step returns new ones.

Every layer is made of `tensor` calls, so it runs on every backend that `tensor` has. On `Trace` tensors it records a graph, which `Trace.value_and_grad` differentiates. `nn` has no math of its own.
