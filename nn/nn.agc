## The `nn/` package: functional layers over `tensor/`, forward only
## (`nn/DESIGN.md`).
##
## A layer is a struct of parameters plus a function `layer(params, input)`;
## nothing holds state and nothing mutates, so a training step later returns
## new structs. Every layer is written over `Tensor.*` alone: it runs on any
## backend the tensors are on, and on `Trace` tensors it records a graph that
## `Trace.value_and_grad` differentiates (`tensor/DESIGN.md` §10).
## `init_*` makes a layer from a seed (`init.agc`); real weights come from a
## checkpoint later.

import Init from "init"
import Tensor from "pkg/tensor/tensor"

## The `eps` a fresh `LayerNorm` gets.
LAYER_NORM_EPS = 0.00001

## The bias `causal_mask` adds where a query must not see a key: finite, so
## no infinity enters the graph, and far enough below any score that its
## softmax weight is zero.
MASKED = -1000000000.0

## Added to `|a|^2 |b|^2` under the root of `cosine_similarity`: the square of the `1e-8` PyTorch clamps a norm to.
COSINE_EPS = 0.0000000000000001

## `y = x w + b` over the last axis: `w` is `[inputs, outputs]`, `b` is
## `[outputs]`. With `out_in`, `w` is `[outputs, inputs]` as a checkpoint
## stores it, and `dense` multiplies by its transpose: ggml does that with no
## copy of `w`, where it copies `[inputs, outputs]` every time it runs.
struct Dense
  w: Tensor.Tensor
  b: Optional[Tensor.Tensor]
  out_in: Bool = false
end

## `layer_norm` over the last axis, then `* gamma + beta` (both `[width]`).
## With `rms` it is `rms_norm`, then `* gamma`: `beta` is ignored, so any
## tensor of the right shape will do (`init_layer_norm` makes zeros).
struct LayerNorm
  gamma: Tensor.Tensor
  beta: Tensor.Tensor
  eps: Float
  rms: Bool = false
end

## A `[rows, width]` table looked up by integer ids; also the learned
## positional table of `positional`.
struct Embedding
  table: Tensor.Tensor
end

## A 1-D convolution over `[channels, length]` or `[batch, channels, length]`:
## `weight` is `[out, in, kernel]`, `bias` is `[out]`, added per channel.
struct Conv1d
  weight: Tensor.Tensor
  bias: Optional[Tensor.Tensor]
  stride: Int
  padding: Int
end

## A 2-D convolution over `[batch, channels, height, width]`: `weight` is `[out, in / groups, kh, kw]`, `bias` is
## `[out]`, added per channel.
struct Conv2d
  weight: Tensor.Tensor
  bias: Optional[Tensor.Tensor]
  stride: Int = 1
  padding: Int = 0
  groups: Int = 1
end

## `Gelu` is the tanh approximation; `GeluErf` the exact form.
enum Activation
  Relu
  Gelu
  GeluErf
  Silu
end

## Rotary position embedding of the queries and keys (`Tensor.rope`): the
## positions are `0...seq` of each input. With `zero_tail` (LFM3, `partial_rotary_factor` times the head width, even) the
## rotation is the one of Hugging Face's `rotate_half` over the whole head whose frequencies are `base^(-2i / head)` for
## the first `zero_tail / 2` pairs and 0 for the others: the pairs are `(i, i + head / 2)`, so the rotated columns are
## `0...zero_tail / 2` and `head / 2...head / 2 + zero_tail / 2`. It is `rope` over the first `zero_tail` columns after
## those are brought together, so the head comes out with its columns reordered (the rotated ones, then the rest). The
## queries and the keys get the same order and `q k^T` does not change, but the heads are not those of the input.
## With `sections` (Qwen3.5's interleaved M-RoPE, `[time, height, width]`) the positions may be `[3, seq]`, one row
## per axis, and the pair `i` of the rotated columns turns by the height row when `i % 3 == 1` and `i < 3 * height`,
## by the width row when `i % 3 == 2` and `i < 3 * width`, and by the time row otherwise. Positions of `[seq]` (or
## none) are the three rows alike: the plain rope.
struct Rope
  base: Float = 10000.0
  neox: Bool = true
  dims: Optional[Int] = none
  zero_tail: Optional[Int] = none
  sections: Optional[List[Int]] = none
end

## Multi-head attention with its four projections: `query` projects to
## `heads` heads and `key` and `value` to `kv_heads` (`heads` when none, fewer
## for grouped-query attention: `kv_heads` must divide `heads`); `output`
## projects the merged heads back. The head width is the width of `query`'s
## output over `heads`, so loaded weights may have a head width other than
## `width / heads` (Qwen3); `init_attention` makes only `width / heads`. `query_norm` and `key_norm` normalize each
## head over its width (Qwen3), before `rope`. With `gate` (Qwen3.5) `query` projects to `2 * heads` heads: per head
## its query and then its gate, and `attend` multiplies the merged heads by `sigmoid(gate)` before `output`. `scale` is
## the factor of `q k^T` when the model sets one (Granite MoE's `attention_multiplier`), `1 / sqrt(head width)` when none.
## `head_gate` (LFM3) is a gate of one number per head: a `Dense` `[heads, width]` (`out_in`) of the input of `query`,
## whose sigmoid scales the merged heads, each head by its own number, before `output`. `project` returns it as a
## `Heads.gate`, so `attend` treats it as the gate of `gate: true`; an attention has one of the two, not both.
struct Attention
  query: Dense
  key: Dense
  value: Dense
  output: Dense
  heads: Int
  kv_heads: Optional[Int] = none
  rope: Optional[Rope] = none
  query_norm: Optional[LayerNorm] = none
  key_norm: Optional[LayerNorm] = none
  gate: Bool = false
  scale: Optional[Float] = none
  head_gate: Optional[Dense] = none
end

## `dense -> activation -> dense`; with a `gate`, the gated form
## `down(activation(gate(x)) * up(x))` (SwiGLU when the activation is `Silu`).
struct Mlp
  up: Dense
  down: Dense
  activation: Activation
  gate: Optional[Dense] = none
end

## A transformer encoder block. Pre-norm (`norm_first`):
## `h = x + attention(norm1(x))`, then `h + mlp(norm2(h))`. Post-norm, as BERT:
## `h = norm1(x + attention(x))`, then `norm2(h + mlp(h))`.
struct EncoderBlock
  norm1: LayerNorm
  attention: Attention
  norm2: LayerNorm
  mlp: Mlp
  norm_first: Bool = true
end

## A transformer decoder block. Pre-norm, as Whisper's decoder runs it:
## `h = x + self_attention(norm1(x))` under a causal mask, then
## `h = h + cross_attention(norm2(h), memory)` over the encoder output, then
## `h + mlp(norm3(h))`. Post-norm: each part is `norm(h + part(h))`.
struct DecoderBlock
  norm1: LayerNorm
  self_attention: Attention
  norm2: LayerNorm
  cross_attention: Attention
  norm3: LayerNorm
  mlp: Mlp
  norm_first: Bool = true
end

## Blocks applied in order, then `final_norm` (pre-norm models end with one).
struct EncoderStack
  blocks: List[EncoderBlock]
  final_norm: Optional[LayerNorm] = none
end

struct DecoderStack
  blocks: List[DecoderBlock]
  final_norm: Optional[LayerNorm] = none
end

## The choices of `init_*` beyond the sizes: `kv_heads`, `rope` and `qk_norm`
## (Qwen3's query and key norms) shape the self-attention, `bias` is whether
## every `Dense` has one, `rms` makes the norms RMS norms with `eps` (Qwen3
## uses `0.000001`), `gated` makes the `Mlp` gated and `norm_first` picks
## pre-norm.
struct BlockOptions
  kv_heads: Optional[Int] = none
  rope: Optional[Rope] = none
  qk_norm: Bool = false
  bias: Bool = true
  rms: Bool = false
  gated: Bool = false
  norm_first: Bool = true
  eps: Float = LAYER_NORM_EPS
end

## The heads of an attention before `Tensor.attention` reads them: `query`
## `[batch, heads, q, head]`, `key` and `value` `[batch, kv_heads, k, head]`,
## with the query and key norms and the rotation applied. A gated attention (`Attention.gate`) also has the `gate`
## `[batch, heads, q, head]`, for `attend`.
struct Heads
  query: Tensor.Tensor
  key: Tensor.Tensor
  value: Tensor.Tensor
  gate: Optional[Tensor.Tensor] = none
end

## The cosine of each pair of rows of `a` and `b` (`[batch, width]` both), as `[batch]`:
## `a.b / sqrt(|a|^2 |b|^2 + COSINE_EPS)`. The small term keeps a row of zeros at a cosine of 0 with a finite
## gradient; for any other row it is below `F32` rounding, so the values are those of PyTorch's `cosine_similarity`.
def cosine_similarity(a: Tensor.Tensor, b: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(a)
  if shape.size != 2 or Tensor.shape(b) != shape
    return error("cosine_similarity needs two [batch, width] tensors, got #{shape} and #{Tensor.shape(b)}")
  end

  dot = try Tensor.sum(try Tensor.mul(a, b), some([1]))
  squares = try Tensor.mul(
    try Tensor.sum(try Tensor.mul(a, a), some([1])),
    try Tensor.sum(try Tensor.mul(b, b), some([1])),
  )

  floor = try Tensor.full([shape[0]], Tensor.Values.Float(COSINE_EPS), Tensor.dtype(a), Tensor.backend(a))
  Tensor.div(dot, try Tensor.sqrt(try Tensor.add(squares, floor)))
end

## `x w + b` on the last axis of an input of any rank >= 1: the leading axes
## are folded into rows for `Tensor.linear` and unfolded after.
def dense(p: Dense, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  return Tensor.linear(x, p.w, p.b, p.out_in) if shape.size == 2
  return error("dense needs rank >= 1, got #{shape}") if shape.size == 0
  lead = shape.take(shape.size - 1)
  rows = lead.reduce(1) do |acc, d| acc * d end
  y = try Tensor.linear(try Tensor.reshape(x, [rows, shape[shape.size - 1]]), p.w, p.b, p.out_in)
  Tensor.reshape(y, lead.push(Tensor.shape(y)[1]))
end

def layer_norm(p: LayerNorm, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  return Tensor.mul(try Tensor.rms_norm(x, p.eps), try Tensor.broadcast_to(p.gamma, shape)) if p.rms

  normed = try Tensor.layer_norm(x, p.eps)
  scaled = try Tensor.mul(normed, try Tensor.broadcast_to(p.gamma, shape))
  Tensor.add(scaled, try Tensor.broadcast_to(p.beta, shape))
end

## The rows of `p.table` for the rank-1 integer `ids`: `[ids, width]`, `F32` also for a half or quantized table (ggml).
def embedding(p: Embedding, ids: Tensor.Tensor) -> Result[Tensor.Tensor]
  rows = try Tensor.embedding(ids, p.table)
  return Tensor.to_dtype(rows, Tensor.Dtype.F32) if Tensor.packed?(Tensor.dtype(p.table))

  ok(rows)
end

## `x` plus the first `seq` rows of the positional table, for `x` of shape
## `[seq, width]` or `[batch, seq, width]`.
def positional(p: Embedding, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  return error("positional needs rank 2 or 3, got #{shape}") if shape.size < 2 or shape.size > 3
  seq = shape[shape.size - 2]
  table = Tensor.shape(p.table)
  return error("positional needs at most #{table[0]} positions, got #{seq}") if seq > table[0]
  rows = try Tensor.slice(p.table, [0, 0], [seq, table[1]])
  Tensor.add(x, try Tensor.broadcast_to(rows, shape))
end

## `Tensor.conv1d` with the layer's stride and padding, plus the bias of each
## output channel: `[out, L_out]` or `[batch, out, L_out]`.
def conv1d(p: Conv1d, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  y = try Tensor.conv1d(x, p.weight, p.stride, p.padding)
  case p.bias
    some(b) ->
      shape = Tensor.shape(y)
      column = try Tensor.reshape(b, [Tensor.shape(b)[0], 1])
      Tensor.add(y, try Tensor.broadcast_to(column, shape))
    none -> ok(y)
  end
end

## `Tensor.conv2d` with the layer's stride, padding and groups, plus the bias of each output channel:
## `[batch, out, h_out, w_out]`.
def conv2d(p: Conv2d, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  y = try Tensor.conv2d(x, p.weight, p.stride, p.padding, p.groups)
  case p.bias
    some(b) ->
      plane = try Tensor.reshape(b, [1, Tensor.shape(b)[0], 1, 1])
      Tensor.add(y, try Tensor.broadcast_to(plane, Tensor.shape(y)))
    none -> ok(y)
  end
end

## `conv` followed by an inference batch norm (`gamma`, `beta`, `mean` and `var`, each `[out]`, and `eps`) as one
## convolution: the weight of each output channel is scaled by `gamma / sqrt(var + eps)`, and the bias becomes
## `beta + (bias - mean) * gamma / sqrt(var + eps)`. The ops are `Tensor` calls, so a loader runs them as one graph on
## ggml. The batch norm is the one of a trained model: its statistics are not updated, so the result is for inference.
def fold_batch_norm(
  conv: Conv2d,
  gamma: Tensor.Tensor,
  beta: Tensor.Tensor,
  mean: Tensor.Tensor,
  var: Tensor.Tensor,
  eps: Float,
) -> Result[Conv2d]
  shape = Tensor.shape(conv.weight)
  channels = [shape[0]]
  for t in [gamma, beta, mean, var]
    if Tensor.shape(t) != channels
      return error("fold_batch_norm needs the #{shape[0]} output channels of the convolution, got #{Tensor.shape(t)}")
    end
  end

  floor = try Tensor.full(channels, Tensor.Values.Float(eps), Tensor.dtype(var), Tensor.backend(var))
  scale = try Tensor.div(gamma, try Tensor.sqrt(try Tensor.add(var, floor)))
  per_channel = try Tensor.broadcast_to(try Tensor.reshape(scale, [shape[0], 1, 1, 1]), shape)
  centered = try Tensor.neg(mean)
  case conv.bias
    some(b) -> centered = try Tensor.sub(b, mean)
    none -> nil
  end

  ok(
    conv.with(
      weight: try Tensor.mul(conv.weight, per_channel),
      bias: some(try Tensor.add(beta, try Tensor.mul(centered, scale))),
    ),
  )
end

## The mean of each channel over its height and width: `[batch, channels, h, w]` to `[batch, channels]`.
def global_avg_pool2d(x: Tensor.Tensor) -> Result[Tensor.Tensor]
  return error("global_avg_pool2d needs a rank-4 input, got #{Tensor.shape(x)}") if Tensor.shape(x).size != 4
  Tensor.mean(x, some([2, 3]), false)
end

def activate(a: Activation, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  case a
    Activation.Relu -> Tensor.relu(x)
    Activation.Gelu -> Tensor.gelu(x)
    Activation.GeluErf -> Tensor.gelu_erf(x)
    Activation.Silu -> Tensor.silu(x)
  end
end

## The heads of `queries` (`[batch, q, width]`) and of `memory`
## (`[batch, k, width]`): the projections split by head, the query and key
## norms, then the rotation if `p` has one. Rope reads the integer vector
## `positions` `[q]` for the queries and `memory_positions` `[k]` for the keys,
## `0...q` and `0...k` when none; with only `positions` given the keys use it
## too, which is self-attention (`q` is `k`). A decoder that keeps the keys and
## values of earlier tokens projects the new token alone, at its own position
## (`positions` of `[n]` for a token at position `n`, the keys cached already
## rotated), and gives `attend` the cache. With a `Rope.zero_tail` the heads come out with their columns reordered (the
## rotated ones first, the same for `query` and `key`), which `attend` does not mind and nothing else may read.
## With a `Rope.sections` and positions of `[3, seq]`, `axes` are the `axis_masks` of the rope, which a decoder makes
## once for all its layers (made here when empty).
def project(
  p: Attention,
  queries: Tensor.Tensor,
  memory: Tensor.Tensor,
  positions: Optional[Tensor.Tensor] = none,
  memory_positions: Optional[Tensor.Tensor] = none,
  axes: List[Tensor.Tensor] = [],
) -> Result[Heads]
  kv_heads = p.kv_heads.unwrap_or(p.heads)
  return error("an attention has a gate or a head gate, not both") if p.gate and p.head_gate.some?
  asked = try query_heads(p, queries)
  q = asked.query
  k = try split_heads(try dense(p.key, memory), kv_heads, p.key_norm)
  v = try split_heads(try dense(p.value, memory), kv_heads)
  case p.rope
    some(r) ->
      q = try rotate(r, q, positions, axes)
      key_positions = memory_positions
      if key_positions.none?
        key_positions = positions
      end

      k = try rotate(r, k, key_positions, axes)
    none -> nil
  end

  ok(Heads(query: q, key: k, value: v, gate: try head_gate_of(p, q, queries, asked.gate)))
end

## `Tensor.attention` of `query` over `key` and `value` (the tensors of a
## `Heads`, or a cache of them), the heads merged and `p.output` applied:
## `[batch, q, width]`. `mask` and `causal` are as for `attention`; `gate` is the `gate` of the `Heads` of a gated
## attention (an error without it, and for any other attention), whose sigmoid scales the merged heads before
## `p.output`.
def attend(
  p: Attention,
  query: Tensor.Tensor,
  key: Tensor.Tensor,
  value: Tensor.Tensor,
  mask: Optional[Tensor.Tensor] = none,
  causal: Bool = false,
  gate: Optional[Tensor.Tensor] = none,
) -> Result[Tensor.Tensor]
  return error("an attention has a gate or a head gate, not both") if p.gate and p.head_gate.some?
  gated = p.gate or p.head_gate.some?
  return error("a gated attention needs the gate of its heads") if gated and gate.none?
  return error("a gate is for a gated attention: this one has no gate") if !gated and gate.some?
  context = try Tensor.attention(query, key, value, mask: mask, causal: causal, scale: p.scale)
  merged = try merge_heads(context)
  case gate
    some(g) -> merged = try Tensor.mul(merged, try Tensor.sigmoid(try merge_heads(g)))
    none -> nil
  end

  dense(p.output, merged)
end

## Attention of `queries` (`[batch, q, width]`) over `memory`
## (`[batch, k, width]`): `project` then `attend`, through `Tensor.attention`.
## `mask` is its additive bias (`[q, k]`, or `[batch|1, heads|1, q, k]`);
## `causal` keeps each query off the keys after it, aligned at the end: with
## `q < k` the last query sees every key (the queries are the last positions of
## the keys, as in decoding with a cache), and `q > k` is an error. On
## `Reference` it needs `q * k` of at most 1,048,576. `positions` and
## `memory_positions` are those of `project`; rope starts at 0 without them.
def attention(
  p: Attention,
  queries: Tensor.Tensor,
  memory: Tensor.Tensor,
  mask: Optional[Tensor.Tensor] = none,
  causal: Bool = false,
  positions: Optional[Tensor.Tensor] = none,
  memory_positions: Optional[Tensor.Tensor] = none,
) -> Result[Tensor.Tensor]
  h = try project(p, queries, memory, positions, memory_positions)
  attend(p, h.query, h.key, h.value, mask, causal, h.gate)
end

def self_attention(
  p: Attention,
  x: Tensor.Tensor,
  mask: Optional[Tensor.Tensor] = none,
  causal: Bool = false,
) -> Result[Tensor.Tensor]
  attention(p, x, x, mask, causal)
end

def mlp(p: Mlp, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  up = try dense(p.up, x)
  case p.gate
    some(g) -> dense(p.down, try Tensor.mul(try activate(p.activation, try dense(g, x)), up))
    none -> dense(p.down, try activate(p.activation, up))
  end
end

## `causal` makes the block a decoder-only one: each position sees itself and
## the ones before it.
def encoder_block(
  p: EncoderBlock,
  x: Tensor.Tensor,
  mask: Optional[Tensor.Tensor] = none,
  causal: Bool = false,
) -> Result[Tensor.Tensor]
  if p.norm_first
    h = try Tensor.add(x, try self_attention(p.attention, try layer_norm(p.norm1, x), mask, causal))
    return Tensor.add(h, try mlp(p.mlp, try layer_norm(p.norm2, h)))
  end

  h = try layer_norm(p.norm1, try Tensor.add(x, try self_attention(p.attention, x, mask, causal)))
  layer_norm(p.norm2, try Tensor.add(h, try mlp(p.mlp, h)))
end

## The additive `[size, size]` mask that keeps query `i` off the keys after
## it: `0.0` where `key <= i`, `MASKED` above the diagonal, the same bias that
## `causal: true` builds inside `Tensor.attention`. With a `window` (a sliding-window layer) it is `0.0` where
## `i - window < key <= i` only: query `i` sees itself and the `window - 1` keys before it. A `window` of `size` or
## more changes nothing.
def causal_mask(
  size: Int,
  dtype: Tensor.Dtype,
  backend: Tensor.Backend,
  window: Optional[Int] = none,
) -> Result[Tensor.Tensor]
  return error("causal_mask needs size >= 1, got #{size}") if size < 1
  square = [size, size]
  positions = try Tensor.arange(0, size, 1, Tensor.Dtype.I32, backend)
  rows = try Tensor.broadcast_to(try Tensor.reshape(positions, [size, 1]), square)
  cols = try Tensor.broadcast_to(try Tensor.reshape(positions, [1, size]), square)
  case window
    some(w) -> return window_bias(rows, cols, w, dtype)
    none -> nil
  end

  blocked = try Tensor.full(square, Tensor.Values.Float(MASKED), dtype, backend)
  Tensor.where(try Tensor.lt(rows, cols), blocked, try Tensor.zeros(square, dtype, backend))
end

## The sliding-window bias of `causal_mask` over the grids `queries` and `keys` (`I32`, one shape): the position of each
## query and of each key it is compared with, so a padded cache can name its own.
def window_bias(queries: Tensor.Tensor, keys: Tensor.Tensor, window: Int, dtype: Tensor.Dtype) -> Result[Tensor.Tensor]
  return error("a window is at least 1 key, got #{window}") if window < 1
  shape = Tensor.shape(queries)
  backend = Tensor.backend(queries)
  edge = try Tensor.full(shape, Tensor.Values.Float(window.to_f - 1.0), Tensor.Dtype.F32, backend)
  blocked = try Tensor.full(shape, Tensor.Values.Float(MASKED), dtype, backend)
  reach = try Tensor.add(try Tensor.to_dtype(keys, Tensor.Dtype.F32), edge)
  older = try Tensor.lt(reach, try Tensor.to_dtype(queries, Tensor.Dtype.F32))
  inside = try Tensor.where(older, blocked, try Tensor.zeros(shape, dtype, backend))
  Tensor.where(try Tensor.lt(queries, keys), blocked, inside)
end

## The additive bias `[batch, 1, seq, seq]` of a `[batch, seq]` mask (1.0 for a token, 0.0 for padding): `MASKED` on
## the keys that are padding, 0.0 elsewhere, for the `mask` of an attention over a padded batch.
def padding_bias(mask: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(mask)
  return error("padding_bias needs a [batch, seq] mask, got #{shape}") if shape.size != 2
  ones = try Tensor.full(shape, Tensor.Values.Float(1.0), Tensor.dtype(mask), Tensor.backend(mask))
  scale = try Tensor.full(shape, Tensor.Values.Float(MASKED), Tensor.dtype(mask), Tensor.backend(mask))
  bias = try Tensor.mul(try Tensor.sub(ones, mask), scale)
  Tensor.broadcast_to(try Tensor.reshape(bias, [shape[0], 1, 1, shape[1]]), [shape[0], 1, shape[1], shape[1]])
end

## `x` (`[batch, t, width]`) attends to itself causally and then to `memory`,
## the encoder output (`[batch, s, width]`; `s` may differ from `t`).
def decoder_block(p: DecoderBlock, x: Tensor.Tensor, memory: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  return error("decoder_block needs a rank-3 input, got #{shape}") if shape.size != 3
  if p.norm_first
    h = try Tensor.add(x, try self_attention(p.self_attention, try layer_norm(p.norm1, x), none, true))
    h = try Tensor.add(h, try attention(p.cross_attention, try layer_norm(p.norm2, h), memory))
    return Tensor.add(h, try mlp(p.mlp, try layer_norm(p.norm3, h)))
  end

  h = try layer_norm(p.norm1, try Tensor.add(x, try self_attention(p.self_attention, x, none, true)))
  h = try layer_norm(p.norm2, try Tensor.add(h, try attention(p.cross_attention, h, memory)))
  layer_norm(p.norm3, try Tensor.add(h, try mlp(p.mlp, h)))
end

def encoder_stack(
  p: EncoderStack,
  x: Tensor.Tensor,
  mask: Optional[Tensor.Tensor] = none,
  causal: Bool = false,
) -> Result[Tensor.Tensor]
  h = x
  for block in p.blocks
    h = try encoder_block(block, h, mask, causal)
  end

  finished(p.final_norm, h)
end

def decoder_stack(p: DecoderStack, x: Tensor.Tensor, memory: Tensor.Tensor) -> Result[Tensor.Tensor]
  h = x
  for block in p.blocks
    h = try decoder_block(block, h, memory)
  end

  finished(p.final_norm, h)
end

## A `Dense` with a Glorot-uniform weight and, unless `bias` is false, a zero
## bias.
def init_dense(inputs: Int, outputs: Int, seed: Int, backend: Tensor.Backend, bias: Bool = true) -> Result[Dense]
  w = try Init.glorot_uniform([inputs, outputs], seed, backend)
  return ok(Dense(w: w, b: none)) if !bias
  b = try Tensor.zeros([outputs], Tensor.Dtype.F32, backend)
  ok(Dense(w: w, b: some(b)))
end

## A `LayerNorm` that starts as the plain normalization: `gamma` ones,
## `beta` zeros.
def init_layer_norm(
  width: Int,
  backend: Tensor.Backend,
  rms: Bool = false,
  eps: Float = LAYER_NORM_EPS,
) -> Result[LayerNorm]
  gamma = try Tensor.ones([width], Tensor.Dtype.F32, backend)
  beta = try Tensor.zeros([width], Tensor.Dtype.F32, backend)
  ok(LayerNorm(gamma: gamma, beta: beta, eps: eps, rms: rms))
end

def init_embedding(rows: Int, width: Int, seed: Int, backend: Tensor.Backend) -> Result[Embedding]
  ok(Embedding(table: try Init.glorot_uniform([rows, width], seed, backend)))
end

## The projections of `heads` heads over `width`, with `options.kv_heads`,
## `rope` and `qk_norm` (a norm of the head width).
def init_attention(
  width: Int,
  heads: Int,
  seed: Int,
  backend: Tensor.Backend,
  options: BlockOptions = BlockOptions(),
) -> Result[Attention]
  return error("attention needs width #{width} divisible by #{heads} heads") if heads < 1 or width % heads != 0
  head_width = width / heads
  kv_heads = options.kv_heads.unwrap_or(heads)
  return error("attention needs #{kv_heads} kv heads dividing #{heads} heads") if kv_heads < 1 or heads % kv_heads != 0
  kv_width = kv_heads * head_width
  query_norm: Optional[LayerNorm] = none
  key_norm: Optional[LayerNorm] = none
  if options.qk_norm
    query_norm = some(try init_layer_norm(head_width, backend, options.rms, options.eps))
    key_norm = some(try init_layer_norm(head_width, backend, options.rms, options.eps))
  end

  ok(
    Attention(
      query: try init_dense(width, width, Init.derive(seed, 0), backend, options.bias),
      key: try init_dense(width, kv_width, Init.derive(seed, 1), backend, options.bias),
      value: try init_dense(width, kv_width, Init.derive(seed, 2), backend, options.bias),
      output: try init_dense(width, width, Init.derive(seed, 3), backend, options.bias),
      heads: heads,
      kv_heads: options.kv_heads,
      rope: options.rope,
      query_norm: query_norm,
      key_norm: key_norm,
    ),
  )
end

def init_mlp(
  width: Int,
  hidden: Int,
  activation: Activation,
  seed: Int,
  backend: Tensor.Backend,
  options: BlockOptions = BlockOptions(),
) -> Result[Mlp]
  up = try init_dense(width, hidden, Init.derive(seed, 0), backend, options.bias)
  down = try init_dense(hidden, width, Init.derive(seed, 1), backend, options.bias)
  gate: Optional[Dense] = none
  if options.gated
    gate = some(try init_dense(width, hidden, Init.derive(seed, 2), backend, options.bias))
  end

  ok(Mlp(up: up, down: down, activation: activation, gate: gate))
end

def init_encoder_block(
  width: Int,
  heads: Int,
  hidden: Int,
  activation: Activation,
  seed: Int,
  backend: Tensor.Backend,
  options: BlockOptions = BlockOptions(),
) -> Result[EncoderBlock]
  ok(
    EncoderBlock(
      norm1: try init_layer_norm(width, backend, options.rms, options.eps),
      attention: try init_attention(width, heads, Init.derive(seed, 0), backend, options),
      norm2: try init_layer_norm(width, backend, options.rms, options.eps),
      mlp: try init_mlp(width, hidden, activation, Init.derive(seed, 1), backend, options),
      norm_first: options.norm_first,
    ),
  )
end

## The cross-attention of a decoder block has plain heads, no rope and no
## query or key norms: only `options.bias` applies to it.
def init_decoder_block(
  width: Int,
  heads: Int,
  hidden: Int,
  activation: Activation,
  seed: Int,
  backend: Tensor.Backend,
  options: BlockOptions = BlockOptions(),
) -> Result[DecoderBlock]
  cross = BlockOptions(bias: options.bias)
  ok(
    DecoderBlock(
      norm1: try init_layer_norm(width, backend, options.rms, options.eps),
      self_attention: try init_attention(width, heads, Init.derive(seed, 0), backend, options),
      norm2: try init_layer_norm(width, backend, options.rms, options.eps),
      cross_attention: try init_attention(width, heads, Init.derive(seed, 1), backend, cross),
      norm3: try init_layer_norm(width, backend, options.rms, options.eps),
      mlp: try init_mlp(width, hidden, activation, Init.derive(seed, 2), backend, options),
      norm_first: options.norm_first,
    ),
  )
end

## `layers` blocks, each from its own `Init.derive(seed, i)`; a pre-norm stack
## (`options.norm_first`) ends in a final norm.
def init_encoder_stack(
  layers: Int,
  width: Int,
  heads: Int,
  hidden: Int,
  activation: Activation,
  seed: Int,
  backend: Tensor.Backend,
  options: BlockOptions = BlockOptions(),
) -> Result[EncoderStack]
  return error("a stack needs layers >= 1, got #{layers}") if layers < 1
  blocks: List[EncoderBlock] = []
  for i in 0...layers
    blocks = blocks.push(
      try init_encoder_block(width, heads, hidden, activation, Init.derive(seed, i), backend, options),
    )
  end

  ok(EncoderStack(blocks: blocks, final_norm: try init_final_norm(width, backend, options)))
end

def init_decoder_stack(
  layers: Int,
  width: Int,
  heads: Int,
  hidden: Int,
  activation: Activation,
  seed: Int,
  backend: Tensor.Backend,
  options: BlockOptions = BlockOptions(),
) -> Result[DecoderStack]
  return error("a stack needs layers >= 1, got #{layers}") if layers < 1
  blocks: List[DecoderBlock] = []
  for i in 0...layers
    blocks = blocks.push(
      try init_decoder_block(width, heads, hidden, activation, Init.derive(seed, i), backend, options),
    )
  end

  ok(DecoderStack(blocks: blocks, final_norm: try init_final_norm(width, backend, options)))
end

## The three masks `[1, 1, 1, width]` of a `Rope` with `sections` over a head of `width` columns (`rotate` with
## positions of `[3, seq]`): the columns that the time, the height and the width row turn, 1 or 0, which sum to 1 on
## every column (those that rope leaves alone belong to the time). Built from fills, so that they are constants of a
## graph. A decoder with many layers makes them once and gives them to `project`.
def axis_masks(r: Rope, width: Int, dtype: Tensor.Dtype, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  return error("axis_masks needs a Rope with sections") if r.sections.none?
  sections = r.sections.unwrap!
  if sections.size != 3 or sections[0] < 0 or sections[1] < 1 or sections[2] < 1
    return error("sections are [time, height, width] with 1 or more for the height and the width, got #{sections}")
  end

  return error("a Rope with sections is the neox layout and takes no zero_tail") if r.zero_tail.some? or !r.neox
  dims = r.dims.unwrap_or(width)
  if dims < 2 or dims % 2 != 0 or dims > width
    return error("a Rope with sections needs an even number of columns to turn, got #{dims}")
  end

  pairs = dims / 2
  height = try axis_mask(1, sections[1], pairs, width, dtype, backend)
  across = try axis_mask(2, sections[2], pairs, width, dtype, backend)
  time = try Tensor.sub(try Tensor.ones([width], dtype, backend), try Tensor.add(height, across))
  masks: List[Tensor.Tensor] = []
  for m in [time, height, across]
    masks = masks.push(try Tensor.reshape(m, [1, 1, 1, width]))
  end

  ok(masks)
end

private

## The query heads of a projection and, for a gated attention, their gates.
struct Asked
  query: Tensor.Tensor
  gate: Optional[Tensor.Tensor]
end

def init_final_norm(width: Int, backend: Tensor.Backend, options: BlockOptions) -> Result[Optional[LayerNorm]]
  return ok(none) if !options.norm_first
  ok(some(try init_layer_norm(width, backend, options.rms, options.eps)))
end

def finished(norm: Optional[LayerNorm], h: Tensor.Tensor) -> Result[Tensor.Tensor]
  case norm
    some(n) -> layer_norm(n, h)
    none -> ok(h)
  end
end

## `[batch, seq, heads * width]` to `[batch, heads, seq, width]`. The norm of each head, when there is one, runs
## before the permute, on the contiguous `[batch, seq, heads, width]`: ggml then lowers the permute to a view.
def split_heads(x: Tensor.Tensor, heads: Int, norm: Optional[LayerNorm] = none) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  return error("attention needs a rank-3 input, got #{shape}") if shape.size != 3
  return error("attention needs width #{shape[2]} divisible by #{heads} heads") if heads < 1 or shape[2] % heads != 0
  split = try Tensor.reshape(x, [shape[0], shape[1], heads, shape[2] / heads])
  Tensor.permute(try normalized(norm, split), [0, 2, 1, 3])
end

## The query heads of `queries`, and with `p.gate` their gates: the projection is `[batch, q, heads * 2 * head]`, per head
## the query then the gate.
def query_heads(p: Attention, queries: Tensor.Tensor) -> Result[Asked]
  projected = try dense(p.query, queries)
  return ok(Asked(query: try split_heads(projected, p.heads, p.query_norm), gate: none)) if !p.gate
  shape = Tensor.shape(projected)
  return error("attention needs a rank-3 input, got #{shape}") if shape.size != 3
  if p.heads < 1 or shape[2] % (2 * p.heads) != 0
    return error("a gated attention needs width #{shape[2]} divisible by 2 * #{p.heads} heads")
  end

  head = shape[2] / (2 * p.heads)
  pairs = try Tensor.reshape(projected, [shape[0], shape[1], p.heads, 2 * head])
  query = try Tensor.slice(pairs, [0, 0, 0, 0], [shape[0], shape[1], p.heads, head])
  gate = try Tensor.slice(pairs, [0, 0, 0, head], [shape[0], shape[1], p.heads, 2 * head])
  ok(
    Asked(
      query: try Tensor.permute(try normalized(p.query_norm, query), [0, 2, 1, 3]),
      gate: some(try Tensor.permute(gate, [0, 2, 1, 3])),
    ),
  )
end

def merge_heads(x: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  Tensor.reshape(try Tensor.permute(x, [0, 2, 1, 3]), [shape[0], shape[2], shape[1] * shape[3]])
end

## `x` (`[batch, heads, seq, width]`) rotated by `positions` if given, otherwise `0...seq`.
def rotate(
  r: Rope,
  x: Tensor.Tensor,
  positions: Optional[Tensor.Tensor],
  axes: List[Tensor.Tensor],
) -> Result[Tensor.Tensor]
  case r.zero_tail
    some(n) -> return rotate_tail(r, n, x, positions)
    none -> nil
  end

  case positions
    some(at) -> return rotate_axes(r, x, at, axes) if Tensor.shape(at).size == 2
    none -> nil
  end

  seq = Tensor.shape(x)[2]
  case positions
    some(at) -> Tensor.rope(x, at, base: r.base, dims: r.dims, neox: r.neox)
    none ->
      at = try Tensor.arange(0, seq, 1, Tensor.Dtype.I32, Tensor.backend(x))
      Tensor.rope(x, at, base: r.base, dims: r.dims, neox: r.neox)
  end
end

def normalized(norm: Optional[LayerNorm], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  case norm
    some(n) -> layer_norm(n, x)
    none -> ok(x)
  end
end

## The gate of `p.head_gate` as the `Heads.gate` of `query`: `[batch, heads, q, head]`, the one number of each head
## repeated over its width. `gate` is the gate that `query_heads` made.
def head_gate_of(
  p: Attention,
  query: Tensor.Tensor,
  queries: Tensor.Tensor,
  gate: Optional[Tensor.Tensor],
) -> Result[Optional[Tensor.Tensor]]
  case p.head_gate
    none -> ok(gate)
    some(d) ->
      shape = Tensor.shape(query)
      per_head = try dense(d, queries)
      if Tensor.shape(per_head) != [shape[0], shape[2], p.heads]
        return error("a head gate needs [#{shape[0]}, #{shape[2]}, #{p.heads}], got #{Tensor.shape(per_head)}")
      end

      column = try Tensor.permute(try Tensor.reshape(per_head, [shape[0], shape[2], p.heads, 1]), [0, 2, 1, 3])
      ok(some(try Tensor.broadcast_to(column, shape)))
  end
end

## `rotate` of a `Rope` with a `zero_tail` of `n` columns: the rotated columns first, then `rope` over them, with the base
## that gives the pair `i` the angle `base^(-2i / head)` over `n` columns.
def rotate_tail(r: Rope, n: Int, x: Tensor.Tensor, positions: Optional[Tensor.Tensor]) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  head = shape[3]
  if r.dims.some? or !r.neox
    return error("a zero_tail rope is the neox layout over the whole head: it takes no dims and no neox: false")
  end

  return error("a zero_tail of the whole head #{head} is a plain rotation: use no zero_tail") if n == head

  if n < 2 or n % 2 != 0 or head % 2 != 0 or n > head
    return error("zero_tail needs an even number of columns in 2 to #{head - 2}, got #{n}")
  end

  half = head / 2
  pairs = n / 2
  cut = ->(from: Int, to: Int) { Tensor.slice(x, [0, 0, 0, from], [shape[0], shape[1], shape[2], to]) }
  brought = try Tensor.concat(
    [
      try cut.call(0, pairs),
      try cut.call(half, half + pairs),
      try cut.call(pairs, half),
      try cut.call(half + pairs, head),
    ],
    3,
  )

  base = Math.exp((Math.log(r.base) * n.to_f) / head.to_f)
  at = case positions
    some(p) -> p
    none -> try Tensor.arange(0, shape[2], 1, Tensor.Dtype.I32, Tensor.backend(x))
  end

  Tensor.rope(brought, at, base: base, dims: some(n), neox: true)
end

## `rotate` of a `Rope` with `sections` by `positions` of `[3, seq]`: `x` rotated by each row, and each pair of columns
## taken from the rotation of its axis (`axes`, the masks over the columns).
def rotate_axes(
  r: Rope,
  x: Tensor.Tensor,
  positions: Tensor.Tensor,
  axes: List[Tensor.Tensor],
) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  seq = shape[2]
  if Tensor.shape(positions) != [3, seq]
    return error("rope positions of [3, #{seq}] are needed, got #{Tensor.shape(positions)}")
  end

  masks = axes
  if masks.empty?
    masks = try axis_masks(r, shape[3], Tensor.dtype(x), Tensor.backend(x))
  end

  out = x
  for axis in 0...3
    row = try Tensor.reshape(try Tensor.slice(positions, [axis, 0], [axis + 1, seq]), [seq])
    turned = try Tensor.rope(x, row, base: r.base, dims: r.dims, neox: true)
    picked = try Tensor.mul(turned, try Tensor.broadcast_to(masks[axis], shape))
    if axis == 0
      out = picked
    else
      out = try Tensor.add(out, picked)
    end
  end

  ok(out)
end

## The columns `[width]` that the pairs of `axis` (1 or 2) rotate: 1 for the columns `i` and `i + pairs` of each pair
## `i` with `i % 3 == axis` below `3 * count`, 0 for all others, including the columns that rope leaves alone.
def axis_mask(
  axis: Int,
  count: Int,
  pairs: Int,
  width: Int,
  dtype: Tensor.Dtype,
  backend: Tensor.Backend,
) -> Result[Tensor.Tensor]
  fill = ->(v: Float, n: Int) { Tensor.full([n], Tensor.Values.Float(v), dtype, backend) }
  flags: List[Tensor.Tensor] = []
  for i in 0...3
    flags = flags.push(try fill.call(selects(axis, i), 1))
  end

  cycle = try Tensor.concat(flags, 0)

  cycles = (pairs + 2) / 3
  tiled = try Tensor.reshape(try Tensor.broadcast_to(try Tensor.reshape(cycle, [1, 3]), [cycles, 3]), [3 * cycles])
  reach = [3 * count, pairs].min.unwrap!
  kept = try Tensor.slice(tiled, [0], [reach])
  per_pair = kept
  if reach < pairs
    per_pair = try Tensor.concat([kept, try fill.call(0.0, pairs - reach)], 0)
  end

  tail = width - 2 * pairs
  return Tensor.concat([per_pair, per_pair, try fill.call(0.0, tail)], 0) if tail > 0

  Tensor.concat([per_pair, per_pair], 0)
end

## 1 when the place `i` of a cycle of three is the one of `axis`, 0 otherwise.
def selects(axis: Int, i: Int) -> Float
  return 1.0 if axis == i
  0.0
end
