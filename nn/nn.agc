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

## `y = x w + b` over the last axis: `w` is `[inputs, outputs]`, `b` is
## `[outputs]`.
struct Dense
  w: Tensor.Tensor
  b: Optional[Tensor.Tensor]
end

## `layer_norm` over the last axis, then `* gamma + beta` (both `[width]`).
struct LayerNorm
  gamma: Tensor.Tensor
  beta: Tensor.Tensor
  eps: Float
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

## `Gelu` is the tanh approximation; `GeluErf` the exact form (no gradient
## through `Trace`).
enum Activation
  Relu
  Gelu
  GeluErf
  Silu
end

## Multi-head attention with its four projections: `query`, `key`, `value`
## project the inputs to `width`, `heads` split it, `output` projects back.
struct Attention
  query: Dense
  key: Dense
  value: Dense
  output: Dense
  heads: Int
end

## `dense -> activation -> dense`.
struct Mlp
  up: Dense
  down: Dense
  activation: Activation
end

## A pre-norm transformer encoder block:
## `h = x + attention(norm1(x))`, then `h + mlp(norm2(h))`.
struct EncoderBlock
  norm1: LayerNorm
  attention: Attention
  norm2: LayerNorm
  mlp: Mlp
end

## A pre-norm transformer decoder block, as Whisper's decoder runs it:
## `h = x + self_attention(norm1(x))` under a causal mask, then
## `h = h + cross_attention(norm2(h), memory)` over the encoder output, then
## `h + mlp(norm3(h))`.
struct DecoderBlock
  norm1: LayerNorm
  self_attention: Attention
  norm2: LayerNorm
  cross_attention: Attention
  norm3: LayerNorm
  mlp: Mlp
end

## `x w + b` on the last axis of an input of any rank >= 1: the leading axes
## are folded into rows for `Tensor.linear` and unfolded after.
def dense(p: Dense, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  return Tensor.linear(x, p.w, p.b) if shape.size == 2
  return error("dense needs rank >= 1, got #{shape}") if shape.size == 0
  lead = shape.take(shape.size - 1)
  rows = lead.reduce(1) do |acc, d| acc * d end
  y = try Tensor.linear(try Tensor.reshape(x, [rows, shape[shape.size - 1]]), p.w, p.b)
  Tensor.reshape(y, lead.push(Tensor.shape(y)[1]))
end

def layer_norm(p: LayerNorm, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  normed = try Tensor.layer_norm(x, p.eps)
  scaled = try Tensor.mul(normed, try Tensor.broadcast_to(p.gamma, shape))
  Tensor.add(scaled, try Tensor.broadcast_to(p.beta, shape))
end

## The rows of `p.table` for the rank-1 integer `ids`: `[ids, width]`.
def embedding(p: Embedding, ids: Tensor.Tensor) -> Result[Tensor.Tensor] = Tensor.embedding(ids, p.table)

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

def activate(a: Activation, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  case a
    Activation.Relu -> Tensor.relu(x)
    Activation.Gelu -> Tensor.gelu(x)
    Activation.GeluErf -> Tensor.gelu_erf(x)
    Activation.Silu -> Tensor.silu(x)
  end
end

## Attention of `queries` (`[batch, q, width]`) over `memory`
## (`[batch, k, width]`); `mask` is the additive bias `Tensor.multi_head_attention`
## takes.
def attention(
  p: Attention,
  queries: Tensor.Tensor,
  memory: Tensor.Tensor,
  mask: Optional[Tensor.Tensor] = none,
) -> Result[Tensor.Tensor]
  q = try dense(p.query, queries)
  k = try dense(p.key, memory)
  v = try dense(p.value, memory)
  dense(p.output, try Tensor.multi_head_attention(q, k, v, p.heads, mask))
end

def self_attention(p: Attention, x: Tensor.Tensor, mask: Optional[Tensor.Tensor] = none) -> Result[Tensor.Tensor]
  attention(p, x, x, mask)
end

def mlp(p: Mlp, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  dense(p.down, try activate(p.activation, try dense(p.up, x)))
end

def encoder_block(p: EncoderBlock, x: Tensor.Tensor, mask: Optional[Tensor.Tensor] = none) -> Result[Tensor.Tensor]
  h = try Tensor.add(x, try self_attention(p.attention, try layer_norm(p.norm1, x), mask))
  Tensor.add(h, try mlp(p.mlp, try layer_norm(p.norm2, h)))
end

## The additive `[size, size]` mask that keeps query `i` off the keys after
## it: `0.0` where `key <= i`, `MASKED` above the diagonal. It is the `mask`
## of `attention`, built from `arange`, `lt` and `where`, so on `Trace` it is
## a constant, never differentiated.
def causal_mask(size: Int, dtype: Tensor.Dtype, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  return error("causal_mask needs size >= 1, got #{size}") if size < 1
  square = [size, size]
  positions = try Tensor.arange(0, size, 1, Tensor.Dtype.I32, backend)
  rows = try Tensor.broadcast_to(try Tensor.reshape(positions, [size, 1]), square)
  cols = try Tensor.broadcast_to(try Tensor.reshape(positions, [1, size]), square)
  blocked = try Tensor.full(square, Tensor.Values.Float(MASKED), dtype, backend)
  Tensor.where(try Tensor.lt(rows, cols), blocked, try Tensor.zeros(square, dtype, backend))
end

## `x` (`[batch, t, width]`) attends to itself causally and then to `memory`,
## the encoder output (`[batch, s, width]`; `s` may differ from `t`).
def decoder_block(p: DecoderBlock, x: Tensor.Tensor, memory: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  return error("decoder_block needs a rank-3 input, got #{shape}") if shape.size != 3
  mask = try causal_mask(shape[1], Tensor.dtype(x), Tensor.backend(x))
  h = try Tensor.add(x, try self_attention(p.self_attention, try layer_norm(p.norm1, x), some(mask)))
  h = try Tensor.add(h, try attention(p.cross_attention, try layer_norm(p.norm2, h), memory))
  Tensor.add(h, try mlp(p.mlp, try layer_norm(p.norm3, h)))
end

## A `Dense` with a Glorot-uniform weight and a zero bias.
def init_dense(inputs: Int, outputs: Int, seed: Int, backend: Tensor.Backend) -> Result[Dense]
  w = try Init.glorot_uniform([inputs, outputs], seed, backend)
  b = try Tensor.zeros([outputs], Tensor.Dtype.F32, backend)
  ok(Dense(w: w, b: some(b)))
end

## A `LayerNorm` that starts as the plain normalization: `gamma` ones,
## `beta` zeros.
def init_layer_norm(width: Int, backend: Tensor.Backend) -> Result[LayerNorm]
  gamma = try Tensor.ones([width], Tensor.Dtype.F32, backend)
  beta = try Tensor.zeros([width], Tensor.Dtype.F32, backend)
  ok(LayerNorm(gamma: gamma, beta: beta, eps: LAYER_NORM_EPS))
end

def init_embedding(rows: Int, width: Int, seed: Int, backend: Tensor.Backend) -> Result[Embedding]
  ok(Embedding(table: try Init.glorot_uniform([rows, width], seed, backend)))
end

def init_attention(width: Int, heads: Int, seed: Int, backend: Tensor.Backend) -> Result[Attention]
  return error("attention needs width #{width} divisible by #{heads} heads") if heads < 1 or width % heads != 0
  ok(
    Attention(
      query: try init_dense(width, width, Init.derive(seed, 0), backend),
      key: try init_dense(width, width, Init.derive(seed, 1), backend),
      value: try init_dense(width, width, Init.derive(seed, 2), backend),
      output: try init_dense(width, width, Init.derive(seed, 3), backend),
      heads: heads,
    ),
  )
end

def init_mlp(width: Int, hidden: Int, activation: Activation, seed: Int, backend: Tensor.Backend) -> Result[Mlp]
  up = try init_dense(width, hidden, Init.derive(seed, 0), backend)
  down = try init_dense(hidden, width, Init.derive(seed, 1), backend)
  ok(Mlp(up: up, down: down, activation: activation))
end

def init_encoder_block(
  width: Int,
  heads: Int,
  hidden: Int,
  activation: Activation,
  seed: Int,
  backend: Tensor.Backend,
) -> Result[EncoderBlock]
  ok(
    EncoderBlock(
      norm1: try init_layer_norm(width, backend),
      attention: try init_attention(width, heads, Init.derive(seed, 0), backend),
      norm2: try init_layer_norm(width, backend),
      mlp: try init_mlp(width, hidden, activation, Init.derive(seed, 1), backend),
    ),
  )
end

def init_decoder_block(
  width: Int,
  heads: Int,
  hidden: Int,
  activation: Activation,
  seed: Int,
  backend: Tensor.Backend,
) -> Result[DecoderBlock]
  ok(
    DecoderBlock(
      norm1: try init_layer_norm(width, backend),
      self_attention: try init_attention(width, heads, Init.derive(seed, 0), backend),
      norm2: try init_layer_norm(width, backend),
      cross_attention: try init_attention(width, heads, Init.derive(seed, 1), backend),
      norm3: try init_layer_norm(width, backend),
      mlp: try init_mlp(width, hidden, activation, Init.derive(seed, 2), backend),
    ),
  )
end
