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

## `Gelu` is the tanh approximation; `GeluErf` the exact form (no gradient
## through `Trace`).
enum Activation
  Relu
  Gelu
  GeluErf
  Silu
end

## Rotary position embedding of the queries and keys (`Tensor.rope`): the
## positions are `0...seq` of each input.
struct Rope
  base: Float = 10000.0
  neox: Bool = true
  dims: Optional[Int] = none
end

## Multi-head attention with its four projections: `query` projects to
## `heads` heads and `key` and `value` to `kv_heads` (`heads` when none, fewer
## for grouped-query attention: `kv_heads` must divide `heads`); `output`
## projects the merged heads back. The head width is the width of `query`'s
## output over `heads`, so loaded weights may have a head width other than
## `width / heads` (Qwen3); `init_attention` makes only `width / heads`. `query_norm` and `key_norm` normalize each
## head over its width (Qwen3), before `rope`.
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
## with the query and key norms and the rotation applied.
struct Heads
  query: Tensor.Tensor
  key: Tensor.Tensor
  value: Tensor.Tensor
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
  return Tensor.mul(try Tensor.rms_norm(x, p.eps), try Tensor.broadcast_to(p.gamma, shape)) if p.rms

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

## The heads of `queries` (`[batch, q, width]`) and of `memory`
## (`[batch, k, width]`): the projections split by head, the query and key
## norms, then the rotation if `p` has one. Rope reads the integer vector
## `positions` `[q]` for the queries and `memory_positions` `[k]` for the keys,
## `0...q` and `0...k` when none; with only `positions` given the keys use it
## too, which is self-attention (`q` is `k`). A decoder that keeps the keys and
## values of earlier tokens projects the new token alone, at its own position
## (`positions` of `[n]` for a token at position `n`, the keys cached already
## rotated), and gives `attend` the cache.
def project(
  p: Attention,
  queries: Tensor.Tensor,
  memory: Tensor.Tensor,
  positions: Optional[Tensor.Tensor] = none,
  memory_positions: Optional[Tensor.Tensor] = none,
) -> Result[Heads]
  kv_heads = p.kv_heads.unwrap_or(p.heads)
  q = try normalized(p.query_norm, try split_heads(try dense(p.query, queries), p.heads))
  k = try normalized(p.key_norm, try split_heads(try dense(p.key, memory), kv_heads))
  v = try split_heads(try dense(p.value, memory), kv_heads)
  case p.rope
    some(r) ->
      q = try rotate(r, q, positions)
      key_positions = memory_positions
      if key_positions.none?
        key_positions = positions
      end

      k = try rotate(r, k, key_positions)
    none -> nil
  end

  ok(Heads(query: q, key: k, value: v))
end

## `Tensor.attention` of `query` over `key` and `value` (the tensors of a
## `Heads`, or a cache of them), the heads merged and `p.output` applied:
## `[batch, q, width]`. `mask` and `causal` are as for `attention`.
def attend(
  p: Attention,
  query: Tensor.Tensor,
  key: Tensor.Tensor,
  value: Tensor.Tensor,
  mask: Optional[Tensor.Tensor] = none,
  causal: Bool = false,
) -> Result[Tensor.Tensor]
  context = try Tensor.attention(query, key, value, mask: mask, causal: causal)
  dense(p.output, try merge_heads(context))
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
  attend(p, h.query, h.key, h.value, mask, causal)
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
## `causal: true` builds inside `Tensor.attention`.
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

private

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

## `[batch, seq, heads * width]` to `[batch, heads, seq, width]`.
def split_heads(x: Tensor.Tensor, heads: Int) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  return error("attention needs a rank-3 input, got #{shape}") if shape.size != 3
  return error("attention needs width #{shape[2]} divisible by #{heads} heads") if heads < 1 or shape[2] % heads != 0
  split = try Tensor.reshape(x, [shape[0], shape[1], heads, shape[2] / heads])
  Tensor.permute(split, [0, 2, 1, 3])
end

def merge_heads(x: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  Tensor.reshape(try Tensor.permute(x, [0, 2, 1, 3]), [shape[0], shape[2], shape[1] * shape[3]])
end

## `x` (`[batch, heads, seq, width]`) rotated by `positions` if given, otherwise `0...seq`.
def rotate(r: Rope, x: Tensor.Tensor, positions: Optional[Tensor.Tensor]) -> Result[Tensor.Tensor]
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
