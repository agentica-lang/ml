## Shared helpers of the `nn/` tests (not a test file itself): tensors from
## lists, and a naive model of every layer over plain `List[List[Float]]`
## matrices, written without `Tensor.*` so it can be the oracle.

import Init from "../init"
import Nn from "../nn"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)

## `t` as a `Trace` constant of `dtype`: the weights of a layer that is
## traced, not differentiated.
def lift(t: Tensor.Tensor, dtype: Tensor.Dtype) -> Result[Tensor.Tensor]
  ok(Trace.constant(try Tensor.to_dtype(t, dtype)))
end

## `t` on `backend`, through the host.
def move(t: Tensor.Tensor, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(try Tensor.to_list(t), Tensor.shape(t), Tensor.dtype(t), backend)
end

## The layer with `f` applied to every tensor it holds.
def map_dense(d: Nn.Dense, f: Fn[Tensor.Tensor, Result[Tensor.Tensor]]) -> Result[Nn.Dense]
  b: Optional[Tensor.Tensor] = case d.b
    some(t) -> some(try f.call(t))
    none -> none
  end

  ok(Nn.Dense(w: try f.call(d.w), b: b))
end

def map_norm(n: Nn.LayerNorm, f: Fn[Tensor.Tensor, Result[Tensor.Tensor]]) -> Result[Nn.LayerNorm]
  ok(n.with(gamma: try f.call(n.gamma), beta: try f.call(n.beta)))
end

def map_optional_norm(
  n: Optional[Nn.LayerNorm],
  f: Fn[Tensor.Tensor, Result[Tensor.Tensor]],
) -> Result[Optional[Nn.LayerNorm]]
  case n
    some(norm) -> ok(some(try map_norm(norm, f)))
    none -> ok(none)
  end
end

def map_attention(a: Nn.Attention, f: Fn[Tensor.Tensor, Result[Tensor.Tensor]]) -> Result[Nn.Attention]
  ok(
    a.with(
      query: try map_dense(a.query, f),
      key: try map_dense(a.key, f),
      value: try map_dense(a.value, f),
      output: try map_dense(a.output, f),
      query_norm: try map_optional_norm(a.query_norm, f),
      key_norm: try map_optional_norm(a.key_norm, f),
    ),
  )
end

def map_mlp(m: Nn.Mlp, f: Fn[Tensor.Tensor, Result[Tensor.Tensor]]) -> Result[Nn.Mlp]
  gate: Optional[Nn.Dense] = case m.gate
    some(g) -> some(try map_dense(g, f))
    none -> none
  end

  ok(m.with(up: try map_dense(m.up, f), down: try map_dense(m.down, f), gate: gate))
end

def map_decoder_block(p: Nn.DecoderBlock, f: Fn[Tensor.Tensor, Result[Tensor.Tensor]]) -> Result[Nn.DecoderBlock]
  ok(
    p.with(
      norm1: try map_norm(p.norm1, f),
      self_attention: try map_attention(p.self_attention, f),
      norm2: try map_norm(p.norm2, f),
      cross_attention: try map_attention(p.cross_attention, f),
      norm3: try map_norm(p.norm3, f),
      mlp: try map_mlp(p.mlp, f),
    ),
  )
end

def map_encoder_block(p: Nn.EncoderBlock, f: Fn[Tensor.Tensor, Result[Tensor.Tensor]]) -> Result[Nn.EncoderBlock]
  ok(
    p.with(
      norm1: try map_norm(p.norm1, f),
      attention: try map_attention(p.attention, f),
      norm2: try map_norm(p.norm2, f),
      mlp: try map_mlp(p.mlp, f),
    ),
  )
end

## The weights of a block made with `kv_heads`, `qk_norm`, `rms`, `gated` and
## no bias, in the order `qwen_block` takes them.
def qwen_params(p: Nn.EncoderBlock) -> List[Tensor.Tensor]
  [
    p.attention.query.w,
    p.attention.key.w,
    p.attention.value.w,
    p.attention.output.w,
    p.attention.query_norm.unwrap!.gamma,
    p.attention.key_norm.unwrap!.gamma,
    p.norm1.gamma,
    p.norm2.gamma,
    p.mlp.gate.unwrap!.w,
    p.mlp.up.w,
    p.mlp.down.w,
  ]
end

## `p` with the weights `ps`.
def qwen_block(p: Nn.EncoderBlock, ps: List[Tensor.Tensor]) -> Nn.EncoderBlock
  attention = p.attention.with(
    query: Nn.Dense(w: ps[0], b: none),
    key: Nn.Dense(w: ps[1], b: none),
    value: Nn.Dense(w: ps[2], b: none),
    output: Nn.Dense(w: ps[3], b: none),
    query_norm: some(p.attention.query_norm.unwrap!.with(gamma: ps[4])),
    key_norm: some(p.attention.key_norm.unwrap!.with(gamma: ps[5])),
  )

  mlp = p.mlp.with(
    gate: some(Nn.Dense(w: ps[8], b: none)),
    up: Nn.Dense(w: ps[9], b: none),
    down: Nn.Dense(w: ps[10], b: none),
  )

  p.with(norm1: p.norm1.with(gamma: ps[6]), attention: attention, norm2: p.norm2.with(gamma: ps[7]), mlp: mlp)
end

## `n` with a random `gamma` (around 1) and `beta` of the same shape, so a
## test sees whether a norm reads them. The seeds of two norms must differ.
def jitter_norm(n: Nn.LayerNorm, seed: Int) -> Result[Nn.LayerNorm]
  width = Tensor.shape(n.gamma)[0]
  gamma = Init.uniform_values(width, 0.5, seed).map { |v| 1.0 + v }
  ok(n.with(gamma: try f32(gamma, [width]), beta: try f32(Init.uniform_values(width, 0.5, seed + 1000), [width])))
end

def jitter_optional_norm(n: Optional[Nn.LayerNorm], seed: Int) -> Result[Optional[Nn.LayerNorm]]
  case n
    some(norm) -> ok(some(try jitter_norm(norm, seed)))
    none -> ok(none)
  end
end

## Every norm of the block, the query and key norms included, with its own
## `gamma` and `beta`.
def jitter_encoder_block(p: Nn.EncoderBlock, seed: Int) -> Result[Nn.EncoderBlock]
  ok(
    p.with(
      norm1: try jitter_norm(p.norm1, seed),
      norm2: try jitter_norm(p.norm2, seed + 1),
      attention: p.attention.with(
        query_norm: try jitter_optional_norm(p.attention.query_norm, seed + 2),
        key_norm: try jitter_optional_norm(p.attention.key_norm, seed + 3),
      ),
    ),
  )
end

def jitter_decoder_block(p: Nn.DecoderBlock, seed: Int) -> Result[Nn.DecoderBlock]
  ok(
    p.with(
      norm1: try jitter_norm(p.norm1, seed),
      norm2: try jitter_norm(p.norm2, seed + 1),
      norm3: try jitter_norm(p.norm3, seed + 2),
    ),
  )
end

def jitter_encoder_stack(p: Nn.EncoderStack, seed: Int) -> Result[Nn.EncoderStack]
  blocks: List[Nn.EncoderBlock] = []
  for i in 0...p.blocks.size
    blocks = blocks.push(try jitter_encoder_block(p.blocks[i], seed + 10 * (i + 1)))
  end

  ok(Nn.EncoderStack(blocks: blocks, final_norm: try jitter_optional_norm(p.final_norm, seed + 99)))
end

def f32(data: List[Float], shape: List[Int], backend: Tensor.Backend = REF) -> Result[Tensor.Tensor]
  Tensor.floats(data, shape, backend)
end

def floats_of(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end

## The rows of a rank-2 tensor.
def matrix_of(t: Tensor.Tensor) -> Result[List[List[Float]]]
  flat = try floats_of(t)
  cols = Tensor.shape(t)[1]
  ok((0...Tensor.shape(t)[0]).to_a.map { |i| flat.drop(i * cols).take(cols) })
end

## `got` matches `want` within `abs + rel * |want|`, element by element.
def expect_close(got: List[Float], want: List[Float], abs: Float, rel: Float, what: String) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    gap = (got[i] - want[i]).abs
    return error("#{what}: value #{i} is #{got[i]}, expected #{want[i]}") if gap > abs + rel * want[i].abs
  end

  ok()
end

## `e^x` by range reduction and a Taylor series: the language has no `exp`.
def exp(x: Float) -> Float
  halvings = 0
  y = x
  while y.abs > 0.5
    y = y / 2.0
    halvings += 1
  end

  term = 1.0
  sum = 1.0
  for k in 1...20
    term = (term * y) / k.to_f
    sum = sum + term
  end

  for _ in 0...halvings
    sum = sum * sum
  end

  sum
end

def matmul(a: List[List[Float]], b: List[List[Float]]) -> List[List[Float]]
  a.map do |row: List[Float]|
    (0...b[0].size).to_a.map do |j: Int| (0...row.size).to_a.reduce(0.0) do |acc, k| acc + row[k] * b[k][j] end end
  end
end

def plus(a: List[List[Float]], b: List[List[Float]]) -> List[List[Float]]
  (0...a.size).to_a.map do |i: Int| (0...a[i].size).to_a.map { |j| a[i][j] + b[i][j] } end
end

def dense(p: Nn.Dense, x: List[List[Float]]) -> Result[List[List[Float]]]
  y = matmul(x, try matrix_of(p.w))
  case p.b
    some(b) ->
      bias = try floats_of(b)
      ok(y.map { |row| (0...row.size).to_a.map { |j| row[j] + bias[j] } })
    none -> ok(y)
  end
end

def layer_norm(p: Nn.LayerNorm, x: List[List[Float]]) -> Result[List[List[Float]]]
  gamma = try floats_of(p.gamma)
  beta = try floats_of(p.beta)
  ok(
    x.map do |row: List[Float]|
      n = row.size.to_f
      mu = row.sum / n
      var = row.reduce(0.0) do |acc, v| acc + (v - mu) * (v - mu) end / n
      sd = Math.sqrt(var + p.eps)
      meansq = row.reduce(0.0) do |acc, v| acc + v * v end / n
      rms = Math.sqrt(meansq + p.eps)
      center = case p.rms
        true -> 0.0
        _ -> mu
      end

      spread = case p.rms
        true -> rms
        _ -> sd
      end

      (0...row.size).to_a.map do |j: Int|
        shift = case p.rms
          true -> 0.0
          _ -> beta[j]
        end

        ((row[j] - center) / spread) * gamma[j] + shift
      end
    end,
  )
end

def softmax_row(row: List[Float]) -> List[Float]
  top = row.max.unwrap!
  e = row.map { |v| exp(v - top) }
  total = e.sum
  e.map { |v| v / total }
end

## The rows of one batch item: `flat` is `[batch, seq, width]` row-major.
def item(flat: List[Float], b: Int, seq: Int, width: Int) -> List[List[Float]]
  (0...seq).to_a.map { |i| flat.drop((b * seq + i) * width).take(width) }
end

## The rows of `x` rotated by position: pair `i` of a row of width `d` turns
## by `position * base^(-2i/d)`; the pair is `(i, i + d/2)` with `neox`, else
## `(2i, 2i + 1)`.
def rotate_rows(r: Nn.Rope, x: List[List[Float]]) -> List[List[Float]]
  (0...x.size).to_a.map do |at: Int|
    row = x[at]
    d = r.dims.unwrap_or(row.size)
    pos = at.to_f
    out = row
    for i in 0...(d / 2)
      angle = pos * exp(((-2.0 * i.to_f) / d.to_f) * Math.log(r.base))
      lo = i
      hi = i + d / 2
      if !r.neox
        lo = 2 * i
        hi = 2 * i + 1
      end

      c = Math.cos(angle)
      s = Math.sin(angle)
      out = out.set(lo, row[lo] * c - row[hi] * s)
      out = out.set(hi, row[hi] * c + row[lo] * s)
    end

    out
  end
end

## The `h`-th head of a projection: its `dh` columns, then the head norm and
## the rotation when the layer has them.
def head_rows(
  proj: List[List[Float]],
  h: Int,
  dh: Int,
  norm: Optional[Nn.LayerNorm],
  rope: Optional[Nn.Rope],
) -> Result[List[List[Float]]]
  rows = proj.map { |row| row.drop(h * dh).take(dh) }
  normed = case norm
    some(n) -> try layer_norm(n, rows)
    none -> rows
  end

  case rope
    some(r) -> ok(rotate_rows(r, normed))
    none -> ok(normed)
  end
end

## Attention of one sequence of `queries` over one of `memory`, head by head;
## `mask` is an additive `[q, k]` bias, `causal` blocks the keys after each
## query. Query head `h` reads key head `h / (heads / kv_heads)`.
def attention(
  p: Nn.Attention,
  queries: List[List[Float]],
  memory: List[List[Float]],
  mask: Optional[List[List[Float]]] = none,
  causal: Bool = false,
) -> Result[List[List[Float]]]
  q = try dense(p.query, queries)
  k = try dense(p.key, memory)
  v = try dense(p.value, memory)
  dh = q[0].size / p.heads
  group = p.heads / p.kv_heads.unwrap_or(p.heads)
  scale = 1.0 / Math.sqrt(dh.to_f)
  empty: List[Float] = []
  ctx = queries.map { |_| empty }
  for h in 0...p.heads
    qh = try head_rows(q, h, dh, p.query_norm, p.rope)
    kh = try head_rows(k, h / group, dh, p.key_norm, p.rope)
    vh = try head_rows(v, h / group, dh, none, none)
    for i in 0...queries.size
      scores = (0...memory.size).to_a.map do |j: Int|
        raw = (0...dh).to_a.reduce(0.0) do |acc, c| acc + qh[i][c] * kh[j][c] end * scale
        blocked = causal and j > i
        with_mask = case mask
          some(m) -> raw + m[i][j]
          none -> raw
        end

        penalty = case blocked
          true -> Nn.MASKED
          _ -> 0.0
        end

        with_mask + penalty
      end

      probs = softmax_row(scores)
      head = (0...dh).to_a.map do |c: Int|
        (0...memory.size).to_a.reduce(0.0) do |acc, j| acc + probs[j] * vh[j][c] end
      end

      ctx = ctx.set(i, ctx[i] + head)
    end
  end

  dense(p.output, ctx)
end

def self_attention(p: Nn.Attention, x: List[List[Float]], causal: Bool = false) -> Result[List[List[Float]]]
  attention(p, x, x, none, causal)
end

def relu(v: Float) -> Float
  return v if v > 0.0
  0.0
end

## `erf` by its Maclaurin series, which holds to 1e-12 for `|x| <= 3`.
def erf(x: Float) -> Float
  term = x
  sum = x
  for n in 1...60
    term = (-term * x * x) / n.to_f
    sum = sum + term / (2 * n + 1).to_f
  end

  (sum * 2.0) / Math.sqrt(3.141592653589793)
end

def activate(a: Nn.Activation, v: Float) -> Float
  case a
    Nn.Activation.Relu -> relu(v)
    Nn.Activation.Silu -> v / (1.0 + exp(-v))
    Nn.Activation.GeluErf -> 0.5 * v * (1.0 + erf(v / Math.sqrt(2.0)))
    Nn.Activation.Gelu -> v
  end
end

def mlp(p: Nn.Mlp, x: List[List[Float]]) -> Result[List[List[Float]]]
  return error("the naive model has no tanh gelu") if p.activation == Nn.Activation.Gelu
  up = try dense(p.up, x)
  case p.gate
    some(g) ->
      gated = try dense(g, x)
      mixed = (0...up.size).to_a.map do |i: Int|
        (0...up[i].size).to_a.map { |j| activate(p.activation, gated[i][j]) * up[i][j] }
      end

      dense(p.down, mixed)
    none -> dense(p.down, up.map { |row| row.map { |v| activate(p.activation, v) } })
  end
end

def encoder_block(p: Nn.EncoderBlock, x: List[List[Float]], causal: Bool = false) -> Result[List[List[Float]]]
  if p.norm_first
    h = plus(x, try self_attention(p.attention, try layer_norm(p.norm1, x), causal))
    return ok(plus(h, try mlp(p.mlp, try layer_norm(p.norm2, h))))
  end

  h = try layer_norm(p.norm1, plus(x, try self_attention(p.attention, x, causal)))
  layer_norm(p.norm2, plus(h, try mlp(p.mlp, h)))
end

def decoder_block(p: Nn.DecoderBlock, x: List[List[Float]], memory: List[List[Float]]) -> Result[List[List[Float]]]
  if p.norm_first
    n1 = try layer_norm(p.norm1, x)
    h = plus(x, try attention(p.self_attention, n1, n1, none, true))
    h = plus(h, try attention(p.cross_attention, try layer_norm(p.norm2, h), memory))
    return ok(plus(h, try mlp(p.mlp, try layer_norm(p.norm3, h))))
  end

  h = try layer_norm(p.norm1, plus(x, try attention(p.self_attention, x, x, none, true)))
  h = try layer_norm(p.norm2, plus(h, try attention(p.cross_attention, h, memory)))
  layer_norm(p.norm3, plus(h, try mlp(p.mlp, h)))
end
