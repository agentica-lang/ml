## Shared helpers of the `nn/` tests (not a test file itself): tensors from
## lists, and a naive model of every layer over plain `List[List[Float]]`
## matrices, written without `Tensor.*` so it can be the oracle.

import Nn from "../nn"
import Tensor from "pkg/tensor/tensor"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)

def f32(data: List[Float], shape: List[Int], backend: Tensor.Backend = REF) -> Result[Tensor.Tensor]
  floats = data.map do |x: Float| Tensor.Values.Float(x) end
  Tensor.from_list(floats, shape, Tensor.Dtype.F32, backend)
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
      (0...row.size).to_a.map { |j| ((row[j] - mu) / sd) * gamma[j] + beta[j] }
    end,
  )
end

def softmax_row(row: List[Float]) -> List[Float]
  top = row.max.unwrap!
  e = row.map { |v| exp(v - top) }
  total = e.sum
  e.map { |v| v / total }
end

## Self-attention of one sequence `x` (`[seq, width]`), head by head.
def self_attention(p: Nn.Attention, x: List[List[Float]]) -> Result[List[List[Float]]]
  q = try dense(p.query, x)
  k = try dense(p.key, x)
  v = try dense(p.value, x)
  dh = q[0].size / p.heads
  scale = 1.0 / Math.sqrt(dh.to_f)
  empty: List[Float] = []
  ctx = x.map { |_| empty }
  for h in 0...p.heads
    cols = ((h * dh)...((h + 1) * dh)).to_a
    for i in 0...x.size
      scores = (0...x.size).to_a.map do |j: Int| cols.reduce(0.0) do |acc, c| acc + q[i][c] * k[j][c] end * scale end
      probs = softmax_row(scores)
      head = cols.map do |c: Int| (0...x.size).to_a.reduce(0.0) do |acc, j| acc + probs[j] * v[j][c] end end
      ctx = ctx.set(i, ctx[i] + head)
    end
  end

  dense(p.output, ctx)
end

def mlp(p: Nn.Mlp, x: List[List[Float]]) -> Result[List[List[Float]]]
  return error("the naive model has no gelu") if p.activation == Nn.Activation.Gelu
  h = try dense(p.up, x)
  act = case p.activation
    Nn.Activation.Relu -> h.map { |row| row.map { |v| v.clamp(0.0, v.abs) } }
    Nn.Activation.Silu -> h.map { |row| row.map { |v| v / (1.0 + exp(-v)) } }
    Nn.Activation.Gelu -> h
  end

  dense(p.down, act)
end

def encoder_block(p: Nn.EncoderBlock, x: List[List[Float]]) -> Result[List[List[Float]]]
  h = plus(x, try self_attention(p.attention, try layer_norm(p.norm1, x)))
  ok(plus(h, try mlp(p.mlp, try layer_norm(p.norm2, h))))
end
