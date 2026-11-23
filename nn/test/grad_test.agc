## `nn/` composes with `Trace`: the gradient of a scalar loss through `dense`
## (on a rank-3 input, so its fold and unfold are traced), a relu `mlp`,
## `embedding` with `positional`, and a gelu encoder block matches a central
## difference of the same loss.

import Nn from "../nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

EPS = 0.000001
TOL = 0.00001
## `[2, 2, 3]`; every pre-activation of the relu `mlp` below stays 0.12 or
## more away from the kink, where a central difference says nothing.
X_DATA = [0.5, -1.0, 2.0, 0.3, 1.2, -0.7, -0.4, 0.9, 0.1, 1.5, -0.2, 0.6]

private

def test_the_gradient_through_dense_matches_a_central_difference() -> Result[Void]
  x = try f64(X_DATA, [2, 2, 3])
  args = [try f64([0.2, -0.4, 0.7, 0.1, -0.3, 0.6], [3, 2]), try f64([0.05, -0.15], [2])]
  check(args, ->(ps: List[Tensor.Tensor]) { dense_loss(ps, x) })
end

def test_the_gradient_through_a_relu_mlp_matches_a_central_difference() -> Result[Void]
  x = try f64(X_DATA, [2, 2, 3])
  args = [
    try f64([0.2, -0.4, 0.7, 0.1, -0.3, 0.6, 0.5, -0.2, 0.4, 0.3, -0.6, 0.8], [3, 4]),
    try f64([0.05, -0.15, 0.1, 0.02], [4]),
    try f64([0.3, -0.5, 0.2, 0.6, 0.1, -0.4, -0.2, 0.7, 0.5, 0.4, -0.3, 0.1], [4, 3]),
    try f64([0.1, 0.0, -0.1], [3]),
  ]

  check(args, ->(ps: List[Tensor.Tensor]) { mlp_loss(ps, x) })
end

def test_the_gradient_through_embedding_and_positional_matches_a_central_difference() -> Result[Void]
  ids = try Tensor.from_list([2, 0, 2].map { |i| Tensor.Values.Int(i) }, [3], Tensor.Dtype.I64, S.REF)
  args = [
    try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7, 0.4, 0.9], [4, 2]),
    try f64([0.1, -0.2, 0.3, 0.6, -0.4, 0.8, 0.2, 0.5], [4, 2]),
  ]

  check(
    args,
    ->(ps: List[Tensor.Tensor]) {
      rows = try Nn.embedding(Nn.Embedding(table: ps[0]), Trace.constant(ids))
      squared(try Nn.positional(Nn.Embedding(table: ps[1]), rows))
    },
  )
end

def test_the_gradient_through_a_gelu_encoder_block_matches_a_central_difference() -> Result[Void]
  args = [try filler([1, 2, 4], 0), try filler([4], 1)]
  fixed: List[Tensor.Tensor] = []
  for k in 2...9
    fixed = fixed.push(try filler([4, 4], k))
  end

  check(args, ->(ps: List[Tensor.Tensor]) { block_loss(ps, fixed.map { |t| Trace.constant(t) }) })
end

## `sum(encoder_block(x)^2)` with a gelu `mlp` and bias-free weights, for
## `ps = [x, norm1 gamma]`: the gradient of `x` crosses every layer of the
## block. `fixed` holds the query, key, value and output weights, the `mlp`
## weights, and a beta and the norm2 gamma (rows 0 and 1).
def block_loss(ps: List[Tensor.Tensor], fixed: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  beta = try Tensor.reshape(try Tensor.slice(fixed[6], [0, 0], [1, 4]), [4])
  gamma2 = try Tensor.reshape(try Tensor.slice(fixed[6], [1, 0], [2, 4]), [4])
  attention = Nn.Attention(
    query: Nn.Dense(w: fixed[0], b: none),
    key: Nn.Dense(w: fixed[1], b: none),
    value: Nn.Dense(w: fixed[2], b: none),
    output: Nn.Dense(w: fixed[3], b: none),
    heads: 2,
  )

  block = Nn.EncoderBlock(
    norm1: Nn.LayerNorm(gamma: ps[1], beta: beta, eps: 0.00001),
    attention: attention,
    norm2: Nn.LayerNorm(gamma: gamma2, beta: beta, eps: 0.00001),
    mlp: Nn.Mlp(
      up: Nn.Dense(w: fixed[4], b: none),
      down: Nn.Dense(w: fixed[5], b: none),
      activation: Nn.Activation.Gelu,
    ),
  )

  squared(try Nn.encoder_block(block, ps[0]))
end

## An `F64` tensor of `shape` with small values that differ by `seed`.
def filler(shape: List[Int], seed: Int) -> Result[Tensor.Tensor]
  n = shape.reduce(1) do |acc, d| acc * d end
  f64((0...n).to_a.map { |j| 0.1 * ((j * 7 + seed * 3) % 11 - 5).to_f + 0.05 * seed.to_f }, shape)
end

## `sum(dense(x)^2)` for the weight and bias in `ps`.
def dense_loss(ps: List[Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  squared(try Nn.dense(Nn.Dense(w: ps[0], b: some(ps[1])), Trace.constant(x)))
end

def mlp_loss(ps: List[Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  m = Nn.Mlp(
    up: Nn.Dense(w: ps[0], b: some(ps[1])),
    down: Nn.Dense(w: ps[2], b: some(ps[3])),
    activation: Nn.Activation.Relu,
  )

  squared(try Nn.mlp(m, Trace.constant(x)))
end

def squared(y: Tensor.Tensor) -> Result[Tensor.Tensor] = Tensor.sum(try Tensor.mul(y, y))

## The gradient of `loss` at `args` against its central difference, value by
## value.
def check(args: List[Tensor.Tensor], loss: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]) -> Result[Void]
  got = try Trace.value_and_grad(args, loss)
  for i in 0...args.size
    analytic = try S.floats_of(got.grads[i])
    base = try S.floats_of(args[i])
    for j in 0...base.size
      up = try at(args, i, base.set(j, base[j] + EPS), loss)
      down = try at(args, i, base.set(j, base[j] - EPS), loss)
      slope = (up - down) / (2.0 * EPS)
      try Test.assert(
        (slope - analytic[j]).abs <= TOL * (1.0 + slope.abs),
        "arg #{i}[#{j}]: slope #{slope}, gradient #{analytic[j]}",
      )
    end
  end

  ok()
end

def f64(data: List[Float], shape: List[Int]) -> Result[Tensor.Tensor]
  Tensor.from_list(data.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F64, S.REF)
end

## The loss with argument `i` replaced by `values`.
def at(
  args: List[Tensor.Tensor],
  i: Int,
  values: List[Float],
  loss: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
) -> Result[Float]
  moved = args.set(i, try f64(values, Tensor.shape(args[i])))
  out = try S.floats_of(try Trace.eval(moved, loss))
  ok(out[0])
end
