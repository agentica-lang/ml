## `nn/` composes with `Trace`: the gradient of a scalar loss through `dense`
## (on a rank-3 input, so its fold and unfold are traced) and through a relu
## `mlp` matches a central difference of the same loss.

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
