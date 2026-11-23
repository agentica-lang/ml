## Loss values against hand-computed ones, and their gradients through `Trace`
## against a central difference.

import Loss from "../loss"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

EPS = 0.000001
TOL = 0.00001
## `log(2)`, `softplus(2)` and `softplus(-3) + 3`.
LOG2 = 0.6931471805599453
SOFTPLUS_2 = 2.1269280110429727
SOFTPLUS_MINUS_3_PLUS_3 = 3.048587351573742

private

def test_mse_is_the_mean_squared_error() -> Result[Void]
  got = try Loss.mse(try S.tensor([1.0, 2.0, -1.0, 0.5], [2, 2]), try S.tensor([0.0, 2.0, 1.0, 1.5], [2, 2]))
  S.expect_close(try S.floats_of(got), [(1.0 + 0.0 + 4.0 + 1.0) / 4.0], TOL, 0.0, "mse")
end

def test_binary_cross_entropy_matches_softplus() -> Result[Void]
  got = try Loss.binary_cross_entropy(try S.tensor([0.0, 2.0, -3.0], [3]), try S.tensor([1.0, 0.0, 1.0], [3]))
  want = (LOG2 + SOFTPLUS_2 + SOFTPLUS_MINUS_3_PLUS_3) / 3.0
  S.expect_close(try S.floats_of(got), [want], 0.000000000001, 0.0, "binary_cross_entropy")
end

## At `z = 0` the gradient is `(sigmoid(0) - t) / n`, not the `-t / n` a
## `relu`-based `|z|` would give.
def test_binary_cross_entropy_has_the_right_gradient_at_zero() -> Result[Void]
  targets = try S.tensor([1.0, 0.0], [2])
  got = try Trace.grad(
    [try S.tensor([0.0, 0.0], [2])],
    ->(ps: List[Tensor.Tensor]) { Loss.binary_cross_entropy(ps[0], Trace.constant(targets)) },
  )

  S.expect_close(try S.floats_of(got[0]), [-0.25, 0.25], 0.000000000001, 0.0, "gradient at 0")
end

def test_the_gradients_match_a_central_difference() -> Result[Void]
  prediction = try S.tensor([0.3, -1.2, 2.5, 0.7, -0.4, 1.1], [2, 3])
  target = try S.tensor([0.0, -1.0, 2.0, 1.0, 0.5, 1.0], [2, 3])
  try check(prediction, ->(ps: List[Tensor.Tensor]) { Loss.mse(ps[0], Trace.constant(target)) })
  labels = try S.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], [2, 3])
  check(prediction, ->(ps: List[Tensor.Tensor]) { Loss.binary_cross_entropy(ps[0], Trace.constant(labels)) })
end

def test_cross_entropy_is_the_mean_negative_log_softmax() -> Result[Void]
  labels = try labels_of([2, 0])
  got = try Loss.cross_entropy(try S.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], [2, 3]), labels)
  # `-log softmax([1, 2, 3])[2]` and `log 3`.
  want = (0.4076059644443803 + 1.0986122886681098) / 2.0
  S.expect_close(try S.floats_of(got), [want], 0.000000000001, 0.0, "cross_entropy")
end

## Logits of 1000 overflow `exp` without the max subtraction; the gradient is
## `(softmax - onehot) / batch`.
def test_cross_entropy_is_stable_and_has_the_softmax_gradient() -> Result[Void]
  labels = try labels_of([0])
  both = try Trace.value_and_grad(
    [try S.tensor([1000.0, 1000.0], [1, 2])],
    ->(ps: List[Tensor.Tensor]) { Loss.cross_entropy(ps[0], Trace.constant(labels)) },
  )

  try S.expect_close(try S.floats_of(both.value), [LOG2], 0.000000000001, 0.0, "value")
  S.expect_close(try S.floats_of(both.grads[0]), [-0.5, 0.5], 0.000000000001, 0.0, "gradient")
end

def test_the_cross_entropy_gradient_matches_a_central_difference() -> Result[Void]
  logits = try S.tensor([0.3, -1.2, 2.5, 0.7, -0.4, 1.1], [2, 3])
  labels = try labels_of([1, 2])
  check(logits, ->(ps: List[Tensor.Tensor]) { Loss.cross_entropy(ps[0], Trace.constant(labels)) })
end

def labels_of(classes: List[Int]) -> Result[Tensor.Tensor]
  Tensor.from_list(classes.map { |c| Tensor.Values.Int(c) }, [classes.size], Tensor.Dtype.I64, S.REF)
end

## The gradient of `loss` at `x` against its central difference.
def check(x: Tensor.Tensor, loss: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]) -> Result[Void]
  analytic = try S.floats_of((try Trace.grad([x], loss))[0])
  base = try S.floats_of(x)
  for j in 0...base.size
    up = try at(base.set(j, base[j] + EPS), x, loss)
    down = try at(base.set(j, base[j] - EPS), x, loss)
    slope = (up - down) / (2.0 * EPS)
    try Test.assert(
      (slope - analytic[j]).abs <= TOL * (1.0 + slope.abs),
      "[#{j}]: slope #{slope}, gradient #{analytic[j]}",
    )
  end

  ok()
end

def at(values: List[Float], x: Tensor.Tensor, loss: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]) -> Result[Float]
  out = try S.floats_of(try Trace.eval([try S.tensor(values, Tensor.shape(x))], loss))
  ok(out[0])
end
