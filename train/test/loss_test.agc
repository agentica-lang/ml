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

## What `cosine_golden.py` printed (torch 2.6.0+cpu, `float64`): the cosines of the rows of `cosine_a` and
## `cosine_b`, and the gradients of their loss against the scores 0.2, 0.9 and 0.4.
COSINES = [-0.26936991816748385, 0.9476099573653523, -0.5952712206103672]
COSINE_GRAD_A = [
  -0.09568339845836529, -0.034082240607375115, 0.002307243366782539, -0.025659682297567005, 0.0028264450812710306,
  -0.003235228460793705, 0.0024760593273944554, -0.0046601305265584324, -0.012115671049715782, -0.1575037236463056,
  -0.21808207889488473, -0.12115671049715815,
]
COSINE_GRAD_B = [
  -0.08886701526564691, 0.08061124269942746, -0.213240564576254, -0.077053877284715, -0.0029577748527240266,
  0.004234995811854858, -0.001863014030611888, 0.002813727376130325, -0.24326160394603322, -0.12726722285266703,
  -0.06068370890988101, -0.21102338358765899,
]
COSINE_TOL = 0.000000000001

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

def test_cross_entropy_refuses_a_label_out_of_range() -> Result[Void]
  logits = try S.tensor([0.3, -1.2, 2.5, 0.7, -0.4, 1.1], [2, 3])
  try Test.error?(Loss.cross_entropy(logits, try labels_of([1, 3])))

  Test.error?(Loss.cross_entropy(logits, try labels_of([-1, 0])))
end

def test_cross_entropy_refuses_bad_shapes() -> Result[Void]
  try Test.error?(Loss.cross_entropy(try S.tensor([0.3, -1.2, 2.5], [3]), try labels_of([1])))

  Test.error?(Loss.cross_entropy(try S.tensor([0.3, -1.2, 2.5, 0.7, -0.4, 1.1], [2, 3]), try labels_of([1, 0, 2])))
end

def cosine_a() -> Result[Tensor.Tensor]
  S.tensor([0.3, -1.2, 2.5, 0.7, -0.4, 1.1, 0.2, -0.9, 1.5, 0.1, -0.6, 0.8], [3, 4])
end

def cosine_b() -> Result[Tensor.Tensor]
  S.tensor([1.0, 0.5, -0.3, 0.2, -0.2, 0.9, 0.4, -1.3, -0.7, 0.6, 1.2, 0.1], [3, 4])
end

## The values and gradients PyTorch gives (`cosine_golden.py`, in `float64`).
def test_the_cosine_loss_and_its_gradients_are_pytorchs() -> Result[Void]
  scores = try S.tensor([0.2, 0.9, 0.4], [3])
  cos = try Loss.cosine_similarity(try cosine_a(), try cosine_b())
  try S.expect_close(try S.floats_of(cos), COSINES, COSINE_TOL, 0.0, "cosine")
  both = try Trace.value_and_grad(
    [try cosine_a(), try cosine_b()],
    ->(ps: List[Tensor.Tensor]) { Loss.cosine_similarity_loss(ps[0], ps[1], Trace.constant(scores)) },
  )

  try S.expect_close(try S.floats_of(both.value), [0.4043798768987104], COSINE_TOL, 0.0, "loss")
  try S.expect_close(try S.floats_of(both.grads[0]), COSINE_GRAD_A, COSINE_TOL, 0.0, "gradient of a")
  S.expect_close(try S.floats_of(both.grads[1]), COSINE_GRAD_B, COSINE_TOL, 0.0, "gradient of b")
end

def test_the_cosine_gradient_matches_a_central_difference() -> Result[Void]
  b = try cosine_b()
  scores = try S.tensor([0.2, 0.9, 0.4], [3])
  check(
    try cosine_a(),
    ->(ps: List[Tensor.Tensor]) { Loss.cosine_similarity_loss(ps[0], Trace.constant(b), Trace.constant(scores)) },
  )
end

## A row of zeros has a cosine of 0 and a finite gradient; other shapes are errors.
def test_a_zero_row_has_cosine_zero_and_bad_shapes_are_errors() -> Result[Void]
  zero = try S.tensor([0.0, 0.0, 3.0, 4.0], [2, 2])
  other = try S.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
  try S.expect_close(try S.floats_of(try Loss.cosine_similarity(zero, other)), [0.0, 1.0], COSINE_TOL, 0.0, "cosine")
  scores = try S.tensor([0.5, 0.5], [2])
  grads = try Trace.grad(
    [zero],
    ->(ps: List[Tensor.Tensor]) { Loss.cosine_similarity_loss(ps[0], Trace.constant(other), Trace.constant(scores)) },
  )

  for g in try S.floats_of(grads[0])
    try Test.assert(g == g and g.abs < 1.0e12, "the gradient #{g} is finite")
  end

  try Test.error?(Loss.cosine_similarity(zero, try S.tensor([1.0, 2.0], [1, 2])))
  try Test.error?(Loss.cosine_similarity(try S.tensor([1.0, 2.0], [2]), try S.tensor([1.0, 2.0], [2])))
  Test.error?(Loss.cosine_similarity_loss(zero, other, try S.tensor([0.5], [1])))
end

## The traced loss on libtorch and ggml, in `F32`, against the reference.
def test_the_cosine_loss_runs_on_the_native_backends() -> Result[Void]
  backends = try S.ggml_backends()
  if Tensor.available(S.TORCH)
    backends = backends.push(S.TORCH)
  end

  for backend in backends
    args = [
      try S.tensor(try S.floats_of(try cosine_a()), [3, 4], Tensor.Dtype.F32, backend),
      try S.tensor(try S.floats_of(try cosine_b()), [3, 4], Tensor.Dtype.F32, backend),
      try S.tensor([0.2, 0.9, 0.4], [3], Tensor.Dtype.F32, backend),
    ]

    both = try Trace.value_and_grad(
      args,
      ->(ps: List[Tensor.Tensor]) { Loss.cosine_similarity_loss(ps[0], ps[1], ps[2]) },
    )

    try S.expect_close(try S.floats_of(both.value), [0.4043798768987104], 0.000001, 0.0, "loss on #{backend}")
    try S.expect_close(try S.floats_of(both.grads[0]), COSINE_GRAD_A, 0.000001, 0.0, "gradient of a on #{backend}")
    try S.expect_close(try S.floats_of(both.grads[1]), COSINE_GRAD_B, 0.000001, 0.0, "gradient of b on #{backend}")
  end

  ok()
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
