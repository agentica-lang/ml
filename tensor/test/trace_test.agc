## Tracing and gradients (`trace.agc`): every backward rule against a central
## difference of the traced function itself, as the M0 spike checked its
## hand-written backward, then a 2-layer MLP loss, the graph's sharing and the
## trace-time errors.

import Conform from "torch_support"
import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

## The step of the central difference: the loss pieces are smooth near every
## point below, and doubles keep the quotient well above rounding.
EPS = 0.000001
## The gap allowed between the slope and the gradient, relative to the slope.
TOL = 0.00001
## Every pre-activation `x w1 + b1` stays 0.04 or more away from the kink of
## `relu`, where a central difference says nothing.
X_DATA = [0.5, -1.2, 0.3, 1.1, 0.8, -0.4, -0.7, 0.2, 1.5, 0.9, -1.1, 0.6]
Y_DATA = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, -1.0]
W1_DATA = [0.2, -0.5, 0.8, 0.1, -0.3, 0.7, 0.4, -0.6, 0.3, 0.9, -0.2, 0.5, 0.6, -0.1, 0.4]
B1_DATA = [0.1, 0.2, -0.1, 0.25, 0.05]
W2_DATA = [0.3, -0.4, 0.6, 0.2, -0.5, 0.7, 0.1, -0.3, 0.8, 0.4]
B2_DATA = [0.05, -0.1]

private

def test_add_matches_a_central_difference() -> Result[Void]
  args = [try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7], [2, 3]), try f64([1.5, 0.2, -0.3, 0.8, -1.1, 0.4], [2, 3])]
  check_grad(args) do |xs| weigh(try Trace.add(xs[0], xs[1])) end
end

def test_sub_matches_a_central_difference() -> Result[Void]
  args = [try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7], [2, 3]), try f64([1.5, 0.2, -0.3, 0.8, -1.1, 0.4], [2, 3])]
  check_grad(args) do |xs| weigh(try Trace.sub(xs[0], xs[1])) end
end

def test_mul_matches_a_central_difference() -> Result[Void]
  args = [try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7], [2, 3]), try f64([1.5, 0.2, -0.3, 0.8, -1.1, 0.4], [2, 3])]
  check_grad(args) do |xs| weigh(try Trace.mul(xs[0], xs[1])) end
end

def test_div_matches_a_central_difference() -> Result[Void]
  args = [try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7], [2, 3]), try f64([1.5, 0.6, -0.9, 0.8, -1.1, 2.4], [2, 3])]
  check_grad(args) do |xs| weigh(try Trace.div(xs[0], xs[1])) end
end

def test_neg_matches_a_central_difference() -> Result[Void]
  check_grad([try f64([0.5, -1.0, 2.0, 0.3], [4])]) do |xs| weigh(try Trace.neg(xs[0])) end
end

def test_exp_matches_a_central_difference() -> Result[Void]
  check_grad([try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7], [3, 2])]) do |xs| weigh(try Trace.exp(xs[0])) end
end

def test_log_matches_a_central_difference() -> Result[Void]
  check_grad([try f64([0.5, 1.0, 2.0, 0.3, 1.2, 3.7], [3, 2])]) do |xs| weigh(try Trace.log(xs[0])) end
end

def test_relu_matches_a_central_difference_away_from_zero() -> Result[Void]
  check_grad([try f64([-1.5, 0.7, 2.0, -0.3, 1.1, -2.2], [2, 3])]) do |xs| weigh(try Trace.relu(xs[0])) end
end

def test_matmul_matches_a_central_difference() -> Result[Void]
  args = [
    try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7], [2, 3]),
    try f64([1.5, 0.2, -0.3, 0.8, -1.1, 0.4, 0.9, -0.6, 0.1, 0.7, -0.2, 1.3], [3, 4]),
  ]

  check_grad(args) do |xs| weigh(try Trace.matmul(xs[0], xs[1])) end
end

def test_transpose_matches_a_central_difference() -> Result[Void]
  check_grad([try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7], [2, 3])]) do |xs| weigh(try Trace.transpose(xs[0])) end
end

def test_sum_of_every_axis_matches_a_central_difference() -> Result[Void]
  check_grad([try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7], [2, 3])]) do |xs| Trace.sum(try Trace.mul(xs[0], xs[0])) end
end

def test_sum_of_named_axes_matches_a_central_difference() -> Result[Void]
  x = try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7, 0.4, 0.9, -1.3, 0.6, -0.2, 1.4], [2, 3, 2])
  try check_grad([x]) do |xs| weigh(try Trace.sum(xs[0], some([2, 0]))) end

  check_grad([x]) do |xs| weigh(try Trace.sum(xs[0], some([1]), true)) end
end

def test_mean_matches_a_central_difference() -> Result[Void]
  x = try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7], [2, 3])
  try check_grad([x]) do |xs| weigh(try Trace.mean(xs[0], some([1]))) end

  check_grad([x]) do |xs| Trace.mean(try Trace.exp(xs[0])) end
end

def test_broadcast_to_matches_a_central_difference() -> Result[Void]
  try check_grad([try f64([0.5, -1.0, 2.0], [3])]) do |xs| weigh(try Trace.broadcast_to(xs[0], [2, 3])) end

  check_grad([try f64([0.5, -1.0], [2, 1])]) do |xs| weigh(try Trace.broadcast_to(xs[0], [2, 2, 3])) end
end

def test_reshape_matches_a_central_difference() -> Result[Void]
  check_grad([try f64([0.5, -1.0, 2.0, 0.3, 1.2, -0.7], [2, 3])]) do |xs| weigh(try Trace.reshape(xs[0], [3, 2])) end
end

def test_a_two_layer_mlp_loss_matches_a_central_difference() -> Result[Void] = check_grad(try mlp_params(), &mlp_loss)

def test_one_sgd_step_on_the_mlp_lowers_the_loss() -> Result[Void]
  params = try mlp_params()
  step = try Trace.value_and_grad(params, &mlp_loss)
  moved: List[Tensor.Tensor] = []
  for i in 0...params.size
    scale = try Tensor.full(
      Tensor.shape(params[i]),
      Tensor.Values.Float(0.1),
      Tensor.Dtype.F64,
      Tensor.Backend.Reference,
    )

    moved = moved.push(try Tensor.sub(params[i], try Tensor.mul(step.grads[i], scale)))
  end

  before = try scalar(step.value)
  after = try scalar(try Trace.eval(moved, &mlp_loss))
  Test.assert(after < before, "the loss went from #{before} to #{after}")
end

def test_eval_gives_what_the_tensor_api_gives() -> Result[Void]
  params = try mlp_params()
  traced = try Trace.eval(params, &mlp_loss)
  x = try f64(X_DATA, [4, 3])
  pre = try Tensor.add(try Tensor.matmul(x, params[0]), try Tensor.broadcast_to(params[1], [4, 5]))
  h = try Tensor.relu(pre)
  o = try Tensor.add(try Tensor.matmul(h, params[2]), try Tensor.broadcast_to(params[3], [4, 2]))
  d = try Tensor.sub(o, try f64(Y_DATA, [4, 2]))
  eager = try Tensor.mean(try Tensor.mul(d, d))
  Test.eq(Support.floats_of(traced), Support.floats_of(eager))
end

def test_a_value_used_twice_is_one_node() -> Result[Void]
  x = try f64([0.5, -1.0, 2.0], [3])
  grads = try Trace.grad([x]) do |xs| Trace.sum(try Trace.add(try Trace.mul(xs[0], xs[0]), xs[0])) end

  Support.expect_floats(grads[0], [2.0, -1.0, 5.0])
end

def test_an_unused_argument_gets_zeros() -> Result[Void]
  args = [try f64([0.5, -1.0], [2]), try f64([1.0, 2.0, 3.0], [3])]
  grads = try Trace.grad(args) do |xs| Trace.sum(xs[0]) end

  try Support.expect_floats(grads[0], [1.0, 1.0])
  Support.expect_floats(grads[1], [0.0, 0.0, 0.0])
end

def test_a_shape_mismatch_fails_at_trace_time() -> Result[Void]
  args = [try f64([0.5, -1.0], [2]), try f64([1.0, 2.0, 3.0], [3])]
  Test.error?(Trace.grad(args) do |xs| Trace.sum(try Trace.add(xs[0], xs[1])) end)
end

def test_a_bad_broadcast_fails_at_trace_time() -> Result[Void]
  Test.error?(Trace.grad([try f64([0.5, -1.0], [2])]) do |xs| Trace.sum(try Trace.broadcast_to(xs[0], [2, 3])) end)
end

def test_a_non_scalar_output_has_no_gradient() -> Result[Void]
  Test.error?(Trace.grad([try f64([0.5, -1.0], [2])]) do |xs| Trace.exp(xs[0]) end)
end

def test_the_mlp_gradient_on_torch_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [W1_DATA, B1_DATA, W2_DATA, B2_DATA]
  shapes = [[3, 5], [5], [5, 2], [2]]
  on_ref: List[Tensor.Tensor] = []
  on_torch: List[Tensor.Tensor] = []
  for i in 0...data.size
    on_ref = on_ref.push(try Conform.on_ref(data[i], shapes[i]))
    on_torch = on_torch.push(try Conform.on_torch(data[i], shapes[i]))
  end

  ref_x = try Conform.on_ref(X_DATA, [4, 3])
  ref_y = try Conform.on_ref(Y_DATA, [4, 2])
  torch_x = try Conform.on_torch(X_DATA, [4, 3])
  torch_y = try Conform.on_torch(Y_DATA, [4, 2])
  want = try Trace.value_and_grad(on_ref, ->(p: List[Trace.Expr]) { mlp(p, ref_x, ref_y) })
  got = try Trace.value_and_grad(on_torch, ->(p: List[Trace.Expr]) { mlp(p, torch_x, torch_y) })
  try near(got.value, want.value, "the loss")
  for i in 0...data.size
    try near(got.grads[i], want.grads[i], "the gradient of parameter #{i}")
  end

  ok()
end

## `data` as an `F64` tensor of `shape` on the reference backend.
def f64(data: List[Float], shape: List[Int]) -> Result[Tensor.Tensor]
  floats = data.map do |x: Float| Tensor.Values.Float(x) end
  Tensor.from_list(floats, shape, Tensor.Dtype.F64, Tensor.Backend.Reference)
end

## The one value of a one-element tensor.
def scalar(t: Tensor.Tensor) -> Result[Float]
  values = try Support.floats_of(t)
  ok(values[0])
end

## A scalar that weighs each element of `y` differently, so every element's
## gradient is checked with its own cotangent.
def weigh(y: Trace.Expr) -> Result[Trace.Expr]
  n = Trace.shape(y).reduce(1) do |acc, d| acc * d end
  weights = (0...n).to_a.map { |k| 0.3 + 0.25 * (k % 4).to_f - 0.1 * k.to_f }
  r = Tensor.from_list(
    weights.map { |w| Tensor.Values.Float(w) },
    Trace.shape(y),
    Trace.dtype(y),
    Tensor.Backend.Reference,
  )

  Trace.sum(try Trace.mul(y, Trace.constant(try r)))
end

## The gradient `f` gives at `args` against the central difference of `f`
## itself, element by element, and the value against `eval`.
def check_grad(args: List[Tensor.Tensor], f: Fn[List[Trace.Expr], Result[Trace.Expr]]) -> Result[Void]
  got = try Trace.value_and_grad(args, f)
  try Test.eq(Support.floats_of(got.value), Support.floats_of(try Trace.eval(args, f)))
  for i in 0...args.size
    try Test.eq(Tensor.shape(got.grads[i]), Tensor.shape(args[i]))
    analytic = try Support.floats_of(got.grads[i])
    base = try Support.floats_of(args[i])
    for j in 0...base.size
      up = try scalar(try Trace.eval(try nudged(args, i, base, j, EPS), f))
      down = try scalar(try Trace.eval(try nudged(args, i, base, j, -EPS), f))
      slope = (up - down) / (2.0 * EPS)
      gap = (slope - analytic[j]).abs
      try Test.assert(
        gap <= TOL * (1.0 + slope.abs),
        "argument #{i}, value #{j}: slope #{slope}, gradient #{analytic[j]}",
      )
    end
  end

  ok()
end

## `args` with value `j` of argument `i` moved by `by`.
def nudged(args: List[Tensor.Tensor], i: Int, base: List[Float], j: Int, by: Float) -> Result[List[Tensor.Tensor]]
  moved = base.set(j, base[j] + by)
  t = try f64(moved, Tensor.shape(args[i]))
  ok(args.set(i, t))
end

## The MLP's parameters: `w1 [3, 5]`, `b1 [5]`, `w2 [5, 2]`, `b2 [2]`.
def mlp_params() -> Result[List[Tensor.Tensor]]
  ok([try f64(W1_DATA, [3, 5]), try f64(B1_DATA, [5]), try f64(W2_DATA, [5, 2]), try f64(B2_DATA, [2])])
end

## The mean squared error of `relu(x w1 + b1) w2 + b2` against `y`, for `F64`
## parameters.
def mlp_loss(p: List[Trace.Expr]) -> Result[Trace.Expr] = mlp(p, try f64(X_DATA, [4, 3]), try f64(Y_DATA, [4, 2]))

## The MLP loss over data `x` and labels `y`, constants of the trace.
def mlp(p: List[Trace.Expr], x: Tensor.Tensor, y: Tensor.Tensor) -> Result[Trace.Expr]
  pre = try Trace.add(try Trace.matmul(Trace.constant(x), p[0]), try Trace.broadcast_to(p[1], [4, 5]))
  h = try Trace.relu(pre)
  o = try Trace.add(try Trace.matmul(h, p[2]), try Trace.broadcast_to(p[3], [4, 2]))
  d = try Trace.sub(o, Trace.constant(y))
  Trace.mean(try Trace.mul(d, d))
end

## Two tensors agree within what `F32` arithmetic on one side allows.
def near(got: Tensor.Tensor, want: Tensor.Tensor, what: String) -> Result[Void]
  try Test.eq(Tensor.shape(got), Tensor.shape(want))
  ours = try Support.floats_of(got)
  theirs = try Support.floats_of(want)
  for i in 0...theirs.size
    gap = (ours[i] - theirs[i]).abs
    if gap > TOL * (1.0 + theirs[i].abs)
      return error("#{what}: value #{i} is #{ours[i]}, the reference says #{theirs[i]}")
    end
  end

  ok()
end
