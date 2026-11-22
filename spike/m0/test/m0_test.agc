## What the spike can prove on its own.
##
## The reference ops against hand-computed values, the hand-written backward
## against a central difference, the demo's loop twice for determinism, and
## the native side against the reference when the module built (skipped, not
## failed, when libtorch is not there: the two tests say so in their name).
##
## ```sh
## agentica test
## ```

import Dataset from "../dataset"
import Model from "../model"
import TensorRef from "../tensor_ref"
import Torch from "../torch_spike"

# The step of the central difference. Small enough for the linear pieces of
# the loss to dominate, large enough to stay out of double rounding.
EPS = 0.000001

def test_from_list_builds_a_row_major_tensor() -> Result[Void]
  Test.eq(
    TensorRef.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
    ok(TensorRef.Tensor(shape: [2, 3], dtype: TensorRef.Dtype.F64, data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
  )
end

def test_zeros_fills_the_shape() -> Result[Void]
  Test.eq(
    TensorRef.zeros([2, 2]),
    ok(TensorRef.Tensor(shape: [2, 2], dtype: TensorRef.Dtype.F64, data: [0.0, 0.0, 0.0, 0.0])),
  )
end

def test_from_list_rejects_a_count_that_does_not_fit_the_shape() -> Result[Void]
  Test.error?(TensorRef.from_list([1.0, 2.0, 3.0], [2, 2]))
end

def test_a_negative_dimension_is_an_error() -> Result[Void] = Test.error?(TensorRef.zeros([2, -1]))

def test_add_is_elementwise() -> Result[Void]
  a = try TensorRef.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
  b = try TensorRef.from_list([10.0, 20.0, 30.0, 40.0], [2, 2])
  Test.eq(
    TensorRef.add(a, b),
    ok(TensorRef.Tensor(shape: [2, 2], dtype: TensorRef.Dtype.F64, data: [11.0, 22.0, 33.0, 44.0])),
  )
end

def test_add_rejects_two_shapes() -> Result[Void]
  a = try TensorRef.from_list([1.0, 2.0], [2])
  b = try TensorRef.from_list([1.0, 2.0], [1, 2])
  Test.error?(TensorRef.add(a, b))
end

def test_mul_is_elementwise() -> Result[Void]
  a = try TensorRef.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
  b = try TensorRef.from_list([2.0, 3.0, 4.0, 5.0], [2, 2])
  Test.eq(
    TensorRef.mul(a, b),
    ok(TensorRef.Tensor(shape: [2, 2], dtype: TensorRef.Dtype.F64, data: [2.0, 6.0, 12.0, 20.0])),
  )
end

def test_matmul_multiplies_two_matrices() -> Result[Void]
  a = try TensorRef.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  b = try TensorRef.from_list([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
  Test.eq(
    TensorRef.matmul(a, b),
    ok(TensorRef.Tensor(shape: [2, 2], dtype: TensorRef.Dtype.F64, data: [58.0, 64.0, 139.0, 154.0])),
  )
end

def test_matmul_rejects_a_mismatched_inner_dimension() -> Result[Void]
  a = try TensorRef.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  b = try TensorRef.from_list([1.0, 2.0], [2, 1])
  Test.error?(TensorRef.matmul(a, b))
end

def test_sum_reduces_to_a_rank_zero_tensor() -> Result[Void]
  t = try TensorRef.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
  Test.eq(TensorRef.sum(t), TensorRef.Tensor(shape: [], dtype: TensorRef.Dtype.F64, data: [10.0]))
end

def test_relu_clamps_at_zero() -> Result[Void]
  t = try TensorRef.from_list([-1.5, 0.0, 2.5], [3])
  Test.eq(TensorRef.relu(t), TensorRef.Tensor(shape: [3], dtype: TensorRef.Dtype.F64, data: [0.0, 0.0, 2.5]))
end

def test_transpose_swaps_rows_and_columns() -> Result[Void]
  t = try TensorRef.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  Test.eq(
    TensorRef.transpose(t),
    ok(TensorRef.Tensor(shape: [3, 2], dtype: TensorRef.Dtype.F64, data: [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])),
  )
end

# The point of a hand-written backward: it can be wrong, so it is checked
# against the forward pass itself. Every element of w is nudged both ways and
# the slope of the loss is compared with what `loss_and_grad` claims.
def test_the_backward_matches_a_central_difference() -> Result[Void]
  x = try TensorRef.from_list(Dataset.X_DATA, Dataset.X_SHAPE)
  w = try TensorRef.from_list(Dataset.W_DATA, Dataset.W_SHAPE)
  step = try Model.loss_and_grad(x, w)
  worst = 0.0
  i = 0
  while i < w.data.size
    up = try bump(w, i, EPS)
    down = try bump(w, i, -EPS)
    slope = (try forward_loss(x, up) - try forward_loss(x, down)) / (2.0 * EPS)
    gap = (slope - step.grad.data[i]).abs
    if gap > worst
      worst = gap
    end

    i += 1
  end

  Test.assert(worst < 0.000001, "the backward and a central difference differ by #{worst}")
end

def test_the_curve_is_the_same_twice() -> Result[Void] = Test.eq(try curve(), try curve())

# The native side, when it built. libtorch is a system dependency: without it
# the module is `optional` and unbuilt, `Native.load` says no, and the test
# states that instead of passing silently.
def test_native_loss_matches_the_reference_or_libtorch_is_absent() -> Result[Void]
  return Test.assert(!ready(), "torchspike is not built: libtorch is not installed (see README)") if !ready()
  x = try TensorRef.from_list(Dataset.X_DATA, Dataset.X_SHAPE)
  w = try TensorRef.from_list(Dataset.W_DATA, Dataset.W_SHAPE)
  native = try Torch.loss(x, w)
  step = try Model.loss_and_grad(x, w)
  Test.assert(
    (native - step.loss).abs < Dataset.TOLERANCE,
    "torchspike gives #{native}, the reference gives #{step.loss}",
  )
end

def test_native_matmul_sums_like_the_reference_or_libtorch_is_absent() -> Result[Void]
  return Test.assert(!ready(), "torchspike is not built: libtorch is not installed (see README)") if !ready()
  x = try TensorRef.from_list(Dataset.X_DATA, Dataset.X_SHAPE)
  w = try TensorRef.from_list(Dataset.W_DATA, Dataset.W_SHAPE)
  reference = try TensorRef.matmul(x, w)
  native = try native_matmul_sum(x, w)
  Test.assert(
    (native - reference.data.sum).abs < Dataset.TOLERANCE,
    "torchspike sums to #{native}, the reference to #{reference.data.sum}",
  )
end

private

def ready() -> Bool = Native.load("torchspike").ok?

# matmul and sum through the module, in one arena: the only readback the five
# exposed functions allow.
def native_matmul_sum(x: TensorRef.Tensor, w: TensorRef.Tensor) -> Result[Float]
  Native.arena do
    tx = try Torch.create(x.shape, x.data)
    tw = try Torch.create(w.shape, w.data)
    y = try Torch.matmul(tx, tw)
    Torch.sum(y)
  end
end

# The forward pass alone, for the central difference.
def forward_loss(x: TensorRef.Tensor, w: TensorRef.Tensor) -> Result[Float]
  y = try TensorRef.matmul(x, w)
  TensorRef.scalar(TensorRef.sum(TensorRef.relu(y)))
end

# `w` with one element moved by `delta`.
def bump(w: TensorRef.Tensor, index: Int, delta: Float) -> Result[TensorRef.Tensor]
  TensorRef.from_list(w.data.set(index, w.data[index] + delta), w.shape)
end

# The demo's loop, for the determinism test.
def curve() -> Result[List[Float]]
  x = try TensorRef.from_list(Dataset.X_DATA, Dataset.X_SHAPE)
  w = try TensorRef.from_list(Dataset.W_DATA, Dataset.W_SHAPE)
  losses: List[Float] = []
  for _step in 0...Dataset.STEPS
    step = try Model.loss_and_grad(x, w)
    losses = losses.push(step.loss)
    w = try Model.sgd_step(w, step.grad, Dataset.LR)
  end

  ok(losses)
end
