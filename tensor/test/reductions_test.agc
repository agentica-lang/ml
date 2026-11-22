## Reductions: values, axes, `keep_dims`, and the error paths.

import Support from "support"
import Tensor from "../tensor"

private

def test_sum_reduces_everything() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  total = try Tensor.sum(t, none, false)
  try Test.eq(Tensor.shape(total), [])
  Support.expect_floats(total, [21.0])
end

def test_sum_reduces_one_axis() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  row = try Tensor.sum(t, some([1]), false)
  try Test.eq(Tensor.shape(row), [2])
  Support.expect_floats(row, [6.0, 15.0])
end

def test_sum_keeps_dims() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  col = try Tensor.sum(t, some([0]), true)
  try Test.eq(Tensor.shape(col), [1, 3])
  Support.expect_floats(col, [5.0, 7.0, 9.0])
end

def test_sum_of_an_empty_tensor_is_zero() -> Result[Void]
  t = try Tensor.zeros([0], Tensor.Dtype.F32, Tensor.Backend.Reference)
  total = try Tensor.sum(t, none, false)
  Support.expect_floats(total, [0.0])
end

def test_sum_rejects_a_wrong_dtype() -> Result[Void]
  t = try Support.i64([1, 2], [2])
  Test.error?(Tensor.sum(t, none, false))
end

def test_sum_rejects_an_axis_out_of_range() -> Result[Void]
  t = try Support.f32([1.0, 2.0], [2])
  Test.error?(Tensor.sum(t, some([2]), false))
end

def test_sum_rejects_a_repeated_axis() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 4.0], [2, 2])
  Test.error?(Tensor.sum(t, some([0, 0]), false))
end

def test_mean_divides_by_the_reduced_count() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  m = try Tensor.mean(t, some([1]), false)
  Support.expect_floats(m, [2.0, 5.0])
end

def test_mean_of_an_empty_tensor_is_an_error() -> Result[Void]
  t = try Tensor.zeros([0, 3], Tensor.Dtype.F32, Tensor.Backend.Reference)
  Test.error?(Tensor.mean(t, none, false))
end

def test_max_takes_the_largest_per_block() -> Result[Void]
  t = try Support.f32([1.0, 5.0, 3.0, 4.0, 2.0, 6.0], [2, 3])
  m = try Tensor.max(t, some([1]), false)
  Support.expect_floats(m, [5.0, 6.0])
end

def test_max_of_an_empty_tensor_is_an_error() -> Result[Void]
  t = try Tensor.zeros([0], Tensor.Dtype.F32, Tensor.Backend.Reference)
  Test.error?(Tensor.max(t, none, false))
end

def test_argmax_takes_the_first_largest_position() -> Result[Void]
  t = try Support.f32([1.0, 5.0, 5.0, 2.0], [4])
  pos = try Tensor.argmax(t, none, false)
  try Test.eq(Tensor.dtype(pos), Tensor.Dtype.I64)
  try Test.eq(Tensor.shape(pos), [])
  Test.eq(try Tensor.to_list(pos), [Tensor.Values.Int(1)])
end

def test_argmax_keeps_dims() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  pos = try Tensor.argmax(t, some([1]), true)
  try Test.eq(Tensor.shape(pos), [2, 1])
  Test.eq(try Tensor.to_list(pos), [Tensor.Values.Int(2), Tensor.Values.Int(2)])
end

def test_argmax_over_two_axes_is_the_flat_offset() -> Result[Void]
  t = try Support.f32([1.0, 5.0, 4.0, 3.0], [2, 2])
  pos = try Tensor.argmax(t, some([0, 1]), false)
  try Test.eq(Tensor.shape(pos), [])
  Test.eq(try Tensor.to_list(pos), [Tensor.Values.Int(1)])
end

def test_argmax_of_an_empty_tensor_is_an_error() -> Result[Void]
  t = try Tensor.zeros([2, 0], Tensor.Dtype.F32, Tensor.Backend.Reference)
  Test.error?(Tensor.argmax(t, some([1]), false))
end
