## Elementwise ops against the reference backend, plus the error paths: the
## shapes, dtypes and non-finite inputs an eager backend would happily compute
## through.

import Ref from "pkg/tensor/tensor"
import Support from "support"
import Torch from "../torch"

private

def test_add_agrees_with_the_reference() -> Result[Void]
  return Support.absent() if !Support.available()

  a = try Support.torch_f32([1.0, 2.0, 3.0, 4.0], [2, 2])
  b = try Support.torch_f32([10.0, 20.0, 30.0, 40.0], [2, 2])
  ra = try Support.ref_f32([1.0, 2.0, 3.0, 4.0], [2, 2])
  rb = try Support.ref_f32([10.0, 20.0, 30.0, 40.0], [2, 2])
  Support.agrees(try Torch.add(Support.CPU, a, b), try Ref.add(ra, rb), "add")
end

def test_mul_agrees_with_the_reference() -> Result[Void]
  return Support.absent() if !Support.available()

  a = try Support.torch_f32([0.5, -2.0, 3.25], [3])
  b = try Support.torch_f32([4.0, 8.0, -2.0], [3])
  ra = try Support.ref_f32([0.5, -2.0, 3.25], [3])
  rb = try Support.ref_f32([4.0, 8.0, -2.0], [3])
  Support.agrees(try Torch.mul(Support.CPU, a, b), try Ref.mul(ra, rb), "mul")
end

def test_relu_agrees_with_the_reference() -> Result[Void]
  return Support.absent() if !Support.available()

  ours = try Torch.relu(Support.CPU, try Support.torch_f32([-2.0, 0.0, 1.5], [3]))
  Support.agrees(ours, try Ref.relu(try Support.ref_f32([-2.0, 0.0, 1.5], [3])), "relu")
end

def test_exp_agrees_with_the_reference_inside_the_float32_range() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [-8.0, -1.0, 0.0, 0.5, 1.0, 7.0]
  ours = try Torch.exp(Support.CPU, try Support.torch_f32(data, [2, 3]))
  Support.agrees(ours, try Ref.exp(try Support.ref_f32(data, [2, 3])), "exp")
end

def test_log_agrees_with_the_reference() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [0.5, 1.0, 2.0, 10.0, 1024.0]
  ours = try Torch.log(Support.CPU, try Support.torch_f32(data, [5]))
  Support.agrees(ours, try Ref.log(try Support.ref_f32(data, [5])), "log")
end

def test_add_rejects_two_shapes() -> Result[Void]
  return Support.absent() if !Support.available()

  # libtorch broadcasts [2] against [1, 2] on its own; the wrapper must not
  # let it (the reference has no implicit broadcasting).
  a = try Support.torch_f32([1.0, 2.0], [2])
  b = try Support.torch_f32([1.0, 2.0], [1, 2])
  try Test.error?(Torch.add(Support.CPU, a, b))
  Test.error?(Ref.add(try Support.ref_f32([1.0, 2.0], [2]), try Support.ref_f32([1.0, 2.0], [1, 2])))
end

def test_add_rejects_two_dtypes() -> Result[Void]
  return Support.absent() if !Support.available()

  a = try Support.torch_f32([1.0], [1])
  b = try Support.torch_i64([1], [1])
  Test.error?(Torch.add(Support.CPU, a, b))
end

def test_add_rejects_int_tensors() -> Result[Void]
  return Support.absent() if !Support.available()

  a = try Support.torch_i64([1, 2], [2])
  b = try Support.torch_i64([3, 4], [2])
  try Test.error?(Torch.add(Support.CPU, a, b))
  Test.error?(Ref.add(try Support.ref_i64([1, 2], [2]), try Support.ref_i64([3, 4], [2])))
end

def test_relu_rejects_int_tensors() -> Result[Void]
  return Support.absent() if !Support.available()

  Test.error?(Torch.relu(Support.CPU, try Support.torch_i64([1], [1])))
end

def test_exp_of_a_saturated_value_is_an_error() -> Result[Void]
  return Support.absent() if !Support.available()

  # exp(1000) is +inf in float32: a finite input with a saturating result,
  # which the reference allows too. Feeding that back in must be an error, not
  # a NaN in the result.
  saturated = try Torch.exp(Support.CPU, try Support.torch_f32([1000.0], [1]))
  try Test.error?(Torch.exp(Support.CPU, saturated))
  try Test.error?(Torch.log(Support.CPU, saturated))
  Test.eq(try Support.floats_of(saturated), [1.0e308 * 10.0])
end

def test_log_of_a_non_positive_value_is_an_error() -> Result[Void]
  return Support.absent() if !Support.available()

  zero = try Support.torch_f32([1.0, 0.0], [2])
  negative = try Support.torch_f32([1.0, -2.0], [2])
  try Test.error?(Torch.log(Support.CPU, zero))
  try Test.error?(Torch.log(Support.CPU, negative))
  Test.error?(Ref.log(try Support.ref_f32([1.0, 0.0], [2])))
end
