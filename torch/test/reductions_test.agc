## `sum` and `mean` against the reference backend: the axes conventions,
## `keep_dims`, the rank-0 case and the reductions the reference refuses.

import Ref from "pkg/tensor/tensor"
import Support from "support"
import Torch from "../torch"

private

def test_sum_of_every_element_agrees() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Torch.sum(Support.CPU, try Support.torch_f32(data, [2, 3]))
  try Test.eq(Torch.shape(ours), [])
  Support.agrees(ours, try Ref.sum(try Support.ref_f32(data, [2, 3])), "sum")
end

def test_sum_over_one_axis_agrees() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Torch.sum(Support.CPU, try Support.torch_f32(data, [2, 3]), some([0]))
  try Test.eq(Torch.shape(ours), [3])
  Support.agrees(ours, try Ref.sum(try Support.ref_f32(data, [2, 3]), some([0])), "sum axis 0")
end

def test_sum_over_two_axes_keeping_dims_agrees() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  ours = try Torch.sum(Support.CPU, try Support.torch_f32(data, [2, 2, 2]), some([0, 2]), keep_dims: true)
  try Test.eq(Torch.shape(ours), [1, 2, 1])
  want = try Ref.sum(try Support.ref_f32(data, [2, 2, 2]), some([0, 2]), keep_dims: true)
  Support.agrees(ours, want, "sum axes 0 and 2, keep_dims")
end

def test_mean_agrees_with_the_reference() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Torch.mean(Support.CPU, try Support.torch_f32(data, [2, 3]), some([1]))
  try Test.eq(Torch.shape(ours), [2])
  Support.agrees(ours, try Ref.mean(try Support.ref_f32(data, [2, 3]), some([1])), "mean axis 1")
end

def test_sum_of_a_rank_zero_tensor_is_itself() -> Result[Void]
  return Support.absent() if !Support.available()

  scalar = try Support.torch_f32([2.5], [])
  ours = try Torch.sum(Support.CPU, scalar)
  try Test.eq(Torch.shape(ours), [])
  Support.agrees(ours, try Ref.sum(try Support.ref_f32([2.5], [])), "sum of a rank-0 tensor")
end

def test_sum_over_no_axis_is_no_reduction() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1.0, 2.0, 3.0, 4.0]
  ours = try Torch.sum(Support.CPU, try Support.torch_f32(data, [2, 2]), some([]))
  Support.agrees(ours, try Ref.sum(try Support.ref_f32(data, [2, 2]), some([])), "sum over no axis")
end

def test_mean_of_zero_elements_is_an_error() -> Result[Void]
  return Support.absent() if !Support.available()

  empty = try Torch.zeros(Support.CPU, [0, 3], Torch.Dtype.F32)
  try Test.error?(Torch.mean(Support.CPU, empty))
  Test.error?(Ref.mean(try Ref.zeros([0, 3], Ref.Dtype.F32, Ref.Backend.Reference)))
end

def test_bad_axes_are_errors() -> Result[Void]
  return Support.absent() if !Support.available()

  t = try Support.torch_f32([1.0, 2.0], [2])
  try Test.error?(Torch.sum(Support.CPU, t, some([1])))
  try Test.error?(Torch.sum(Support.CPU, t, some([0, 0])))
  Test.error?(Torch.mean(Support.CPU, t, some([-1])))
end

def test_sum_rejects_int_tensors() -> Result[Void]
  return Support.absent() if !Support.available()

  Test.error?(Torch.sum(Support.CPU, try Support.torch_i64([1, 2], [2])))
end
