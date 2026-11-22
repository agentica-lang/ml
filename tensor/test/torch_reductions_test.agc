## `sum` and `mean` through the dispatch: the axes conventions, `keep_dims`,
## the rank-0 case, and the reductions the reference refuses.

import Conform from "torch_support"
import Tensor from "../tensor"

private

def test_sum_of_every_element_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.sum(try Conform.on_torch(data, [2, 3]))
  try Test.eq(Tensor.shape(ours), [])
  Conform.agrees(ours, try Tensor.sum(try Conform.on_ref(data, [2, 3])), "sum")
end

def test_sum_over_one_axis_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.sum(try Conform.on_torch(data, [2, 3]), some([0]))
  try Test.eq(Tensor.shape(ours), [3])
  Conform.agrees(ours, try Tensor.sum(try Conform.on_ref(data, [2, 3]), some([0])), "sum axis 0")
end

def test_sum_over_two_axes_keeping_dims_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  ours = try Tensor.sum(try Conform.on_torch(data, [2, 2, 2]), some([0, 2]), keep_dims: true)
  try Test.eq(Tensor.shape(ours), [1, 2, 1])
  Conform.agrees(
    ours,
    try Tensor.sum(try Conform.on_ref(data, [2, 2, 2]), some([0, 2]), keep_dims: true),
    "sum two axes",
  )
end

def test_mean_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.mean(try Conform.on_torch(data, [2, 3]), some([1]))
  try Test.eq(Tensor.shape(ours), [2])
  Conform.agrees(ours, try Tensor.mean(try Conform.on_ref(data, [2, 3]), some([1])), "mean axis 1")
end

def test_sum_of_a_rank_zero_tensor_is_itself() -> Result[Void]
  return Conform.absent() if !Conform.available()

  ours = try Tensor.sum(try Conform.on_torch([2.5], []))
  try Test.eq(Tensor.shape(ours), [])
  Conform.agrees(ours, try Tensor.sum(try Conform.on_ref([2.5], [])), "sum of a rank-0 tensor")
end

def test_sum_over_no_axis_is_no_reduction() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0]
  ours = try Tensor.sum(try Conform.on_torch(data, [2, 2]), some([]))
  Conform.agrees(ours, try Tensor.sum(try Conform.on_ref(data, [2, 2]), some([])), "sum over no axis")
end

def test_mean_of_zero_elements_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  empty_ours = try Tensor.zeros([0, 3], Tensor.Dtype.F32, Conform.TORCH)
  empty_theirs = try Tensor.zeros([0, 3], Tensor.Dtype.F32, Conform.REF)
  try Test.error?(Tensor.mean(empty_ours))
  Test.error?(Tensor.mean(empty_theirs))
end

def test_bad_axes_are_errors_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  ours = try Conform.on_torch([1.0, 2.0], [2])
  theirs = try Conform.on_ref([1.0, 2.0], [2])
  try Test.error?(Tensor.sum(ours, some([1])))
  try Test.error?(Tensor.sum(theirs, some([1])))
  try Test.error?(Tensor.sum(ours, some([0, 0])))
  try Test.error?(Tensor.sum(theirs, some([0, 0])))
  Test.error?(Tensor.mean(ours, some([-1])))
end
