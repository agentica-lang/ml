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

def test_max_of_every_element_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-3.0, 7.5, 2.0, 0.0, 7.25, -1.0]
  ours = try Tensor.max(try Conform.on_torch(data, [2, 3]))
  try Test.eq(Tensor.shape(ours), [])
  Conform.agrees(ours, try Tensor.max(try Conform.on_ref(data, [2, 3])), "max")
end

def test_max_over_one_axis_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-3.0, 7.5, 2.0, 0.0, 7.25, -1.0]
  ours = try Tensor.max(try Conform.on_torch(data, [2, 3]), some([1]))
  try Test.eq(Tensor.shape(ours), [2])
  Conform.agrees(ours, try Tensor.max(try Conform.on_ref(data, [2, 3]), some([1])), "max axis 1")
end

def test_max_over_two_axes_keeping_dims_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0, 8.0]
  ours = try Tensor.max(try Conform.on_torch(data, [2, 2, 2]), some([0, 2]), keep_dims: true)
  try Test.eq(Tensor.shape(ours), [1, 2, 1])
  Conform.agrees(
    ours,
    try Tensor.max(try Conform.on_ref(data, [2, 2, 2]), some([0, 2]), keep_dims: true),
    "max two axes",
  )
end

def test_max_of_an_empty_tensor_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.max(try Tensor.zeros([0, 3], Tensor.Dtype.F32, Conform.TORCH)))
  Test.error?(Tensor.max(try Tensor.zeros([0, 3], Tensor.Dtype.F32, Conform.REF)))
end

def test_argmax_over_one_axis_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 9.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.argmax(try Conform.on_torch(data, [2, 3]), some([1]))
  try Test.eq(Tensor.dtype(ours), Tensor.Dtype.I64)
  try Test.eq(Tensor.shape(ours), [2])
  Conform.agrees(ours, try Tensor.argmax(try Conform.on_ref(data, [2, 3]), some([1])), "argmax axis 1")
end

def test_argmax_over_every_axis_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.argmax(try Conform.on_torch(data, [2, 3]))
  try Test.eq(Tensor.shape(ours), [])
  Conform.agrees(ours, try Tensor.argmax(try Conform.on_ref(data, [2, 3])), "argmax of everything")
end

def test_argmax_over_two_axes_keeping_dims_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
  ours = try Tensor.argmax(try Conform.on_torch(data, [2, 3, 2]), some([0, 2]), keep_dims: true)
  try Test.eq(Tensor.shape(ours), [1, 3, 1])
  Conform.agrees(
    ours,
    try Tensor.argmax(try Conform.on_ref(data, [2, 3, 2]), some([0, 2]), keep_dims: true),
    "argmax two axes",
  )
end

def test_argmax_keeps_the_lowest_position_on_a_tie() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # The reference keeps the first position on a tie (`>` against the best), and
  # the per-axis path here must answer the same index, not the last one.
  data = [1.0, 3.0, 3.0, 3.0, 3.0, 0.0]
  ours = try Tensor.argmax(try Conform.on_torch(data, [2, 3]), some([1]))
  Conform.agrees(ours, try Tensor.argmax(try Conform.on_ref(data, [2, 3]), some([1])), "argmax tie")
end

def test_argmax_over_no_axis_is_zeros_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0]
  ours = try Tensor.argmax(try Conform.on_torch(data, [2, 2]), some([]))
  try Test.eq(Tensor.shape(ours), [2, 2])
  Conform.agrees(ours, try Tensor.argmax(try Conform.on_ref(data, [2, 2]), some([])), "argmax of no axis")
end

def test_argmax_of_an_empty_tensor_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.argmax(try Tensor.zeros([0, 3], Tensor.Dtype.F32, Conform.TORCH)))
  Test.error?(Tensor.argmax(try Tensor.zeros([0, 3], Tensor.Dtype.F32, Conform.REF)))
end

def test_argmax_rejects_an_int_tensor_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.argmax(try Conform.on_torch_i64([1, 2], [2])))
  Test.error?(Tensor.argmax(try Conform.on_ref_i64([1, 2], [2])))
end

def test_argmax_with_unsorted_axes_breaks_ties_in_ascending_order() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # `[2, 1, 2]` with a tie between (i0=0, i2=1) and (i0=1, i2=0), asked for with
  # `axes = [2, 0]`. The wrapper reads the axes as the ascending set {0, 2}, so
  # the position inside the block is the row-major offset of that order: 1 for
  # (i0=0, i2=1) and 2 for (i0=1, i2=0), and the tie goes to 1.
  data = [1.0, 5.0, 5.0, 0.0]
  ours = try Tensor.argmax(try Conform.on_torch(data, [2, 1, 2]), some([2, 0]))
  theirs = try Tensor.argmax(try Conform.on_ref(data, [2, 1, 2]), some([2, 0]))
  try Test.eq(Tensor.shape(ours), [1])
  try Test.eq(try Conform.ints_of(ours), [1])
  Conform.agrees(ours, theirs, "argmax with unsorted axes")
end

def test_argmax_with_unsorted_axes_numbers_the_block_by_the_ascending_order() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # The two reduced axes have different sizes, which is where the order of the
  # axes is visible in the number, not just in the tie. `[2, 1, 3]` under
  # `{0, 2}` numbers the block as `i0 * 3 + i2`, so the unique maximum at
  # (i0=1, i2=2) is 5. Reading `[2, 0]` literally would number it `i2 * 2 + i0`
  # on one side and `i0 * 2 + i2` on the other, and the two backends would
  # answer 5 and 4.
  data = [1.0, 2.0, 3.0, 4.0, 5.0, 9.0]
  ours = try Tensor.argmax(try Conform.on_torch(data, [2, 1, 3]), some([2, 0]), keep_dims: true)
  theirs = try Tensor.argmax(try Conform.on_ref(data, [2, 1, 3]), some([2, 0]), keep_dims: true)
  try Test.eq(Tensor.shape(ours), [1, 1, 1])
  try Test.eq(try Conform.ints_of(ours), [5])
  Conform.agrees(ours, theirs, "argmax with unsorted axes and no tie")
end

def test_argmax_with_unsorted_axes_breaks_a_tie_over_uneven_axes() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # A tie between (i0=0, i2=2) and (i0=1, i2=0), with `[2, 1, 3]` and `{0, 2}`:
  # their offsets are 2 and 3, so the lower one wins. Read literally, the two
  # backends collide on those two elements and answer 1 and 2 instead.
  data = [1.0, 2.0, 9.0, 9.0, 5.0, 6.0]
  ours = try Tensor.argmax(try Conform.on_torch(data, [2, 1, 3]), some([2, 0]))
  theirs = try Tensor.argmax(try Conform.on_ref(data, [2, 1, 3]), some([2, 0]))
  try Test.eq(try Conform.ints_of(ours), [2])
  Conform.agrees(ours, theirs, "argmax tie over uneven axes")
end
