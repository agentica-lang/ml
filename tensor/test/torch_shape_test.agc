## `reshape` through the dispatch, plus the volume rule that keeps a native
## reshape from reinterpreting memory the wrapper did not expect.

import Conform from "torch_support"
import Tensor from "../tensor"

private

def test_reshape_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.reshape(try Conform.on_torch(data, [2, 3]), [3, 2])
  try Test.eq(Tensor.shape(ours), [3, 2])
  Conform.agrees(ours, try Tensor.reshape(try Conform.on_ref(data, [2, 3]), [3, 2]), "reshape")
end

def test_reshape_to_a_rank_zero_tensor_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  ours = try Tensor.reshape(try Conform.on_torch([4.5], [1]), [])
  try Test.eq(Tensor.shape(ours), [])
  Conform.agrees(ours, try Tensor.reshape(try Conform.on_ref([4.5], [1]), []), "reshape to rank 0")
end

def test_reshape_to_a_bigger_rank_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0]
  Conform.agrees(
    try Tensor.reshape(try Conform.on_torch(data, [4]), [2, 1, 2]),
    try Tensor.reshape(try Conform.on_ref(data, [4]), [2, 1, 2]),
    "reshape to rank 3",
  )
end

def test_reshape_with_a_wrong_volume_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0]
  try Test.error?(Tensor.reshape(try Conform.on_torch(data, [3]), [2, 2]))
  Test.error?(Tensor.reshape(try Conform.on_ref(data, [3]), [2, 2]))
end

def test_reshape_with_a_negative_dimension_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0]
  try Test.error?(Tensor.reshape(try Conform.on_torch(data, [2]), [2, -1]))
  Test.error?(Tensor.reshape(try Conform.on_ref(data, [2]), [2, -1]))
end

def test_squeeze_dropping_every_size_one_axis_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0]
  ours = try Tensor.squeeze(try Conform.on_torch(data, [1, 3, 1]), none)
  try Test.eq(Tensor.shape(ours), [3])
  Conform.agrees(ours, try Tensor.squeeze(try Conform.on_ref(data, [1, 3, 1]), none), "squeeze every axis")
end

def test_squeeze_of_the_named_axis_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0]
  ours = try Tensor.squeeze(try Conform.on_torch(data, [1, 2, 2]), some(0))
  try Test.eq(Tensor.shape(ours), [2, 2])
  Conform.agrees(ours, try Tensor.squeeze(try Conform.on_ref(data, [1, 2, 2]), some(0)), "squeeze axis 0")
end

def test_squeeze_of_a_dimension_that_is_not_one_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.squeeze(try Conform.on_torch([1.0, 2.0], [2]), some(0)))
  Test.error?(Tensor.squeeze(try Conform.on_ref([1.0, 2.0], [2]), some(0)))
end

def test_unsqueeze_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0]
  ours = try Tensor.unsqueeze(try Conform.on_torch(data, [3]), 0)
  try Test.eq(Tensor.shape(ours), [1, 3])
  try Conform.agrees(ours, try Tensor.unsqueeze(try Conform.on_ref(data, [3]), 0), "unsqueeze at 0")
  Conform.agrees(
    try Tensor.unsqueeze(try Conform.on_torch(data, [3]), 1),
    try Tensor.unsqueeze(try Conform.on_ref(data, [3]), 1),
    "unsqueeze at 1",
  )
end

def test_unsqueeze_past_the_rank_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.unsqueeze(try Conform.on_torch([1.0], [1]), 2))
  Test.error?(Tensor.unsqueeze(try Conform.on_ref([1.0], [1]), 2))
end

def test_slice_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
  ours = try Tensor.slice(try Conform.on_torch(data, [3, 4]), [0, 1], [2, 3])
  try Test.eq(Tensor.shape(ours), [2, 2])
  Conform.agrees(ours, try Tensor.slice(try Conform.on_ref(data, [3, 4]), [0, 1], [2, 3]), "slice 2x2 window")
end

def test_slice_of_one_row_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.slice(try Conform.on_torch(data, [2, 3]), [1, 0], [2, 3])
  try Test.eq(Tensor.shape(ours), [1, 3])
  Conform.agrees(ours, try Tensor.slice(try Conform.on_ref(data, [2, 3]), [1, 0], [2, 3]), "slice last row")
end

def test_slice_past_the_edge_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.slice(try Conform.on_torch([1.0, 2.0], [2]), [0], [3]))
  Test.error?(Tensor.slice(try Conform.on_ref([1.0, 2.0], [2]), [0], [3]))
end

def test_concat_along_an_existing_axis_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  wide = [1.0, 2.0, 3.0, 4.0]
  row = [5.0, 6.0]
  other = [7.0, 8.0]
  # Three tensors, so the fold through the binary native call is exercised.
  ours = try Tensor.concat(
    [try Conform.on_torch(wide, [2, 2]), try Conform.on_torch(row, [1, 2]), try Conform.on_torch(other, [1, 2])],
    0,
  )

  try Test.eq(Tensor.shape(ours), [4, 2])
  Conform.agrees(
    ours,
    try Tensor.concat(
      [try Conform.on_ref(wide, [2, 2]), try Conform.on_ref(row, [1, 2]), try Conform.on_ref(other, [1, 2])],
      0,
    ),
    "concat along 0",
  )
end

def test_concat_along_the_last_axis_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  left = [1.0, 2.0, 3.0, 4.0]
  right = [5.0, 6.0]
  ours = try Tensor.concat([try Conform.on_torch(left, [2, 2]), try Conform.on_torch(right, [2, 1])], 1)
  try Test.eq(Tensor.shape(ours), [2, 3])
  Conform.agrees(
    ours,
    try Tensor.concat([try Conform.on_ref(left, [2, 2]), try Conform.on_ref(right, [2, 1])], 1),
    "concat along 1",
  )
end

def test_concat_with_a_mismatched_dimension_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  square = try Conform.on_torch([1.0, 2.0, 3.0, 4.0], [2, 2])
  wide = try Conform.on_torch([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  try Test.error?(Tensor.concat([square, wide], 0))
  Test.error?(Tensor.concat([try Conform.on_ref([1.0, 2.0, 3.0, 4.0], [2, 2]), try Conform.on_ref([1.0], [1])], 0))
end

def test_stack_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0]
  ours = try Tensor.stack([try Conform.on_torch(data, [2, 2]), try Conform.on_torch(data, [2, 2])], 0)
  try Test.eq(Tensor.shape(ours), [2, 2, 2])
  Conform.agrees(
    ours,
    try Tensor.stack([try Conform.on_ref(data, [2, 2]), try Conform.on_ref(data, [2, 2])], 0),
    "stack along 0",
  )
end

def test_stack_along_the_last_axis_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0]
  ours = try Tensor.stack([try Conform.on_torch(data, [3]), try Conform.on_torch(data, [3])], 1)
  try Test.eq(Tensor.shape(ours), [3, 2])
  Conform.agrees(
    ours,
    try Tensor.stack([try Conform.on_ref(data, [3]), try Conform.on_ref(data, [3])], 1),
    "stack along 1",
  )
end

def test_stack_of_two_shapes_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.stack([try Conform.on_torch([1.0, 2.0], [2]), try Conform.on_torch([1.0, 2.0], [1, 2])], 0))
  Test.error?(Tensor.stack([try Conform.on_ref([1.0, 2.0], [2]), try Conform.on_ref([1.0, 2.0], [1, 2])], 0))
end

def test_broadcast_to_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0]
  ours = try Tensor.broadcast_to(try Conform.on_torch(data, [1, 3]), [2, 3])
  try Test.eq(Tensor.shape(ours), [2, 3])
  try Conform.agrees(ours, try Tensor.broadcast_to(try Conform.on_ref(data, [1, 3]), [2, 3]), "broadcast a row")
  Conform.agrees(
    try Tensor.broadcast_to(try Conform.on_torch([5.0], [1]), [2, 2]),
    try Tensor.broadcast_to(try Conform.on_ref([5.0], [1]), [2, 2]),
    "broadcast a scalar",
  )
end

def test_broadcast_to_that_drops_an_axis_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.broadcast_to(try Conform.on_torch([1.0, 2.0], [2]), []))
  Test.error?(Tensor.broadcast_to(try Conform.on_ref([1.0, 2.0], [2]), []))
end

def test_broadcast_to_that_narrows_an_axis_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # libtorch's `expand` refuses both of these on its own; the wrapper must not
  # let either reach it, and the reference refuses them too.
  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  try Test.error?(Tensor.broadcast_to(try Conform.on_torch(data, [2, 3]), [2, 2]))
  Test.error?(Tensor.broadcast_to(try Conform.on_ref(data, [2, 3]), [2, 2]))
end

def test_gather_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ids = [2, 0, 2]
  ours = try Tensor.gather(try Conform.on_torch(table, [3, 2]), try Conform.on_torch_i64(ids, [3]))
  try Test.eq(Tensor.shape(ours), [3, 2])
  Conform.agrees(
    ours,
    try Tensor.gather(try Conform.on_ref(table, [3, 2]), try Conform.on_ref_i64(ids, [3])),
    "gather rows",
  )
end

def test_gather_with_a_rank_two_index_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  table = [1.0, 2.0, 3.0, 4.0]
  ids = [1, 0, 0, 1]
  ours = try Tensor.gather(try Conform.on_torch(table, [2, 2]), try Conform.on_torch_i64(ids, [2, 2]))
  try Test.eq(Tensor.shape(ours), [2, 2, 2])
  Conform.agrees(
    ours,
    try Tensor.gather(try Conform.on_ref(table, [2, 2]), try Conform.on_ref_i64(ids, [2, 2])),
    "gather a rank-2 index",
  )
end

def test_gather_with_an_index_out_of_range_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  table = [1.0, 2.0, 3.0, 4.0]
  try Test.error?(Tensor.gather(try Conform.on_torch(table, [2, 2]), try Conform.on_torch_i64([0, 2], [2])))
  try Test.error?(Tensor.gather(try Conform.on_torch(table, [2, 2]), try Conform.on_torch_i64([-1], [1])))
  try Test.error?(Tensor.gather(try Conform.on_ref(table, [2, 2]), try Conform.on_ref_i64([0, 2], [2])))
  Test.error?(Tensor.gather(try Conform.on_ref(table, [2, 2]), try Conform.on_ref_i64([-1], [1])))
end

def test_gather_with_float_indices_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  table = [1.0, 2.0, 3.0, 4.0]
  try Test.error?(Tensor.gather(try Conform.on_torch(table, [2, 2]), try Conform.on_torch([0.0], [1])))
  Test.error?(Tensor.gather(try Conform.on_ref(table, [2, 2]), try Conform.on_ref([0.0], [1])))
end

def test_concat_of_one_tensor_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0]
  ours = try Tensor.concat([try Conform.on_torch(data, [2, 2])], 0)
  try Test.eq(Tensor.shape(ours), [2, 2])
  Conform.agrees(ours, try Tensor.concat([try Conform.on_ref(data, [2, 2])], 0), "concat of one tensor")
end

def test_stack_of_one_tensor_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0]
  ours = try Tensor.stack([try Conform.on_torch(data, [3])], 0)
  try Test.eq(Tensor.shape(ours), [1, 3])
  Conform.agrees(ours, try Tensor.stack([try Conform.on_ref(data, [3])], 0), "stack of one tensor")
end
