## Shape ops: values and error paths.

import Support from "support"
import Tensor from "../tensor"

private

def test_reshape_reorders_the_same_values() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  u = try Tensor.reshape(t, [3, 2])
  try Test.eq(Tensor.shape(u), [3, 2])
  Support.expect_floats(u, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
end

def test_reshape_rejects_a_volume_change() -> Result[Void]
  t = try Support.f32([1.0, 2.0], [2])
  Test.error?(Tensor.reshape(t, [2, 2]))
end

def test_squeeze_drops_every_size_one_dimension() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0], [1, 3, 1])
  u = try Tensor.squeeze(t, none)
  try Test.eq(Tensor.shape(u), [3])
  Support.expect_floats(u, [1.0, 2.0, 3.0])
end

def test_squeeze_drops_the_named_axis() -> Result[Void]
  t = try Support.f32([1.0, 2.0], [1, 2])
  u = try Tensor.squeeze(t, some(0))
  Support.expect_floats(u, [1.0, 2.0])
end

def test_squeeze_rejects_a_dimension_that_is_not_one() -> Result[Void]
  t = try Support.f32([1.0, 2.0], [2])
  Test.error?(Tensor.squeeze(t, some(0)))
end

def test_unsqueeze_inserts_a_dimension() -> Result[Void]
  t = try Support.f32([1.0, 2.0], [2])
  first = try Tensor.unsqueeze(t, 0)
  try Test.eq(Tensor.shape(first), [1, 2])
  last = try Tensor.unsqueeze(t, 1)
  try Test.eq(Tensor.shape(last), [2, 1])
  Test.eq(try Tensor.to_list(last), try Tensor.to_list(first))
end

def test_unsqueeze_rejects_an_axis_out_of_range() -> Result[Void]
  t = try Support.f32([1.0], [1])
  Test.error?(Tensor.unsqueeze(t, 2))
end

def test_slice_takes_a_window() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  u = try Tensor.slice(t, [0, 1], [2, 3])
  try Test.eq(Tensor.shape(u), [2, 2])
  Support.expect_floats(u, [2.0, 3.0, 5.0, 6.0])
end

def test_slice_rejects_a_window_outside_its_dimension() -> Result[Void]
  t = try Support.f32([1.0, 2.0], [2])
  Test.error?(Tensor.slice(t, [0], [3]))
end

def test_concat_joins_along_the_axis() -> Result[Void]
  a = try Support.f32([1.0, 2.0], [1, 2])
  b = try Support.f32([3.0, 4.0], [1, 2])
  down = try Tensor.concat([a, b], 0)
  try Test.eq(Tensor.shape(down), [2, 2])
  try Support.expect_floats(down, [1.0, 2.0, 3.0, 4.0])
  across = try Tensor.concat([a, b], 1)
  try Test.eq(Tensor.shape(across), [1, 4])
  Support.expect_floats(across, [1.0, 2.0, 3.0, 4.0])
end

def test_concat_rejects_nothing_to_join() -> Result[Void]
  empty: List[Tensor.Tensor] = []
  Test.error?(Tensor.concat(empty, 0))
end

def test_concat_rejects_other_dimensions_that_differ() -> Result[Void]
  a = try Support.f32([1.0, 2.0], [1, 2])
  b = try Support.f32([1.0, 2.0, 3.0], [1, 3])
  Test.error?(Tensor.concat([a, b], 0))
end

def test_stack_joins_along_a_new_axis() -> Result[Void]
  a = try Support.f32([1.0, 2.0], [2])
  b = try Support.f32([3.0, 4.0], [2])
  first = try Tensor.stack([a, b], 0)
  try Test.eq(Tensor.shape(first), [2, 2])
  try Support.expect_floats(first, [1.0, 2.0, 3.0, 4.0])
  second = try Tensor.stack([a, b], 1)
  try Test.eq(Tensor.shape(second), [2, 2])
  Support.expect_floats(second, [1.0, 3.0, 2.0, 4.0])
end

def test_stack_rejects_shapes_that_differ() -> Result[Void]
  a = try Support.f32([1.0, 2.0], [2])
  b = try Support.f32([1.0, 2.0, 3.0], [3])
  Test.error?(Tensor.stack([a, b], 0))
end

def test_broadcast_to_widens_size_one_dimensions() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0], [3])
  u = try Tensor.broadcast_to(t, [2, 3])
  try Test.eq(Tensor.shape(u), [2, 3])
  Support.expect_floats(u, [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
end

def test_broadcast_to_leaves_matching_shapes_alone() -> Result[Void]
  t = try Support.f32([1.0, 2.0], [1, 2])
  u = try Tensor.broadcast_to(t, [1, 2])
  Support.expect_floats(u, [1.0, 2.0])
end

def test_broadcast_to_rejects_an_incompatible_dimension() -> Result[Void]
  t = try Support.f32([1.0, 2.0], [2])
  Test.error?(Tensor.broadcast_to(t, [3]))
end

def test_broadcast_to_rejects_dropped_axes() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 4.0], [2, 2])
  Test.error?(Tensor.broadcast_to(t, [2]))
end

def test_gather_takes_rows_by_index() -> Result[Void]
  table = try Support.f32([10.0, 11.0, 20.0, 21.0, 30.0, 31.0], [3, 2])
  ids = try Support.i64([2, 0], [2])
  rows = try Tensor.gather(table, ids)
  try Test.eq(Tensor.shape(rows), [2, 2])
  Support.expect_floats(rows, [30.0, 31.0, 10.0, 11.0])
end

def test_gather_keeps_index_shape() -> Result[Void]
  table = try Support.f32([10.0, 11.0, 20.0, 21.0], [2, 2])
  ids = try Support.i64([1, 0], [2, 1])
  rows = try Tensor.gather(table, ids)
  try Test.eq(Tensor.shape(rows), [2, 1, 2])
  Support.expect_floats(rows, [20.0, 21.0, 10.0, 11.0])
end

def test_gather_rejects_an_index_out_of_range() -> Result[Void]
  table = try Support.f32([1.0, 2.0], [2, 1])
  ids = try Support.i64([5], [1])
  Test.error?(Tensor.gather(table, ids))
end

def test_gather_rejects_non_integer_indices() -> Result[Void]
  table = try Support.f32([1.0, 2.0], [2, 1])
  ids = try Support.f32([0.0], [1])
  Test.error?(Tensor.gather(table, ids))
end
