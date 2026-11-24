## The padded cache helpers on the reference backend: the capacity buckets, the
## mask, and `written` for the rows axis of a `[1, rows, d]` cache and of a
## `[1, heads, rows, width]` one, the latter against `Tensor.set_rows`.

import Padded from "../kv"
import Nn from "../nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"

private

def positions(at: List[Int]) -> Result[Tensor.Tensor]
  Tensor.from_list(at.map { |i| Tensor.Values.Int(i) }, [at.size], Tensor.Dtype.I32, S.REF)
end

def keep(i: Int) -> Float
  row = (i / 3) % 4
  return 0.0 if row == 0 or row == 2
  i.to_f
end

def test_the_capacity_is_a_power_of_two_up_to_the_limit() -> Result[Void]
  try Test.eq(Padded.capacity(1, 100), 8)
  try Test.eq(Padded.capacity(8, 100), 8)
  try Test.eq(Padded.capacity(9, 100), 16)
  try Test.eq(Padded.capacity(33, 100), 64)
  Test.eq(Padded.capacity(70, 100), 100)
end

## The rows an attention reads: every row of a small cache, steps of 256 rows up to a capacity of 2048 and of an
## eighth of the capacity above, never fewer than the rows in use.
def test_the_reach_is_the_rows_in_use_rounded_up_to_a_step() -> Result[Void]
  try Test.eq(Padded.reach(1, 8), 8)
  try Test.eq(Padded.reach(200, 256), 256)
  try Test.eq(Padded.reach(257, 512), 512)
  try Test.eq(Padded.reach(513, 1024), 768)
  try Test.eq(Padded.reach(768, 1024), 768)
  try Test.eq(Padded.reach(769, 1024), 1024)
  try Test.eq(Padded.reach(1025, 2048), 1280)
  try Test.eq(Padded.reach(2049, 4096), 2560)
  try Test.eq(Padded.reach(4096, 4096), 4096)
  try Test.eq(Padded.reach(4097, 8192), 5120)
  Test.eq(Padded.reach(70, 100), 100)
end

def test_the_mask_hides_the_later_rows() -> Result[Void]
  slots = Padded.Slots(positions: try positions([1, 3]), capacity: 4)
  got = try S.floats_of(try Padded.padded_mask(slots, Tensor.Dtype.F32))
  m = Nn.MASKED
  Test.eq(got, [0.0, 0.0, m, m, 0.0, 0.0, 0.0, 0.0])
end

def test_written_places_rows_and_grows_the_cache() -> Result[Void]
  slots = Padded.Slots(positions: try positions([3, 0]), capacity: 4)
  spread = try Padded.spread_of(slots, Tensor.Dtype.F32)
  fresh = try S.f32([1.0, 2.0, 3.0, 4.0], [1, 2, 2])
  first = try Padded.written(none, fresh, spread, 1)
  try Test.eq(try S.floats_of(first), [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0])

  later = Padded.Slots(positions: try positions([1]), capacity: 8)
  more = try Padded.written(
    some(first),
    try S.f32([9.0, 8.0], [1, 1, 2]),
    try Padded.spread_of(later, Tensor.Dtype.F32),
    1,
  )

  expected = [3.0, 4.0, 9.0, 8.0, 0.0, 0.0, 1.0, 2.0] + (0...8).map { |_| 0.0 }
  try Test.eq(Tensor.shape(more), [1, 8, 2])
  Test.eq(try S.floats_of(more), expected)
end

## A position of `capacity` (the padding of a prompt) names no row: the row is dropped.
def test_written_drops_a_row_at_the_capacity() -> Result[Void]
  slots = Padded.Slots(positions: try positions([4, 0]), capacity: 4)
  spread = try Padded.spread_of(slots, Tensor.Dtype.F32)
  got = try Padded.written(none, try S.f32([1.0, 2.0, 3.0, 4.0], [1, 2, 2]), spread, 1)
  try Test.eq(try S.floats_of(got), [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
  Test.error?(Padded.written(none, try S.f32([1.0, 2.0], [1, 1, 2]), spread, 3))
end

def test_written_on_a_four_axis_cache_is_set_rows() -> Result[Void]
  # `written` adds into rows that are zero: the cache is empty at the positions it fills.
  old = try S.f32((0...24).map { |i| keep(i) }, [1, 2, 4, 3])
  fresh = try S.f32((0...12).map { |i| 100.0 + i.to_f }, [1, 2, 2, 3])
  at = try positions([2, 0])
  slots = Padded.Slots(positions: at, capacity: 4)
  got = try Padded.written(some(old), fresh, try Padded.spread_of(slots, Tensor.Dtype.F32), 2)
  Test.eq(try S.floats_of(got), try S.floats_of(try Tensor.set_rows(old, fresh, at)))
end
