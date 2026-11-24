## Creation and readback through the dispatch: the same call on `Reference` and
## on `Torch`, compared, plus the error paths an unchecked native backend would
## let through.

import Conform from "torch_support"
import Tensor from "../tensor"

private

def test_zeros_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  theirs = try Tensor.zeros([2, 3], Tensor.Dtype.F32, Conform.REF)
  Conform.agrees(try Tensor.zeros([2, 3], Tensor.Dtype.F32, Conform.TORCH), theirs, "zeros")
end

def test_ones_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  theirs = try Tensor.ones([3], Tensor.Dtype.F32, Conform.REF)
  Conform.agrees(try Tensor.ones([3], Tensor.Dtype.F32, Conform.TORCH), theirs, "ones")
end

def test_floats_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.5, -2.0, 3.0, 0.25]
  theirs = try Tensor.floats(data, [2, 2], Conform.REF)
  try Test.error?(Tensor.floats(data, [3], Conform.TORCH))
  Conform.agrees(try Tensor.floats(data, [2, 2], Conform.TORCH), theirs, "floats")
end

def test_empty_is_deterministic_zeros() -> Result[Void]
  return Conform.absent() if !Conform.available()

  theirs = try Tensor.empty([2], Tensor.Dtype.F32, Conform.REF)
  Conform.agrees(try Tensor.empty([2], Tensor.Dtype.F32, Conform.TORCH), theirs, "empty")
end

def test_full_repeats_the_value() -> Result[Void]
  return Conform.absent() if !Conform.available()

  value = Tensor.Values.Float(-1.5)
  theirs = try Tensor.full([2, 2], value, Tensor.Dtype.F32, Conform.REF)
  Conform.agrees(try Tensor.full([2, 2], value, Tensor.Dtype.F32, Conform.TORCH), theirs, "full")
end

def test_zeros_and_full_are_not_capped_at_two_to_the_twentieth() -> Result[Void]
  return Conform.absent() if !Conform.available()

  zeros = try Tensor.zeros([1048577], Tensor.Dtype.F32, Conform.TORCH)
  try Test.eq(Tensor.shape(zeros), [1048577])
  try Conform.expect_floats(try Tensor.sum(zeros), [0.0], "sum of zeros")
  threes = try Tensor.full([2, 1048576], Tensor.Values.Float(3.0), Tensor.Dtype.F32, Conform.TORCH)
  try Test.eq(Tensor.shape(threes), [2, 1048576])
  Conform.expect_floats(try Tensor.sum(threes), [6291456.0], "sum of threes")
end

def test_like_keeps_shape_and_dtype() -> Result[Void]
  return Conform.absent() if !Conform.available()

  seed = try Conform.on_torch_i64([1, 2, 3], [3])
  ours = try Tensor.like(seed)
  try Test.eq(Tensor.shape(ours), [3])
  try Test.eq(Tensor.dtype(ours), Tensor.Dtype.I64)
  Conform.agrees(ours, try Tensor.like(try Conform.on_ref_i64([1, 2, 3], [3])), "like")
end

def test_arange_counts_up_and_down() -> Result[Void]
  return Conform.absent() if !Conform.available()

  up = try Tensor.arange(0, 5, 2, Tensor.Dtype.I64, Conform.TORCH)
  try Conform.agrees(up, try Tensor.arange(0, 5, 2, Tensor.Dtype.I64, Conform.REF), "arange up")
  down = try Tensor.arange(5, 0, -2, Tensor.Dtype.I32, Conform.TORCH)
  try Test.eq(Tensor.dtype(down), Tensor.Dtype.I32)
  Test.eq(try Conform.ints_of(down), [5, 3, 1])
end

def test_arange_past_the_int_width_diverges_and_is_pinned() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # `arange` builds its storage itself on both backends and neither checks the
  # int width there, so `[0, 100, 200]` reaches an `I8` tensor on both sides.
  # Reading it back then differs: libtorch's conversion to `char` wraps 200 to
  # -56, while the reference refuses a value its own dtype cannot hold. This is
  # the current behaviour of each side, pinned so a fix on either is visible.
  wrapped = try Tensor.arange(0, 300, 100, Tensor.Dtype.I8, Conform.TORCH)
  try Test.eq(try Conform.ints_of(wrapped), [0, 100, -56])
  refused = try Tensor.arange(0, 300, 100, Tensor.Dtype.I8, Conform.REF)
  Test.error?(Tensor.to_list(refused))
end

def test_scalar_to_tensor_is_rank_zero() -> Result[Void]
  return Conform.absent() if !Conform.available()

  ours = try Tensor.scalar_to_tensor(Tensor.Values.Float(2.5), Tensor.Dtype.F32, Conform.TORCH)
  try Test.eq(Tensor.shape(ours), [])
  Conform.agrees(ours, try Tensor.scalar_to_tensor(Tensor.Values.Float(2.5), Tensor.Dtype.F32, Conform.REF), "scalar")
end

def test_from_list_round_trips_through_the_readback() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.5, -2.25, 3.0, 4.5, 5.0, 6.25]
  ours = try Conform.on_torch(data, [2, 3])
  Conform.agrees(ours, try Conform.on_ref(data, [2, 3]), "round trip")
end

def test_int_readback_keeps_integers_exact() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-9007199254740992, 9007199254740992]
  ours = try Conform.on_torch_i64(data, [2])
  Test.eq(try Conform.ints_of(ours), data)
end

def test_negative_dimension_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.zeros([2, -1], Tensor.Dtype.F32, Conform.TORCH))
  try Test.error?(Tensor.zeros([2, -1], Tensor.Dtype.F32, Conform.REF))
  Test.error?(Tensor.full([-2], Tensor.Values.Float(1.0), Tensor.Dtype.F32, Conform.TORCH))
end

def test_from_list_count_mismatch_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  values = [Tensor.Values.Float(1.0), Tensor.Values.Float(2.0)]
  try Test.error?(Tensor.from_list(values, [3], Tensor.Dtype.F32, Conform.TORCH))
  Test.error?(Tensor.from_list(values, [3], Tensor.Dtype.F32, Conform.REF))
end

def test_a_value_of_the_wrong_class_is_an_error() -> Result[Void]
  return Conform.absent() if !Conform.available()

  ints = [Tensor.Values.Int(1)]
  try Test.error?(Tensor.from_list(ints, [1], Tensor.Dtype.F32, Conform.TORCH))
  try Test.error?(Tensor.from_list(ints, [1], Tensor.Dtype.F32, Conform.REF))
  try Test.error?(Tensor.full([1], Tensor.Values.Float(1.0), Tensor.Dtype.I64, Conform.TORCH))
  try Test.error?(Tensor.full([1], Tensor.Values.Float(1.0), Tensor.Dtype.I64, Conform.REF))
  try Test.error?(Tensor.full([1], Tensor.Values.Bool(true), Tensor.Dtype.U8, Conform.TORCH))
  Test.error?(Tensor.full([1], Tensor.Values.Bool(true), Tensor.Dtype.U8, Conform.REF))
end

def test_arange_rejects_a_bad_dtype_or_step_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.arange(0, 3, 1, Tensor.Dtype.F32, Conform.TORCH))
  try Test.error?(Tensor.arange(0, 3, 1, Tensor.Dtype.F32, Conform.REF))
  try Test.error?(Tensor.arange(0, 3, 0, Tensor.Dtype.I64, Conform.TORCH))
  Test.error?(Tensor.arange(0, 3, 0, Tensor.Dtype.I64, Conform.REF))
end
