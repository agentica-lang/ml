## Creation and readback against the reference backend, plus the error paths
## an unchecked native backend would let through.

import Support from "support"
import Torch from "../torch"

private

def test_zeros_agrees_with_the_reference() -> Result[Void]
  return Support.absent() if !Support.available()

  ours = try Torch.zeros(Support.CPU, [2, 3], Torch.Dtype.F32)
  theirs = try Support.ref_f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2, 3])
  Support.agrees(ours, theirs, "zeros")
end

def test_ones_agrees_with_the_reference() -> Result[Void]
  return Support.absent() if !Support.available()

  ours = try Torch.ones(Support.CPU, [3], Torch.Dtype.F32)
  Support.agrees(ours, try Support.ref_f32([1.0, 1.0, 1.0], [3]), "ones")
end

def test_empty_is_deterministic_zeros() -> Result[Void]
  return Support.absent() if !Support.available()

  ours = try Torch.empty(Support.CPU, [2], Torch.Dtype.F32)
  Support.agrees(ours, try Support.ref_f32([0.0, 0.0], [2]), "empty")
end

def test_full_repeats_the_value() -> Result[Void]
  return Support.absent() if !Support.available()

  ours = try Torch.full(Support.CPU, [2, 2], Torch.Values.Float(-1.5), Torch.Dtype.F32)
  Support.agrees(ours, try Support.ref_f32([-1.5, -1.5, -1.5, -1.5], [2, 2]), "full")
end

def test_like_keeps_shape_and_dtype() -> Result[Void]
  return Support.absent() if !Support.available()

  seed = try Support.torch_i64([1, 2, 3], [3])
  ours = try Torch.like(Support.CPU, seed)
  try Test.eq(Torch.shape(ours), [3])
  try Test.eq(Torch.dtype(ours), Torch.Dtype.I64)
  Support.agrees(ours, try Support.ref_i64([0, 0, 0], [3]), "like")
end

def test_arange_counts_up_and_down() -> Result[Void]
  return Support.absent() if !Support.available()

  up = try Torch.arange(Support.CPU, 0, 5, 2, Torch.Dtype.I64)
  try Support.agrees(up, try Support.ref_i64([0, 2, 4], [3]), "arange up")
  down = try Torch.arange(Support.CPU, 5, 0, -2, Torch.Dtype.I32)
  try Test.eq(Torch.dtype(down), Torch.Dtype.I32)
  Test.eq(try Support.ints_of(down), [5, 3, 1])
end

def test_scalar_to_tensor_is_rank_zero() -> Result[Void]
  return Support.absent() if !Support.available()

  ours = try Torch.scalar_to_tensor(Support.CPU, Torch.Values.Float(2.5), Torch.Dtype.F32)
  try Test.eq(Torch.shape(ours), [])
  Support.agrees(ours, try Support.ref_f32([2.5], []), "scalar_to_tensor")
end

def test_from_list_round_trips_through_the_readback() -> Result[Void]
  return Support.absent() if !Support.available()

  ours = try Support.torch_f32([1.5, -2.25, 3.0, 4.5, 5.0, 6.25], [2, 3])
  Support.agrees(ours, try Support.ref_f32([1.5, -2.25, 3.0, 4.5, 5.0, 6.25], [2, 3]), "round trip")
end

def test_int_readback_keeps_integers_exact() -> Result[Void]
  return Support.absent() if !Support.available()

  ours = try Support.torch_i64([-9007199254740992, 9007199254740992], [2])
  Test.eq(try Support.ints_of(ours), [-9007199254740992, 9007199254740992])
end

def test_negative_dimension_is_an_error() -> Result[Void]
  return Support.absent() if !Support.available()

  try Test.error?(Torch.zeros(Support.CPU, [2, -1], Torch.Dtype.F32))
  Test.error?(Torch.full(Support.CPU, [-2], Torch.Values.Float(1.0), Torch.Dtype.F32))
end

def test_from_list_count_mismatch_is_an_error() -> Result[Void]
  return Support.absent() if !Support.available()

  values = [Torch.Values.Float(1.0), Torch.Values.Float(2.0)]
  try Test.error?(Torch.from_list(Support.CPU, values, [3], Torch.Dtype.F32))
  Test.error?(Support.ref_f32([1.0, 2.0], [3]))
end

def test_a_value_of_the_wrong_class_is_an_error() -> Result[Void]
  return Support.absent() if !Support.available()

  try Test.error?(Torch.from_list(Support.CPU, [Torch.Values.Int(1)], [1], Torch.Dtype.F32))
  try Test.error?(Torch.full(Support.CPU, [1], Torch.Values.Float(1.0), Torch.Dtype.I64))
  Test.error?(Torch.full(Support.CPU, [1], Torch.Values.Bool(true), Torch.Dtype.U8))
end

def test_arange_rejects_a_bad_dtype_or_step() -> Result[Void]
  return Support.absent() if !Support.available()

  try Test.error?(Torch.arange(Support.CPU, 0, 3, 1, Torch.Dtype.F32))
  Test.error?(Torch.arange(Support.CPU, 0, 3, 0, Torch.Dtype.I64))
end

def test_cuda_is_not_served() -> Result[Void]
  return Support.absent() if !Support.available()

  Test.error?(Torch.zeros(Torch.Device.Cuda(0), [2], Torch.Dtype.F32))
end
