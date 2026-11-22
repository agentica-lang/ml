## `to_dtype` against the reference backend: the truncation rule, the `Bool`
## rule and the width rule that libtorch would otherwise wrap silently.

import Ref from "pkg/tensor/tensor"
import Support from "support"
import Torch from "../torch"

private

def test_float_to_float_agrees() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [0.5, -1.25, 1000000.0]
  ours = try Torch.to_dtype(Support.CPU, try Support.torch_f32(data, [3]), Torch.Dtype.F64)
  try Test.eq(Torch.dtype(ours), Torch.Dtype.F64)
  Support.agrees(ours, try Ref.to_dtype(try Support.ref_f32(data, [3]), Ref.Dtype.F64), "F32 to F64")
end

def test_float_to_int_truncates() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1.9, -1.9, 3.0]
  ours = try Torch.to_dtype(Support.CPU, try Support.torch_f32(data, [3]), Torch.Dtype.I64)
  try Test.eq(Torch.dtype(ours), Torch.Dtype.I64)
  try Test.eq(try Support.ints_of(ours), [1, -1, 3])
  Support.agrees(ours, try Ref.to_dtype(try Support.ref_f32(data, [3]), Ref.Dtype.I64), "F32 to I64")
end

def test_float_to_bool_is_nonzero() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [0.0, 2.5, -0.25]
  ours = try Torch.to_dtype(Support.CPU, try Support.torch_f32(data, [3]), Torch.Dtype.Bool)
  try Test.eq(Torch.dtype(ours), Torch.Dtype.Bool)
  Support.agrees(ours, try Ref.to_dtype(try Support.ref_f32(data, [3]), Ref.Dtype.Bool), "F32 to Bool")
end

def test_int_to_float_agrees() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1, -2, 3]
  ours = try Torch.to_dtype(Support.CPU, try Support.torch_i64(data, [3]), Torch.Dtype.F32)
  Support.agrees(ours, try Ref.to_dtype(try Support.ref_i64(data, [3]), Ref.Dtype.F32), "I64 to F32")
end

def test_int_to_a_narrower_int_agrees_inside_the_width() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1, -2, 127]
  ours = try Torch.to_dtype(Support.CPU, try Support.torch_i64(data, [3]), Torch.Dtype.I8)
  try Test.eq(Torch.dtype(ours), Torch.Dtype.I8)
  Support.agrees(ours, try Ref.to_dtype(try Support.ref_i64(data, [3]), Ref.Dtype.I8), "I64 to I8")
end

def test_int_to_a_narrower_int_refuses_to_wrap() -> Result[Void]
  return Support.absent() if !Support.available()

  wide = try Support.torch_i64([200], [1])
  try Test.error?(Torch.to_dtype(Support.CPU, wide, Torch.Dtype.I8))
  Test.error?(Ref.to_dtype(try Support.ref_i64([200], [1]), Ref.Dtype.I8))
end

def test_a_negative_value_does_not_fit_u8() -> Result[Void]
  return Support.absent() if !Support.available()

  Test.error?(Torch.to_dtype(Support.CPU, try Support.torch_i64([-1], [1]), Torch.Dtype.U8))
end

def test_bool_to_int_agrees() -> Result[Void]
  return Support.absent() if !Support.available()

  flags = try Torch.to_dtype(Support.CPU, try Support.torch_f32([0.0, 3.0], [2]), Torch.Dtype.Bool)
  ours = try Torch.to_dtype(Support.CPU, flags, Torch.Dtype.I64)
  try Test.eq(Torch.dtype(ours), Torch.Dtype.I64)
  try Test.eq(try Support.ints_of(ours), [0, 1])
  ref_flags = try Ref.to_dtype(try Support.ref_f32([0.0, 3.0], [2]), Ref.Dtype.Bool)
  Support.agrees(ours, try Ref.to_dtype(ref_flags, Ref.Dtype.I64), "Bool to I64")
end

def test_bool_to_float_agrees() -> Result[Void]
  return Support.absent() if !Support.available()

  floats = try Support.torch_f32([0.0, 3.0], [2])
  flags = try Torch.to_dtype(Support.CPU, floats, Torch.Dtype.Bool)
  ours = try Torch.to_dtype(Support.CPU, flags, Torch.Dtype.F32)
  Support.expect_floats(ours, [0.0, 1.0], "Bool to F32")
end
