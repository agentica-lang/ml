## Movement between dtypes/devices, copies, and the two stubs.

import Support from "support"
import Tensor from "../tensor"

private

def test_to_dtype_truncates_float_to_int() -> Result[Void]
  t = try Support.f32([1.9, -2.5, 3.0], [3])
  i = try Tensor.to_dtype(t, Tensor.Dtype.I64)
  try Test.eq(Tensor.dtype(i), Tensor.Dtype.I64)
  Test.eq(try Tensor.to_list(i), [Tensor.Values.Int(1), Tensor.Values.Int(-2), Tensor.Values.Int(3)])
end

def test_to_dtype_keeps_int_to_float_exact() -> Result[Void]
  t = try Support.i64([1, -2, 3], [3])
  f = try Tensor.to_dtype(t, Tensor.Dtype.F32)
  Support.expect_floats(f, [1.0, -2.0, 3.0])
end

def test_to_dtype_tests_nonzero_for_bool() -> Result[Void]
  t = try Support.f32([0.0, 2.5, -1.0], [3])
  b = try Tensor.to_dtype(t, Tensor.Dtype.Bool)
  Test.eq(try Tensor.to_list(b), [Tensor.Values.Bool(false), Tensor.Values.Bool(true), Tensor.Values.Bool(true)])
end

def test_to_dtype_reads_bool_as_one_and_zero() -> Result[Void]
  t = try Support.bools([true, false], [2])
  i = try Tensor.to_dtype(t, Tensor.Dtype.I32)
  Test.eq(try Tensor.to_list(i), [Tensor.Values.Int(1), Tensor.Values.Int(0)])
end

def test_to_dtype_range_checks_narrower_widths() -> Result[Void]
  t = try Support.i64([127], [1])
  _ = try Tensor.to_dtype(t, Tensor.Dtype.I8)
  wide = try Support.i64([128], [1])
  try Test.error?(Tensor.to_dtype(wide, Tensor.Dtype.I8))
  neg = try Support.i64([-1], [1])
  Test.error?(Tensor.to_dtype(neg, Tensor.Dtype.U8))
end

def test_to_dtype_keeps_f16_and_f64_values() -> Result[Void]
  t = try Support.f32([1.25], [1])
  half = try Tensor.to_dtype(t, Tensor.Dtype.F16)
  try Test.eq(Tensor.dtype(half), Tensor.Dtype.F16)
  try Support.expect_floats(half, [1.25])
  double = try Tensor.to_dtype(t, Tensor.Dtype.F64)
  Test.eq(Tensor.dtype(double), Tensor.Dtype.F64)
end

def test_to_dtype_rejects_infinity_for_int() -> Result[Void]
  huge = try Tensor.exp(try Support.f32([1000.0], [1]))
  Test.error?(Tensor.to_dtype(huge, Tensor.Dtype.I32))
end

def test_to_device_keeps_cpu_tensors() -> Result[Void]
  t = try Support.f32([1.0], [1])
  u = try Tensor.to_device(t, Tensor.Device.Cpu)
  Support.expect_floats(u, [1.0])
end

def test_copy_and_contiguous_keep_the_values() -> Result[Void]
  t = try Support.f32([1.0, 2.0], [2])
  try Support.expect_floats(try Tensor.copy(t), [1.0, 2.0])
  Support.expect_floats(try Tensor.contiguous(t), [1.0, 2.0])
end
