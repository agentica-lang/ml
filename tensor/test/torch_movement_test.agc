## `to_dtype` through the dispatch: the truncation rule, the `Bool` rule and
## the width rule that libtorch would otherwise wrap silently.

import Conform from "torch_support"
import Tensor from "../tensor"

private

def test_float_to_float_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [0.5, -1.25, 1000000.0]
  ours = try Tensor.to_dtype(try Conform.on_torch(data, [3]), Tensor.Dtype.F64)
  try Test.eq(Tensor.dtype(ours), Tensor.Dtype.F64)
  Conform.agrees(ours, try Tensor.to_dtype(try Conform.on_ref(data, [3]), Tensor.Dtype.F64), "F32 to F64")
end

def test_float_to_int_truncates() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.9, -1.9, 3.0]
  ours = try Tensor.to_dtype(try Conform.on_torch(data, [3]), Tensor.Dtype.I64)
  try Test.eq(Tensor.dtype(ours), Tensor.Dtype.I64)
  try Test.eq(try Conform.ints_of(ours), [1, -1, 3])
  Conform.agrees(ours, try Tensor.to_dtype(try Conform.on_ref(data, [3]), Tensor.Dtype.I64), "F32 to I64")
end

def test_float_to_bool_is_nonzero_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [0.0, 2.5, -0.25]
  ours = try Tensor.to_dtype(try Conform.on_torch(data, [3]), Tensor.Dtype.Bool)
  try Test.eq(Tensor.dtype(ours), Tensor.Dtype.Bool)
  Conform.agrees(ours, try Tensor.to_dtype(try Conform.on_ref(data, [3]), Tensor.Dtype.Bool), "F32 to Bool")
end

def test_int_to_float_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1, -2, 3]
  Conform.agrees(
    try Tensor.to_dtype(try Conform.on_torch_i64(data, [3]), Tensor.Dtype.F32),
    try Tensor.to_dtype(try Conform.on_ref_i64(data, [3]), Tensor.Dtype.F32),
    "I64 to F32",
  )
end

def test_int_to_a_narrower_int_agrees_inside_the_width() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1, -2, 127]
  ours = try Tensor.to_dtype(try Conform.on_torch_i64(data, [3]), Tensor.Dtype.I8)
  try Test.eq(Tensor.dtype(ours), Tensor.Dtype.I8)
  Conform.agrees(ours, try Tensor.to_dtype(try Conform.on_ref_i64(data, [3]), Tensor.Dtype.I8), "I64 to I8")
end

def test_int_to_a_narrower_int_refuses_to_wrap_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.to_dtype(try Conform.on_torch_i64([200], [1]), Tensor.Dtype.I8))
  Test.error?(Tensor.to_dtype(try Conform.on_ref_i64([200], [1]), Tensor.Dtype.I8))
end

def test_a_negative_value_does_not_fit_u8_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.to_dtype(try Conform.on_torch_i64([-1], [1]), Tensor.Dtype.U8))
  Test.error?(Tensor.to_dtype(try Conform.on_ref_i64([-1], [1]), Tensor.Dtype.U8))
end

def test_bool_to_int_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [0.0, 3.0]
  ours = try Tensor.to_dtype(try Tensor.to_dtype(try Conform.on_torch(data, [2]), Tensor.Dtype.Bool), Tensor.Dtype.I64)
  theirs = try Tensor.to_dtype(try Tensor.to_dtype(try Conform.on_ref(data, [2]), Tensor.Dtype.Bool), Tensor.Dtype.I64)
  try Test.eq(Tensor.dtype(ours), Tensor.Dtype.I64)
  try Test.eq(try Conform.ints_of(ours), [0, 1])
  Conform.agrees(ours, theirs, "Bool to I64")
end

def test_bool_to_float_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  flags = try Tensor.to_dtype(try Conform.on_torch([0.0, 3.0], [2]), Tensor.Dtype.Bool)
  ours = try Tensor.to_dtype(flags, Tensor.Dtype.F32)
  Conform.expect_floats(ours, [0.0, 1.0], "Bool to F32")
end

def test_copy_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, -2.5, 3.25]
  ours = try Tensor.copy(try Conform.on_torch(data, [3]))
  try Conform.agrees(ours, try Tensor.copy(try Conform.on_ref(data, [3])), "copy")
  # A copy shares no buffer: the original is still readable through its own
  # handle, and reading both back gives the same values.
  Conform.expect_floats(ours, data, "copy")
end

def test_contiguous_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  # A transposed tensor is a view in libtorch; `contiguous` must hand back the
  # same row-major values the reference produces.
  ours = try Tensor.contiguous(try Tensor.transpose(try Conform.on_torch(data, [2, 3])))
  theirs = try Tensor.contiguous(try Tensor.transpose(try Conform.on_ref(data, [2, 3])))
  Conform.agrees(ours, theirs, "contiguous of a transpose")
end
