## Creation, dtypes, readback and introspection.

import Support from "support"
import Tensor from "../tensor"

private

def test_zeros_fills_the_shape() -> Result[Void]
  t = try Tensor.zeros([2, 2], Tensor.Dtype.F32, Tensor.Backend.Reference)
  try Test.eq(Tensor.shape(t), [2, 2])
  Support.expect_floats(t, [0.0, 0.0, 0.0, 0.0])
end

def test_ones_fills_the_shape() -> Result[Void]
  t = try Tensor.ones([3], Tensor.Dtype.F32, Tensor.Backend.Reference)
  Support.expect_floats(t, [1.0, 1.0, 1.0])
end

def test_empty_is_deterministic_zeros() -> Result[Void]
  t = try Tensor.empty([2], Tensor.Dtype.F32, Tensor.Backend.Reference)
  Support.expect_floats(t, [0.0, 0.0])
end

def test_full_repeats_the_value() -> Result[Void]
  t = try Tensor.full([2, 2], Tensor.Values.Float(2.5), Tensor.Dtype.F32, Tensor.Backend.Reference)
  Support.expect_floats(t, [2.5, 2.5, 2.5, 2.5])
end

def test_like_keeps_shape_and_dtype() -> Result[Void]
  t = try Support.i64([1, 2, 3], [3])
  same = try Tensor.like(t)
  try Test.eq(Tensor.shape(same), [3])
  try Test.eq(Tensor.dtype(same), Tensor.Dtype.I64)
  Test.eq(try Tensor.to_list(same), [Tensor.Values.Int(0), Tensor.Values.Int(0), Tensor.Values.Int(0)])
end

def test_arange_counts_up() -> Result[Void]
  t = try Tensor.arange(0, 5, 2, Tensor.Dtype.I64, Tensor.Backend.Reference)
  Test.eq(try Tensor.to_list(t), [Tensor.Values.Int(0), Tensor.Values.Int(2), Tensor.Values.Int(4)])
end

def test_arange_counts_down() -> Result[Void]
  t = try Tensor.arange(5, 0, -2, Tensor.Dtype.I32, Tensor.Backend.Reference)
  Test.eq(try Tensor.to_list(t), [Tensor.Values.Int(5), Tensor.Values.Int(3), Tensor.Values.Int(1)])
end

def test_arange_rejects_a_zero_step() -> Result[Void]
  Test.error?(Tensor.arange(0, 3, 0, Tensor.Dtype.I64, Tensor.Backend.Reference))
end

def test_arange_rejects_float_dtypes() -> Result[Void]
  Test.error?(Tensor.arange(0, 3, 1, Tensor.Dtype.F32, Tensor.Backend.Reference))
end

def test_scalar_to_tensor_builds_a_rank_zero_tensor() -> Result[Void]
  t = try Tensor.scalar_to_tensor(Tensor.Values.Float(1.5), Tensor.Dtype.F32, Tensor.Backend.Reference)
  try Test.eq(Tensor.shape(t), [])
  Support.expect_floats(t, [1.5])
end

def test_from_list_builds_a_row_major_tensor() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  try Test.eq(Tensor.shape(t), [2, 3])
  Support.expect_floats(t, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
end

def test_from_list_rejects_a_count_that_does_not_fit_the_shape() -> Result[Void]
  vals = [Tensor.Values.Float(1.0), Tensor.Values.Float(2.0)]
  Test.error?(Tensor.from_list(vals, [2, 2], Tensor.Dtype.F32, Tensor.Backend.Reference))
end

def test_from_list_rejects_a_negative_dimension() -> Result[Void]
  vals = [Tensor.Values.Float(1.0)]
  Test.error?(Tensor.from_list(vals, [1, -1], Tensor.Dtype.F32, Tensor.Backend.Reference))
end

def test_from_list_rejects_a_mistagged_value() -> Result[Void]
  vals = [Tensor.Values.Int(1)]
  Test.error?(Tensor.from_list(vals, [1], Tensor.Dtype.F32, Tensor.Backend.Reference))
end

def test_from_list_range_checks_int_widths() -> Result[Void]
  good = [Tensor.Values.Int(127)]
  try Test.eq((try Tensor.from_list(good, [1], Tensor.Dtype.I8, Tensor.Backend.Reference)).shape, [1])
  wide = [Tensor.Values.Int(128)]
  try Test.error?(Tensor.from_list(wide, [1], Tensor.Dtype.I8, Tensor.Backend.Reference))
  neg = [Tensor.Values.Int(-1)]
  Test.error?(Tensor.from_list(neg, [1], Tensor.Dtype.U8, Tensor.Backend.Reference))
end

def test_to_list_round_trips_each_class() -> Result[Void]
  f = try Support.f32([1.5], [1])
  try Test.eq(try Tensor.to_list(f), [Tensor.Values.Float(1.5)])
  b = try Support.bools([true, false], [2])
  try Test.eq(try Tensor.to_list(b), [Tensor.Values.Bool(true), Tensor.Values.Bool(false)])
  i = try Support.i64([-3], [1])
  Test.eq(try Tensor.to_list(i), [Tensor.Values.Int(-3)])
end

def test_introspection_reads_fields_without_a_call() -> Result[Void]
  t = try Support.f32([1.0], [1])
  try Test.eq(Tensor.shape(t), [1])
  try Test.eq(Tensor.dtype(t), Tensor.Dtype.F32)
  Test.eq(Tensor.backend(t), Tensor.Backend.Reference)
end

def test_zeros_on_torch_answers_the_module_state() -> Result[Void]
  # The libtorch backend is the one arm that runs or not depending on this
  # machine (`[native.torch] optional`), so the same call is a tensor where the
  # module is built and an error naming the missing backend where it is not.
  case Tensor.zeros([1], Tensor.Dtype.F32, Tensor.Backend.Torch(Tensor.Device.Cpu))
    ok(t) -> Test.eq(Tensor.shape(t), [1])
    error(e) -> Test.assert(e.message.include?("backend not available"), "the Torch arm said \"#{e.message}\"")
  end
end

def test_ones_on_xla_is_not_available() -> Result[Void]
  Test.error?(Tensor.ones([1], Tensor.Dtype.F32, Tensor.Backend.Xla(Tensor.Device.Cpu)))
end

def test_from_list_on_candle_is_not_available() -> Result[Void]
  vals = [Tensor.Values.Float(1.0)]
  Test.error?(Tensor.from_list(vals, [1], Tensor.Dtype.F32, Tensor.Backend.Candle(Tensor.Device.Cpu)))
end

def test_cuda_construction_is_an_error() -> Result[Void]
  Test.error?(Tensor.zeros([1], Tensor.Dtype.F32, Tensor.Backend.Torch(Tensor.Device.Cuda(0))))
end
