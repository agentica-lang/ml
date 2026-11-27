import IR from "pkg/agc-ir/src/ir"
import Pool from "../src/stablehlo_pool"

private

def test_avg_pool1d_builds_reduce_window_with_rank_three_padding() -> Result[Void]
  input = IR.TensorType(shape: [2, 3, 8], element: IR.Element.F32)
  b = IR.block([input])
  built = try Pool.avg_pool1d(b, b.args[0], 3, 2, 1)
  try Test.eq(built.values[0].type, IR.TensorType(shape: [2, 3, 4], element: IR.Element.F32))
  text = try IR.function("pool_1d", built.block, built.values)
  try Test.assert(text.include?("window_dimensions = array<i64: 1, 1, 3>"), text)
  try Test.assert(text.include?("window_strides = array<i64: 1, 1, 2>"), text)
  try Test.assert(text.include?("padding = dense<[[0, 0], [0, 0], [1, 1]]> : tensor<3x2xi64>"), text)
  Test.assert(text.include?("\"stablehlo.add\""), text)
end

def test_avg_pool1d_builds_unbatched_rank_two_window() -> Result[Void]
  input = IR.TensorType(shape: [2, 8], element: IR.Element.F32)
  b = IR.block([input])
  built = try Pool.avg_pool1d(b, b.args[0], 3, 2, 1)
  try Test.eq(built.values[0].type, IR.TensorType(shape: [2, 4], element: IR.Element.F32))
  text = try IR.function("pool_1d_unbatched", built.block, built.values)
  Test.assert(text.include?("padding = dense<[[0, 0], [1, 1]]> : tensor<2x2xi64>"), text)
end

def test_avg_pool2d_builds_nchw_windows_and_infers_output() -> Result[Void]
  input = IR.TensorType(shape: [2, 3, 5, 6], element: IR.Element.F32)
  b = IR.block([input])
  built = try Pool.avg_pool2d(b, b.args[0], 3, 2, 1)
  try Test.eq(built.values[0].type, IR.TensorType(shape: [2, 3, 3, 3], element: IR.Element.F32))
  text = try IR.function("pool_2d", built.block, built.values)
  try Test.assert(text.include?("window_dimensions = array<i64: 1, 1, 3, 3>"), text)
  try Test.assert(text.include?("window_strides = array<i64: 1, 1, 2, 2>"), text)
  Test.assert(text.include?("padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>"), text)
end

def test_average_pools_reject_invalid_windows() -> Result[Void]
  vector = IR.TensorType(shape: [2, 3], element: IR.Element.F32)
  image = IR.TensorType(shape: [1, 2, 3, 3], element: IR.Element.F32)
  one = IR.block([vector])
  two = IR.block([image])
  try Test.error?(Pool.avg_pool1d(one, one.args[0], 3, 1, 2))
  try Test.error?(Pool.avg_pool2d(two, two.args[0], 3, 0, 1))
  Test.error?(Pool.avg_pool2d(two, two.args[0], 5, 1, 3))
end
