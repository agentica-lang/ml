import Conv from "../src/stablehlo_conv"
import IR from "pkg/agc-ir/src/ir"

private

def test_conv2d_infers_the_grouped_rectangular_output() -> Result[Void]
  input = IR.TensorType(shape: [2, 4, 5, 7], element: IR.Element.F32)
  weight = IR.TensorType(shape: [6, 2, 3, 2], element: IR.Element.F32)
  b = IR.block([input, weight])
  built = try Conv.conv2d(b, b.args[0], b.args[1], 2, 1, 2)
  Test.eq(built.values[0].type, IR.TensorType(shape: [2, 6, 3, 4], element: IR.Element.F32))
end

def test_conv2d_rejects_invalid_groups_and_windows() -> Result[Void]
  input = IR.TensorType(shape: [1, 4, 2, 2], element: IR.Element.F32)
  weight = IR.TensorType(shape: [6, 2, 3, 3], element: IR.Element.F32)
  b = IR.block([input, weight])
  try Test.error?(Conv.conv2d(b, b.args[0], b.args[1], 1, 1, 3))
  try Test.error?(Conv.conv2d(b, b.args[0], b.args[1], 1, 1, 1))
  try Test.error?(Conv.conv2d(b, b.args[0], b.args[1], 0, 1, 2))
  Test.error?(Conv.conv2d(b, b.args[0], b.args[1], 1, 0, 2))
end

def test_conv1d_infers_batched_and_unbatched_outputs() -> Result[Void]
  batched = IR.TensorType(shape: [2, 3, 12], element: IR.Element.F32)
  weight = IR.TensorType(shape: [5, 3, 3], element: IR.Element.F32)
  b = IR.block([batched, weight])
  made = try Conv.conv1d(b, b.args[0], b.args[1], 2, 1)
  try Test.eq(made.values[0].type, IR.TensorType(shape: [2, 5, 6], element: IR.Element.F32))
  try Test.eq(made.block.ops.size, 1)
  text = try IR.operation(made.block.ops[0], {b.args[0].id: "%x", b.args[1].id: "%w"}, "conv")
  try Test.assert(text.include?("#stablehlo.conv<[b, f, 0]x[o, i, 0]->[b, f, 0]>"), text)
  Test.assert(text.include?("padding = dense<1> : tensor<1x2xi64>"), text)
end

def test_conv1d_restores_the_rank_two_input_shape() -> Result[Void]
  input = IR.TensorType(shape: [3, 12], element: IR.Element.F32)
  weight = IR.TensorType(shape: [5, 3, 3], element: IR.Element.F32)
  b = IR.block([input, weight])
  made = try Conv.conv1d(b, b.args[0], b.args[1], 2, 1)
  try Test.eq(made.values[0].type, IR.TensorType(shape: [5, 6], element: IR.Element.F32))
  Test.eq(made.block.ops.size, 3)
end

def test_conv1d_rejects_invalid_shapes_and_parameters() -> Result[Void]
  input = IR.TensorType(shape: [3, 2], element: IR.Element.F32)
  weight = IR.TensorType(shape: [5, 4, 3], element: IR.Element.F32)
  b = IR.block([input, weight])
  try Test.error?(Conv.conv1d(b, b.args[0], b.args[1], 1, 1))
  try Test.error?(Conv.conv1d(b, b.args[0], b.args[1], 0, 1))
  Test.error?(Conv.conv1d(b, b.args[0], b.args[1], 1, 0))
end
