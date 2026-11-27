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
