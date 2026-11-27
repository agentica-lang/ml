import Delta from "../src/stablehlo_delta"
import IR from "pkg/agc-ir/ir"

private

I32 = IR.TensorType(shape: [], element: IR.Element.I32)

def test_while_requires_a_boolean_condition() -> Result[Void]
  outer = IR.block([I32])
  cond = IR.block([I32])
  body = IR.block([I32])
  bad_cond = try IR.append(cond, "stablehlo.return", cond.args, [])
  yielded = try IR.append(body, "stablehlo.return", body.args, [])
  result = Delta.while_loop(outer, outer.args, bad_cond.block, yielded.block)
  try Test.error?(result)
  Test.assert(result.message.include?("return types"), result.message)
end
