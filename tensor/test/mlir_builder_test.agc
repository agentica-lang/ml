import Delta from "../src/stablehlo_delta"
import IR from "../src/mlir_builder"

private

I32 = IR.TensorType(shape: [], element: IR.Element.I32)

def test_a_value_from_a_sibling_region_is_not_in_scope() -> Result[Void]
  a = IR.block([I32])
  b = IR.block([I32])
  Test.error?(IR.append(b, "stablehlo.add", [a.args[0], b.args[0]], [I32]))
end

def test_a_region_cannot_capture_a_value_from_another_function() -> Result[Void]
  outer = IR.block([I32])
  other = IR.block([I32])
  region = IR.block([], some(other))
  ended = try IR.append(region, "stablehlo.return", other.args, [])
  nested = try IR.append(outer, "test.region", [], [], regions: [ended.block])
  result = IR.function("wrong_capture", nested.block, outer.args)
  try Test.error?(result)
  Test.assert(result.message.include?("lexical scope"), result.message)
end

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
