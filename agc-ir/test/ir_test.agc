import IR from "../src/ir"

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
