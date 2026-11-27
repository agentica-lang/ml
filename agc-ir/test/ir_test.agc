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

def test_an_operation_can_use_existing_ssa_names() -> Result[Void]
  b = IR.block([I32, I32])
  added = try IR.append(b, "test.binary", b.args, [I32])
  op = added.block.ops[0]
  names: Map[Ref, String] = {}
  try Test.error?(IR.operation(op, names, "unused"))
  names = names.put(b.args[0].id, "%4").put(b.args[1].id, "%arg2")
  text = try IR.operation(op, names, "unused")
  Test.eq(text, "\"test.binary\"(%4, %arg2) : (tensor<i32>, tensor<i32>) -> (tensor<i32>)")
end

def test_a_bool_attribute_prints_true_or_false() -> Result[Void]
  b = IR.block([I32])
  added = try IR.append(
    b,
    "test.flags",
    b.args,
    [I32],
    [IR.Attribute(key: "sorted", value: IR.Attr.Bool(false)), IR.Attribute(key: "unique", value: IR.Attr.Bool(true))],
  )

  names: Map[Ref, String] = {}
  names = names.put(b.args[0].id, "%arg0")
  text = try IR.operation(added.block.ops[0], names, "unused")
  Test.eq(text, "\"test.flags\"(%arg0) {sorted = false, unique = true} : (tensor<i32>) -> (tensor<i32>)")
end

def test_the_narrow_integer_elements_print_their_mlir_names() -> Result[Void]
  shown = [IR.Element.I8, IR.Element.I16, IR.Element.U8, IR.Element.BF16].map { |e|
    IR.show_type(IR.TensorType(shape: [2], element: e))
  }

  Test.eq(shown, ["tensor<2xi8>", "tensor<2xi16>", "tensor<2xui8>", "tensor<2xbf16>"])
end

def test_a_dense_literal_prints_its_payload_as_given() -> Result[Void]
  scalar = IR.TensorType(shape: [], element: IR.Element.F16)
  table = IR.TensorType(shape: [2], element: IR.Element.F32)
  b = IR.block([])
  added = try IR.append(
    b,
    "test.constants",
    [],
    [scalar, table],
    [
      IR.Attribute(key: "low", value: IR.Attr.DenseLiteral("0xFC00", scalar)),
      IR.Attribute(key: "turns", value: IR.Attr.DenseLiteral("[1.0, 1.0e-2]", table)),
    ],
  )

  text = try IR.operation(added.block.ops[0], {}, "unused")
  Test.eq(
    text,
    "\"test.constants\"() {low = dense<0xFC00> : tensor<f16>, turns = dense<[1.0, 1.0e-2]> : tensor<2xf32>} : () -> (tensor<f16>, tensor<2xf32>)",
  )
end

def test_a_function_prints_region_arguments_captures_and_attributes() -> Result[Void]
  outer = IR.block([I32])
  region = IR.block([I32], some(outer))
  tail = try IR.append(region, "test.yield", region.args + outer.args, [])
  attrs = [
    IR.Attribute(key: "count", value: IR.Attr.I64(2)),
    IR.Attribute(key: "axes", value: IR.Attr.I64Array([0, 1])),
    IR.Attribute(key: "none", value: IR.Attr.I64Array([])),
    IR.Attribute(key: "pad", value: IR.Attr.DenseInt(1, IR.TensorType(shape: [2, 2], element: IR.Element.I64))),
    IR.Attribute(
      key: "padding",
      value: IR.Attr.DenseIntRows([[0, 0], [1, 1]], IR.TensorType(shape: [2, 2], element: IR.Element.I64)),
    ),
    IR.Attribute(key: "scale", value: IR.Attr.DenseF32(0.5, IR.TensorType(shape: [], element: IR.Element.F32))),
    IR.Attribute(key: "mode", value: IR.Attr.DialectEnum("test", "mode", "SIGNED")),
    IR.Attribute(key: "layout", value: IR.Attr.DialectAttr("test", "layout", "[0, 1]")),
  ]

  nested = try IR.append(outer, "test.region", outer.args, [I32], attrs, regions: [tail.block])
  text = try IR.function("with_region", nested.block, nested.values)
  try Test.assert(text.include?("%v0_0 = \"test.region\"(%arg0)"), text)
  try Test.assert(text.include?("^bb0(%v0r0_a0: tensor<i32>):"), text)
  try Test.assert(text.include?("\"test.yield\"(%v0r0_a0, %arg0)"), text)
  try Test.assert(
    text.include?("count = 2 : i64, axes = array<i64: 0, 1>, none = array<i64>, pad = dense<1> : tensor<2x2xi64>"),
    text,
  )

  try Test.assert(text.include?("padding = dense<[[0, 0], [1, 1]]> : tensor<2x2xi64>"), text)
  try Test.assert(
    text.include?("scale = dense<0.5> : tensor<f32>, mode = #test<mode SIGNED>, layout = #test.layout<[0, 1]>"),
    text,
  )

  Test.assert(text.include?("\"func.return\"(%v0_0)"), text)
end
