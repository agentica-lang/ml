## Small, programmatic MLIR builder. Dialects supply operation-specific checks; the builder owns SSA scope,
## result identities, region arguments, and generic-form printing. No PJRT or tensor graph dependency.

struct TensorType
  shape: List[Int]
  element: Element
end

enum Element
  I1
  I32
  I64
  F16
  F32
  F64
end

struct Value
  id: Ref
  type: TensorType
end

enum Attr
  I64(Int)
  I64Array(List[Int])
  Bool(Bool)
  DenseInt(Int, TensorType)
  DenseIntRows(List[List[Int]], TensorType)
  DenseF32(Float, TensorType)
  DialectEnum(String, String, String)
  DialectAttr(String, String, String)
end

struct Attribute
  key: String
  value: Attr
end

struct Operation
  name: String
  operands: List[Value]
  results: List[Value]
  attributes: List[Attribute]
  regions: List[Block]
end

struct Block
  args: List[Value]
  ops: List[Operation]
  visible: Map[Ref, TensorType]
end

struct Added
  block: Block
  values: List[Value]
end

## A single-block region. Its arguments are in scope along with values from the enclosing block at this point.
def block(types: List[TensorType], parent: Optional[Block] = none) -> Block
  args = types.map { |t| fresh(t) }
  visible: Map[Ref, TensorType] = {}
  if !parent.none?
    visible = parent.unwrap!.visible
  end

  for arg in args
    visible = visible.put(arg.id, arg.type)
  end

  Block(args: args, ops: [], visible: visible)
end

## Results become visible only in the enclosing block after the operation. Regions are built from a snapshot
## of that block; the printer also checks the region's actual lexical scope at its point of insertion.
def append(
  b: Block,
  name: String,
  operands: List[Value],
  types: List[TensorType],
  attributes: List[Attribute] = [],
  regions: List[Block] = [],
) -> Result[Added]
  for v in operands
    if !b.visible.has_key?(v.id) or b.visible[v.id] != v.type
      return error("#{name}: operand is not in scope or has the wrong type")
    end
  end

  results = types.map { |t| fresh(t) }
  visible = b.visible
  for r in results
    visible = visible.put(r.id, r.type)
  end

  op = Operation(name: name, operands: operands, results: results, attributes: attributes, regions: regions)
  ok(Added(block: b.with(ops: b.ops.push(op), visible: visible), values: results))
end

def show_type(t: TensorType) -> String
  ty = case t.element
    Element.I1 -> "i1"
    Element.I32 -> "i32"
    Element.I64 -> "i64"
    Element.F16 -> "f16"
    Element.F32 -> "f32"
    Element.F64 -> "f64"
  end

  dims = t.shape.map { |d| "#{d}" }.push(ty)
  "tensor<#{dims.join("x")}>"
end

## Prints a private helper function; its signature is derived from its argument and return values.
def function(name: String, b: Block, returns: List[Value]) -> Result[String]
  ended = try append(b, "func.return", returns, [])
  names: Map[Ref, String] = {}
  for i in 0...b.args.size
    names = names.put(b.args[i].id, "%arg#{i}")
  end

  args = b.args.map { |a| "#{names[a.id]}: #{show_type(a.type)}" }.join(", ")
  text = try render(ended.block, names, "v", "    ")
  ok("  func.func private @#{name}(#{args}) -> (#{types_of(returns)}) {\n#{text}\n  }")
end

## Prints an operation without its SSA assignment, for embedding in an existing emitter.
## `names` binds visible operands; `base` is a fresh name stem for any nested region values.
def operation(op: Operation, names: Map[Ref, String], base: String, indent: String = "") -> Result[String]
  for v in op.operands
    return error("#{op.name}: operand escapes its lexical scope") if !names.has_key?(v.id)
  end

  operands = op.operands.map { |v| names[v.id] }.join(", ")
  regions: List[String] = []
  for j in 0...op.regions.size
    region = op.regions[j]
    stem = "#{base}r#{j}_"
    local = names
    for k in 0...region.args.size
      local = local.put(region.args[k].id, "%#{stem}a#{k}")
    end

    args = region.args.map { |a| "#{local[a.id]}: #{show_type(a.type)}" }.join(", ")
    text = try render(region, local, stem, indent + "  ")
    regions = regions.push("{\n#{indent}^bb0(#{args}):\n#{text}\n#{indent}}")
  end

  region_list = ""
  if !regions.empty?
    region_list = " (#{regions.join(", ")})"
  end

  attrs = ""
  if !op.attributes.empty?
    attrs = " {#{op.attributes.map { |a| attribute(a) }.join(", ")}}"
  end

  ok("\"#{op.name}\"(#{operands})#{region_list}#{attrs} : (#{types_of(op.operands)}) -> (#{types_of(op.results)})")
end

private

def fresh(t: TensorType) -> Value = Value(id: Ref.new(), type: t)

def boolean(x: Bool) -> String
  return "true" if x
  "false"
end

def types_of(values: List[Value]) -> String = values.map { |v| show_type(v.type) }.join(", ")

def attribute(a: Attribute) -> String
  value = case a.value
    Attr.I64(x) -> "#{x} : i64"
    Attr.I64Array(xs) -> "array<i64: #{xs.map { |x| "#{x}" }.join(", ")}>"
    Attr.Bool(x) -> boolean(x)
    Attr.DenseInt(x, t) -> "dense<#{x}> : #{show_type(t)}"
    Attr.DenseIntRows(rows, t) ->
      "dense<[#{rows.map { |row| "[#{row.map { |x| "#{x}" }.join(", ")}]" }.join(", ")}]> : #{show_type(t)}"
    Attr.DenseF32(x, t) -> "dense<#{x}> : #{show_type(t)}"
    Attr.DialectEnum(dialect, kind, x) -> "##{dialect}<#{kind} #{x}>"
    Attr.DialectAttr(dialect, kind, payload) -> "##{dialect}.#{kind}<#{payload}>"
  end

  "#{a.key} = #{value}"
end

def render(b: Block, inherited: Map[Ref, String], prefix: String, indent: String) -> Result[String]
  names = inherited
  lines: List[String] = []
  for i in 0...b.ops.size
    op = b.ops[i]
    base = "#{prefix}#{i}"
    text = try operation(op, names, base, indent)

    lhs: List[String] = []
    for j in 0...op.results.size
      name = "%#{base}_#{j}"
      lhs = lhs.push(name)
      names = names.put(op.results[j].id, name)
    end

    assignment = ""
    if !lhs.empty?
      assignment = "#{lhs.join(", ")} = "
    end

    lines = lines.push("#{indent}#{assignment}#{text}")
  end

  ok(lines.join("\n"))
end
