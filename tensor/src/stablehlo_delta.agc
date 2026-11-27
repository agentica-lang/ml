# StableHLO dialect builders for the gated-delta kernel. The recurrence uses SSA values, not textual names.
import IR from "pkg/agc-ir/ir"

I32 = IR.TensorType(shape: [], element: IR.Element.I32)
BOOL = IR.TensorType(shape: [], element: IR.Element.I1)
F32 = IR.TensorType(shape: [], element: IR.Element.F32)

## A private MLIR function implementing Tensor.gated_delta_composed, without unrolling over tokens.
def emit(name: String, tokens: Int, heads: Int, width: Int) -> Result[String]
  seq = f32([tokens, heads, width])
  gates = f32([tokens, heads])
  state = f32([heads, width, width])
  entry = IR.block([seq, seq, seq, gates, gates, state])
  init = try integer(entry, 0)
  empty = try floating(init.block, 0.0, seq)
  carried = [init.value, entry.args[5], empty.value]
  types = carried.map { |v| v.type }
  cond = IR.block(types, some(empty.block))
  limit = try integer(cond, tokens)
  more = try less(limit.block, cond.args[0], limit.value)
  condition = try yield_values(more.block, [more.value], [BOOL])
  body = try step(IR.block(types, some(empty.block)), entry.args, heads, width)
  loop = try while_loop(empty.block, carried, condition, body)
  answers = try reshape(loop.block, loop.values[2], [tokens, heads * width])
  held = try reshape(answers.block, loop.values[1], [width, heads * width])
  packed = try concat_rows(held.block, answers.value, held.value)
  IR.function(name, packed.block, [packed.value])
end

## Adds a StableHLO while with region arguments and body results matching the carried values.
def while_loop(b: IR.Block, carried: List[IR.Value], cond: IR.Block, body: IR.Block) -> Result[IR.Added]
  types = carried.map { |v| v.type }
  for region in [cond, body]
    return error("while region arguments must match the carried types") if region.args.map { |v| v.type } != types
  end

  try check_yield(cond, [BOOL])
  try check_yield(body, types)
  IR.append(b, "stablehlo.while", carried, types, regions: [cond, body])
end

private

struct Added
  block: IR.Block
  value: IR.Value
end

enum Binary
  Add
  Subtract
  Multiply
end

def check_yield(region: IR.Block, types: List[IR.TensorType]) -> Result[Void]
  return error("while region must end in stablehlo.return") if region.ops.empty?
  tail = region.ops[region.ops.size - 1]
  return error("while region must end in stablehlo.return") if tail.name != "stablehlo.return"
  return error("while region return types do not match the signature") if tail.operands.map { |v| v.type } != types
  ok()
end

def step(b: IR.Block, inputs: List[IR.Value], heads: Int, width: Int) -> Result[IR.Block]
  i = b.args[0]
  held = b.args[1]
  out = b.args[2]
  zero = try integer(b, 0)
  one = try integer(zero.block, 1)
  q = try row(one.block, inputs[0], i, zero.value)
  k = try row(q.block, inputs[1], i, zero.value)
  v = try row(k.block, inputs[2], i, zero.value)
  g = try row(v.block, inputs[3], i, zero.value)
  beta = try row(g.block, inputs[4], i, zero.value)
  decay = try single(beta.block, "stablehlo.exponential", [g.value], g.value.type)
  factor = try broadcast(decay.block, decay.value, held.type, [0])
  decayed = try binary(factor.block, Binary.Multiply, held, factor.value)
  key = try broadcast(decayed.block, k.value, held.type, [0, 2])
  read_key = try binary(key.block, Binary.Multiply, decayed.value, key.value)
  seen = try sum_keys(read_key.block, read_key.value)
  rate = try broadcast(seen.block, beta.value, v.value.type, [0])
  diff = try binary(rate.block, Binary.Subtract, v.value, seen.value)
  change = try binary(diff.block, Binary.Multiply, diff.value, rate.value)
  value = try broadcast(change.block, change.value, held.type, [0, 1])
  update = try binary(value.block, Binary.Multiply, value.value, key.value)
  next_state = try binary(update.block, Binary.Add, decayed.value, update.value)
  query = try broadcast(next_state.block, q.value, held.type, [0, 2])
  read_query = try binary(query.block, Binary.Multiply, next_state.value, query.value)
  raw = try sum_keys(read_query.block, read_query.value)
  scale = try floating(raw.block, 1.0 / Math.sqrt(width.to_f), f32([heads, width]))
  scaled = try binary(scale.block, Binary.Multiply, raw.value, scale.value)
  answer = try reshape(scaled.block, scaled.value, [1, heads, width])
  next_out = try update_row(answer.block, out, answer.value, [i, zero.value, zero.value])
  next_i = try binary(next_out.block, Binary.Add, i, one.value)
  yield_values(next_i.block, [next_i.value, next_state.value, next_out.value], b.args.map { |a| a.type })
end

def f32(shape: List[Int]) -> IR.TensorType = IR.TensorType(shape: shape, element: IR.Element.F32)

def attr(key: String, value: IR.Attr) -> IR.Attribute = IR.Attribute(key: key, value: value)

def single(
  b: IR.Block,
  name: String,
  args: List[IR.Value],
  result: IR.TensorType,
  attrs: List[IR.Attribute] = [],
  regions: List[IR.Block] = [],
) -> Result[Added]
  added = try IR.append(b, name, args, [result], attrs, regions)
  ok(Added(block: added.block, value: added.values[0]))
end

def integer(b: IR.Block, x: Int) -> Result[Added]
  single(b, "stablehlo.constant", [], I32, [attr("value", IR.Attr.DenseInt(x, I32))])
end

def floating(b: IR.Block, x: Float, ty: IR.TensorType) -> Result[Added]
  return error("expected an f32 constant type") if ty.element != IR.Element.F32
  single(b, "stablehlo.constant", [], ty, [attr("value", IR.Attr.DenseF32(x, ty))])
end

def less(b: IR.Block, a: IR.Value, c: IR.Value) -> Result[Added]
  return error("expected scalar i32 loop bounds") if a.type != I32 or c.type != I32
  attrs = [
    attr("comparison_direction", IR.Attr.DialectEnum("stablehlo", "comparison_direction", "LT")),
    attr("compare_type", IR.Attr.DialectEnum("stablehlo", "comparison_type", "SIGNED")),
  ]

  single(b, "stablehlo.compare", [a, c], BOOL, attrs)
end

def binary(b: IR.Block, op: Binary, a: IR.Value, c: IR.Value) -> Result[Added]
  return error("expected equal arithmetic operand types") if a.type != c.type or a.type.element == IR.Element.I1
  name = case op
    Binary.Add -> "stablehlo.add"
    Binary.Subtract -> "stablehlo.subtract"
    Binary.Multiply -> "stablehlo.multiply"
  end

  single(b, name, [a, c], a.type)
end

def reshape(b: IR.Block, x: IR.Value, shape: List[Int]) -> Result[Added]
  return error("reshape changes the element count") if elements(x.type.shape) != elements(shape)
  single(b, "stablehlo.reshape", [x], x.type.with(shape: shape))
end

def elements(shape: List[Int]) -> Int
  n = 1
  for d in shape
    n = n * d
  end

  n
end

def row(b: IR.Block, x: IR.Value, i: IR.Value, zero: IR.Value) -> Result[Added]
  if x.type.shape.empty? or x.type.shape[0] < 1 or i.type != I32 or zero.type != I32
    return error("expected a nonempty sequence and i32 row indices")
  end

  tail = x.type.shape.drop(1)
  sizes = [1] + tail
  args = [x, i] + tail.map { |_| zero }
  slice = try single(
    b,
    "stablehlo.dynamic_slice",
    args,
    x.type.with(shape: sizes),
    [attr("slice_sizes", IR.Attr.I64Array(sizes))],
  )

  reshape(slice.block, slice.value, tail)
end

def broadcast(b: IR.Block, x: IR.Value, ty: IR.TensorType, dims: List[Int]) -> Result[Added]
  if dims.size != x.type.shape.size or x.type.element != ty.element
    return error("invalid broadcast rank or element type")
  end

  last = -1
  for i in 0...dims.size
    d = dims[i]
    return error("invalid broadcast dimensions") if d <= last or d >= ty.shape.size
    return error("incompatible broadcast extent") if x.type.shape[i] != 1 and x.type.shape[i] != ty.shape[d]
    last = d
  end

  single(b, "stablehlo.broadcast_in_dim", [x], ty, [attr("broadcast_dimensions", IR.Attr.I64Array(dims))])
end

def sum_keys(b: IR.Block, x: IR.Value) -> Result[Added]
  return error("expected a rank-three f32 state") if x.type.shape.size != 3 or x.type.element != IR.Element.F32
  zero = try floating(b, 0.0, F32)
  region = IR.block([F32, F32])
  summed = try binary(region, Binary.Add, region.args[0], region.args[1])
  body = try yield_values(summed.block, [summed.value], [F32])
  single(
    zero.block,
    "stablehlo.reduce",
    [x, zero.value],
    f32(x.type.shape.take(2)),
    [attr("dimensions", IR.Attr.I64Array([2]))],
    [body],
  )
end

def update_row(b: IR.Block, x: IR.Value, row: IR.Value, starts: List[IR.Value]) -> Result[Added]
  shape = x.type.shape
  if (
    shape.empty? or
    shape[0] < 1 or
    row.type != x.type.with(shape: [1] + shape.drop(1)) or
    starts.size != shape.size or
    starts.any? { |i| i.type != I32 }
  )
    return error("invalid row update signature")
  end

  single(b, "stablehlo.dynamic_update_slice", [x, row] + starts, x.type)
end

def concat_rows(b: IR.Block, a: IR.Value, c: IR.Value) -> Result[Added]
  if (
    a.type.shape.size != 2 or
    c.type.shape.size != 2 or
    a.type.element != c.type.element or
    a.type.shape[1] != c.type.shape[1]
  )
    return error("invalid row concatenation signature")
  end

  single(
    b,
    "stablehlo.concatenate",
    [a, c],
    a.type.with(shape: [a.type.shape[0] + c.type.shape[0], a.type.shape[1]]),
    [attr("dimension", IR.Attr.I64(0))],
  )
end

def yield_values(b: IR.Block, values: List[IR.Value], expected: List[IR.TensorType]) -> Result[IR.Block]
  return error("region return types do not match the signature") if values.map { |a| a.type } != expected
  ended = try IR.append(b, "stablehlo.return", values, [])
  ok(ended.block)
end
