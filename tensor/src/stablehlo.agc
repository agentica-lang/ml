## A traced graph as StableHLO text (`tensor/DESIGN.md` §12): one `func.func @main` that takes the inputs, then the
## lifted constants, and returns the outputs. It is the second lowering of a trace, beside `lower.agc`, and it knows no
## backend or runtime: the text is what XLA (PJRT) and IREE read.
##
## Shapes are static and every elementwise op takes operands of one shape, as the `Tensor.*` wrappers check, so the
## text needs no implicit broadcast: `broadcast_to` is a `broadcast_in_dim`, and a scalar of an op is a splat constant.
## An op the emitter does not know is an error naming it.

import IR from "pkg/agc-ir/src/ir"
import Conv from "stablehlo_conv"
import Graph from "graph"
import Lower from "lower"
import Reference from "reference"
import Pool from "stablehlo_pool"
import Reduce from "stablehlo_reduce"
import Delta from "stablehlo_delta"
import Types from "types"

## Starts the text of a node that has two results.
TWO = "<2>"

## The program: its text, and the tensors of its constants in the order of the arguments that follow the inputs.
struct Module
  text: String
  constants: List[Types.Tensor]
  ## The positions of the inputs the program updates in place (`emit` with `donate`), each aliased to its output.
  donated: List[Int] = []
  ## True when the program returns one more output after the traced ones (`emit` with `checked`): an `i1` that is
  ## true when an index of a `gather` or a `scatter_add` is outside its rows.
  checked: Bool = false
end

## The module of the traced tensors `outs` over the input nodes `inputs` (traced `Input` tensors, in argument order).
## With `donate`, the moments an `adam_step` updates in place (`Lower.adam_moments`) are aliased to their outputs
## (`tf.aliasing_output`), so XLA writes the new moment over the old one: a run consumes those inputs. With `checked`,
## a program with a `gather` or a `scatter_add` also answers whether one of its indices is outside the rows
## (`Module.checked`), which XLA would clamp or drop: the caller makes that the error the other backends give.
def emit(
  inputs: List[Types.Tensor],
  outs: List[Types.Tensor],
  donate: Bool = false,
  checked: Bool = false,
) -> Result[Module]
  return error("a StableHLO program needs at least one output") if outs.empty?
  positions: Map[Ref, Int] = {}
  names: Map[Ref, String] = {}
  for i in 0...inputs.size
    id = Graph.id_of(inputs[i])
    positions = positions.put(id, i)
    names = names.put(id, "%arg#{i}")
  end

  aliased: Map[Int, Int] = {}
  if donate
    moments = try Lower.adam_moments(outs, positions)
    for j in 0...moments.size
      if moments[j] >= 0
        aliased = aliased.put(moments[j], j)
      end
    end
  end

  params: List[String] = []
  for i in 0...inputs.size
    shown = try type_of(inputs[i])
    attr = ""
    if aliased.has_key?(i)
      attr = " {tf.aliasing_output = #{aliased[i]} : i32}"
    end

    params = params.push("%arg#{i}: #{shown}#{attr}")
  end

  constants: List[Types.Tensor] = []
  body: List[String] = []
  flags: List[String] = []
  helpers: List[String] = []
  next = 0
  for t in try Lower.walk(outs, positions)
    n = try Graph.node_of(t)
    case n.op
      Types.Op.Input -> nil
      Types.Op.Constant(c) ->
        arg = "%arg#{inputs.size + constants.size}"
        names = names.put(n.id, arg)
        shown = try type_of(t)
        params = params.push("#{arg}: #{shown}")
        constants = constants.push(c)
      _ ->
        case n.op
          Types.Op.GatedDeltaRule ->
            shape = n.args[0].shape
            helpers = helpers.push(try Delta.emit("gated_delta_#{next}", shape[0], shape[1], shape[2]))
          _ -> nil
        end

        defs = try define(t, n, n.args.map { |a| names[Graph.id_of(a)] }, next)
        for i in 0...defs.size
          body = body.push(numbered(next + i, defs[i]))
        end

        next = next + defs.size
        if defs.empty?
          names = names.put(n.id, names[Graph.id_of(n.args[0])])
        else
          names = names.put(n.id, "%#{next - 1}")
        end

        bound = indexed(n)
        if checked and !bound.none?
          at = bound.unwrap!
          lines = try outside(at.index, names[Graph.id_of(at.index)], at.rows, next)
          for i in 0...lines.size
            body = body.push("    %#{next + i} = #{lines[i]}")
          end

          next = next + lines.size
          if !lines.empty?
            flags = flags.push("%#{next - 1}")
          end
        end
    end
  end

  results = outs.map { |o| names[Graph.id_of(o)] }
  types = try types_of(outs)
  if !flags.empty?
    flag = flags[0]
    for f in flags.drop(1)
      bit = IR.TensorType(shape: [], element: IR.Element.I1)
      body = body.push(numbered(next, try ir_op("stablehlo.or", [flag, f], [bit, bit], bit, [], next)))
      flag = "%#{next}"
      next = next + 1
    end

    results = results.push(flag)
    types = types.push("tensor<i1>")
  end

  text = (
    ["module {", "  func.func public @main(#{params.join(", ")}) -> (#{types.join(", ")}) {"] +
    body +
    ["    func.return #{results.join(", ")} : #{types.join(", ")}", "  }"] +
    helpers +
    ["}", ""]
  ).join("\n")

  ok(Module(text: text, constants: constants, donated: aliased.keys.sort, checked: !flags.empty?))
end

private

# The indices of a node that reads or writes rows (`gather`, `scatter_add`) and how many rows there are.
struct Bound
  index: Types.Tensor
  rows: Int
end

# The lines of an op that takes several, the first named `%first`.
struct Lines
  first: Int
  defs: List[String]
end

# The lines so far and the name of the tensor that the last of them made (or the one that was given).
struct Side
  lines: Lines
  name: String
end

# One text per SSA value of the node `t`, the first named `%first`; none means the node is its first argument.
def define(t: Types.Tensor, n: Types.Node, a: List[String], first: Int) -> Result[List[String]]
  out = try ir_type(t.shape, t.dtype)
  ins = try ir_types(n.args)
  case n.op
    Types.Op.Full(v) -> ok([try splat(v, t.shape, t.dtype)])
    Types.Op.Copy -> ok([])
    Types.Op.Contiguous -> ok([])
    Types.Op.ToDevice(Types.Device.Cpu) -> ok([])
    Types.Op.ToDevice(device) ->
      error("XLA runs on the CPU only; cannot lower to_device to #{Types.device_name(device)}")
    Types.Op.Add -> elementwise("add", a, out, first)
    Types.Op.Sub -> elementwise("subtract", a, out, first)
    Types.Op.Mul -> elementwise("multiply", a, out, first)
    Types.Op.Div -> elementwise("divide", a, out, first)
    Types.Op.Neg -> elementwise("negate", a, out, first)
    Types.Op.Exp -> elementwise("exponential", a, out, first)
    Types.Op.Log -> elementwise("log", a, out, first)
    Types.Op.Sqrt -> elementwise("sqrt", a, out, first)
    Types.Op.Tanh -> elementwise("tanh", a, out, first)
    Types.Op.Sigmoid -> elementwise("logistic", a, out, first)
    Types.Op.GeluErf -> gelu_erf(t, a[0], first)
    Types.Op.LayerNorm(eps) -> normalized(t, a[0], eps, true, first)
    Types.Op.RmsNorm(eps) -> normalized(t, a[0], eps, false, first)
    Types.Op.Rope(base, dims, neox, inverse) -> rope(t, n.args, a, base, dims, neox, inverse, first)
    Types.Op.Pow -> elementwise("power", a, out, first)
    Types.Op.Clamp(lo, hi) ->
      ok([
        try fill(lo, t),
        try fill(hi, t),
        try ir_op("stablehlo.clamp", ["%#{first}", a[0], "%#{first + 1}"], [out, out, out], out, [], first + 2),
      ])
    Types.Op.ToDtype -> convert(t, n.args[0], a[0], first)
    Types.Op.Relu ->
      ok([try fill(0.0, t), try ir_op("stablehlo.maximum", [a[0], "%#{first}"], [out, out], out, [], first + 1)])
    Types.Op.Gelu -> gelu(t, a[0], first)
    Types.Op.StopGrad -> ok([])
    Types.Op.Matmul -> matmul(t, n.args, a, first)
    Types.Op.Transpose -> ok([try transpose_op(a[0], ins[0], out, (0...t.shape.size).to_a.reverse, first)])
    Types.Op.Reshape -> ok([try ir_op("stablehlo.reshape", a, ins, out, [], first)])
    Types.Op.Squeeze(_) -> ok([try ir_op("stablehlo.reshape", a, ins, out, [], first)])
    Types.Op.Unsqueeze(_) -> ok([try ir_op("stablehlo.reshape", a, ins, out, [], first)])
    Types.Op.Permute(axes) -> ok([try transpose_op(a[0], ins[0], out, axes, first)])
    Types.Op.Concat(axis) -> ok([try ir_op("stablehlo.concatenate", a, ins, out, concat_dim(axis), first)])
    Types.Op.Stack(axis) -> stack(t, n.args, a, axis, first)
    Types.Op.BatchMatmul -> batch_matmul(t, n.args, a, first)
    Types.Op.Conv1d(stride, padding) -> conv1d(t, n.args, a, stride, padding, first)
    Types.Op.Conv2d(stride, padding, groups) -> conv2d(t, n.args, a, stride, padding, groups, first)
    Types.Op.GatedDeltaRule ->
      ok(["func.call @gated_delta_#{first}(#{a.join(", ")}) : #{try signature(n.args, try type_of(t))}"])
    Types.Op.MaxPool2d(kernel, stride, padding) -> max_pool2d(t, n.args[0], a[0], kernel, stride, padding, first)
    Types.Op.AvgPool1d(kernel, stride, padding) -> avg_pool1d(t, n.args[0], a[0], kernel, stride, padding, first)
    Types.Op.AvgPool2d(kernel, stride, padding) -> avg_pool2d(t, n.args[0], a[0], kernel, stride, padding, first)
    Types.Op.ExpertMatmul -> expert_matmul(t, n.args, a, first)
    Types.Op.Argmax(axes, keep_dims) -> argmax(t, n.args[0], a[0], axes, keep_dims, first)
    Types.Op.Slice(starts, ends) -> ok([try slice_op(a[0], ins[0], starts, ends, first)])
    Types.Op.BroadcastTo -> broadcast(t, n.args[0], a[0], first)
    Types.Op.Sum(axes, keep_dims) -> reduction(t, n.args[0], a[0], axes, keep_dims, "add", first)
    Types.Op.Mean(axes, keep_dims) -> mean(t, n.args[0], a[0], axes, keep_dims, first)
    Types.Op.Max(axes, keep_dims) -> reduction(t, n.args[0], a[0], axes, keep_dims, "maximum", first)
    Types.Op.Lt -> ok([try ir_op("stablehlo.compare", a, ins, out, compare_attrs("LT", n.args[0].dtype), first)])
    Types.Op.Eq -> ok([try ir_op("stablehlo.compare", a, ins, out, compare_attrs("EQ", n.args[0].dtype), first)])
    Types.Op.Where -> ok([try ir_op("stablehlo.select", a, [ins[0], out, out], out, [], first)])
    Types.Op.Arange(start, _, step) -> arange(t, start, step, first)
    Types.Op.Gather -> gather(t, n.args, a, first)
    Types.Op.SetRows -> set_rows(t, n.args, a, first)
    Types.Op.TopK(k) -> top_k(t, n.args[0], a[0], k, first)
    Types.Op.ScatterAdd(_) -> scatter_add(t, n.args, a, first)
    Types.Op.Attention(scale) -> attention(t, n.args, a, scale, first)
    Types.Op.AdamStep -> adam(t, n.args, a, first)
    Types.Op.AdamPart(i) -> adam_part(t, n.args[0], a[0], i, first)
    _ -> error("StableHLO has no emitter for #{Types.op_name(n.op)} yet")
  end
end

def elementwise(op: String, a: List[String], out: IR.TensorType, first: Int) -> Result[List[String]]
  ok([try ir_op("stablehlo.#{op}", a, a.map { |_| out }, out, [], first)])
end

def transpose_op(name: String, x: IR.TensorType, out: IR.TensorType, axes: List[Int], first: Int) -> Result[String]
  ir_op(
    "stablehlo.transpose",
    [name],
    [x],
    out,
    [IR.Attribute(key: "permutation", value: IR.Attr.I64Array(axes))],
    first,
  )
end

def ir_types(ts: List[Types.Tensor]) -> Result[List[IR.TensorType]]
  out: List[IR.TensorType] = []
  for t in ts
    out = out.push(try ir_type(t.shape, t.dtype))
  end

  ok(out)
end

def avg_pool1d(
  t: Types.Tensor,
  input: Types.Tensor,
  a: String,
  kernel: Int,
  stride: Int,
  padding: Int,
  first: Int,
) -> Result[List[String]]
  entry = IR.block([try float_ir_type(input)])
  built = try Pool.avg_pool1d(entry, entry.args[0], kernel, stride, padding)
  return error("avg_pool1d IR result type does not match the graph") if built.values[0].type != try float_ir_type(t)
  pool_lines(built.block, built.block.ops, a, first)
end

def avg_pool2d(
  t: Types.Tensor,
  input: Types.Tensor,
  a: String,
  kernel: Int,
  stride: Int,
  padding: Int,
  first: Int,
) -> Result[List[String]]
  entry = IR.block([try float_ir_type(input)])
  built = try Pool.avg_pool2d(entry, entry.args[0], kernel, stride, padding)
  return error("avg_pool2d IR result type does not match the graph") if built.values[0].type != try float_ir_type(t)
  pool_lines(built.block, built.block.ops, a, first)
end

def pool_lines(b: IR.Block, ops: List[IR.Operation], source: String, first: Int) -> Result[List[String]]
  entry_value = b.args[0]
  names: Map[Ref, String] = {}
  names = names.put(entry_value.id, source)
  lines: List[String] = []
  for i in 0...ops.size
    op = ops[i]
    lines = lines.push(try IR.operation(op, names, "pool#{first}_#{i}"))
    for value in op.results
      names = names.put(value.id, "%#{first + i}")
    end
  end

  ok(lines)
end

def conv1d(
  t: Types.Tensor,
  args: List[Types.Tensor],
  a: List[String],
  stride: Int,
  padding: Int,
  first: Int,
) -> Result[List[String]]
  entry = IR.block([try float_ir_type(args[0]), try float_ir_type(args[1])])
  input = entry.args[0]
  weight = entry.args[1]
  built = try Conv.conv1d(entry, input, weight, stride, padding)
  out = try float_ir_type(t)
  return error("conv1d IR result type does not match the graph") if built.values[0].type != out
  names: Map[Ref, String] = {}
  names = names.put(input.id, a[0]).put(weight.id, a[1])
  lines: List[String] = []
  for i in 0...built.block.ops.size
    op = built.block.ops[i]
    lines = lines.push(try IR.operation(op, names, "conv1d_#{first}_#{i}"))
    for value in op.results
      names = names.put(value.id, "%#{first + i}")
    end
  end

  ok(lines)
end

# Bind the typed builder arguments to the SSA names already assigned by the graph emitter.
def conv2d(
  t: Types.Tensor,
  args: List[Types.Tensor],
  a: List[String],
  stride: Int,
  padding: Int,
  groups: Int,
  first: Int,
) -> Result[List[String]]
  entry = IR.block([try float_ir_type(args[0]), try float_ir_type(args[1])])
  input = entry.args[0]
  weight = entry.args[1]
  built = try Conv.conv2d(entry, input, weight, stride, padding, groups)
  out = try float_ir_type(t)
  return error("conv2d IR result type does not match the graph") if built.values[0].type != out
  names: Map[Ref, String] = {}
  names = names.put(input.id, a[0]).put(weight.id, a[1])
  ok([try IR.operation(built.block.ops[0], names, "conv#{first}_")])
end

def float_ir_type(t: Types.Tensor) -> Result[IR.TensorType]
  case t.dtype
    Types.Dtype.F16 -> ok(IR.TensorType(shape: t.shape, element: IR.Element.F16))
    Types.Dtype.F32 -> ok(IR.TensorType(shape: t.shape, element: IR.Element.F32))
    Types.Dtype.F64 -> ok(IR.TensorType(shape: t.shape, element: IR.Element.F64))
    _ -> error("StableHLO IR needs a floating-point tensor")
  end
end

# Rows of the source at the indices: one row slice per index, the indices' axes first, as `Reference.gather` answers.
# An index outside the rows is clamped, as XLA does.
def gather(t: Types.Tensor, args: List[Types.Tensor], a: List[String], first: Int) -> Result[List[String]]
  types = [try ir_type(args[0].shape, args[0].dtype), try ir_type(args[1].shape, args[1].dtype)]
  attrs = gather_attrs(args[0].shape, args[1].shape.size)
  ok([try ir_op("stablehlo.gather", a, types, try ir_type(t.shape, t.dtype), attrs, first)])
end

# See StableHLO reduce_window: padding takes the initial value, so padded cells must not win over negative inputs.
def max_pool2d(
  t: Types.Tensor,
  x: Types.Tensor,
  a: String,
  kernel: Int,
  stride: Int,
  pad: Int,
  first: Int,
) -> Result[List[String]]
  input = try ir_type(x.shape, x.dtype)
  entry = IR.block([input, IR.TensorType(shape: [], element: input.element)])
  built = try Pool.max_pool2d(entry, entry.args[0], entry.args[1], kernel, stride, pad)
  if built.values[0].type != try ir_type(t.shape, t.dtype)
    return error("max_pool2d IR result type does not match the graph")
  end

  ok([try fill_scalar("maximum", t.dtype)] + try ir_lines(built.block, [a, "%#{first}"], first + 1))
end

def expert_matmul(t: Types.Tensor, args: List[Types.Tensor], a: List[String], first: Int) -> Result[List[String]]
  weights = args[0]
  x = args[1]
  ids = args[2]
  rows = weights.shape[1]
  inner = weights.shape[2]
  entry = IR.block([
    try ir_type(weights.shape, weights.dtype),
    try ir_type(x.shape, x.dtype),
    try ir_type(ids.shape, ids.dtype),
  ])

  selected = try ir_type(ids.shape + [rows, inner], weights.dtype)
  broadcast_x = try ir_type(ids.shape + [inner], x.dtype)
  out = try ir_type(t.shape, t.dtype)
  gathered = try IR.append(
    entry,
    "stablehlo.gather",
    [entry.args[0], entry.args[2]],
    [selected],
    gather_attrs(weights.shape, ids.shape.size),
  )

  spread = try IR.append(
    gathered.block,
    "stablehlo.broadcast_in_dim",
    [entry.args[1]],
    [broadcast_x],
    [IR.Attribute(key: "broadcast_dimensions", value: IR.Attr.I64Array([0, 1, 2]))],
  )

  product = try IR.append(
    spread.block,
    "stablehlo.dot_general",
    [spread.values[0], gathered.values[0]],
    [out],
    [dot_numbers([0, 1], [0, 1], [2], [3])],
  )

  ir_lines(product.block, [a[0], a[1], a[2]], first)
end

def gather_attrs(source: List[Int], index_rank: Int) -> List[IR.Attribute]
  rest = source.drop(1)
  fields = ["collapsed_slice_dims = [0]", "start_index_map = [0]", "index_vector_dim = #{index_rank}"]
  if !rest.empty?
    fields = ["offset_dims = #{ints((0...rest.size).to_a.map { |i| index_rank + i })}"] + fields
  end

  [
    IR.Attribute(key: "dimension_numbers", value: IR.Attr.DialectAttr("stablehlo", "gather", fields.join(", "))),
    IR.Attribute(key: "slice_sizes", value: IR.Attr.I64Array([1] + rest)),
    IR.Attribute(key: "indices_are_sorted", value: IR.Attr.Bool(false)),
  ]
end

def dot_numbers(lhs_batch: List[Int], rhs_batch: List[Int], lhs: List[Int], rhs: List[Int]) -> IR.Attribute
  fields: List[String] = []
  if !lhs_batch.empty?
    fields = ["lhs_batching_dimensions = #{ints(lhs_batch)}", "rhs_batching_dimensions = #{ints(rhs_batch)}"]
  end

  fields = fields + ["lhs_contracting_dimensions = #{ints(lhs)}", "rhs_contracting_dimensions = #{ints(rhs)}"]
  IR.Attribute(key: "dot_dimension_numbers", value: IR.Attr.DialectAttr("stablehlo", "dot", fields.join(", ")))
end

def ir_op(
  name: String,
  sources: List[String],
  types: List[IR.TensorType],
  result: IR.TensorType,
  attributes: List[IR.Attribute],
  first: Int,
  regions: List[IR.Block] = [],
) -> Result[String]
  entry = IR.block(types)
  added = try IR.append(entry, name, entry.args, [result], attributes, regions)
  lines = try ir_lines(added.block, sources, first)
  ok(lines[0])
end

def ir_lines(b: IR.Block, sources: List[String], first: Int) -> Result[List[String]]
  names: Map[Ref, String] = {}
  for i in 0...sources.size
    names = names.put(b.args[i].id, sources[i])
  end

  lines: List[String] = []
  for i in 0...b.ops.size
    op = b.ops[i]
    lines = lines.push(try IR.operation(op, names, "ir#{first}_#{i}", "    "))
    for value in op.results
      names = names.put(value.id, "%#{first + i}")
    end
  end

  ok(lines)
end

# `adam_step` over `[p, g, m, v, hyper]`: the new `[p, m, v]` stacked on a new first axis, from the ops of
# `Tensor.adam_composed` over the 7 numbers of `hyper` (`lr, beta1, beta2, eps, decay, correct1, correct2`).
def adam(t: Types.Tensor, args: List[Types.Tensor], a: List[String], first: Int) -> Result[List[String]]
  dtype = args[0].dtype
  shape = try ir_type(args[0].shape, dtype)
  scalar = try ir_type([], dtype)
  row = try ir_type([1], dtype)
  hyper = try ir_type(args[4].shape, args[4].dtype)
  l = Lines(first: first, defs: [])
  at: List[String] = []
  for i in 0...7
    l = try put_slice(l, a[4], hyper, [i], [i + 1])
    l = try put_ir(l, "stablehlo.reshape", [last(l)], [row], scalar)
    l = try put_ir(l, "stablehlo.broadcast_in_dim", [last(l)], [scalar], shape, broadcast_dims([]))
    at = at.push(last(l))
  end

  l = put(l, try fill(1.0, args[0]))
  one = last(l)
  l = try put_binary(l, "subtract", one, at[1], shape)
  keep1 = last(l)
  l = try put_binary(l, "subtract", one, at[2], shape)
  keep2 = last(l)
  l = try put_binary(l, "multiply", at[0], at[4], shape)
  l = try put_binary(l, "subtract", one, last(l), shape)
  shrink = last(l)
  l = try put_binary(l, "multiply", a[2], at[1], shape)
  mb = last(l)
  l = try put_binary(l, "multiply", a[1], keep1, shape)
  l = try put_binary(l, "add", mb, last(l), shape)
  m = last(l)
  l = try put_binary(l, "multiply", a[3], at[2], shape)
  vb = last(l)
  l = try put_binary(l, "multiply", a[1], a[1], shape)
  l = try put_binary(l, "multiply", last(l), keep2, shape)
  l = try put_binary(l, "add", vb, last(l), shape)
  v = last(l)
  l = try put_binary(l, "multiply", v, at[6], shape)
  l = try put_unary(l, "sqrt", last(l), shape)
  l = try put_binary(l, "add", last(l), at[3], shape)
  denominator = last(l)
  l = try put_binary(l, "multiply", m, at[5], shape)
  l = try put_binary(l, "divide", last(l), denominator, shape)
  l = try put_binary(l, "multiply", at[0], last(l), shape)
  step = last(l)
  l = try put_binary(l, "multiply", a[0], shrink, shape)
  l = try put_binary(l, "subtract", last(l), step, shape)
  p = last(l)
  layer = try ir_type([1] + args[0].shape, dtype)
  rows: List[String] = []
  for x in [p, m, v]
    l = try put_ir(l, "stablehlo.reshape", [x], [shape], layer)
    rows = rows.push(last(l))
  end

  l = try put_ir(l, "stablehlo.concatenate", rows, [layer, layer, layer], try ir_type(t.shape, t.dtype), concat_dim(0))
  ok(l.defs)
end

# Part `i` of an `adam_step`: its layer `i` of the stacked result.
def adam_part(t: Types.Tensor, stacked: Types.Tensor, name: String, i: Int, first: Int) -> Result[List[String]]
  layer = try ir_type([1] + t.shape, t.dtype)
  l = try put_slice(
    Lines(first: first, defs: []),
    name,
    try ir_type(stacked.shape, stacked.dtype),
    [i] + t.shape.map { |_| 0 },
    [i + 1] + t.shape,
  )

  l = try put_ir(l, "stablehlo.reshape", [last(l)], [layer], try ir_type(t.shape, t.dtype))
  ok(l.defs)
end

def indexed(n: Types.Node) -> Optional[Bound]
  case n.op
    Types.Op.ExpertMatmul -> some(Bound(index: n.args[2], rows: n.args[0].shape[0]))
    Types.Op.Gather -> some(Bound(index: n.args[1], rows: n.args[0].shape[0]))
    Types.Op.ScatterAdd(rows) -> some(Bound(index: n.args[0], rows: rows))
    Types.Op.SetRows -> some(Bound(index: n.args[2], rows: n.args[0].shape[n.args[0].shape.size - 2]))
    _ -> none
  end
end

def set_rows(t: Types.Tensor, args: List[Types.Tensor], a: List[String], first: Int) -> Result[List[String]]
  rank = t.shape.size
  axis = rank - 2
  axes = [axis] + (0...rank).select { |i| i != axis }
  updates_shape = axes.map { |i| args[1].shape[i] }
  indices_shape = [args[2].shape[0], 1]
  entry = IR.block([
    try ir_type(t.shape, t.dtype),
    try ir_type(indices_shape, args[2].dtype),
    try ir_type(updates_shape, args[1].dtype),
  ])

  scalar = IR.TensorType(shape: [], element: entry.args[0].type.element)
  region = IR.block([scalar, scalar])
  ended = try IR.append(region, "stablehlo.return", [region.args[1]], [])
  dims = (
    "update_window_dims = #{ints((1...rank).to_a)}, inserted_window_dims = [#{axis}], scatter_dims_to_operand_dims = [#{axis}], index_vector_dim = 1"
  )

  attrs = [
    IR.Attribute(key: "scatter_dimension_numbers", value: IR.Attr.DialectAttr("stablehlo", "scatter", dims)),
    IR.Attribute(key: "indices_are_sorted", value: IR.Attr.Bool(false)),
    IR.Attribute(key: "unique_indices", value: IR.Attr.Bool(false)),
  ]

  scattered = try IR.append(
    entry,
    "stablehlo.scatter",
    entry.args,
    [entry.args[0].type],
    attrs,
    regions: [ended.block],
  )

  names: Map[Ref, String] = {}
  names = names.put(entry.args[0].id, a[0]).put(entry.args[1].id, "%#{first}").put(entry.args[2].id, "%#{first + 1}")
  scatter = try IR.operation(scattered.block.ops[0], names, "set_rows#{first}_", "    ")
  ok([
    try ir_op("stablehlo.reshape", [a[2]], [try ir_type(args[2].shape, args[2].dtype)], entry.args[1].type, [], first),
    try transpose_op(a[1], try ir_type(args[1].shape, args[1].dtype), entry.args[2].type, axes, first + 1),
    scatter,
  ])
end

def ir_type(shape: List[Int], dtype: Types.Dtype) -> Result[IR.TensorType]
  case dtype
    Types.Dtype.F16 -> ok(IR.TensorType(shape: shape, element: IR.Element.F16))
    Types.Dtype.F32 -> ok(IR.TensorType(shape: shape, element: IR.Element.F32))
    Types.Dtype.F64 -> ok(IR.TensorType(shape: shape, element: IR.Element.F64))
    Types.Dtype.I8 -> ok(IR.TensorType(shape: shape, element: IR.Element.I8))
    Types.Dtype.I16 -> ok(IR.TensorType(shape: shape, element: IR.Element.I16))
    Types.Dtype.I32 -> ok(IR.TensorType(shape: shape, element: IR.Element.I32))
    Types.Dtype.I64 -> ok(IR.TensorType(shape: shape, element: IR.Element.I64))
    Types.Dtype.U8 -> ok(IR.TensorType(shape: shape, element: IR.Element.U8))
    Types.Dtype.Bool -> ok(IR.TensorType(shape: shape, element: IR.Element.I1))
    _ -> error("StableHLO IR has no element type for #{Types.dtype_name(dtype)}")
  end
end

def put_ir(
  l: Lines,
  name: String,
  sources: List[String],
  types: List[IR.TensorType],
  result: IR.TensorType,
  attributes: List[IR.Attribute] = [],
  regions: List[IR.Block] = [],
) -> Result[Lines]
  ok(put(l, try ir_op(name, sources, types, result, attributes, l.first + l.defs.size, regions)))
end

def put_unary(l: Lines, op: String, x: String, t: IR.TensorType) -> Result[Lines]
  put_ir(l, "stablehlo.#{op}", [x], [t], t)
end

def put_binary(l: Lines, op: String, x: String, y: String, t: IR.TensorType) -> Result[Lines]
  put_ir(l, "stablehlo.#{op}", [x, y], [t, t], t)
end

def put_slice(l: Lines, name: String, x: IR.TensorType, starts: List[Int], ends: List[Int]) -> Result[Lines]
  ok(put(l, try slice_op(name, x, starts, ends, l.first + l.defs.size)))
end

def concat_dim(axis: Int) -> List[IR.Attribute] = [IR.Attribute(key: "dimension", value: IR.Attr.I64(axis))]

def broadcast_dims(dims: List[Int]) -> List[IR.Attribute]
  [IR.Attribute(key: "broadcast_dimensions", value: IR.Attr.I64Array(dims))]
end

def put_reduce(
  l: Lines,
  name: String,
  x: IR.TensorType,
  init: String,
  axes: List[Int],
  combine: String,
) -> Result[Lines]
  entry = IR.block([x, IR.TensorType(shape: [], element: x.element)])
  built = try Reduce.reduce_op(entry, entry.args[0], entry.args[1], axes, combine)
  lines = try ir_lines(built.block, [name, init], l.first + l.defs.size)
  ok(l.with(defs: l.defs + lines))
end

# Whether one of the indices `name` (the tensor `idx`) is outside `[0, rows)`: an `i1`, the last line. No lines when
# the dtype of `idx` cannot hold such an index: a compare needs its bound in that dtype.
def outside(idx: Types.Tensor, name: String, rows: Int, first: Int) -> Result[List[String]]
  shape = try ir_type(idx.shape, idx.dtype)
  flags = try ir_type(idx.shape, Types.Dtype.Bool)
  l = Lines(first: first, defs: [])
  found: List[String] = []
  if idx.dtype != Types.Dtype.U8
    l = put(l, try splat(Types.Values.Int(0), idx.shape, idx.dtype))
    l = try put_ir(l, "stablehlo.compare", [name, last(l)], [shape, shape], flags, compare_attrs("LT", idx.dtype))
    found = found.push(last(l))
  end

  if Types.check_int_range(rows, idx.dtype).ok?
    l = put(l, try splat(Types.Values.Int(rows), idx.shape, idx.dtype))
    l = try put_ir(l, "stablehlo.compare", [name, last(l)], [shape, shape], flags, compare_attrs("GE", idx.dtype))
    found = found.push(last(l))
  end

  return ok([]) if found.empty?
  if found.size == 2
    l = try put_binary(l, "or", found[0], found[1], flags)
  end

  return ok(l.defs) if idx.shape.empty?
  any = last(l)
  l = put(l, try splat(Types.Values.Bool(false), [], Types.Dtype.Bool))
  l = try put_reduce(l, any, flags, last(l), (0...idx.shape.size).to_a, "or")
  ok(l.defs)
end

def put(l: Lines, text: String) -> Lines = l.with(defs: l.defs.push(text))

# The name of the last line of `l`.
def last(l: Lines) -> String = "%#{l.first + l.defs.size - 1}"

def matmul(t: Types.Tensor, args: List[Types.Tensor], a: List[String], first: Int) -> Result[List[String]]
  if args[0].dtype != args[1].dtype
    kinds = "#{Types.dtype_name(args[0].dtype)} by #{Types.dtype_name(args[1].dtype)}"
    return error("StableHLO has no emitter for matmul of #{kinds} yet")
  end

  dot(t, args, a, [], [1], [0], first)
end

def dot(
  t: Types.Tensor,
  args: List[Types.Tensor],
  a: List[String],
  batch: List[Int],
  lhs: List[Int],
  rhs: List[Int],
  first: Int,
) -> Result[List[String]]
  types = [try ir_type(args[0].shape, args[0].dtype), try ir_type(args[1].shape, args[1].dtype)]
  attrs = [dot_numbers(batch, batch, lhs, rhs)]
  ok([try ir_op("stablehlo.dot_general", a, types, try ir_type(t.shape, t.dtype), attrs, first)])
end

# The line `n` of the program for the text of a node: an op with two results (`TWO`) is `%n:2`, and its second
# result is
# `%n#1`.
def numbered(n: Int, text: String) -> String
  return "    %#{n}:2 = #{text.slice(TWO.size, text.size - TWO.size)}" if text.start_with?(TWO)
  "    %#{n} = #{text}"
end

# The ids of the `k` largest values of each row of `x` (a vector or a matrix), the largest first and the lowest
# index of
# equal values first: a stable descending sort of the values with an `iota`, and the first `k` ids.
def top_k(t: Types.Tensor, x: Types.Tensor, name: String, k: Int, first: Int) -> Result[List[String]]
  axis = x.shape.size - 1
  values = try ir_type(x.shape, x.dtype)
  ids = try ir_type(x.shape, Types.Dtype.I32)
  value = IR.TensorType(shape: [], element: values.element)
  index = IR.TensorType(shape: [], element: IR.Element.I32)
  flag = IR.TensorType(shape: [], element: IR.Element.I1)
  region = IR.block([value, value, index, index])
  compared = try IR.append(
    region,
    "stablehlo.compare",
    [region.args[0], region.args[1]],
    [flag],
    compare_attrs("GT", x.dtype),
  )

  yielded = try IR.append(compared.block, "stablehlo.return", compared.values, [])
  entry = IR.block([values, ids])
  attrs = [
    IR.Attribute(key: "dimension", value: IR.Attr.I64(axis)),
    IR.Attribute(key: "is_stable", value: IR.Attr.Bool(true)),
  ]

  sorted = try IR.append(entry, "stablehlo.sort", entry.args, [values, ids], attrs, [yielded.block])
  names: Map[Ref, String] = {}
  names = names.put(entry.args[0].id, name).put(entry.args[1].id, "%#{first}")
  ok([
    try iota_op(ids, axis, first),
    "#{TWO}#{try IR.operation(sorted.block.ops[0], names, "ir#{first + 1}_0", "    ")}",
    try slice_op("%#{first + 1}#1", ids, x.shape.map { |_| 0 }, x.shape.take(axis) + [k], first + 2),
  ])
end

def iota_op(t: IR.TensorType, dim: Int, first: Int) -> Result[String]
  ir_op("stablehlo.iota", [], [], t, [IR.Attribute(key: "iota_dimension", value: IR.Attr.I64(dim))], first)
end

def slice_op(name: String, x: IR.TensorType, starts: List[Int], ends: List[Int], first: Int) -> Result[String]
  out = IR.TensorType(shape: (0...starts.size).map { |i| ends[i] - starts[i] }, element: x.element)
  attrs = [
    IR.Attribute(key: "start_indices", value: IR.Attr.I64Array(starts)),
    IR.Attribute(key: "limit_indices", value: IR.Attr.I64Array(ends)),
    IR.Attribute(key: "strides", value: IR.Attr.I64Array(starts.map { |_| 1 })),
  ]

  ir_op("stablehlo.slice", [name], [x], out, attrs, first)
end

def compare_attrs(direction: String, dtype: Types.Dtype) -> List[IR.Attribute]
  [
    IR.Attribute(
      key: "comparison_direction",
      value: IR.Attr.DialectEnum("stablehlo", "comparison_direction", direction),
    ),
    IR.Attribute(key: "compare_type", value: IR.Attr.DialectEnum("stablehlo", "comparison_type", compare_type(dtype))),
  ]
end

# `scatter_add`: the rows of `updates` added into zeros `[rows, ...]` at `indices`, duplicates summing. An index
# outside the rows is dropped by XLA, so for `Xla` the program also answers a flag that one is (`checked`).
def scatter_add(t: Types.Tensor, args: List[Types.Tensor], a: List[String], first: Int) -> Result[List[String]]
  k = args[0].shape.size
  rest = t.shape.drop(1)
  fields = ["inserted_window_dims = [0]", "scatter_dims_to_operand_dims = [0]", "index_vector_dim = #{k}"]
  if !rest.empty?
    fields = ["update_window_dims = #{ints((0...rest.size).to_a.map { |i| k + i })}"] + fields
  end

  out = try ir_type(t.shape, t.dtype)
  scalar = IR.TensorType(shape: [], element: out.element)
  region = IR.block([scalar, scalar])
  total = try IR.append(region, "stablehlo.add", region.args, [scalar])
  yielded = try IR.append(total.block, "stablehlo.return", total.values, [])
  attrs = [
    IR.Attribute(
      key: "scatter_dimension_numbers",
      value: IR.Attr.DialectAttr("stablehlo", "scatter", fields.join(", ")),
    ),
    IR.Attribute(key: "indices_are_sorted", value: IR.Attr.Bool(false)),
    IR.Attribute(key: "unique_indices", value: IR.Attr.Bool(false)),
  ]

  types = [out, try ir_type(args[0].shape, args[0].dtype), try ir_type(args[1].shape, args[1].dtype)]
  ok([
    try fill(0.0, t),
    try ir_op("stablehlo.scatter", ["%#{first}", a[0], a[1]], types, out, attrs, first + 1, [yielded.block]),
  ])
end

# Attention over `[batch, heads, seq, width]` queries, keys and values (the keys and values may have fewer heads,
# each serving a group of consecutive query heads): `softmax(q k^T scale + mask) v`, with the softmax over the
# keys written as the maximum, `exp` and a sum, as `Reference.attention` does. A key of another dtype than the
# query is converted.
def attention(
  t: Types.Tensor,
  args: List[Types.Tensor],
  a: List[String],
  scale: Float,
  first: Int,
) -> Result[List[String]]
  q = args[0]
  heads = q.shape[1]
  group = heads / args[1].shape[1]
  l = Lines(first: first, defs: [])
  side = try attention_side(l, args[1], q, group, a[1])
  keys = side.name
  side = try attention_side(side.lines, args[2], q, group, a[2])
  values = side.name
  l = side.lines
  shape = [q.shape[0], heads, q.shape[2], args[1].shape[2]]
  scores = try ir_type(shape, q.dtype)
  kt = try ir_type([q.shape[0], heads, args[1].shape[2], q.shape[3]], q.dtype)
  vt = try ir_type([q.shape[0], heads, args[2].shape[2], args[2].shape[3]], q.dtype)
  l = try put_ir(
    l,
    "stablehlo.dot_general",
    [a[0], keys],
    [try ir_type(q.shape, q.dtype), kt],
    scores,
    [dot_numbers([0, 1], [0, 1], [3], [3])],
  )

  product = last(l)
  l = put(l, try splat(Types.Values.Float(scale), shape, q.dtype))
  l = try put_ir(l, "stablehlo.multiply", [product, last(l)], [scores, scores], scores)
  if args.size == 4
    mask = args[3]
    scaled = last(l)
    masked = a[3]
    if mask.shape != shape
      spread = (0...mask.shape.size).to_a.map { |i| shape.size - mask.shape.size + i }
      l = try put_ir(
        l,
        "stablehlo.broadcast_in_dim",
        [a[3]],
        [try ir_type(mask.shape, mask.dtype)],
        scores,
        broadcast_dims(spread),
      )

      masked = last(l)
    end

    l = try put_ir(l, "stablehlo.add", [scaled, masked], [scores, scores], scores)
  end

  l = try softmax_last(l, last(l), shape, q.dtype)
  l = try put_ir(
    l,
    "stablehlo.dot_general",
    [last(l), values],
    [scores, vt],
    try ir_type(t.shape, t.dtype),
    [dot_numbers([0, 1], [0, 1], [3], [2])],
  )

  ok(l.defs)
end

# `x` (keys or values of fewer heads) as one tensor of the query's heads, in the dtype of the query: each head repeated
# for its group.
def attention_side(l: Lines, x: Types.Tensor, q: Types.Tensor, group: Int, name: String) -> Result[Side]
  source = name
  shown = try ir_type(x.shape, x.dtype)
  if x.dtype != q.dtype
    wide = try ir_type(x.shape, q.dtype)
    l = try put_ir(l, "stablehlo.convert", [source], [shown], wide)
    shown = wide
    source = last(l)
  end

  return ok(Side(lines: l, name: source)) if group == 1
  b = x.shape[0]
  spread = try ir_type([b, x.shape[1], group, x.shape[2], x.shape[3]], q.dtype)
  merged = try ir_type([b, q.shape[1], x.shape[2], x.shape[3]], q.dtype)
  l = try put_ir(l, "stablehlo.broadcast_in_dim", [source], [shown], spread, broadcast_dims([0, 1, 3, 4]))
  l = try put_ir(l, "stablehlo.reshape", [last(l)], [spread], merged)
  ok(Side(lines: l, name: last(l)))
end

# The softmax over the last axis of `name` (of `shape`): `exp(x - max) / sum(exp(x - max))`. The last line.
def softmax_last(l: Lines, name: String, shape: List[Int], dtype: Types.Dtype) -> Result[Lines]
  q = shape.size - 1
  full = try ir_type(shape, dtype)
  lead = try ir_type(shape.take(q), dtype)
  dims = broadcast_dims((0...q).to_a)
  l = put(l, try fill_scalar("maximum", dtype))
  l = try put_reduce(l, name, full, last(l), [q], "maximum")
  l = try put_ir(l, "stablehlo.broadcast_in_dim", [last(l)], [lead], full, dims)
  l = try put_ir(l, "stablehlo.subtract", [name, last(l)], [full, full], full)
  l = try put_ir(l, "stablehlo.exponential", [last(l)], [full], full)
  powers = last(l)
  l = put(l, try fill_scalar("add", dtype))
  l = try put_reduce(l, powers, full, last(l), [q], "add")
  l = try put_ir(l, "stablehlo.broadcast_in_dim", [last(l)], [lead], full, dims)
  put_ir(l, "stablehlo.divide", [powers, last(l)], [full, full], full)
end

# The exact GELU, `0.5 x (1 + erf(x / sqrt(2)))`, the form `Reference` computes: `erf` is CHLO's (XLA lowers it).
def gelu_erf(t: Types.Tensor, x: String, first: Int) -> Result[List[String]]
  out = try ir_type(t.shape, t.dtype)
  l = put(Lines(first: first, defs: []), try fill(Reference.INV_SQRT2, t))
  l = try put_binary(l, "multiply", x, last(l), out)
  l = try put_ir(l, "chlo.erf", [last(l)], [out], out)
  erf = last(l)
  l = put(l, try fill(1.0, t))
  l = try put_binary(l, "add", last(l), erf, out)
  sum = last(l)
  l = put(l, try fill(0.5, t))
  l = try put_binary(l, "multiply", last(l), x, out)
  l = try put_binary(l, "multiply", last(l), sum, out)
  ok(l.defs)
end

# A layer norm (`center`) or an RMS norm of `x` over its last axis, no affine: `centered / sqrt(mean(centered^2) +
# eps)`, where `centered` is `x - mean(x)` for the layer norm and `x` for the RMS norm, as
# `Tensor.layer_norm_composed` and `rms_norm_composed` write them. The sums run in the dtype of `x`.
def normalized(t: Types.Tensor, x: String, eps: Float, center: Bool, first: Int) -> Result[List[String]]
  out = try ir_type(t.shape, t.dtype)
  q = t.shape.size - 1
  dims = broadcast_dims((0...q).to_a)
  lead = try ir_type(t.shape.take(q), t.dtype)
  l = Lines(first: first, defs: [])
  centered = x
  if center
    l = try last_mean(l, x, t)
    l = try put_ir(l, "stablehlo.broadcast_in_dim", [last(l)], [lead], out, dims)
    l = try put_binary(l, "subtract", x, last(l), out)
    centered = last(l)
  end

  l = try put_binary(l, "multiply", centered, centered, out)
  l = try last_mean(l, last(l), t)
  l = try put_ir(l, "stablehlo.broadcast_in_dim", [last(l)], [lead], out, dims)
  var = last(l)
  l = put(l, try fill(eps, t))
  l = try put_binary(l, "add", var, last(l), out)
  l = try put_unary(l, "sqrt", last(l), out)
  l = try put_binary(l, "divide", centered, last(l), out)
  ok(l.defs)
end

# The mean of `name` (of the shape of `like`) over the last axis, as the last line of `l`: the shape without that axis.
def last_mean(l: Lines, name: String, like: Types.Tensor) -> Result[Lines]
  q = like.shape.size - 1
  lead = like.shape.take(q)
  l = put(l, try fill_scalar("add", like.dtype))
  l = try put_reduce(l, name, try ir_type(like.shape, like.dtype), last(l), [q], "add")
  total = last(l)
  l = put(l, try splat(Types.Values.Float(like.shape[q].to_f), lead, like.dtype))
  put_binary(l, "divide", total, last(l), try ir_type(lead, like.dtype))
end

# Rotary embedding of `x` `[..., seq, width]` by the integer `positions` `[seq]`. The cosines and sines of the
# angles `position * base^(-2i/dims)` are written in the program (the frequencies are constants), then the pairs
# are rotated as `Reference.rope` does: `(i, i + dims/2)` with `neox`, `(2i, 2i + 1)` without; the columns past
# `dims` pass through.
def rope(
  t: Types.Tensor,
  args: List[Types.Tensor],
  a: List[String],
  base: Float,
  dims: Int,
  neox: Bool,
  inverse: Bool,
  first: Int,
) -> Result[List[String]]
  x = args[0]
  rank = x.shape.size
  seq = x.shape[rank - 2]
  width = x.shape[rank - 1]
  half = dims / 2
  grid = try ir_type([seq, half], x.dtype)
  table = try ir_type([half], x.dtype)
  line = try ir_type([seq], x.dtype)
  l = Lines(first: first, defs: [])
  l = try put_ir(l, "stablehlo.convert", [a[1]], [try ir_type(args[1].shape, args[1].dtype)], line)
  l = try put_ir(l, "stablehlo.broadcast_in_dim", [last(l)], [line], grid, broadcast_dims([0]))
  positions = last(l)
  turns = (0...half).map { |i| decimal(Math.exp(-((2.0 * i.to_f) / dims.to_f) * Math.log(base))) }
  l = put(l, try constant_op("[#{turns.join(", ")}]", table))
  l = try put_ir(l, "stablehlo.broadcast_in_dim", [last(l)], [table], grid, broadcast_dims([1]))
  l = try put_binary(l, "multiply", positions, last(l), grid)
  angle = last(l)
  l = try put_unary(l, "cosine", angle, grid)
  cosine = last(l)
  l = try put_unary(l, "sine", angle, grid)
  if inverse
    l = try put_unary(l, "negate", last(l), grid)
  end

  sine = last(l)
  lead = x.shape.take(rank - 1)
  pairs = try ir_type(lead + [half], x.dtype)
  spread = broadcast_dims([rank - 2, rank - 1])
  l = try put_ir(l, "stablehlo.broadcast_in_dim", [cosine], [grid], pairs, spread)
  cosine = last(l)
  l = try put_ir(l, "stablehlo.broadcast_in_dim", [sine], [grid], pairs, spread)
  sine = last(l)
  full = try ir_type(lead + [dims], x.dtype)
  parts: List[String] = []
  part_types: List[IR.TensorType] = []
  if neox
    halves: List[String] = []
    for c in [[0, half], [half, dims]]
      l = try put_columns(l, x, a[0], c[0], c[1])
      halves = halves.push(last(l))
    end

    l = try put_binary(l, "multiply", halves[0], cosine, pairs)
    turned = last(l)
    l = try put_binary(l, "multiply", halves[1], sine, pairs)
    l = try put_binary(l, "subtract", turned, last(l), pairs)
    parts = parts.push(last(l))
    l = try put_binary(l, "multiply", halves[1], cosine, pairs)
    turned = last(l)
    l = try put_binary(l, "multiply", halves[0], sine, pairs)
    l = try put_binary(l, "add", turned, last(l), pairs)
    parts = parts.push(last(l))
    part_types = [pairs, pairs]
  else
    twos = try ir_type(lead + [half, 2], x.dtype)
    one = try ir_type(lead + [half, 1], x.dtype)
    l = try put_columns(l, x, a[0], 0, dims)
    l = try put_ir(l, "stablehlo.reshape", [last(l)], [full], twos)
    folded = last(l)
    cells: List[String] = []
    for k in 0...2
      l = try put_slice(l, folded, twos, lead.map { |_| 0 } + [0, k], lead + [half, k + 1])
      l = try put_ir(l, "stablehlo.reshape", [last(l)], [one], pairs)
      cells = cells.push(last(l))
    end

    l = try put_binary(l, "multiply", cells[0], cosine, pairs)
    turned = last(l)
    l = try put_binary(l, "multiply", cells[1], sine, pairs)
    l = try put_binary(l, "subtract", turned, last(l), pairs)
    even = last(l)
    l = try put_binary(l, "multiply", cells[1], cosine, pairs)
    turned = last(l)
    l = try put_binary(l, "multiply", cells[0], sine, pairs)
    l = try put_binary(l, "add", turned, last(l), pairs)
    odd = last(l)
    rows: List[String] = []
    for r in [even, odd]
      l = try put_ir(l, "stablehlo.reshape", [r], [pairs], one)
      rows = rows.push(last(l))
    end

    l = try put_ir(l, "stablehlo.concatenate", rows, [one, one], twos, concat_dim(rank))
    l = try put_ir(l, "stablehlo.reshape", [last(l)], [twos], full)
    parts = [last(l)]
    part_types = [full]
  end

  if dims < width
    l = try put_columns(l, x, a[0], dims, width)
    parts = parts.push(last(l))
    part_types = part_types.push(try ir_type(lead + [width - dims], x.dtype))
  end

  l = try put_ir(l, "stablehlo.concatenate", parts, part_types, try ir_type(t.shape, t.dtype), concat_dim(rank - 1))
  ok(l.defs)
end

# The columns `from...to` of the last axis of `x`.
def put_columns(l: Lines, x: Types.Tensor, name: String, from: Int, to: Int) -> Result[Lines]
  rank = x.shape.size
  starts = x.shape.map { |_| 0 }.take(rank - 1) + [from]
  put_slice(l, name, try ir_type(x.shape, x.dtype), starts, x.shape.take(rank - 1) + [to])
end

# The batched product `[B..., m, k] . [B..., k, n]`: the leading axes are the batch, the rest is `matmul`.
def batch_matmul(t: Types.Tensor, args: List[Types.Tensor], a: List[String], first: Int) -> Result[List[String]]
  if args[0].dtype != args[1].dtype
    kinds = "#{Types.dtype_name(args[0].dtype)} by #{Types.dtype_name(args[1].dtype)}"
    return error("StableHLO has no emitter for batch_matmul of #{kinds} yet")
  end

  rank = args[0].shape.size
  dot(t, args, a, (0...(rank - 2)).to_a, [rank - 1], [rank - 2], first)
end

# `x` as another dtype: a conversion truncates a float to an integer, and a boolean is `x != 0`.
def convert(t: Types.Tensor, x: Types.Tensor, name: String, first: Int) -> Result[List[String]]
  from = try ir_type(x.shape, x.dtype)
  to = try ir_type(t.shape, t.dtype)
  if t.dtype == Types.Dtype.Bool and x.dtype != Types.Dtype.Bool
    zero = Types.Values.Int(0)
    if Types.is_float_dtype(x.dtype)
      zero = Types.Values.Float(0.0)
    end

    return ok([
      try splat(zero, x.shape, x.dtype),
      try ir_op("stablehlo.compare", [name, "%#{first}"], [from, from], to, compare_attrs("NE", x.dtype), first + 1),
    ])
  end

  ok([try ir_op("stablehlo.convert", [name], [from], to, [], first)])
end

# The tensors of `args` joined on a new `axis`: each gets a 1 there, and they are concatenated.
def stack(t: Types.Tensor, args: List[Types.Tensor], a: List[String], axis: Int, first: Int) -> Result[List[String]]
  layer = try ir_type(args[0].shape.take(axis) + [1] + args[0].shape.drop(axis), args[0].dtype)
  l = Lines(first: first, defs: [])
  rows: List[String] = []
  for i in 0...args.size
    l = try put_ir(l, "stablehlo.reshape", [a[i]], [try ir_type(args[i].shape, args[i].dtype)], layer)
    rows = rows.push(last(l))
  end

  l = try put_ir(
    l,
    "stablehlo.concatenate",
    rows,
    rows.map { |_| layer },
    try ir_type(t.shape, t.dtype),
    concat_dim(axis),
  )

  ok(l.defs)
end

# The index of the largest value of each block of `x` that the reduced `axes` make, numbered row-major by the ascending
# axes, the lowest one of a tie: the axes go last and are flattened into one `block`, the maximum is found, and the
# smallest index among the positions that hold it (the others hold `block`) is the answer.
def argmax(
  t: Types.Tensor,
  x: Types.Tensor,
  name: String,
  axes: List[Int],
  keep_dims: Bool,
  first: Int,
) -> Result[List[String]]
  kept = (0...x.shape.size).to_a.select { |i| !axes.include?(i) }
  order = kept + axes
  block = try Types.volume(axes.map { |ax| x.shape[ax] })
  lead = kept.map { |i| x.shape[i] }
  flat = lead + [block]
  q = lead.size
  permuted = try ir_type(order.map { |i| x.shape[i] }, x.dtype)
  l = Lines(first: first, defs: [])
  source = name
  if order != (0...x.shape.size).to_a
    l = put(l, try transpose_op(source, try ir_type(x.shape, x.dtype), permuted, order, first + l.defs.size))
    source = last(l)
  end

  rows = try ir_type(flat, x.dtype)
  l = try put_ir(l, "stablehlo.reshape", [source], [permuted], rows)
  source = last(l)
  top = try ir_type(lead, x.dtype)
  l = put(l, try fill_scalar("maximum", x.dtype))
  l = try put_reduce(l, source, rows, last(l), [q], "maximum")
  l = try put_ir(l, "stablehlo.broadcast_in_dim", [last(l)], [top], rows, broadcast_dims((0...q).to_a))
  flags = try ir_type(flat, Types.Dtype.Bool)
  l = try put_ir(l, "stablehlo.compare", [source, last(l)], [rows, rows], flags, compare_attrs("EQ", x.dtype))
  hit = last(l)
  positions = try ir_type(flat, t.dtype)
  l = put(l, try iota_op(positions, q, first + l.defs.size))
  index = last(l)
  l = put(l, try splat(Types.Values.Int(block), flat, t.dtype))
  l = try put_ir(l, "stablehlo.select", [hit, index, last(l)], [flags, positions, positions], positions)
  chosen = last(l)
  l = put(l, try splat(Types.Values.Int(block), [], t.dtype))
  l = try put_reduce(l, chosen, positions, last(l), [q], "minimum")
  if keep_dims
    l = try put_ir(l, "stablehlo.reshape", [last(l)], [try ir_type(lead, t.dtype)], try ir_type(t.shape, t.dtype))
  end

  ok(l.defs)
end

# `0.5 x (1 + tanh(c (x + k x^3)))`, the form `Reference` computes.
def gelu(t: Types.Tensor, x: String, first: Int) -> Result[List[String]]
  out = try ir_type(t.shape, t.dtype)
  l = try put_binary(Lines(first: first, defs: []), "multiply", x, x, out)
  l = try put_binary(l, "multiply", last(l), x, out)
  cube = last(l)
  l = put(l, try fill(Reference.GELU_C, t))
  l = try put_binary(l, "multiply", last(l), cube, out)
  l = try put_binary(l, "add", x, last(l), out)
  inner = last(l)
  l = put(l, try fill(Reference.SQRT2_OVER_PI, t))
  l = try put_binary(l, "multiply", last(l), inner, out)
  l = try put_unary(l, "tanh", last(l), out)
  curve = last(l)
  l = put(l, try fill(1.0, t))
  l = try put_binary(l, "add", last(l), curve, out)
  sum = last(l)
  l = put(l, try fill(0.5, t))
  l = try put_binary(l, "multiply", last(l), x, out)
  l = try put_binary(l, "multiply", last(l), sum, out)
  ok(l.defs)
end

def broadcast(t: Types.Tensor, from: Types.Tensor, name: String, first: Int) -> Result[List[String]]
  return ok([]) if t.shape == from.shape
  lead = t.shape.size - from.shape.size
  dims = (0...from.shape.size).to_a.map { |i| lead + i }
  x = try ir_type(from.shape, from.dtype)
  ok([
    try ir_op("stablehlo.broadcast_in_dim", [name], [x], try ir_type(t.shape, t.dtype), broadcast_dims(dims), first),
  ])
end

def reduction(
  t: Types.Tensor,
  x: Types.Tensor,
  name: String,
  axes: List[Int],
  keep_dims: Bool,
  combine: String,
  first: Int,
) -> Result[List[String]]
  dropped = Types.reduced_shape(x.shape, axes, false)
  l = put(Lines(first: first, defs: []), try fill_scalar(combine, x.dtype))
  l = try put_reduce(l, name, try ir_type(x.shape, x.dtype), last(l), axes, combine)
  return ok(l.defs) if !keep_dims
  l = try put_ir(l, "stablehlo.reshape", [last(l)], [try ir_type(dropped, x.dtype)], try ir_type(t.shape, t.dtype))
  ok(l.defs)
end

def mean(
  t: Types.Tensor,
  x: Types.Tensor,
  name: String,
  axes: List[Int],
  keep_dims: Bool,
  first: Int,
) -> Result[List[String]]
  dropped = Types.reduced_shape(x.shape, axes, false)
  reduced = try ir_type(dropped, x.dtype)
  l = Lines(first: first, defs: try reduction(t, x, name, axes, false, "add", first))
  total = last(l)
  l = put(l, try splat(Types.Values.Float(Types.reduced_count(x.shape, axes).to_f), dropped, x.dtype))
  l = try put_binary(l, "divide", total, last(l), reduced)
  return ok(l.defs) if !keep_dims
  l = try put_ir(l, "stablehlo.reshape", [last(l)], [reduced], try ir_type(t.shape, t.dtype))
  ok(l.defs)
end

def arange(t: Types.Tensor, start: Int, step: Int, first: Int) -> Result[List[String]]
  out = try ir_type(t.shape, t.dtype)
  l = put(Lines(first: first, defs: []), try iota_op(out, 0, first))
  if step != 1
    counted = last(l)
    l = put(l, try splat(Types.Values.Int(step), t.shape, t.dtype))
    l = try put_binary(l, "multiply", counted, last(l), out)
  end

  if start != 0
    scaled = last(l)
    l = put(l, try splat(Types.Values.Int(start), t.shape, t.dtype))
    l = try put_binary(l, "add", scaled, last(l), out)
  end

  ok(l.defs)
end

def fill(x: Float, like: Types.Tensor) -> Result[String] = splat(Types.Values.Float(x), like.shape, like.dtype)

def fill_scalar(combine: String, dtype: Types.Dtype) -> Result[String]
  return splat(Types.Values.Float(0.0), [], dtype) if combine == "add"
  constant_op(try infinity(dtype, true), try ir_type([], dtype))
end

def splat(v: Types.Values, shape: List[Int], dtype: Types.Dtype) -> Result[String]
  constant_op(try literal(v, dtype), try ir_type(shape, dtype))
end

def constant_op(payload: String, t: IR.TensorType) -> Result[String]
  ir_op("stablehlo.constant", [], [], t, [IR.Attribute(key: "value", value: IR.Attr.DenseLiteral(payload, t))], 0)
end

# A float literal always has its decimal point in MLIR, and a NaN or an infinity is its bit pattern.
def literal(v: Types.Values, dtype: Types.Dtype) -> Result[String]
  case v
    Types.Values.Float(x) ->
      return nan(dtype) if x != x
      return ok(try infinity(dtype, x < 0.0)) if !Types.is_finite(x)
      ok(decimal(x))
    Types.Values.Int(i) -> ok(i.to_s)
    Types.Values.Bool(b) -> ok(b.to_s)
  end
end

def decimal(x: Float) -> String
  text = x.to_s
  parts = text.split("e")
  return text if parts[0].include?(".")
  return "#{text}.0" if parts.size == 1
  "#{parts[0]}.0e#{parts[1]}"
end

def nan(dtype: Types.Dtype) -> Result[String]
  case dtype
    Types.Dtype.F32 -> ok("0x7FC00000")
    Types.Dtype.F16 -> ok("0x7E00")
    Types.Dtype.F64 -> ok("0x7FF8000000000000")
    _ -> error("#{Types.dtype_name(dtype)} has no NaN")
  end
end

def infinity(dtype: Types.Dtype, negative: Bool) -> Result[String]
  case dtype
    Types.Dtype.F32 -> ok(sign(negative, "0xFF800000", "0x7F800000"))
    Types.Dtype.F16 -> ok(sign(negative, "0xFC00", "0x7C00"))
    Types.Dtype.F64 -> ok(sign(negative, "0xFFF0000000000000", "0x7FF0000000000000"))
    _ -> error("#{Types.dtype_name(dtype)} has no infinity")
  end
end

def sign(negative: Bool, minus: String, plus: String) -> String
  return minus if negative
  plus
end

# `Bool` is a one-bit unsigned integer to `compare`.
def compare_type(dtype: Types.Dtype) -> String
  case dtype
    Types.Dtype.F32 -> "FLOAT"
    Types.Dtype.F16 -> "FLOAT"
    Types.Dtype.F64 -> "FLOAT"
    Types.Dtype.Q8_0 -> "FLOAT"
    Types.Dtype.Q4_K -> "FLOAT"
    Types.Dtype.Q6_K -> "FLOAT"
    Types.Dtype.I8 -> "SIGNED"
    Types.Dtype.I16 -> "SIGNED"
    Types.Dtype.I32 -> "SIGNED"
    Types.Dtype.I64 -> "SIGNED"
    Types.Dtype.U8 -> "UNSIGNED"
    Types.Dtype.Bool -> "UNSIGNED"
  end
end

def signature(args: List[Types.Tensor], out: String) -> Result[String]
  types = try types_of(args)
  ok("(#{types.join(", ")}) -> #{out}")
end

def types_of(ts: List[Types.Tensor]) -> Result[List[String]]
  out: List[String] = []
  for t in ts
    out = out.push(try type_of(t))
  end

  ok(out)
end

def type_of(t: Types.Tensor) -> Result[String] = tensor_type(t.shape, t.dtype)

def tensor_type(shape: List[Int], dtype: Types.Dtype) -> Result[String]
  element = try element_type(dtype)
  ok("tensor<#{(shape.map { |d| d.to_s } + [element]).join("x")}>")
end

def element_type(dtype: Types.Dtype) -> Result[String]
  case dtype
    Types.Dtype.F32 -> ok("f32")
    Types.Dtype.F16 -> ok("f16")
    Types.Dtype.F64 -> ok("f64")
    Types.Dtype.I8 -> ok("i8")
    Types.Dtype.I16 -> ok("i16")
    Types.Dtype.I32 -> ok("i32")
    Types.Dtype.I64 -> ok("i64")
    Types.Dtype.U8 -> ok("ui8")
    Types.Dtype.Bool -> ok("i1")
    Types.Dtype.Q8_0 -> error("StableHLO has no element type for Q8_0")
    Types.Dtype.Q4_K -> error("StableHLO has no element type for Q4_K")
    Types.Dtype.Q6_K -> error("StableHLO has no element type for Q6_K")
  end
end

def ints(xs: List[Int]) -> String = "[#{xs.map { |x| x.to_s }.join(", ")}]"
