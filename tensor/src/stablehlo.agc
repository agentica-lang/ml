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
      body = body.push("    %#{next} = stablehlo.or #{flag}, #{f} : tensor<i1>")
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
  out = try type_of(t)
  sig = try signature(n.args, out)
  case n.op
    Types.Op.Full(v) -> ok([try splat(v, t.shape, t.dtype)])
    Types.Op.Copy -> ok([])
    Types.Op.Contiguous -> ok([])
    Types.Op.ToDevice(Types.Device.Cpu) -> ok([])
    Types.Op.ToDevice(device) -> error("XLA runs on the CPU only; cannot lower to_device to #{Types.device_name(device)}")
    Types.Op.Add -> ok(["stablehlo.add #{a[0]}, #{a[1]} : #{out}"])
    Types.Op.Sub -> ok(["stablehlo.subtract #{a[0]}, #{a[1]} : #{out}"])
    Types.Op.Mul -> ok(["stablehlo.multiply #{a[0]}, #{a[1]} : #{out}"])
    Types.Op.Div -> ok(["stablehlo.divide #{a[0]}, #{a[1]} : #{out}"])
    Types.Op.Neg -> ok(["stablehlo.negate #{a[0]} : #{out}"])
    Types.Op.Exp -> ok(["stablehlo.exponential #{a[0]} : #{out}"])
    Types.Op.Log -> ok(["stablehlo.log #{a[0]} : #{out}"])
    Types.Op.Sqrt -> ok(["stablehlo.sqrt #{a[0]} : #{out}"])
    Types.Op.Tanh -> ok(["stablehlo.tanh #{a[0]} : #{out}"])
    Types.Op.Sigmoid -> ok(["stablehlo.logistic #{a[0]} : #{out}"])
    Types.Op.GeluErf -> gelu_erf(t, a[0], first)
    Types.Op.LayerNorm(eps) -> normalized(t, a[0], eps, true, first)
    Types.Op.RmsNorm(eps) -> normalized(t, a[0], eps, false, first)
    Types.Op.Rope(base, dims, neox, inverse) -> rope(t, n.args, a, base, dims, neox, inverse, first)
    Types.Op.Pow -> ok(["stablehlo.power #{a[0]}, #{a[1]} : #{out}"])
    Types.Op.Clamp(lo, hi) ->
      ok([try fill(lo, t), try fill(hi, t), "stablehlo.clamp %#{first}, #{a[0]}, %#{first + 1} : #{out}"])
    Types.Op.ToDtype -> convert(t, n.args[0], a[0], first)
    Types.Op.Relu -> ok([try fill(0.0, t), "stablehlo.maximum #{a[0]}, %#{first} : #{out}"])
    Types.Op.Gelu -> gelu(t, a[0], first)
    Types.Op.StopGrad -> ok([])
    Types.Op.Matmul -> matmul(n.args, a, sig)
    Types.Op.Transpose -> ok(["stablehlo.transpose #{a[0]}, dims = #{ints((0...t.shape.size).to_a.reverse)} : #{sig}"])
    Types.Op.Reshape -> ok(["stablehlo.reshape #{a[0]} : #{sig}"])
    Types.Op.Squeeze(_) -> ok(["stablehlo.reshape #{a[0]} : #{sig}"])
    Types.Op.Unsqueeze(_) -> ok(["stablehlo.reshape #{a[0]} : #{sig}"])
    Types.Op.Permute(axes) -> ok(["stablehlo.transpose #{a[0]}, dims = #{ints(axes)} : #{sig}"])
    Types.Op.Concat(axis) -> ok(["stablehlo.concatenate #{a.join(", ")}, dim = #{axis} : #{sig}"])
    Types.Op.Stack(axis) -> stack(t, n.args, a, axis, first)
    Types.Op.BatchMatmul -> batch_matmul(n.args, a, sig)
    Types.Op.Conv1d(stride, padding) -> conv1d(t, n.args, a, stride, padding, first)
    Types.Op.Conv2d(stride, padding, groups) -> conv2d(t, n.args, a, stride, padding, groups, first)
    Types.Op.GatedDeltaRule -> ok(["func.call @gated_delta_#{first}(#{a.join(", ")}) : #{sig}"])
    Types.Op.MaxPool2d(kernel, stride, padding) -> max_pool2d(t, n.args[0], a[0], kernel, stride, padding, first)
    Types.Op.AvgPool1d(kernel, stride, padding) -> avg_pool1d(t, n.args[0], a[0], kernel, stride, padding, first)
    Types.Op.AvgPool2d(kernel, stride, padding) -> avg_pool2d(t, n.args[0], a[0], kernel, stride, padding, first)
    Types.Op.ExpertMatmul -> expert_matmul(t, n.args, a, first)
    Types.Op.Argmax(axes, keep_dims) -> argmax(t, n.args[0], a[0], axes, keep_dims, first)
    Types.Op.Slice(starts, ends) -> ok(["stablehlo.slice #{a[0]} #{extents(starts, ends)} : #{sig}"])
    Types.Op.BroadcastTo -> broadcast(t, n.args[0], a[0])
    Types.Op.Sum(axes, keep_dims) -> reduction(t, n.args[0], a[0], axes, keep_dims, "add", first)
    Types.Op.Mean(axes, keep_dims) -> mean(t, n.args[0], a[0], axes, keep_dims, first)
    Types.Op.Max(axes, keep_dims) -> reduction(t, n.args[0], a[0], axes, keep_dims, "maximum", first)
    Types.Op.Lt -> ok(["stablehlo.compare LT, #{a[0]}, #{a[1]}, #{compare_type(n.args[0].dtype)} : #{sig}"])
    Types.Op.Eq -> ok(["stablehlo.compare EQ, #{a[0]}, #{a[1]}, #{compare_type(n.args[0].dtype)} : #{sig}"])
    Types.Op.Where -> ok(["stablehlo.select #{a[0]}, #{a[1]}, #{a[2]} : #{try type_of(n.args[0])}, #{out}"])
    Types.Op.Arange(start, _, step) -> arange(t, start, step, first)
    Types.Op.Gather -> gather(t, n.args, a, sig)
    Types.Op.SetRows -> set_rows(t, n.args, a, first)
    Types.Op.TopK(k) -> top_k(t, n.args[0], a[0], k, first)
    Types.Op.ScatterAdd(_) -> scatter_add(t, n.args, a, first)
    Types.Op.Attention(scale) -> attention(t, n.args, a, scale, first)
    Types.Op.AdamStep -> adam(t, n.args, a, first)
    Types.Op.AdamPart(i) -> adam_part(t, n.args[0], a[0], i, first)
    _ -> error("StableHLO has no emitter for #{Types.op_name(n.op)} yet")
  end
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
  return error("avg_pool1d IR result type does not match the graph") if built.values[0].type != (try float_ir_type(t))
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
  return error("avg_pool2d IR result type does not match the graph") if built.values[0].type != (try float_ir_type(t))
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
def gather(t: Types.Tensor, args: List[Types.Tensor], a: List[String], sig: String) -> Result[List[String]]
  k = args[1].shape.size
  rest = args[0].shape.drop(1)
  fields = ["collapsed_slice_dims = [0]", "start_index_map = [0]", "index_vector_dim = #{k}"]
  if !rest.empty?
    fields = ["offset_dims = #{ints((0...rest.size).to_a.map { |i| k + i })}"] + fields
  end

  sizes = ([1] + rest).map { |d| d.to_s }.join(", ")
  attrs = "dimension_numbers = #stablehlo.gather<#{fields.join(", ")}>, slice_sizes = array<i64: #{sizes}>"
  ok(["\"stablehlo.gather\"(#{a[0]}, #{a[1]}) {#{attrs}, indices_are_sorted = false} : #{sig}"])
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
  scalar = try tensor_type([], t.dtype)
  dims = (
    "window_dimensions = array<i64: 1, 1, #{kernel}, #{kernel}>, window_strides = array<i64: 1, 1, #{stride}, #{stride}>"
  )

  padding = "padding = dense<[[0, 0], [0, 0], [#{pad}, #{pad}], [#{pad}, #{pad}]]> : tensor<4x2xi64>"
  ok([
    try fill_scalar("maximum", t.dtype),
    "\"stablehlo.reduce_window\"(#{a}, %#{first}) ({\n      ^bb0(%lhs: #{scalar}, %rhs: #{scalar}):\n        %max = stablehlo.maximum %lhs, %rhs : #{scalar}\n        stablehlo.return %max : #{scalar}\n    }) {#{dims}, #{padding}} : (#{try type_of(x)}, #{scalar}) -> #{try type_of(t)}",
  ])
end

# Gather the selected expert matrices, then contract each input row with its own expert.
def expert_matmul(t: Types.Tensor, args: List[Types.Tensor], a: List[String], first: Int) -> Result[List[String]]
  weights = args[0]
  x = args[1]
  ids = args[2]
  rows = weights.shape[1]
  inner = weights.shape[2]
  selected = weights.with(shape: ids.shape + [rows, inner])
  broadcast_x = try tensor_type(ids.shape + [inner], x.dtype)
  selected_ty = try type_of(selected)
  gathered = try gather(
    selected,
    [weights, ids],
    [a[0], a[2]],
    "(#{try type_of(weights)}, #{try type_of(ids)}) -> #{selected_ty}",
  )

  ok(
    gathered +
    [
      "stablehlo.broadcast_in_dim #{a[1]}, dims = [0, 1, 2] : (#{try type_of(x)}) -> #{broadcast_x}",
      "stablehlo.dot_general %#{first + gathered.size}, %#{first + gathered.size - 1}, batching_dims = [0, 1] x [0, 1], contracting_dims = [2] x [3] : (#{broadcast_x}, #{selected_ty}) -> #{try type_of(t)}",
    ],
  )
end

# `adam_step` over `[p, g, m, v, hyper]`: the new `[p, m, v]` stacked on a new first axis, from the ops of
# `Tensor.adam_composed` over the 7 numbers of `hyper` (`lr, beta1, beta2, eps, decay, correct1, correct2`).
def adam(t: Types.Tensor, args: List[Types.Tensor], a: List[String], first: Int) -> Result[List[String]]
  shape = try type_of(args[0])
  scalar = try tensor_type([], args[0].dtype)
  row = try tensor_type([1], args[0].dtype)
  hyper = try type_of(args[4])
  l = Lines(first: first, defs: [])
  at: List[String] = []
  for i in 0...7
    l = put(l, "stablehlo.slice #{a[4]} #{extents([i], [i + 1])} : (#{hyper}) -> #{row}")
    l = put(l, "stablehlo.reshape #{last(l)} : (#{row}) -> #{scalar}")
    l = put(l, "stablehlo.broadcast_in_dim #{last(l)}, dims = [] : (#{scalar}) -> #{shape}")
    at = at.push(last(l))
  end

  l = put(l, try fill(1.0, args[0]))
  one = last(l)
  l = put(l, "stablehlo.subtract #{one}, #{at[1]} : #{shape}")
  keep1 = last(l)
  l = put(l, "stablehlo.subtract #{one}, #{at[2]} : #{shape}")
  keep2 = last(l)
  l = put(l, "stablehlo.multiply #{at[0]}, #{at[4]} : #{shape}")
  l = put(l, "stablehlo.subtract #{one}, #{last(l)} : #{shape}")
  shrink = last(l)
  l = put(l, "stablehlo.multiply #{a[2]}, #{at[1]} : #{shape}")
  mb = last(l)
  l = put(l, "stablehlo.multiply #{a[1]}, #{keep1} : #{shape}")
  l = put(l, "stablehlo.add #{mb}, #{last(l)} : #{shape}")
  m = last(l)
  l = put(l, "stablehlo.multiply #{a[3]}, #{at[2]} : #{shape}")
  vb = last(l)
  l = put(l, "stablehlo.multiply #{a[1]}, #{a[1]} : #{shape}")
  l = put(l, "stablehlo.multiply #{last(l)}, #{keep2} : #{shape}")
  l = put(l, "stablehlo.add #{vb}, #{last(l)} : #{shape}")
  v = last(l)
  l = put(l, "stablehlo.multiply #{v}, #{at[6]} : #{shape}")
  l = put(l, "stablehlo.sqrt #{last(l)} : #{shape}")
  l = put(l, "stablehlo.add #{last(l)}, #{at[3]} : #{shape}")
  denominator = last(l)
  l = put(l, "stablehlo.multiply #{m}, #{at[5]} : #{shape}")
  l = put(l, "stablehlo.divide #{last(l)}, #{denominator} : #{shape}")
  l = put(l, "stablehlo.multiply #{at[0]}, #{last(l)} : #{shape}")
  step = last(l)
  l = put(l, "stablehlo.multiply #{a[0]}, #{shrink} : #{shape}")
  l = put(l, "stablehlo.subtract #{last(l)}, #{step} : #{shape}")
  p = last(l)
  layer = try tensor_type([1] + args[0].shape, args[0].dtype)
  rows: List[String] = []
  for x in [p, m, v]
    l = put(l, "stablehlo.reshape #{x} : (#{shape}) -> #{layer}")
    rows = rows.push(last(l))
  end

  l = put(
    l,
    "stablehlo.concatenate #{rows.join(", ")}, dim = 0 : (#{[layer, layer, layer].join(", ")}) -> #{try type_of(t)}",
  )

  ok(l.defs)
end

# Part `i` of an `adam_step`: its layer `i` of the stacked result.
def adam_part(t: Types.Tensor, stacked: Types.Tensor, name: String, i: Int, first: Int) -> Result[List[String]]
  layer = try tensor_type([1] + t.shape, t.dtype)
  window = extents([i] + t.shape.map { |_| 0 }, [i + 1] + t.shape)
  ok([
    "stablehlo.slice #{name} #{window} : (#{try type_of(stacked)}) -> #{layer}",
    "stablehlo.reshape %#{first} : (#{layer}) -> #{try type_of(t)}",
  ])
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

# Replace rows along the penultimate dimension. Move that dimension to the front so the updates have the
# scatter layout [indices, window...], then scatter-replace into the original cache.
def set_rows(t: Types.Tensor, args: List[Types.Tensor], a: List[String], first: Int) -> Result[List[String]]
  rank = t.shape.size
  axis = rank - 2
  axes = [axis] + (0...rank).select { |i| i != axis }
  updates_shape = axes.map { |i| args[1].shape[i] }
  indices_shape = [args[2].shape[0], 1]
  indices_type = try tensor_type(indices_shape, args[2].dtype)
  updates_type = try tensor_type(updates_shape, args[1].dtype)
  cache_type = try type_of(t)
  scalar = try tensor_type([], t.dtype)
  window_dims = ints((1...rank).to_a)
  region = [
    "^bb0(%old: #{scalar}, %update: #{scalar}):",
    "        stablehlo.return %update : #{scalar}",
  ].join("\n      ")
  types = "(#{cache_type}, #{indices_type}, #{updates_type}) -> #{cache_type}"
  ok([
    "stablehlo.reshape #{a[2]} : (#{try type_of(args[2])}) -> #{indices_type}",
    "stablehlo.transpose #{a[1]}, dims = #{ints(axes)} : (#{try type_of(args[1])}) -> #{updates_type}",
    "\"stablehlo.scatter\"(#{a[0]}, %#{first}, %#{first + 1}) ({\n      #{region}\n    }) {scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = #{window_dims}, inserted_window_dims = [#{axis}], scatter_dims_to_operand_dims = [#{axis}], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = false} : #{types}",
  ])
end

# Whether one of the indices `name` (the tensor `idx`) is outside `[0, rows)`: an `i1`, the last line. No lines when
# the dtype of `idx` cannot hold such an index: a compare needs its bound in that dtype.
def outside(idx: Types.Tensor, name: String, rows: Int, first: Int) -> Result[List[String]]
  shape = try type_of(idx)
  flags = try tensor_type(idx.shape, Types.Dtype.Bool)
  kind = compare_type(idx.dtype)
  l = Lines(first: first, defs: [])
  found: List[String] = []
  if idx.dtype != Types.Dtype.U8
    l = put(l, try splat(Types.Values.Int(0), idx.shape, idx.dtype))
    l = put(l, "stablehlo.compare LT, #{name}, #{last(l)}, #{kind} : (#{shape}, #{shape}) -> #{flags}")
    found = found.push(last(l))
  end

  if Types.check_int_range(rows, idx.dtype).ok?
    l = put(l, try splat(Types.Values.Int(rows), idx.shape, idx.dtype))
    l = put(l, "stablehlo.compare GE, #{name}, #{last(l)}, #{kind} : (#{shape}, #{shape}) -> #{flags}")
    found = found.push(last(l))
  end

  return ok([]) if found.empty?
  if found.size == 2
    l = put(l, "stablehlo.or #{found[0]}, #{found[1]} : #{flags}")
  end

  return ok(l.defs) if idx.shape.empty?
  any = last(l)
  l = put(l, try splat(Types.Values.Bool(false), [], Types.Dtype.Bool))
  axes = ints((0...idx.shape.size).to_a)
  ok(
    l.defs.push(
      "stablehlo.reduce(#{any} init: #{last(l)}) applies stablehlo.or across dimensions = #{axes} : (#{flags}, tensor<i1>) -> tensor<i1>",
    ),
  )
end

def put(l: Lines, text: String) -> Lines = l.with(defs: l.defs.push(text))

# The name of the last line of `l`.
def last(l: Lines) -> String = "%#{l.first + l.defs.size - 1}"

def matmul(args: List[Types.Tensor], a: List[String], sig: String) -> Result[List[String]]
  if args[0].dtype != args[1].dtype
    kinds = "#{Types.dtype_name(args[0].dtype)} by #{Types.dtype_name(args[1].dtype)}"
    return error("StableHLO has no emitter for matmul of #{kinds} yet")
  end

  ok(["stablehlo.dot_general #{a[0]}, #{a[1]}, contracting_dims = [1] x [0] : #{sig}"])
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
  ids = try tensor_type(x.shape, Types.Dtype.I32)
  shown = try type_of(x)
  value = try tensor_type([], x.dtype)
  index = try tensor_type([], Types.Dtype.I32)
  body = [
    "^bb0(%a: #{value}, %b: #{value}, %i: #{index}, %j: #{index}):",
    "        %c = stablehlo.compare GT, %a, %b, #{compare_type(x.dtype)} : (#{value}, #{value}) -> tensor<i1>",
    "        stablehlo.return %c : tensor<i1>",
  ].join("\n      ")

  sorted = "\"stablehlo.sort\"(#{name}, %#{first}) ({\n      #{body}\n    })"
  attrs = "{dimension = #{axis} : i64, is_stable = true} : (#{shown}, #{ids}) -> (#{shown}, #{ids})"
  window = extents(x.shape.map { |_| 0 }, x.shape.take(axis) + [k])
  ok([
    "stablehlo.iota dim = #{axis} : #{ids}",
    "#{TWO}#{sorted} #{attrs}",
    "stablehlo.slice %#{first + 1}#1 #{window} : (#{ids}) -> #{try type_of(t)}",
  ])
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

  scalar = try tensor_type([], t.dtype)
  body = [
    "^bb0(%acc: #{scalar}, %upd: #{scalar}):",
    "        %sum = stablehlo.add %acc, %upd : #{scalar}",
    "        stablehlo.return %sum : #{scalar}",
  ].join("\n      ")

  attrs = "scatter_dimension_numbers = #stablehlo.scatter<#{fields.join(", ")}>, indices_are_sorted = false"
  types = try signature([t, args[0], args[1]], try type_of(t))
  ok([
    try fill(0.0, t),
    "\"stablehlo.scatter\"(%#{first}, #{a[0]}, #{a[1]}) ({\n      #{body}\n    }) {#{attrs}, unique_indices = false} : #{types}",
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
  scores = try tensor_type(shape, q.dtype)
  qt = try type_of(q)
  kt = try tensor_type([q.shape[0], heads, args[1].shape[2], q.shape[3]], q.dtype)
  vt = try tensor_type([q.shape[0], heads, args[2].shape[2], args[2].shape[3]], q.dtype)
  batch = "batching_dims = [0, 1] x [0, 1]"
  l = put(
    l,
    "stablehlo.dot_general #{a[0]}, #{keys}, #{batch}, contracting_dims = [3] x [3] : (#{qt}, #{kt}) -> #{scores}",
  )

  l = put(l, try splat(Types.Values.Float(scale), shape, q.dtype))
  l = put(l, "stablehlo.multiply %#{first + l.defs.size - 2}, #{last(l)} : #{scores}")
  if args.size == 4
    mask = args[3]
    if mask.shape != shape
      spread = ints((0...mask.shape.size).to_a.map { |i| shape.size - mask.shape.size + i })
      l = put(l, "stablehlo.broadcast_in_dim #{a[3]}, dims = #{spread} : (#{try type_of(mask)}) -> #{scores}")
      l = put(l, "stablehlo.add %#{first + l.defs.size - 2}, #{last(l)} : #{scores}")
    else
      l = put(l, "stablehlo.add #{last(l)}, #{a[3]} : #{scores}")
    end
  end

  l = try softmax_last(l, last(l), shape, q.dtype)
  l = put(
    l,
    "stablehlo.dot_general #{last(l)}, #{values}, #{batch}, contracting_dims = [3] x [2] : (#{scores}, #{vt}) -> #{try type_of(t)}",
  )

  ok(l.defs)
end

# `x` (keys or values of fewer heads) as one tensor of the query's heads, in the dtype of the query: each head repeated
# for its group.
def attention_side(l: Lines, x: Types.Tensor, q: Types.Tensor, group: Int, name: String) -> Result[Side]
  source = name
  shown = try type_of(x)
  if x.dtype != q.dtype
    shown = try tensor_type(x.shape, q.dtype)
    l = put(l, "stablehlo.convert #{source} : (#{try type_of(x)}) -> #{shown}")
    source = last(l)
  end

  return ok(Side(lines: l, name: source)) if group == 1
  b = x.shape[0]
  spread = try tensor_type([b, x.shape[1], group, x.shape[2], x.shape[3]], q.dtype)
  merged = try tensor_type([b, q.shape[1], x.shape[2], x.shape[3]], q.dtype)
  l = put(l, "stablehlo.broadcast_in_dim #{source}, dims = [0, 1, 3, 4] : (#{shown}) -> #{spread}")
  l = put(l, "stablehlo.reshape #{last(l)} : (#{spread}) -> #{merged}")
  ok(Side(lines: l, name: last(l)))
end

# The softmax over the last axis of `name` (of `shape`): `exp(x - max) / sum(exp(x - max))`. The last line.
def softmax_last(l: Lines, name: String, shape: List[Int], dtype: Types.Dtype) -> Result[Lines]
  q = shape.size - 1
  full = try tensor_type(shape, dtype)
  lead = try tensor_type(shape.take(q), dtype)
  scalar = try tensor_type([], dtype)
  dims = ints((0...q).to_a)
  l = put(l, try fill_scalar("maximum", dtype))
  l = put(
    l,
    "stablehlo.reduce(#{name} init: #{last(l)}) applies stablehlo.maximum across dimensions = [#{q}] : (#{full}, #{scalar}) -> #{lead}",
  )

  l = put(l, "stablehlo.broadcast_in_dim #{last(l)}, dims = #{dims} : (#{lead}) -> #{full}")
  l = put(l, "stablehlo.subtract #{name}, #{last(l)} : #{full}")
  l = put(l, "stablehlo.exponential #{last(l)} : #{full}")
  powers = last(l)
  l = put(l, try fill_scalar("add", dtype))
  l = put(
    l,
    "stablehlo.reduce(#{powers} init: #{last(l)}) applies stablehlo.add across dimensions = [#{q}] : (#{full}, #{scalar}) -> #{lead}",
  )

  l = put(l, "stablehlo.broadcast_in_dim #{last(l)}, dims = #{dims} : (#{lead}) -> #{full}")
  ok(put(l, "stablehlo.divide #{powers}, #{last(l)} : #{full}"))
end

# The exact GELU, `0.5 x (1 + erf(x / sqrt(2)))`, the form `Reference` computes: `erf` is CHLO's (XLA lowers it).
def gelu_erf(t: Types.Tensor, x: String, first: Int) -> Result[List[String]]
  out = try type_of(t)
  ok([
    try fill(Reference.INV_SQRT2, t),
    "stablehlo.multiply #{x}, %#{first} : #{out}",
    "\"chlo.erf\"(%#{first + 1}) : (#{out}) -> #{out}",
    try fill(1.0, t),
    "stablehlo.add %#{first + 3}, %#{first + 2} : #{out}",
    try fill(0.5, t),
    "stablehlo.multiply %#{first + 5}, #{x} : #{out}",
    "stablehlo.multiply %#{first + 6}, %#{first + 4} : #{out}",
  ])
end

# A layer norm (`center`) or an RMS norm of `x` over its last axis, no affine: `centered / sqrt(mean(centered^2) +
# eps)`, where `centered` is `x - mean(x)` for the layer norm and `x` for the RMS norm, as
# `Tensor.layer_norm_composed` and `rms_norm_composed` write them. The sums run in the dtype of `x`.
def normalized(t: Types.Tensor, x: String, eps: Float, center: Bool, first: Int) -> Result[List[String]]
  out = try type_of(t)
  q = t.shape.size - 1
  dims = ints((0...q).to_a)
  lead = try tensor_type(t.shape.take(q), t.dtype)
  l = Lines(first: first, defs: [])
  centered = x
  if center
    l = try last_mean(l, x, t)
    l = put(l, "stablehlo.broadcast_in_dim #{last(l)}, dims = #{dims} : (#{lead}) -> #{out}")
    l = put(l, "stablehlo.subtract #{x}, #{last(l)} : #{out}")
    centered = last(l)
  end

  l = put(l, "stablehlo.multiply #{centered}, #{centered} : #{out}")
  l = try last_mean(l, last(l), t)
  l = put(l, "stablehlo.broadcast_in_dim #{last(l)}, dims = #{dims} : (#{lead}) -> #{out}")
  var = last(l)
  l = put(l, try fill(eps, t))
  l = put(l, "stablehlo.add #{var}, #{last(l)} : #{out}")
  l = put(l, "stablehlo.sqrt #{last(l)} : #{out}")
  l = put(l, "stablehlo.divide #{centered}, #{last(l)} : #{out}")
  ok(l.defs)
end

# The mean of `name` (of the shape of `like`) over the last axis, as the last line of `l`: the shape without that axis.
def last_mean(l: Lines, name: String, like: Types.Tensor) -> Result[Lines]
  q = like.shape.size - 1
  lead = like.shape.take(q)
  scalar = try tensor_type([], like.dtype)
  reduced = try tensor_type(lead, like.dtype)
  l = put(l, try fill_scalar("add", like.dtype))
  l = put(
    l,
    "stablehlo.reduce(#{name} init: #{last(l)}) applies stablehlo.add across dimensions = [#{q}] : (#{try type_of(like)}, #{scalar}) -> #{reduced}",
  )

  total = last(l)
  l = put(l, try splat(Types.Values.Float(like.shape[q].to_f), lead, like.dtype))
  ok(put(l, "stablehlo.divide #{total}, #{last(l)} : #{reduced}"))
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
  grid = try tensor_type([seq, half], x.dtype)
  table = try tensor_type([half], x.dtype)
  l = Lines(first: first, defs: [])
  l = put(l, "stablehlo.convert #{a[1]} : (#{try type_of(args[1])}) -> #{try tensor_type([seq], x.dtype)}")
  l = put(l, "stablehlo.broadcast_in_dim #{last(l)}, dims = [0] : (#{try tensor_type([seq], x.dtype)}) -> #{grid}")
  turns = (0...half).map { |i| decimal(Math.exp(-((2.0 * i.to_f) / dims.to_f) * Math.log(base))) }
  l = put(l, "stablehlo.constant dense<[#{turns.join(", ")}]> : #{table}")
  l = put(l, "stablehlo.broadcast_in_dim #{last(l)}, dims = [1] : (#{table}) -> #{grid}")
  l = put(l, "stablehlo.multiply %#{first + 1}, #{last(l)} : #{grid}")
  angle = last(l)
  l = put(l, "stablehlo.cosine #{angle} : #{grid}")
  cosine = last(l)
  l = put(l, "stablehlo.sine #{angle} : #{grid}")
  if inverse
    l = put(l, "stablehlo.negate #{last(l)} : #{grid}")
  end

  sine = last(l)
  lead = x.shape.take(rank - 1)
  pairs = try tensor_type(lead + [half], x.dtype)
  spread = "dims = [#{rank - 2}, #{rank - 1}] : (#{grid}) -> #{pairs}"
  l = put(l, "stablehlo.broadcast_in_dim #{cosine}, #{spread}")
  cosine = last(l)
  l = put(l, "stablehlo.broadcast_in_dim #{sine}, #{spread}")
  sine = last(l)
  parts: List[String] = []
  part_types: List[String] = []
  if neox
    columns = [[0, half], [half, dims]]
    halves: List[String] = []
    for c in columns
      l = put(l, try columns_of(x, a[0], c[0], c[1]))
      halves = halves.push(last(l))
    end

    l = put(l, "stablehlo.multiply #{halves[0]}, #{cosine} : #{pairs}")
    l = put(l, "stablehlo.multiply #{halves[1]}, #{sine} : #{pairs}")
    l = put(l, "stablehlo.subtract %#{first + l.defs.size - 2}, #{last(l)} : #{pairs}")
    parts = parts.push(last(l))
    l = put(l, "stablehlo.multiply #{halves[1]}, #{cosine} : #{pairs}")
    l = put(l, "stablehlo.multiply #{halves[0]}, #{sine} : #{pairs}")
    l = put(l, "stablehlo.add %#{first + l.defs.size - 2}, #{last(l)} : #{pairs}")
    parts = parts.push(last(l))
    part_types = [pairs, pairs]
  else
    twos = try tensor_type(lead + [half, 2], x.dtype)
    l = put(l, try columns_of(x, a[0], 0, dims))
    l = put(l, "stablehlo.reshape #{last(l)} : (#{try tensor_type(lead + [dims], x.dtype)}) -> #{twos}")
    folded = last(l)
    cells: List[String] = []
    for k in 0...2
      starts = lead.map { |_| 0 } + [0, k]
      ends = lead + [half, k + 1]
      one = try tensor_type(lead + [half, 1], x.dtype)
      l = put(l, "stablehlo.slice #{folded} #{extents(starts, ends)} : (#{twos}) -> #{one}")
      l = put(l, "stablehlo.reshape #{last(l)} : (#{one}) -> #{pairs}")
      cells = cells.push(last(l))
    end

    l = put(l, "stablehlo.multiply #{cells[0]}, #{cosine} : #{pairs}")
    l = put(l, "stablehlo.multiply #{cells[1]}, #{sine} : #{pairs}")
    l = put(l, "stablehlo.subtract %#{first + l.defs.size - 2}, #{last(l)} : #{pairs}")
    even = last(l)
    l = put(l, "stablehlo.multiply #{cells[1]}, #{cosine} : #{pairs}")
    l = put(l, "stablehlo.multiply #{cells[0]}, #{sine} : #{pairs}")
    l = put(l, "stablehlo.add %#{first + l.defs.size - 2}, #{last(l)} : #{pairs}")
    odd = last(l)
    one = try tensor_type(lead + [half, 1], x.dtype)
    rows: List[String] = []
    for r in [even, odd]
      l = put(l, "stablehlo.reshape #{r} : (#{pairs}) -> #{one}")
      rows = rows.push(last(l))
    end

    l = put(l, "stablehlo.concatenate #{rows.join(", ")}, dim = #{rank} : (#{one}, #{one}) -> #{twos}")
    l = put(l, "stablehlo.reshape #{last(l)} : (#{twos}) -> #{try tensor_type(lead + [dims], x.dtype)}")
    parts = [last(l)]
    part_types = [try tensor_type(lead + [dims], x.dtype)]
  end

  if dims < width
    l = put(l, try columns_of(x, a[0], dims, width))
    parts = parts.push(last(l))
    part_types = part_types.push(try tensor_type(lead + [width - dims], x.dtype))
  end

  l = put(
    l,
    "stablehlo.concatenate #{parts.join(", ")}, dim = #{rank - 1} : (#{part_types.join(", ")}) -> #{try type_of(t)}",
  )

  ok(l.defs)
end

# The columns `from...to` of the last axis of `x`.
def columns_of(x: Types.Tensor, name: String, from: Int, to: Int) -> Result[String]
  rank = x.shape.size
  starts = x.shape.map { |_| 0 }.take(rank - 1) + [from]
  ends = x.shape.take(rank - 1) + [to]
  shape = x.shape.take(rank - 1) + [to - from]
  ok("stablehlo.slice #{name} #{extents(starts, ends)} : (#{try type_of(x)}) -> #{try tensor_type(shape, x.dtype)}")
end

# The batched product `[B..., m, k] . [B..., k, n]`: the leading axes are the batch, the rest is `matmul`.
def batch_matmul(args: List[Types.Tensor], a: List[String], sig: String) -> Result[List[String]]
  if args[0].dtype != args[1].dtype
    kinds = "#{Types.dtype_name(args[0].dtype)} by #{Types.dtype_name(args[1].dtype)}"
    return error("StableHLO has no emitter for batch_matmul of #{kinds} yet")
  end

  rank = args[0].shape.size
  batch = ints((0...(rank - 2)).to_a)
  dims = "batching_dims = #{batch} x #{batch}, contracting_dims = [#{rank - 1}] x [#{rank - 2}]"
  ok(["stablehlo.dot_general #{a[0]}, #{a[1]}, #{dims} : #{sig}"])
end

# `x` as another dtype: a conversion truncates a float to an integer, and a boolean is `x != 0`.
def convert(t: Types.Tensor, x: Types.Tensor, name: String, first: Int) -> Result[List[String]]
  from = try type_of(x)
  if t.dtype == Types.Dtype.Bool and x.dtype != Types.Dtype.Bool
    zero = Types.Values.Int(0)
    if Types.is_float_dtype(x.dtype)
      zero = Types.Values.Float(0.0)
    end

    return ok([
      try splat(zero, x.shape, x.dtype),
      "stablehlo.compare NE, #{name}, %#{first}, #{compare_type(x.dtype)} : (#{from}, #{from}) -> #{try type_of(t)}",
    ])
  end

  ok(["stablehlo.convert #{name} : (#{from}) -> #{try type_of(t)}"])
end

# The tensors of `args` joined on a new `axis`: each gets a 1 there, and they are concatenated.
def stack(t: Types.Tensor, args: List[Types.Tensor], a: List[String], axis: Int, first: Int) -> Result[List[String]]
  one = args[0].shape.take(axis) + [1] + args[0].shape.drop(axis)
  layer = try tensor_type(one, args[0].dtype)
  l = Lines(first: first, defs: [])
  rows: List[String] = []
  for i in 0...args.size
    l = put(l, "stablehlo.reshape #{a[i]} : (#{try type_of(args[i])}) -> #{layer}")
    rows = rows.push(last(l))
  end

  l = put(
    l,
    "stablehlo.concatenate #{rows.join(", ")}, dim = #{axis} : (#{rows.map { |_| layer }.join(", ")}) -> #{try type_of(t)}",
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
  l = Lines(first: first, defs: [])
  source = name
  if order != (0...x.shape.size).to_a
    permuted = try tensor_type(order.map { |i| x.shape[i] }, x.dtype)
    l = put(l, "stablehlo.transpose #{source}, dims = #{ints(order)} : (#{try type_of(x)}) -> #{permuted}")
    source = last(l)
  end

  rows = try tensor_type(flat, x.dtype)
  l = put(l, "stablehlo.reshape #{source} : (#{try tensor_type(order.map { |i| x.shape[i] }, x.dtype)}) -> #{rows}")
  source = last(l)
  top = try tensor_type(lead, x.dtype)
  l = put(l, try fill_scalar("maximum", x.dtype))
  l = put(
    l,
    "stablehlo.reduce(#{source} init: #{last(l)}) applies stablehlo.maximum across dimensions = [#{q}] : (#{rows}, #{try tensor_type([], x.dtype)}) -> #{top}",
  )

  l = put(l, "stablehlo.broadcast_in_dim #{last(l)}, dims = #{ints((0...q).to_a)} : (#{top}) -> #{rows}")

  flags = try tensor_type(flat, Types.Dtype.Bool)
  l = put(l, "stablehlo.compare EQ, #{source}, #{last(l)}, #{compare_type(x.dtype)} : (#{rows}, #{rows}) -> #{flags}")

  hit = last(l)
  positions = try tensor_type(flat, t.dtype)
  l = put(l, "stablehlo.iota dim = #{q} : #{positions}")
  index = last(l)
  l = put(l, try splat(Types.Values.Int(block), flat, t.dtype))
  l = put(l, "stablehlo.select #{hit}, #{index}, #{last(l)} : #{flags}, #{positions}")
  chosen = last(l)
  l = put(l, try splat(Types.Values.Int(block), [], t.dtype))
  found = try tensor_type(lead, t.dtype)
  l = put(
    l,
    "stablehlo.reduce(#{chosen} init: #{last(l)}) applies stablehlo.minimum across dimensions = [#{q}] : (#{positions}, #{try tensor_type([], t.dtype)}) -> #{found}",
  )

  if keep_dims
    l = put(l, "stablehlo.reshape #{last(l)} : (#{found}) -> #{try type_of(t)}")
  end

  ok(l.defs)
end

# `0.5 x (1 + tanh(c (x + k x^3)))`, the form `Reference` computes.
def gelu(t: Types.Tensor, x: String, first: Int) -> Result[List[String]]
  out = try type_of(t)
  ok([
    "stablehlo.multiply #{x}, #{x} : #{out}",
    "stablehlo.multiply %#{first}, #{x} : #{out}",
    try fill(Reference.GELU_C, t),
    "stablehlo.multiply %#{first + 2}, %#{first + 1} : #{out}",
    "stablehlo.add #{x}, %#{first + 3} : #{out}",
    try fill(Reference.SQRT2_OVER_PI, t),
    "stablehlo.multiply %#{first + 5}, %#{first + 4} : #{out}",
    "stablehlo.tanh %#{first + 6} : #{out}",
    try fill(1.0, t),
    "stablehlo.add %#{first + 8}, %#{first + 7} : #{out}",
    try fill(0.5, t),
    "stablehlo.multiply %#{first + 10}, #{x} : #{out}",
    "stablehlo.multiply %#{first + 11}, %#{first + 9} : #{out}",
  ])
end

def broadcast(t: Types.Tensor, from: Types.Tensor, name: String) -> Result[List[String]]
  return ok([]) if t.shape == from.shape
  lead = t.shape.size - from.shape.size
  dims = ints((0...from.shape.size).to_a.map { |i| lead + i })
  sig = try signature([from], try type_of(t))
  ok(["stablehlo.broadcast_in_dim #{name}, dims = #{dims} : #{sig}"])
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
  scalar = try tensor_type([], x.dtype)
  reduced = try tensor_type(dropped, x.dtype)
  sig = "(#{try type_of(x)}, #{scalar}) -> #{reduced}"
  apply = "applies stablehlo.#{combine} across dimensions = #{ints(axes)}"
  defs = [try fill_scalar(combine, x.dtype), "stablehlo.reduce(#{name} init: %#{first}) #{apply} : #{sig}"]

  return ok(defs) if !keep_dims
  ok(defs.push("stablehlo.reshape %#{first + 1} : (#{reduced}) -> #{try type_of(t)}"))
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
  sum = try reduction(t, x, name, axes, false, "add", first)
  count = try splat(Types.Values.Float(Types.reduced_count(x.shape, axes).to_f), dropped, x.dtype)
  reduced = try tensor_type(dropped, x.dtype)
  defs = sum + [count, "stablehlo.divide %#{first + 1}, %#{first + 2} : #{reduced}"]
  return ok(defs) if !keep_dims
  ok(defs.push("stablehlo.reshape %#{first + 3} : (#{reduced}) -> #{try type_of(t)}"))
end

def arange(t: Types.Tensor, start: Int, step: Int, first: Int) -> Result[List[String]]
  out = try type_of(t)
  defs = ["stablehlo.iota dim = 0 : #{out}"]
  if step != 1
    scale = try splat(Types.Values.Int(step), t.shape, t.dtype)
    defs = defs + [scale, "stablehlo.multiply %#{first}, %#{first + 1} : #{out}"]
  end

  if start != 0
    last = first + defs.size - 1
    offset = try splat(Types.Values.Int(start), t.shape, t.dtype)
    defs = defs + [offset, "stablehlo.add %#{last}, %#{last + 1} : #{out}"]
  end

  ok(defs)
end

def fill(x: Float, like: Types.Tensor) -> Result[String] = splat(Types.Values.Float(x), like.shape, like.dtype)

def fill_scalar(combine: String, dtype: Types.Dtype) -> Result[String]
  return splat(Types.Values.Float(0.0), [], dtype) if combine == "add"
  ok("stablehlo.constant dense<#{try infinity(dtype, true)}> : #{try tensor_type([], dtype)}")
end

def splat(v: Types.Values, shape: List[Int], dtype: Types.Dtype) -> Result[String]
  ok("stablehlo.constant dense<#{try literal(v, dtype)}> : #{try tensor_type(shape, dtype)}")
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

# `[0:1, 2:5]`: the start and the end of each dimension.
def extents(starts: List[Int], ends: List[Int]) -> String
  "[#{(0...starts.size).map { |i| "#{starts[i]}:#{ends[i]}" }.join(", ")}]"
end

def ints(xs: List[Int]) -> String = "[#{xs.map { |x| x.to_s }.join(", ")}]"
