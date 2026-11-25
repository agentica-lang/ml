## A traced graph as StableHLO text (`tensor/DESIGN.md` §12): one `func.func @main` that takes the inputs, then the
## lifted constants, and returns the outputs. It is the second lowering of a trace, beside `lower.agc`, and it knows no
## backend or runtime: the text is what XLA (PJRT) and IREE read.
##
## Shapes are static and every elementwise op takes operands of one shape, as the `Tensor.*` wrappers check, so the
## text needs no implicit broadcast: `broadcast_to` is a `broadcast_in_dim`, and a scalar of an op is a splat constant.
## An op the emitter does not know is an error naming it.

import Graph from "graph"
import Lower from "lower"
import Reference from "reference"
import Types from "types"

## The program: its text, and the tensors of its constants in the order of the arguments that follow the inputs.
struct Module
  text: String
  constants: List[Types.Tensor]
  ## The positions of the inputs the program updates in place (`emit` with `donate`), each aliased to its output.
  donated: List[Int] = []
  ## True when the program returns one more output after the traced ones (`emit` with `checked`): an `i1` that is
  ## true when an index of a `gather` is outside the rows of its source.
  checked: Bool = false
end

## The module of the traced tensors `outs` over the input nodes `inputs` (traced `Input` tensors, in argument order).
## With `donate`, the moments an `adam_step` updates in place (`Lower.adam_moments`) are aliased to their outputs
## (`tf.aliasing_output`), so XLA writes the new moment over the old one: a run consumes those inputs. With `checked`,
## a program with a `gather` also answers whether one of its indices is outside the rows (`Module.checked`), which
## XLA would clamp: the caller makes that the error the other backends give.
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
        defs = try define(t, n, n.args.map { |a| names[Graph.id_of(a)] }, next)
        for i in 0...defs.size
          body = body.push("    %#{next + i} = #{defs[i]}")
        end

        next = next + defs.size
        if defs.empty?
          names = names.put(n.id, names[Graph.id_of(n.args[0])])
        else
          names = names.put(n.id, "%#{next - 1}")
        end

        if checked and gather?(n)
          lines = try outside(n.args[1], names[Graph.id_of(n.args[1])], n.args[0].shape[0], next)
          for i in 0...lines.size
            body = body.push("    %#{next + i} = #{lines[i]}")
          end

          next = next + lines.size
          flags = flags.push("%#{next - 1}")
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
    ["    func.return #{results.join(", ")} : #{types.join(", ")}", "  }", "}", ""]
  ).join("\n")

  ok(Module(text: text, constants: constants, donated: aliased.keys.sort, checked: !flags.empty?))
end

private

# The lines of an op that takes several, the first named `%first`.
struct Lines
  first: Int
  defs: List[String]
end

# One text per SSA value of the node `t`, the first named `%first`; none means the node is its first argument.
def define(t: Types.Tensor, n: Types.Node, a: List[String], first: Int) -> Result[List[String]]
  out = try type_of(t)
  sig = try signature(n.args, out)
  case n.op
    Types.Op.Full(v) -> ok([try splat(v, t.shape, t.dtype)])
    Types.Op.Add -> ok(["stablehlo.add #{a[0]}, #{a[1]} : #{out}"])
    Types.Op.Sub -> ok(["stablehlo.subtract #{a[0]}, #{a[1]} : #{out}"])
    Types.Op.Mul -> ok(["stablehlo.multiply #{a[0]}, #{a[1]} : #{out}"])
    Types.Op.Div -> ok(["stablehlo.divide #{a[0]}, #{a[1]} : #{out}"])
    Types.Op.Neg -> ok(["stablehlo.negate #{a[0]} : #{out}"])
    Types.Op.Exp -> ok(["stablehlo.exponential #{a[0]} : #{out}"])
    Types.Op.Log -> ok(["stablehlo.log #{a[0]} : #{out}"])
    Types.Op.Sqrt -> ok(["stablehlo.sqrt #{a[0]} : #{out}"])
    Types.Op.Tanh -> ok(["stablehlo.tanh #{a[0]} : #{out}"])
    Types.Op.Relu -> ok([try fill(0.0, t), "stablehlo.maximum #{a[0]}, %#{first} : #{out}"])
    Types.Op.Gelu -> gelu(t, a[0], first)
    Types.Op.StopGrad -> ok([])
    Types.Op.Matmul -> matmul(n.args, a, sig)
    Types.Op.Transpose -> ok(["stablehlo.transpose #{a[0]}, dims = #{ints((0...t.shape.size).to_a.reverse)} : #{sig}"])
    Types.Op.Reshape -> ok(["stablehlo.reshape #{a[0]} : #{sig}"])
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
    Types.Op.AdamStep -> adam(t, n.args, a, first)
    Types.Op.AdamPart(i) -> adam_part(t, n.args[0], a[0], i, first)
    _ -> error("StableHLO has no emitter for #{Types.op_name(n.op)} yet")
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
  l = put(l, "stablehlo.multiply #{a[1]}, #{keep1} : #{shape}")
  l = put(l, "stablehlo.add %#{first + l.defs.size - 2}, #{last(l)} : #{shape}")
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

def gather?(n: Types.Node) -> Bool
  case n.op
    Types.Op.Gather -> true
    _ -> false
  end
end

# Whether one of the indices `name` (the tensor `idx`) is outside `[0, rows)`: an `i1`, the last line.
def outside(idx: Types.Tensor, name: String, rows: Int, first: Int) -> Result[List[String]]
  shape = try type_of(idx)
  flags = try tensor_type(idx.shape, Types.Dtype.Bool)
  kind = compare_type(idx.dtype)
  lines = [
    try splat(Types.Values.Int(0), idx.shape, idx.dtype),
    try splat(Types.Values.Int(rows), idx.shape, idx.dtype),
    "stablehlo.compare LT, #{name}, %#{first}, #{kind} : (#{shape}, #{shape}) -> #{flags}",
    "stablehlo.compare GE, #{name}, %#{first + 1}, #{kind} : (#{shape}, #{shape}) -> #{flags}",
    "stablehlo.or %#{first + 2}, %#{first + 3} : #{flags}",
  ]

  return ok(lines) if idx.shape.empty?
  axes = ints((0...idx.shape.size).to_a)
  ok(
    lines +
    [
      try splat(Types.Values.Bool(false), [], Types.Dtype.Bool),
      "stablehlo.reduce(%#{first + 4} init: %#{first + 5}) applies stablehlo.or across dimensions = #{axes} : (#{flags}, tensor<i1>) -> tensor<i1>",
    ],
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
  end
end

# `[0:1, 2:5]`: the start and the end of each dimension.
def extents(starts: List[Int], ends: List[Int]) -> String
  "[#{(0...starts.size).map { |i| "#{starts[i]}:#{ends[i]}" }.join(", ")}]"
end

def ints(xs: List[Int]) -> String = "[#{xs.map { |x| x.to_s }.join(", ")}]"
