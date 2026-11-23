## Tracing and reverse-mode gradients (`tensor/DESIGN.md` §10).
##
## A differentiable function is written against this module instead of
## `tensor.agc`: the same names, argument order and checks, over `Expr`, a
## node of a graph, instead of `Tensor`. Calling it with input nodes traces
## it: each op checks shapes and dtypes and returns a new node, and computes
## nothing. `gradients` turns that graph into the graph of its gradients,
## written with the same ops, and `evaluate` runs any of those graphs on the
## inputs' backend through the public `Tensor.*` calls, so every backend stays
## forward-only. `value_and_grad`, `grad` and `eval` are the three entry points.
##
## A node's `id` is unique within a run and grows with creation, so an
## argument always has a smaller id than its user: sorting by id is a
## topological order, and two uses of one value are one node.

import Shapes from "src/torch_shapes"
import Tensor from "tensor"
import Types from "src/types"

## The source of node ids: the one counter every trace of a run shares.
IDS: ATS[String, Int] = ATS.new()

## What a node computes. `Input` is an argument of the traced function and
## `Constant` a tensor captured from outside; `Lt` and `Where` appear only in
## gradient graphs (the backward of `relu`), and have no backward rule.
enum Op
  Input
  Constant(Types.Tensor)
  Full(Float)
  Add
  Sub
  Mul
  Div
  Neg
  Exp
  Log
  Relu
  Matmul
  Transpose
  Sum(List[Int], Bool)
  Mean(List[Int], Bool)
  BroadcastTo
  Reshape
  Lt
  Where
end

## One node of a traced graph: its op, its arguments and the shape and dtype
## it will have. Shape-changing ops (`BroadcastTo`, `Reshape`) read their
## target from `shape`.
struct Expr
  id: Int
  op: Op
  args: List[Expr]
  shape: List[Int]
  dtype: Types.Dtype
end

## The value of a scalar function and its gradient for each argument, in
## argument order.
struct ValueAndGrad
  value: Types.Tensor
  grads: List[Types.Tensor]
end

## The shape `e` will have: no computation.
def shape(e: Expr) -> List[Int] = e.shape

## The dtype `e` will have: no computation.
def dtype(e: Expr) -> Types.Dtype = e.dtype

## A tensor from outside the function as a node: data, labels, any value the
## gradient is not taken for.
def constant(t: Types.Tensor) -> Expr = node(Op.Constant(t), [], t.shape, t.dtype)

## Every element `value`, float dtypes only.
def full(shape: List[Int], value: Float, dtype: Types.Dtype) -> Result[Expr]
  _ = try Types.volume(shape)
  return error("full needs a float dtype, got #{Types.dtype_name(dtype)}") if !Types.is_float_dtype(dtype)
  ok(node(Op.Full(value), [], shape, dtype))
end

## Elementwise `a + b`, float only, exact shapes.
def add(a: Expr, b: Expr) -> Result[Expr] = binary(Op.Add, a, b, "add")

## Elementwise `a - b`, float only, exact shapes.
def sub(a: Expr, b: Expr) -> Result[Expr] = binary(Op.Sub, a, b, "sub")

## Elementwise `a * b`, float only, exact shapes.
def mul(a: Expr, b: Expr) -> Result[Expr] = binary(Op.Mul, a, b, "mul")

## Elementwise `a / b`, float only, exact shapes.
def div(a: Expr, b: Expr) -> Result[Expr] = binary(Op.Div, a, b, "div")

## Elementwise `-a`, float only.
def neg(t: Expr) -> Result[Expr] = unary(Op.Neg, t, "neg")

## Elementwise `exp`, float only.
def exp(t: Expr) -> Result[Expr] = unary(Op.Exp, t, "exp")

## Elementwise natural `log`, float only.
def log(t: Expr) -> Result[Expr] = unary(Op.Log, t, "log")

## Elementwise `max(x, 0)`, float only. Its derivative at `0` is taken as `0`.
def relu(t: Expr) -> Result[Expr] = unary(Op.Relu, t, "relu")

## The rank-2 product `[rows, inner] . [inner, cols]`, float only.
def matmul(a: Expr, b: Expr) -> Result[Expr]
  _ = try same_dtype(a, b, "matmul")
  _ = try float_only(a, "matmul")
  return error("matmul needs rank-2 tensors, got #{a.shape} and #{b.shape}") if a.shape.size != 2 or b.shape.size != 2
  return error("matmul needs #{a.shape[1]} in the inner dimension, got #{b.shape[0]}") if a.shape[1] != b.shape[0]
  ok(node(Op.Matmul, [a, b], [a.shape[0], b.shape[1]], a.dtype))
end

## Reverses every axis.
def transpose(t: Expr) -> Result[Expr] = ok(node(Op.Transpose, [t], t.shape.reverse, t.dtype))

## Sums every element (`none`) or the named `axes`, float only.
def sum(t: Expr, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Expr]
  axs = try reduction_axes(t, axes, "sum")
  ok(node(Op.Sum(axs, keep_dims), [t], Shapes.reduced_shape(t.shape, axs, keep_dims), t.dtype))
end

## `sum` divided by the reduced count, float only.
def mean(t: Expr, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Expr]
  axs = try reduction_axes(t, axes, "mean")
  ok(node(Op.Mean(axs, keep_dims), [t], Shapes.reduced_shape(t.shape, axs, keep_dims), t.dtype))
end

## Views `t` as `shape`: trailing axes align, size-1 dimensions widen.
def broadcast_to(t: Expr, shape: List[Int]) -> Result[Expr]
  _ = try Types.check_broadcast(t.shape, shape, "broadcast_to")
  ok(node(Op.BroadcastTo, [t], shape, t.dtype))
end

## A tensor of `shape` with the same values: the volumes must match.
def reshape(t: Expr, shape: List[Int]) -> Result[Expr]
  n = try Types.volume(shape)
  have = try Types.volume(t.shape)
  return error("reshape needs #{have} elements for #{shape}, got #{n}") if n != have
  ok(node(Op.Reshape, [t], shape, t.dtype))
end

## The graph of the gradient of the scalar `out` for each node of `wrt`, in
## order: a node `out` does not depend on gets zeros of its shape. `out` is a
## float node of one element.
def gradients(out: Expr, wrt: List[Expr]) -> Result[List[Expr]]
  _ = try float_only(out, "gradients")
  return error("gradients needs a scalar output, got shape #{out.shape}") if try Types.volume(out.shape) != 1
  nodes = reachable([out])
  wanted = Set.new(wrt.map { |w| w.id })
  needed: Set[Int] = Set.new([])
  for e in nodes
    if (wanted.include?(e.id) or e.args.any? { |a| needed.include?(a.id) })
      needed = needed.add(e.id)
    end
  end

  cts: Map[Int, Expr] = {}
  cts = cts.put(out.id, try full(out.shape, 1.0, out.dtype))
  for e in nodes.reverse
    case cts.get(e.id)
      some(ct) -> cts = try propagate(e, ct, needed, cts)
      none -> nil
    end
  end

  out_grads: List[Expr] = []
  for w in wrt
    case cts.get(w.id)
      some(g) -> out_grads = out_grads.push(g)
      none -> out_grads = out_grads.push(try full(w.shape, 0.0, w.dtype))
    end
  end

  ok(out_grads)
end

## Runs the graphs of `outs` on the backend of `args`: `env` holds the value
## of every node already known (the inputs), keyed by node id.
def evaluate(outs: List[Expr], env: Map[Int, Types.Tensor], backend: Types.Backend) -> Result[List[Types.Tensor]]
  values = env
  for e in reachable(outs)
    if !values.key?(e.id)
      values = values.put(e.id, try apply(e, values, backend))
    end
  end

  ok(outs.map { |o| values[o.id] })
end

## Traces `f` over `args` and runs it: the forward value, as `Tensor.*` calls
## would give it.
def eval(args: List[Types.Tensor], f: Fn[List[Expr], Result[Expr]]) -> Result[Types.Tensor]
  inputs = try inputs_for(args)
  out = try f.call(inputs)
  values = try evaluate([out], bind(inputs, args), args[0].backend)
  ok(values[0])
end

## The value of the scalar function `f` at `args` and its gradient for every
## argument. The trace is built once, the gradient graph is derived from it,
## and one evaluation runs both, sharing the forward values.
def value_and_grad(args: List[Types.Tensor], f: Fn[List[Expr], Result[Expr]]) -> Result[ValueAndGrad]
  inputs = try inputs_for(args)
  out = try f.call(inputs)
  grads = try gradients(out, inputs)
  values = try evaluate([out] + grads, bind(inputs, args), args[0].backend)
  ok(ValueAndGrad(value: values[0], grads: values.drop(1)))
end

## The gradient of the scalar function `f` at `args`, one per argument.
def grad(args: List[Types.Tensor], f: Fn[List[Expr], Result[Expr]]) -> Result[List[Types.Tensor]]
  both = try value_and_grad(args, f)
  ok(both.grads)
end

private

## A new node: the next id of the run.
def node(op: Op, args: List[Expr], shape: List[Int], dtype: Types.Dtype) -> Expr
  Expr(id: IDS.add("next", 1), op: op, args: args, shape: shape, dtype: dtype)
end

## An elementwise op of two operands of one shape and one float dtype.
def binary(op: Op, a: Expr, b: Expr, name: String) -> Result[Expr]
  return error("#{name} needs two tensors of one shape, got #{a.shape} and #{b.shape}") if a.shape != b.shape
  _ = try same_dtype(a, b, name)
  _ = try float_only(a, name)
  ok(node(op, [a, b], a.shape, a.dtype))
end

## An elementwise op of one float operand.
def unary(op: Op, t: Expr, name: String) -> Result[Expr]
  _ = try float_only(t, name)
  ok(node(op, [t], t.shape, t.dtype))
end

## Two nodes share their dtype.
def same_dtype(a: Expr, b: Expr, op: String) -> Result[Bool]
  if a.dtype != b.dtype
    return error("#{op} needs two #{Types.dtype_name(a.dtype)} tensors, got #{Types.dtype_name(b.dtype)}")
  end

  ok(true)
end

## `t` computes: float dtypes only.
def float_only(t: Expr, op: String) -> Result[Bool]
  return error("#{op} needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  ok(true)
end

## The ascending axes a reduction of `t` names: every axis for `none`.
def reduction_axes(t: Expr, axes: Optional[List[Int]], op: String) -> Result[List[Int]]
  _ = try float_only(t, op)
  axs = try Shapes.normalize_axes(axes, t.shape.size, op)
  ok(axs.sort)
end

## One `Input` node per argument, with its shape and dtype; every argument on
## one backend.
def inputs_for(args: List[Types.Tensor]) -> Result[List[Expr]]
  return error("a traced function needs at least one argument") if args.empty?
  for t in args
    return error("a traced function needs its arguments on one backend") if t.backend != args[0].backend
  end

  ok(args.map { |t| node(Op.Input, [], t.shape, t.dtype) })
end

## The value of each input node, keyed by its id.
def bind(inputs: List[Expr], args: List[Types.Tensor]) -> Map[Int, Types.Tensor]
  env: Map[Int, Types.Tensor] = {}
  for i in 0...inputs.size
    env = env.put(inputs[i].id, args[i])
  end

  env
end

## Every node `outs` depend on, each once, in id order: a topological order,
## since an argument is always made before its user.
def reachable(outs: List[Expr]) -> List[Expr]
  seen: Set[Int] = Set.new([])
  found: List[Expr] = []
  work = outs
  while !work.empty?
    e = work.last
    work = work.take(work.size - 1)
    if !seen.include?(e.id)
      seen = seen.add(e.id)
      found = found.push(e)
      work = work + e.args
    end
  end

  found.sort_by { |e| e.id }
end

## `cts` with the cotangent `ct` of `e` added to each argument of `e` that
## leads to a node the gradient is taken for.
def propagate(e: Expr, ct: Expr, needed: Set[Int], cts: Map[Int, Expr]) -> Result[Map[Int, Expr]]
  out = cts
  for i in 0...e.args.size
    a = e.args[i]
    if needed.include?(a.id)
      out = out.put(a.id, try accumulate(out.get(a.id), try vjp(e, ct, i)))
    end
  end

  ok(out)
end

## The sum of the cotangents reaching one node.
def accumulate(prev: Optional[Expr], part: Expr) -> Result[Expr]
  case prev
    some(p) -> add(p, part)
    none -> ok(part)
  end
end

## The cotangent of argument `i` of `e`, given the cotangent `ct` of `e`:
## each op's backward rule, written with the traced ops themselves.
def vjp(e: Expr, ct: Expr, i: Int) -> Result[Expr]
  x = e.args[i]
  case e.op
    Op.Add -> ok(ct)
    Op.Sub ->
      return ok(ct) if i == 0
      neg(ct)
    Op.Mul -> mul(ct, e.args[1 - i])
    Op.Div ->
      return div(ct, e.args[1]) if i == 0
      neg(try mul(ct, try div(e, e.args[1])))
    Op.Neg -> neg(ct)
    Op.Exp -> mul(ct, e)
    Op.Log -> div(ct, x)
    Op.Relu ->
      zeros = try full(x.shape, 0.0, x.dtype)
      where(try lt(zeros, x), ct, zeros)
    Op.Matmul ->
      return matmul(ct, try transpose(e.args[1])) if i == 0
      matmul(try transpose(e.args[0]), ct)
    Op.Transpose -> transpose(ct)
    Op.Sum(axes, keep_dims) -> unreduce(ct, x.shape, axes, keep_dims)
    Op.Mean(axes, keep_dims) ->
      spread = try unreduce(ct, x.shape, axes, keep_dims)
      count = Shapes.reduced_count(x.shape, axes).to_f
      mul(spread, try full(x.shape, 1.0 / count, x.dtype))
    Op.BroadcastTo -> unbroadcast(ct, x.shape)
    Op.Reshape -> reshape(ct, x.shape)
    _ -> error("node #{e.id} has no backward rule")
  end
end

## The backward of a reduction: `ct` back to the input's rank, then widened
## over the reduced axes.
def unreduce(ct: Expr, shape: List[Int], axes: List[Int], keep_dims: Bool) -> Result[Expr]
  return broadcast_to(ct, shape) if keep_dims
  broadcast_to(try reshape(ct, Shapes.reduced_shape(shape, axes, true)), shape)
end

## The backward of `broadcast_to`: `ct` summed over the axes the broadcast
## added or widened from 1, then shaped as the original.
def unbroadcast(ct: Expr, from: List[Int]) -> Result[Expr]
  lead = ct.shape.size - from.size
  axes: List[Int] = (0...lead).to_a
  for i in 0...from.size
    if from[i] == 1 and ct.shape[lead + i] != 1
      axes = axes.push(lead + i)
    end
  end

  return reshape(ct, from) if axes.empty?
  reshape(try sum(ct, some(axes), true), from)
end

## Elementwise `a < b` as a `Bool` node.
def lt(a: Expr, b: Expr) -> Result[Expr]
  _ = try same_dtype(a, b, "lt")
  ok(node(Op.Lt, [a, b], a.shape, Types.Dtype.Bool))
end

## `cond ? a : b` elementwise.
def where(cond: Expr, a: Expr, b: Expr) -> Result[Expr] = ok(node(Op.Where, [cond, a, b], a.shape, a.dtype))

## The value of `e` from the values of its arguments, by the public API.
def apply(e: Expr, values: Map[Int, Types.Tensor], backend: Types.Backend) -> Result[Types.Tensor]
  args = e.args.map { |a| values[a.id] }
  case e.op
    Op.Input -> error("node #{e.id} is an input of another trace")
    Op.Constant(t) -> ok(t)
    Op.Full(x) -> Tensor.full(e.shape, Types.Values.Float(x), e.dtype, backend)
    Op.Add -> Tensor.add(args[0], args[1])
    Op.Sub -> Tensor.sub(args[0], args[1])
    Op.Mul -> Tensor.mul(args[0], args[1])
    Op.Div -> Tensor.div(args[0], args[1])
    Op.Neg -> Tensor.neg(args[0])
    Op.Exp -> Tensor.exp(args[0])
    Op.Log -> Tensor.log(args[0])
    Op.Relu -> Tensor.relu(args[0])
    Op.Matmul -> Tensor.matmul(args[0], args[1])
    Op.Transpose -> Tensor.transpose(args[0])
    Op.Sum(axes, keep_dims) -> Tensor.sum(args[0], some(axes), keep_dims)
    Op.Mean(axes, keep_dims) -> Tensor.mean(args[0], some(axes), keep_dims)
    Op.BroadcastTo -> Tensor.broadcast_to(args[0], e.shape)
    Op.Reshape -> Tensor.reshape(args[0], e.shape)
    Op.Lt -> Tensor.lt(args[0], args[1])
    Op.Where -> Tensor.where(args[0], args[1], args[2])
  end
end
