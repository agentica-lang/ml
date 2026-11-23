## Tracing and reverse-mode gradients (`tensor/DESIGN.md` §10).
##
## A function written against `tensor.agc` is traced by calling it with
## tensors of the `Trace` backend: each `Tensor.*` call checks its arguments as
## it always does and records a node instead of computing. `gradients` turns
## the graph into the graph of its gradients, built with the same `Tensor.*`
## calls, and `evaluate` runs any of those graphs on a real backend through
## the public API, so every backend stays forward-only. `value_and_grad`,
## `grad`, `eval` and `eval_many` are the entry points; a tensor from outside
## enters a trace through `constant`.

import Graph from "src/graph"
import Lower from "src/lower"
import Tensor from "tensor"
import Types from "src/types"

## The constants of `gelu_tanh_approx` (`src/reference.agc`).
SQRT2_OVER_PI = 0.7978845608028654
GELU_CUBIC = 0.044715

## The value of a scalar function and its gradient for each argument, in
## argument order.
struct ValueAndGrad
  value: Types.Tensor
  grads: List[Types.Tensor]
end

## A tensor from outside the function as a node of the trace: data, labels,
## any value the gradient is not taken for. A traced tensor is returned as it
## is.
def constant(t: Types.Tensor) -> Types.Tensor
  return t if t.backend == Types.Backend.Trace
  Graph.record(Types.Op.Constant(t), [], t.shape, t.dtype)
end

## The graph of the gradient of the traced scalar `out` for each traced tensor
## of `wrt`, in order: a tensor `out` does not depend on gets zeros of its
## shape. `out` is a float tensor of one element.
def gradients(out: Types.Tensor, wrt: List[Types.Tensor]) -> Result[List[Types.Tensor]]
  _ = try Graph.node_of(out)
  if !Types.is_float_dtype(out.dtype) or try Types.volume(out.shape) != 1
    return error("gradients needs a float scalar output, got #{Types.dtype_name(out.dtype)} #{out.shape}")
  end

  for w in wrt
    _ = try Graph.node_of(w)
  end

  wanted = Set.new(wrt.map { |w| id_of(w) })
  nodes = try reachable([out], Set.new([]))
  needed: Set[Int] = Set.new([])
  for t in nodes
    args = try args_of(t)
    depends = args.any? { |a| needed.include?(id_of(a)) }
    if Types.is_float_dtype(t.dtype) and (wanted.include?(id_of(t)) or depends)
      needed = needed.add(id_of(t))
    end
  end

  cts: Map[Int, Types.Tensor] = {}
  cts = cts.put(id_of(out), try Graph.filled(out.shape, 1.0, out.dtype))
  for t in nodes.reverse
    case cts.get(id_of(t))
      some(ct) -> cts = try propagate(t, ct, needed, cts)
      none -> nil
    end
  end

  out_grads: List[Types.Tensor] = []
  for w in wrt
    case cts.get(id_of(w))
      some(g) -> out_grads = out_grads.push(g)
      none -> out_grads = out_grads.push(try Graph.filled(w.shape, 0.0, w.dtype))
    end
  end

  ok(out_grads)
end

## Runs the traced tensors `outs` on `backend`. `env` holds the value of every
## node already known (the inputs), keyed by node id; the walk stops there.
## A backend that compiles (`Ggml`) runs the whole graph as one graph of its
## own and never calls `Tensor.*`; every other backend computes node by node
## through the public API (§10.3).
def evaluate(
  outs: List[Types.Tensor],
  env: Map[Int, Types.Tensor],
  backend: Types.Backend,
) -> Result[List[Types.Tensor]]
  case backend
    Types.Backend.Ggml(device) -> return Lower.evaluate(outs, env, device)
    _ -> nil
  end

  values = env
  for t in try reachable(outs, Set.new(env.keys))
    values = values.put(id_of(t), try apply(t, values, backend))
  end

  ok(outs.map { |o| values[id_of(o)] })
end

## The number of distinct nodes `outs` reach: what `evaluate` computes, once
## each.
def node_count(outs: List[Types.Tensor]) -> Result[Int]
  nodes = try reachable(outs, Set.new([]))
  ok(nodes.size)
end

## Traces `f` over `args` and runs it on their backend: the value the same
## `Tensor.*` calls give eagerly.
def eval(args: List[Types.Tensor], f: Fn[List[Types.Tensor], Result[Types.Tensor]]) -> Result[Types.Tensor]
  inputs = try inputs_for(args)
  out = try f.call(inputs)
  values = try evaluate([out], bind(inputs, args), args[0].backend)
  ok(values[0])
end

## Traces `f` over `args` and runs every tensor it answers in one evaluation
## on their backend: on `Ggml`, one compiled graph.
def eval_many(
  args: List[Types.Tensor],
  f: Fn[List[Types.Tensor], Result[List[Types.Tensor]]],
) -> Result[List[Types.Tensor]]
  inputs = try inputs_for(args)
  outs = try f.call(inputs)
  evaluate(outs, bind(inputs, args), args[0].backend)
end

## The value of the scalar function `f` at `args` and its gradient for every
## argument. The trace is built once, the gradient graph is derived from it,
## and one evaluation runs both, sharing the forward values.
def value_and_grad(args: List[Types.Tensor], f: Fn[List[Types.Tensor], Result[Types.Tensor]]) -> Result[ValueAndGrad]
  inputs = try inputs_for(args)
  out = try f.call(inputs)
  grads = try gradients(out, inputs)
  values = try evaluate([out] + grads, bind(inputs, args), args[0].backend)
  ok(ValueAndGrad(value: values[0], grads: values.drop(1)))
end

## The gradient of the scalar function `f` at `args`, one per argument.
def grad(args: List[Types.Tensor], f: Fn[List[Types.Tensor], Result[Types.Tensor]]) -> Result[List[Types.Tensor]]
  both = try value_and_grad(args, f)
  ok(both.grads)
end

private

## The id of a traced tensor's node; `-1` for a tensor that is not traced.
def id_of(t: Types.Tensor) -> Int
  case t.handle
    Types.Handle.Traced(n) -> n.id
    _ -> -1
  end
end

## The arguments of a traced tensor's node.
def args_of(t: Types.Tensor) -> Result[List[Types.Tensor]]
  n = try Graph.node_of(t)
  ok(n.args)
end

## One `Input` node per argument, with its shape and dtype; every argument on
## one backend that computes.
def inputs_for(args: List[Types.Tensor]) -> Result[List[Types.Tensor]]
  return error("a traced function needs at least one argument") if args.empty?
  for t in args
    return error("a traced function needs its arguments on one backend") if t.backend != args[0].backend
  end

  return error("a traced function needs computed arguments, got traced ones") if args[0].backend == Types.Backend.Trace
  ok(args.map { |t| Graph.record(Types.Op.Input, [], t.shape, t.dtype) })
end

## The value of each input node, keyed by its id.
def bind(inputs: List[Types.Tensor], args: List[Types.Tensor]) -> Map[Int, Types.Tensor]
  env: Map[Int, Types.Tensor] = {}
  for i in 0...inputs.size
    env = env.put(id_of(inputs[i]), args[i])
  end

  env
end

## Every traced tensor `outs` reach, each node once, in id order (a
## topological order: an argument is made before its user). The walk does not
## enter a node of `known`. `stack[0...top]` is the work left.
def reachable(outs: List[Types.Tensor], known: Set[Int]) -> Result[List[Types.Tensor]]
  seen = known
  found: List[Types.Tensor] = []
  stack = outs
  top = outs.size
  while top > 0
    top -= 1
    t = stack[top]
    args = try args_of(t)
    if !seen.include?(id_of(t))
      seen = seen.add(id_of(t))
      found = found.push(t)
      for a in args
        if top < stack.size
          stack = stack.set(top, a)
        else
          stack = stack.push(a)
        end

        top += 1
      end
    end
  end

  ok(found.sort_by { |t| id_of(t) })
end

## `cts` with the cotangent `ct` of `t` added to each argument of `t` that
## leads to a tensor the gradient is taken for.
def propagate(
  t: Types.Tensor,
  ct: Types.Tensor,
  needed: Set[Int],
  cts: Map[Int, Types.Tensor],
) -> Result[Map[Int, Types.Tensor]]
  n = try Graph.node_of(t)
  out = cts
  for i in 0...n.args.size
    a = n.args[i]
    if needed.include?(id_of(a))
      case try vjp(t, n, ct, i)
        some(part) -> out = out.put(id_of(a), try accumulate(out.get(id_of(a)), part))
        none -> nil
      end
    end
  end

  ok(out)
end

## The sum of the cotangents reaching one node.
def accumulate(prev: Optional[Types.Tensor], part: Types.Tensor) -> Result[Types.Tensor]
  case prev
    some(p) -> Tensor.add(p, part)
    none -> ok(part)
  end
end

## The cotangent of argument `i` of `t` (whose node is `n`), given the
## cotangent `ct` of `t`: each op's backward rule, built with `Tensor.*` on
## traced tensors. `none` cuts the gradient (a comparison, a condition).
def vjp(t: Types.Tensor, n: Types.Node, ct: Types.Tensor, i: Int) -> Result[Optional[Types.Tensor]]
  x = n.args[i]
  case n.op
    Types.Op.Input -> ok(none)
    Types.Op.Constant(_) -> ok(none)
    Types.Op.Full(_) -> ok(none)
    Types.Op.Lt -> ok(none)
    Types.Op.Where -> where_rule(n, ct, i)
    Types.Op.Add -> ok(some(ct))
    Types.Op.Sub -> part(sub_rule(ct, i))
    Types.Op.Mul -> part(Tensor.mul(ct, n.args[1 - i]))
    Types.Op.Div -> part(div_rule(t, n, ct, i))
    Types.Op.Neg -> part(Tensor.neg(ct))
    Types.Op.Exp -> part(Tensor.mul(ct, t))
    Types.Op.Log -> part(Tensor.div(ct, x))
    Types.Op.Relu -> part(relu_rule(x, ct))
    Types.Op.Matmul -> part(matmul_rule(n, ct, i))
    Types.Op.Transpose -> part(Tensor.transpose(ct))
    Types.Op.Sum(axes, keep_dims) -> part(unreduce(ct, x.shape, axes, keep_dims))
    Types.Op.Mean(axes, keep_dims) -> part(mean_rule(ct, x, axes, keep_dims))
    Types.Op.BroadcastTo -> part(unbroadcast(ct, x.shape))
    Types.Op.Reshape -> part(Tensor.reshape(ct, x.shape))
    Types.Op.Copy -> ok(some(ct))
    Types.Op.Contiguous -> ok(some(ct))
    # Every backend that computes today lives on `Cpu`, so a move to `Cpu`
    # stays on its device.
    Types.Op.ToDevice(Types.Device.Cpu) -> ok(some(ct))
    Types.Op.ToDevice(_) -> error("to_device has no backward rule across devices yet")
    Types.Op.Eq -> ok(none)
    Types.Op.Argmax(_, _) -> ok(none)
    Types.Op.Arange(_, _, _) -> ok(none)
    Types.Op.StopGrad -> ok(none)
    Types.Op.Sqrt -> part(Tensor.div(ct, try Tensor.add(t, t)))
    Types.Op.Pow -> no_rule("pow")
    Types.Op.Sigmoid -> part(Tensor.mul(ct, try Tensor.mul(t, try one_minus(t))))
    Types.Op.Tanh -> part(Tensor.mul(ct, try one_minus(try Tensor.mul(t, t))))
    Types.Op.Gelu -> part(Tensor.mul(ct, try gelu_slope(x)))
    Types.Op.GeluErf -> no_rule("gelu_erf")
    Types.Op.Clamp(_, _) -> no_rule("clamp")
    Types.Op.Max(axes, keep_dims) -> part(max_rule(t, x, ct, axes, keep_dims))
    Types.Op.BatchMatmul -> no_rule("batch_matmul")
    Types.Op.Squeeze(_) -> part(Tensor.reshape(ct, x.shape))
    Types.Op.Unsqueeze(_) -> part(Tensor.reshape(ct, x.shape))
    Types.Op.Slice(starts, _) -> part(slice_rule(ct, x.shape, starts))
    Types.Op.Concat(axis) -> part(concat_rule(n, ct, axis, i))
    Types.Op.Stack(axis) -> part(stack_rule(ct, x.shape, axis, i))
    Types.Op.Gather -> part(gather_rule(n.args[0], n.args[1], ct))
    Types.Op.ToDtype -> no_rule("to_dtype")
    Types.Op.Conv1d(_, _) -> no_rule("conv1d")
    Types.Op.AvgPool1d(_, _, _) -> no_rule("avg_pool1d")
  end
end

## The answer of an op the gradient has to pass through but has no rule for.
def no_rule(op: String) -> Result[Optional[Types.Tensor]] = error("#{op} has no backward rule yet")

## A rule's tensor as a contribution.
def part(r: Result[Types.Tensor]) -> Result[Optional[Types.Tensor]] = ok(some(try r))

def sub_rule(ct: Types.Tensor, i: Int) -> Result[Types.Tensor]
  return ok(ct) if i == 0
  Tensor.neg(ct)
end

## `d(a/b)/da = 1/b`, `d(a/b)/db = -(a/b)/b`.
def div_rule(t: Types.Tensor, n: Types.Node, ct: Types.Tensor, i: Int) -> Result[Types.Tensor]
  return Tensor.div(ct, n.args[1]) if i == 0
  Tensor.neg(try Tensor.mul(ct, try Tensor.div(t, n.args[1])))
end

## The cotangent where `x > 0`, zero elsewhere (`0` included).
def relu_rule(x: Types.Tensor, ct: Types.Tensor) -> Result[Types.Tensor]
  zeros = try Graph.filled(x.shape, 0.0, x.dtype)
  Tensor.where(try Tensor.lt(zeros, x), ct, zeros)
end

def matmul_rule(n: Types.Node, ct: Types.Tensor, i: Int) -> Result[Types.Tensor]
  return Tensor.matmul(ct, try Tensor.transpose(n.args[1])) if i == 0
  Tensor.matmul(try Tensor.transpose(n.args[0]), ct)
end

## The cotangent of the branch `cond` chose; the condition gets none.
def where_rule(n: Types.Node, ct: Types.Tensor, i: Int) -> Result[Optional[Types.Tensor]]
  return ok(none) if i == 0
  zeros = try Graph.filled(ct.shape, 0.0, ct.dtype)
  return part(Tensor.where(n.args[0], ct, zeros)) if i == 1
  part(Tensor.where(n.args[0], zeros, ct))
end

def mean_rule(ct: Types.Tensor, x: Types.Tensor, axes: List[Int], keep_dims: Bool) -> Result[Types.Tensor]
  spread = try unreduce(ct, x.shape, axes, keep_dims)
  count = Types.reduced_count(x.shape, axes).to_f
  Tensor.div(spread, try Graph.filled(x.shape, count, x.dtype))
end

## The backward of a reduction: `ct` back to the input's rank, then widened
## over the reduced axes.
def unreduce(ct: Types.Tensor, shape: List[Int], axes: List[Int], keep_dims: Bool) -> Result[Types.Tensor]
  return Tensor.broadcast_to(ct, shape) if keep_dims
  Tensor.broadcast_to(try Tensor.reshape(ct, Types.reduced_shape(shape, axes, true)), shape)
end

## The backward of `broadcast_to`: `ct` summed over the axes the broadcast
## added or widened from 1, then shaped as the original.
def unbroadcast(ct: Types.Tensor, from: List[Int]) -> Result[Types.Tensor]
  lead = ct.shape.size - from.size
  axes: List[Int] = (0...lead).to_a
  for i in 0...from.size
    if from[i] == 1 and ct.shape[lead + i] != 1
      axes = axes.push(lead + i)
    end
  end

  return Tensor.reshape(ct, from) if axes.empty?
  Tensor.reshape(try Tensor.sum(ct, some(axes), true), from)
end

## A traced fill of `like`'s shape and dtype with `x`.
def const(like: Types.Tensor, x: Float) -> Result[Types.Tensor] = Graph.filled(like.shape, x, like.dtype)

def one_minus(t: Types.Tensor) -> Result[Types.Tensor] = Tensor.sub(try const(t, 1.0), t)

## The derivative of `gelu_tanh_approx`, `0.5 x (1 + tanh u)` with
## `u = c (x + k x^3)`: `0.5 (1 + tanh u) + 0.5 x (1 - tanh^2 u) c (1 + 3 k x^2)`.
def gelu_slope(x: Types.Tensor) -> Result[Types.Tensor]
  x2 = try Tensor.mul(x, x)
  inner = try Tensor.add(x, try Tensor.mul(try const(x, GELU_CUBIC), try Tensor.mul(x, x2)))
  th = try Tensor.tanh(try Tensor.mul(try const(x, SQRT2_OVER_PI), inner))
  half = try const(x, 0.5)
  outer = try Tensor.mul(half, try Tensor.add(try const(x, 1.0), th))
  du = try Tensor.mul(
    try const(x, SQRT2_OVER_PI),
    try Tensor.add(try const(x, 1.0), try Tensor.mul(try const(x, 3.0 * GELU_CUBIC), x2)),
  )

  Tensor.add(outer, try Tensor.mul(try Tensor.mul(half, x), try Tensor.mul(try one_minus(try Tensor.mul(th, th)), du)))
end

## The cotangent of each block goes to the positions that hold its maximum,
## split evenly between them when several tie (as `jax.numpy.max` and
## `torch.amax` do).
def max_rule(
  t: Types.Tensor,
  x: Types.Tensor,
  ct: Types.Tensor,
  axes: List[Int],
  keep_dims: Bool,
) -> Result[Types.Tensor]
  hit = try Tensor.eq(x, try unreduce(t, x.shape, axes, keep_dims))
  mask = try Tensor.to_dtype(hit, x.dtype)
  count = try Tensor.broadcast_to(try Tensor.sum(mask, some(axes), true), x.shape)
  spread = try unreduce(ct, x.shape, axes, keep_dims)
  Tensor.div(try Tensor.mul(spread, mask), count)
end

## The backward of `slice`: `ct` padded with zeros back to `shape`, one axis
## at a time.
def slice_rule(ct: Types.Tensor, shape: List[Int], starts: List[Int]) -> Result[Types.Tensor]
  return Graph.filled(shape, 0.0, ct.dtype) if try Types.volume(ct.shape) == 0
  out = ct
  for axis in 0...shape.size
    before = starts[axis]
    after = shape[axis] - before - ct.shape[axis]
    parts = [out]
    if before > 0
      parts = [try Graph.filled(out.shape.set(axis, before), 0.0, ct.dtype)] + parts
    end

    if after > 0
      parts = parts.push(try Graph.filled(out.shape.set(axis, after), 0.0, ct.dtype))
    end

    if parts.size > 1
      out = try Tensor.concat(parts, axis)
    end
  end

  ok(out)
end

## The backward of `concat` for its argument `i`: its window of `ct`.
def concat_rule(n: Types.Node, ct: Types.Tensor, axis: Int, i: Int) -> Result[Types.Tensor]
  offset = n.args.take(i).reduce(0) do |acc, a| acc + a.shape[axis] end
  starts = ct.shape.map { |_| 0 }.set(axis, offset)
  Tensor.slice(ct, starts, ct.shape.set(axis, offset + n.args[i].shape[axis]))
end

## The backward of `stack` for its argument `i`: entry `i` of `ct` along
## `axis`.
def stack_rule(ct: Types.Tensor, shape: List[Int], axis: Int, i: Int) -> Result[Types.Tensor]
  starts = ct.shape.map { |_| 0 }.set(axis, i)
  Tensor.reshape(try Tensor.slice(ct, starts, ct.shape.set(axis, i + 1)), shape)
end

## The backward of `gather`: each row of `ct` added into the row of `source`
## it was read from, as `onehot . ct` where
## `onehot[r, k] = (indices[k] == r)`. A repeated index sums its rows. It
## costs `[rows, k]` grids, which a scatter-add primitive would avoid (ml#25).
def gather_rule(source: Types.Tensor, indices: Types.Tensor, ct: Types.Tensor) -> Result[Types.Tensor]
  rows = source.shape[0]
  k = try Types.volume(indices.shape)
  width = try Types.volume(source.shape.drop(1))
  return Graph.filled(source.shape, 0.0, ct.dtype) if rows == 0 or k == 0 or width == 0
  grid = [rows, k]
  positions = try Tensor.arange(0, rows, 1, indices.dtype, Types.Backend.Trace)
  r = try Tensor.broadcast_to(try Tensor.reshape(positions, [rows, 1]), grid)
  c = try Tensor.broadcast_to(try Tensor.reshape(indices, [1, k]), grid)
  onehot = try Tensor.to_dtype(try Tensor.eq(r, c), ct.dtype)
  Tensor.reshape(try Tensor.matmul(onehot, try Tensor.reshape(ct, [k, width])), source.shape)
end

## The value of `t` on `backend` from the values of its node's arguments, by
## the public API. A lifted constant must live on `backend`.
def apply(t: Types.Tensor, values: Map[Int, Types.Tensor], backend: Types.Backend) -> Result[Types.Tensor]
  n = try Graph.node_of(t)
  r = try compute(t, n, values, backend)
  if r.shape != t.shape or r.dtype != t.dtype
    return error(
      "node #{n.id} (#{Types.op_name(n.op)}) computed #{Types.dtype_name(r.dtype)} #{r.shape}, the trace recorded #{Types.dtype_name(t.dtype)} #{t.shape}",
    )
  end

  ok(r)
end

## The value of the node `n` behind `t`, by the public API.
def compute(
  t: Types.Tensor,
  n: Types.Node,
  values: Map[Int, Types.Tensor],
  backend: Types.Backend,
) -> Result[Types.Tensor]
  args = n.args.map { |a| values[id_of(a)] }
  case n.op
    Types.Op.Input -> error("node #{n.id} is an input of another trace")
    Types.Op.Constant(c) ->
      if c.backend != backend
        return error("a constant on #{Types.backend_name(c.backend)} in a trace run on #{Types.backend_name(backend)}")
      end

      ok(c)
    Types.Op.Full(v) -> Tensor.full(t.shape, v, t.dtype, backend)
    Types.Op.Add -> Tensor.add(args[0], args[1])
    Types.Op.Sub -> Tensor.sub(args[0], args[1])
    Types.Op.Mul -> Tensor.mul(args[0], args[1])
    Types.Op.Div -> Tensor.div(args[0], args[1])
    Types.Op.Neg -> Tensor.neg(args[0])
    Types.Op.Exp -> Tensor.exp(args[0])
    Types.Op.Log -> Tensor.log(args[0])
    Types.Op.Relu -> Tensor.relu(args[0])
    Types.Op.Matmul -> Tensor.matmul(args[0], args[1])
    Types.Op.Transpose -> Tensor.transpose(args[0])
    Types.Op.Sum(axes, keep_dims) -> Tensor.sum(args[0], some(axes), keep_dims)
    Types.Op.Mean(axes, keep_dims) -> Tensor.mean(args[0], some(axes), keep_dims)
    Types.Op.BroadcastTo -> Tensor.broadcast_to(args[0], t.shape)
    Types.Op.Reshape -> Tensor.reshape(args[0], t.shape)
    Types.Op.Lt -> Tensor.lt(args[0], args[1])
    Types.Op.Where -> Tensor.where(args[0], args[1], args[2])
    Types.Op.Sqrt -> Tensor.sqrt(args[0])
    Types.Op.Pow -> Tensor.pow(args[0], args[1])
    Types.Op.Sigmoid -> Tensor.sigmoid(args[0])
    Types.Op.Tanh -> Tensor.tanh(args[0])
    Types.Op.Gelu -> Tensor.gelu_tanh_approx(args[0])
    Types.Op.GeluErf -> Tensor.gelu_erf(args[0])
    Types.Op.Eq -> Tensor.eq(args[0], args[1])
    Types.Op.Clamp(lo, hi) -> Tensor.clamp(args[0], lo, hi)
    Types.Op.Max(axes, keep_dims) -> Tensor.max(args[0], some(axes), keep_dims)
    Types.Op.Argmax(axes, keep_dims) -> Tensor.argmax(args[0], some(axes), keep_dims)
    Types.Op.BatchMatmul -> Tensor.batch_matmul(args[0], args[1])
    Types.Op.Squeeze(axis) -> Tensor.squeeze(args[0], axis)
    Types.Op.Unsqueeze(axis) -> Tensor.unsqueeze(args[0], axis)
    Types.Op.Slice(starts, ends) -> Tensor.slice(args[0], starts, ends)
    Types.Op.Concat(axis) -> Tensor.concat(args, axis)
    Types.Op.Stack(axis) -> Tensor.stack(args, axis)
    Types.Op.Gather -> Tensor.gather(args[0], args[1])
    Types.Op.ToDtype -> Tensor.to_dtype(args[0], t.dtype)
    Types.Op.ToDevice(device) -> Tensor.to_device(args[0], device)
    Types.Op.Copy -> Tensor.copy(args[0])
    Types.Op.Contiguous -> Tensor.contiguous(args[0])
    Types.Op.Conv1d(stride, padding) -> Tensor.conv1d(args[0], args[1], stride, padding)
    Types.Op.AvgPool1d(kernel, stride, padding) -> Tensor.avg_pool1d(args[0], kernel, stride, padding)
    Types.Op.Arange(start, stop, step) -> Tensor.arange(start, stop, step, t.dtype, backend)
    Types.Op.StopGrad -> Tensor.stop_grad(args[0])
  end
end
