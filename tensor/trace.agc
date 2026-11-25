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

import GgmlRaw from "src/ggml_native"
import Graph from "src/graph"
import Lower from "src/lower"
import Stablehlo from "src/stablehlo"
import Tensor from "tensor"
import Types from "src/types"
import Windows from "src/windows"
import Xla from "src/xla"

export Stablehlo.Module

## The constants of `gelu_tanh_approx` (`src/reference.agc`).
SQRT2_OVER_PI = 0.7978845608028654
GELU_CUBIC = 0.044715
## `1 / sqrt(2 pi)`, the peak of the normal density.
INV_SQRT_2PI = 0.3989422804014327

## How many nodes `Diff.worst` lists.
WORST = 5

## How many outputs one ggml graph returns at most is 512: `diff` asks for
## 400 nodes at a time there.
GGML_SLICE = 400

## The value of a scalar function and its gradient for each argument, in
## argument order.
struct ValueAndGrad
  value: Types.Tensor
  grads: List[Types.Tensor]
end

## What a compiled graph takes for one input: the shape, the dtype and the
## backend of the tensor `run` must be given.
struct Spec
  shape: List[Int]
  dtype: Types.Dtype
  backend: Types.Backend
end

## A function traced and lowered once for inputs of fixed shapes (`compile`),
## to run any number of times (`run`).
struct Compiled
  specs: List[Spec]
  inputs: List[Types.Tensor]
  outs: List[Types.Tensor]
  backend: Types.Backend
  program: Optional[Lower.Program]
  ## On ggml, the pin that keeps the graph in the cache (`close`).
  pin: Optional[GgmlRaw.Pin]
  ## On XLA, the compiled program.
  xla: Optional[Xla.Program]
end

## One computed node of a traced function, run on two backends: where it is
## in the graph, what it does, and how far its two values are apart.
struct NodeDiff
  ## The id of the node in the graph.
  node: Ref
  op: String
  shape: List[Int]
  dtype: Types.Dtype
  ## The shapes of the node's arguments.
  inputs: List[List[Int]]
  ## The largest `|a - b|` and `|a - b| / max(|a|, |b|)` over the elements. A
  ## non-finite value that the other side does not have counts as infinite.
  max_abs: Float
  max_rel: Float
  ## Every float element has `|a - b| <= atol + rtol * max(|a|, |b|)`, every integer or `Bool` one is equal, and
  ## a non-finite one is the same on both sides.
  within: Bool
end

## The result of `diff`: how many nodes it compared, the first node, in graph order, whose output is outside the
## tolerance while its inputs are within, and the nodes with the largest `max_abs`, largest first (at most `WORST`).
struct Diff
  nodes: Int
  first: Optional[NodeDiff]
  worst: List[NodeDiff]
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
    n = try Graph.node_of(w)
    case n.op
      Types.Op.Constant(c) -> _ = try not_half_weight(c, "a constant of wrt")
      _ -> nil
    end
  end

  wanted = Set.new(wrt.map { |w| Graph.id_of(w) })
  nodes = try reachable([out], Set.new([]))
  needed: Set[Ref] = Set.new([])
  for t in nodes
    args = try args_of(t)
    depends = args.any? { |a| needed.include?(Graph.id_of(a)) }
    if Types.is_float_dtype(t.dtype) and (wanted.include?(Graph.id_of(t)) or depends)
      needed = needed.add(Graph.id_of(t))
    end
  end

  cts: Map[Ref, Types.Tensor] = {}
  cts = cts.put(Graph.id_of(out), try Graph.filled(out.shape, 1.0, out.dtype))
  for t in nodes.reverse
    case cts.get(Graph.id_of(t))
      some(ct) -> cts = try propagate(t, ct, needed, cts)
      none -> nil
    end
  end

  out_grads: List[Types.Tensor] = []
  for w in wrt
    case cts.get(Graph.id_of(w))
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
## through the public API (§10.3). With `keep` false, a graph `Ggml` compiles
## for this call takes no slot of its cache (a graph used once).
def evaluate(
  outs: List[Types.Tensor],
  env: Map[Ref, Types.Tensor],
  backend: Types.Backend,
  keep: Bool = true,
) -> Result[List[Types.Tensor]]
  case backend
    Types.Backend.Ggml(device) -> return Lower.evaluate(outs, env, device, keep)
    Types.Backend.Xla(_) -> return xla_evaluate(outs, env)
    _ -> nil
  end

  values = env
  for t in try reachable(outs, Set.new(env.keys))
    values = values.put(Graph.id_of(t), try apply(t, values, backend))
  end

  ok(outs.map { |o| values[Graph.id_of(o)] })
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
  out = try System.Profile.measure("ml.trace") do f.call(inputs) end
  values = try evaluate([out], bind(inputs, args), args[0].backend)
  ok(values[0])
end

## Traces `f` over `args` and runs every tensor it answers in one evaluation
## on their backend: on `Ggml`, one compiled graph, kept in the cache unless
## `keep` is false.
def eval_many(
  args: List[Types.Tensor],
  f: Fn[List[Types.Tensor], Result[List[Types.Tensor]]],
  keep: Bool = true,
) -> Result[List[Types.Tensor]]
  inputs = try inputs_for(args)
  outs = try System.Profile.measure("ml.trace") do f.call(inputs) end
  evaluate(outs, bind(inputs, args), args[0].backend, keep)
end

## The spec of the tensor `t`.
def spec_of(t: Types.Tensor) -> Spec = Spec(shape: t.shape, dtype: t.dtype, backend: t.backend)

## Traces `f` once over inputs of `specs` and lowers it, so that `run` neither
## traces nor plans. On `Ggml` the function is one ggml graph, built now and
## pinned in the structural cache of the process (32 graphs), so that the cache
## does not evict it while the pin lives: `run` finds it, and sends the nodes
## again only when another run holds it at that moment. The pin belongs to the
## task that compiles, like any native resource: it ends with that task or at
## `close`, and the graph then stays in the cache as any other, evictable. When
## no slot is free (every graph is in use or pinned), nothing is pinned and the
## function runs as an ordinary graph: the compile does not fail. On the other
## backends the result is the traced graph, evaluated node by node on every
## run.
##
## What `f` reads from outside the inputs enters through `constant`, and is
## fixed: the same tensors on every run. On ggml they stay bound to the pinned
## graph between runs, so a run binds only its inputs, until the pin ends or
## one of them is closed. A run holds their memory from the bind to its end, so
## a constant that its owner closes meanwhile is still there for that run; a
## run that starts after the close is an `error` naming the leaf. Whatever
## changes between runs (the weights of a training step, a KV cache) must be an
## input. `f` may build any graph, a gradient (`gradients`) and an update of the
## parameters included, and answers every tensor `run` gives back.
def compile(specs: List[Spec], f: Fn[List[Types.Tensor], Result[List[Types.Tensor]]]) -> Result[Compiled]
  return error("a compiled function needs at least one input") if specs.empty?
  backend = specs[0].backend
  for s in specs
    return error("a compiled function needs its inputs on one backend") if s.backend != backend
  end

  return error("a compiled function needs computed inputs, got traced ones") if backend == Types.Backend.Trace
  inputs = specs.map { |s| Graph.record(Types.Op.Input, [], s.shape, s.dtype) }
  outs = try System.Profile.measure("ml.trace") do f.call(inputs) end
  case backend
    Types.Backend.Ggml(device) ->
      # The program is all `run` needs: it keeps no trace.
      p = try Lower.program(outs, inputs, device)
      held = try Lower.pin(p)
      ok(
        Compiled(
          specs: specs,
          inputs: [],
          outs: [],
          backend: backend,
          program: some(Lower.pinned(p, held)),
          pin: some(held),
          xla: none,
        ),
      )
    Types.Backend.Xla(_) ->
      m = try Stablehlo.emit(inputs, outs, true, true)
      p = try Xla.compile(m.text, m.constants, outs, m.donated, m.checked)
      ok(Compiled(specs: specs, inputs: [], outs: [], backend: backend, program: none, pin: none, xla: some(p)))
    _ -> ok(Compiled(specs: specs, inputs: inputs, outs: outs, backend: backend, program: none, pin: none, xla: none))
  end
end

## Traces `f` once over inputs of `specs` (the backend of a spec is not read) and writes the traced outputs as one
## StableHLO program (`src/stablehlo.agc`). Its arguments are the inputs, then what `f` reads through `constant`: the
## tensors of those constants are in the module's `constants`, in argument order.
def stablehlo(specs: List[Spec], f: Fn[List[Types.Tensor], Result[List[Types.Tensor]]]) -> Result[Stablehlo.Module]
  return error("a StableHLO program needs at least one input") if specs.empty?
  inputs = specs.map { |s| Graph.record(Types.Op.Input, [], s.shape, s.dtype) }
  Stablehlo.emit(inputs, try unfused(inputs, try f.call(inputs)))
end

## `outs` with each `adam_step` node written as the ops of `Tensor.adam_composed`, which StableHLO has an emitter for;
## `outs` itself when the graph has no such node.
def unfused(inputs: List[Types.Tensor], outs: List[Types.Tensor]) -> Result[List[Types.Tensor]]
  positions: Map[Ref, Int] = {}
  for i in 0...inputs.size
    positions = positions.put(Graph.id_of(inputs[i]), i)
  end

  nodes = try Lower.walk(outs, positions)
  has_step = false
  for t in nodes
    n = try Graph.node_of(t)
    case n.op
      Types.Op.AdamStep -> has_step = true
      _ -> nil
    end
  end

  return ok(outs) if !has_step
  values: Map[Ref, Types.Tensor] = {}
  steps: Map[Ref, List[Types.Tensor]] = {}
  for t in nodes
    n = try Graph.node_of(t)
    args = n.args.map { |a| values.get(Graph.id_of(a), a) }
    case n.op
      Types.Op.AdamStep ->
        steps = steps.put(n.id, try Tensor.adam_composed(args[0], args[1], args[2], args[3], args[4]))
      Types.Op.AdamPart(i) -> values = values.put(n.id, steps[Graph.id_of(n.args[0])][i])
      Types.Op.Input -> values = values.put(n.id, t)
      Types.Op.Constant(_) -> values = values.put(n.id, t)
      _ -> values = values.put(n.id, Graph.record(n.op, args, t.shape, t.dtype))
    end
  end

  ok(outs.map { |o| values[Graph.id_of(o)] })
end

## One run of `compiled` over `args`, which must match its specs (shape,
## dtype and backend, one tensor per spec): the tensors `f` answered, with
## the semantics of `eval_many`. Any number of tasks may run one compiled
## function; on `Ggml` a graph in use by one run is built again for another
## (the nodes are sent again, nothing is traced or planned).
##
## `into` (ggml only) gives, for the outputs it names, the tensor to write
## there instead of a new one: one entry per output, `none` for a new tensor.
## Such a tensor has the shape, dtype and backend of its output, is not an
## input of the run, and is returned as it is. The run overwrites it, so
## whoever else reads it must be done.
##
## On XLA the moments that an `adam_step` updates in place (as on ggml) are
## consumed: XLA writes the new moment over the old one, so the input tensor
## cannot be read after the run (free it), and the output is a new tensor.
## They are consumed even when the run then fails its gather check (an index
## outside the rows of its source).
def run(
  compiled: Compiled,
  args: List[Types.Tensor],
  into: List[Optional[Types.Tensor]] = [],
) -> Result[List[Types.Tensor]]
  try matches(compiled.specs, args)
  case compiled.xla
    some(p) ->
      return error("run: into needs a compiled ggml function") if !into.empty?
      return Xla.execute(p, args)
    none -> nil
  end

  case compiled.program
    some(p) -> Lower.execute(p, args, "run", into)
    none ->
      return error("run: into needs a compiled ggml function") if !into.empty?
      evaluate(compiled.outs, bind(compiled.inputs, args), compiled.backend)
  end
end

## Per output of `compiled`, whether a run answers the input tensor itself, which it wrote in place (the cache of a
## `set_rows`, the moments of an `adam_step`), and not a new tensor: whoever freed the inputs of the run after it must
## not free that one. Only a compiled ggml function does it.
def donated(compiled: Compiled) -> List[Bool]
  case compiled.program
    some(p) -> p.donates.map { |d| d >= 0 }
    none ->
      case compiled.xla
        some(p) -> p.outs.map { |_| false }
        none -> compiled.outs.map { |_| false }
      end
  end
end

## Ends the pin of `compiled` (its graph stays in the cache, evictable). The
## `Compiled` still runs: a run builds the graph again if the cache lost it.
## On XLA it lets go of the compiled program, which stays in the process's
## cache of programs, evictable, as a ggml graph does.
def close(compiled: Compiled) -> Void
  case compiled.pin
    some(p) -> Native.close(p)
    none -> nil
  end

  case compiled.xla
    some(x) -> Native.close(x.program)
    none -> nil
  end
end

## `close` and then the constants of `compiled` (what its function read from
## outside its inputs through `constant`) closed too: for a function that owns
## them, as a log-mel graph owns its basis. A run afterwards is an error naming
## a closed constant. A second call is harmless. Do not call it when the
## constants are tensors somebody else still reads.
def close_constants(compiled: Compiled) -> Void
  close(compiled)
  case compiled.xla
    some(p) ->
      for t in p.constants
        case t.handle
          Types.Handle.Xla(h) -> Native.close(h)
          _ -> nil
        end
      end

    none -> nil
  end

  case compiled.program
    some(program) ->
      for leaf in program.leaves
        case leaf.fixed
          some(t) ->
            case t.handle
              Types.Handle.Ggml(h) -> Native.close(h)
              _ -> nil
            end

          none -> nil
        end
      end

    none -> nil
  end
end

## The value of the scalar function `f` at `args` and its gradient for every
## argument. The trace is built once, the gradient graph is derived from it,
## and one evaluation runs both, sharing the forward values.
def value_and_grad(args: List[Types.Tensor], f: Fn[List[Types.Tensor], Result[Types.Tensor]]) -> Result[ValueAndGrad]
  inputs = try inputs_for(args)
  for i in 0...args.size
    _ = try not_half_weight(args[i], "argument #{i}")
  end

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

## Runs `f` on `args` (any backend that computes) and compares the value of every node it computes on the backends `a`
## and `b`. `f` is traced once; only the nodes `f`'s result reaches are compared. Each backend runs the graph as `eval`
## does, with `args` copied to it when they live elsewhere. The answer lists the first node, in graph order, whose
## output is outside the tolerance while its inputs are within, and the nodes with the largest errors. A float element
## agrees when `|a - b| <= atol + rtol * max(|a|, |b|)`, an integer or `Bool` one only when equal, and a non-finite
## one only when the other side has the same. Everything `f` needs from outside must be an argument: a
## `Trace.constant` lives on one backend, and running it on the other is an `error`. An op that overflows to infinity
## on `Reference` or `Torch` makes the next node an `error` there, which `diff` returns. The comparison runs on the
## host, so it is slow on big activations.
##
## ggml runs the nodes 400 at a time, since one graph returns at most 512 outputs, and each slice runs its ancestors
## again from the inputs. `Reference` and `Torch` run all the nodes at once.
def diff(
  args: List[Types.Tensor],
  a: Types.Backend,
  b: Types.Backend,
  rtol: Float = 0.0001,
  atol: Float = 0.000001,
  f: Fn[List[Types.Tensor], Result[Types.Tensor]],
) -> Result[Diff]
  if !Types.is_finite(rtol) or !Types.is_finite(atol) or rtol < 0.0 or atol < 0.0
    return error("diff takes an rtol and an atol that are finite and 0 or more, got #{rtol} and #{atol}")
  end

  Native.arena do
    inputs = try inputs_for(args)
    out = try f.call(inputs)
    env_a = bind(inputs, try moved(args, a))
    env_b = bind(inputs, try moved(args, b))
    computed: List[Types.Tensor] = []
    for t in try reachable([out], Set.new([]))
      n = try Graph.node_of(t)
      case n.op
        Types.Op.Input -> nil
        _ -> computed = computed.push(t)
      end
    end

    size = slice_limit(a)
    if slice_limit(b) < size
      size = slice_limit(b)
    end

    whole_a = try at_once(computed, env_a, a, size)
    whole_b = try at_once(computed, env_b, b, size)
    found: List[NodeDiff] = []
    start = 0
    while start < computed.size
      part = computed.drop(start).take(size)
      found = found.concat(try slice_diffs(part, [whole_a, whole_b], start, [env_a, env_b], [a, b], rtol, atol))
      start += size
    end

    first: Optional[NodeDiff] = none
    for d in found
      if !d.within and first == none
        first = some(d)
      end
    end

    ok(Diff(nodes: found.size, first: first, worst: found.sort_by { |d| -d.max_abs }.take(WORST)))
  end
end

private

## `evaluate` on XLA: the graph as one StableHLO program, compiled for this call and run once. Its inputs are the
## nodes of `env` that `outs` reach, in node order.
def xla_evaluate(outs: List[Types.Tensor], env: Map[Ref, Types.Tensor]) -> Result[List[Types.Tensor]]
  positions: Map[Ref, Int] = {}
  for id in env.keys
    positions = positions.put(id, positions.size)
  end

  inputs = (try Lower.walk(outs, positions)).select { |t| env.has_key?(Graph.id_of(t)) }
  m = try Stablehlo.emit(inputs, outs, false, true)
  p = try Xla.compile(m.text, m.constants, outs, [], m.checked)
  values = Xla.execute(p, inputs.map { |t| env[Graph.id_of(t)] })
  Native.close(p.program)
  values
end

## A half or quantized weight (`F16` or `Q8_0` on ggml, `Tensor.from_file` with that `dtype`) is read-only:
## no gradient is taken for it, which an `F16` tensor of the other backends keeps.
def not_half_weight(t: Types.Tensor, what: String) -> Result[Bool]
  case t.backend
    Types.Backend.Ggml(_) ->
      if Types.is_packed_dtype(t.dtype)
        kind = "half"
        if t.dtype == Types.Dtype.Q8_0
          kind = "quantized"
        end

        return error(
          "#{what} is a #{kind} weight (#{Types.dtype_name(t.dtype)} on ggml): no gradient is taken for it, weights loaded as F16 or Q8_0 are for inference",
        )
      end

    _ -> nil
  end

  ok(true)
end

## An error unless `args` are the tensors `specs` describe.
def matches(specs: List[Spec], args: List[Types.Tensor]) -> Result[Void]
  return error("a compiled function takes #{specs.size} inputs, got #{args.size}") if args.size != specs.size

  for i in 0...specs.size
    s = specs[i]
    t = args[i]
    if t.shape != s.shape or t.dtype != s.dtype or t.backend != s.backend
      return error(
        "input #{i}: the compiled function takes #{Types.dtype_name(s.dtype)} #{s.shape} on #{Types.backend_name(s.backend)}, " +
        "got #{Types.dtype_name(t.dtype)} #{t.shape} on #{Types.backend_name(t.backend)}",
      )
    end
  end

  ok()
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
def bind(inputs: List[Types.Tensor], args: List[Types.Tensor]) -> Map[Ref, Types.Tensor]
  env: Map[Ref, Types.Tensor] = {}
  for i in 0...inputs.size
    env = env.put(Graph.id_of(inputs[i]), args[i])
  end

  env
end

## Every traced tensor `outs` reach, each node once, in id order (a
## topological order: an argument is made before its user). The walk does not
## enter a node of `known`. `stack[0...top]` is the work left.
def reachable(outs: List[Types.Tensor], known: Set[Ref]) -> Result[List[Types.Tensor]]
  seen = known
  found: List[Types.Tensor] = []
  stack = outs
  top = outs.size
  while top > 0
    top -= 1
    t = stack[top]
    args = try args_of(t)
    if !seen.include?(Graph.id_of(t))
      seen = seen.add(Graph.id_of(t))
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

  ok(found.sort_by { |t| Graph.id_of(t) })
end

## `cts` with the cotangent `ct` of `t` added to each argument of `t` that
## leads to a tensor the gradient is taken for.
def propagate(
  t: Types.Tensor,
  ct: Types.Tensor,
  needed: Set[Ref],
  cts: Map[Ref, Types.Tensor],
) -> Result[Map[Ref, Types.Tensor]]
  n = try Graph.node_of(t)
  out = cts
  for i in 0...n.args.size
    a = n.args[i]
    if needed.include?(Graph.id_of(a))
      case try vjp(t, n, ct, i)
        some(part) -> out = out.put(Graph.id_of(a), try accumulate(out.get(Graph.id_of(a)), part))
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
    Types.Op.GeluErf -> part(Tensor.mul(ct, try gelu_erf_slope(t, x)))
    Types.Op.LayerNorm(eps) -> part(layer_norm_rule(t, x, ct, eps))
    Types.Op.RmsNorm(eps) -> part(rms_norm_rule(t, x, ct, eps))
    Types.Op.Clamp(_, _) -> no_rule("clamp")
    Types.Op.Max(axes, keep_dims) -> part(max_rule(t, x, ct, axes, keep_dims))
    Types.Op.BatchMatmul -> no_rule("batch_matmul")
    Types.Op.Squeeze(_) -> part(Tensor.reshape(ct, x.shape))
    Types.Op.Unsqueeze(_) -> part(Tensor.reshape(ct, x.shape))
    Types.Op.Slice(starts, _) -> part(slice_rule(ct, x.shape, starts))
    Types.Op.Concat(axis) -> part(concat_rule(n, ct, axis, i))
    Types.Op.Stack(axis) -> part(stack_rule(ct, x.shape, axis, i))
    Types.Op.Gather -> part(gather_rule(n.args[0], n.args[1], ct))
    Types.Op.SetRows -> no_rule("set_rows")
    Types.Op.AdamStep -> no_rule("adam_step")
    Types.Op.AdamPart(_) -> no_rule("adam_part")
    Types.Op.TopK(_) -> no_rule("top_ids")
    Types.Op.ScatterAdd(_) -> scatter_add_rule(n, ct, i)
    Types.Op.ToDtype -> no_rule("to_dtype")
    Types.Op.Conv1d(_, _) -> no_rule("conv1d")
    Types.Op.AvgPool1d(_, _, _) -> no_rule("avg_pool1d")
    Types.Op.Conv2d(stride, padding, groups) ->
      part(Windows.conv2d_rule(n.args[0], n.args[1], ct, stride, padding, groups, i))
    Types.Op.MaxPool2d(kernel, stride, padding) -> part(Windows.max_pool2d_rule(x, t, ct, kernel, stride, padding))
    Types.Op.AvgPool2d(kernel, stride, padding) -> part(Windows.avg_pool2d_rule(x, ct, kernel, stride, padding))
    Types.Op.Permute(axes) -> part(Tensor.permute(ct, Types.inverse_axes(axes)))
    Types.Op.Attention(scale) -> attention_rule(n, ct, scale, i)
    Types.Op.Rope(base, dims, neox, inverse) -> rope_rule(n, ct, base, dims, neox, inverse, i)
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
  if i == 0
    if n.args[1].dtype == Types.Dtype.Q8_0
      return error("the gradient of a product with a Q8_0 weight is not supported: the weight is for inference")
    end

    return Tensor.matmul(ct, try Tensor.transpose(n.args[1]))
  end

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
  Tensor.reshape(try sum_in_blocks(ct, axes), from)
end

## `ct` summed over `axes` with the dims kept: a bias `[out, 1, 1]` of an image batch sums axes 0, 2 and 3, which ggml
## cannot reduce at once (a reduction takes a leading or a trailing block of axes), so the leading axes are a sum of
## their own when the rest are the trailing ones.
def sum_in_blocks(ct: Types.Tensor, axes: List[Int]) -> Result[Types.Tensor]
  rank = ct.shape.size
  lead = 0
  while lead < axes.size and axes[lead] == lead
    lead = lead + 1
  end

  rest = axes.drop(lead)
  trailing = (0...rest.size).all? { |i| rest[i] == rank - rest.size + i }
  return Tensor.sum(ct, some(axes), true) if lead == 0 or rest.empty? or !trailing

  Tensor.sum(try Tensor.sum(ct, some(rest), true), some(axes.take(lead)), true)
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

## The derivative of `gelu_erf`, `Phi(x) + x phi(x)`, with `t` its value: `Phi(x) = t / x`, and `0.5` where `x` is `0`;
## `phi(x) = exp(-x^2 / 2) / sqrt(2 pi)`.
def gelu_erf_slope(t: Types.Tensor, x: Types.Tensor) -> Result[Types.Tensor]
  zero = try Tensor.to_dtype(try Tensor.eq(x, try const(x, 0.0)), x.dtype)
  cdf = try Tensor.div(try Tensor.add(t, try Tensor.mul(try const(x, 0.5), zero)), try Tensor.add(x, zero))
  density = try Tensor.mul(
    try const(x, INV_SQRT_2PI),
    try Tensor.exp(try Tensor.mul(try const(x, -0.5), try Tensor.mul(x, x))),
  )

  Tensor.add(cdf, try Tensor.mul(x, density))
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

## The backward of `layer_norm`, with `y = t` its value, `sigma = sqrt(variance + eps)` and the means over the last axis:
## `(ct - mean(ct) - y * mean(ct * y)) / sigma`.
def layer_norm_rule(t: Types.Tensor, x: Types.Tensor, ct: Types.Tensor, eps: Float) -> Result[Types.Tensor]
  last = x.shape.size - 1
  centered = try Tensor.sub(x, try Tensor.broadcast_to(try Tensor.mean(x, some([last]), true), x.shape))
  var = try Tensor.mean(try Tensor.mul(centered, centered), some([last]), true)
  floor = try Tensor.full(var.shape, Types.Values.Float(eps), x.dtype, x.backend)
  sigma = try Tensor.broadcast_to(try Tensor.sqrt(try Tensor.add(var, floor)), x.shape)
  along = try Tensor.broadcast_to(try Tensor.mean(ct, some([last]), true), x.shape)
  with_y = try Tensor.broadcast_to(try Tensor.mean(try Tensor.mul(ct, t), some([last]), true), x.shape)
  Tensor.div(try Tensor.sub(try Tensor.sub(ct, along), try Tensor.mul(t, with_y)), sigma)
end

## The backward of `rms_norm`, with `y = t` its value, `rms = sqrt(mean(x * x) + eps)` and the means over the last axis:
## `(ct - y * mean(ct * y)) / rms`.
def rms_norm_rule(t: Types.Tensor, x: Types.Tensor, ct: Types.Tensor, eps: Float) -> Result[Types.Tensor]
  last = x.shape.size - 1
  ms = try Tensor.mean(try Tensor.mul(x, x), some([last]), true)
  floor = try Tensor.full(ms.shape, Types.Values.Float(eps), x.dtype, x.backend)
  rms = try Tensor.broadcast_to(try Tensor.sqrt(try Tensor.add(ms, floor)), x.shape)
  with_y = try Tensor.broadcast_to(try Tensor.mean(try Tensor.mul(ct, t), some([last]), true), x.shape)
  Tensor.div(try Tensor.sub(ct, try Tensor.mul(t, with_y)), rms)
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
## it was read from, which is `scatter_add`. A repeated index sums its rows.
def gather_rule(source: Types.Tensor, indices: Types.Tensor, ct: Types.Tensor) -> Result[Types.Tensor]
  rows = source.shape[0]
  empty = rows == 0 or try Types.volume(ct.shape) == 0
  return Graph.filled(source.shape, 0.0, ct.dtype) if empty
  Tensor.scatter_add(indices, ct, rows)
end

## The backward of `scatter_add` for its updates: the rows of `ct` at the
## indices, which is `gather`. The indices have no gradient.
def scatter_add_rule(n: Types.Node, ct: Types.Tensor, i: Int) -> Result[Optional[Types.Tensor]]
  return ok(none) if i == 0
  part(Tensor.gather(ct, n.args[0]))
end

## The backward of `attention` for its query (`i` 0), key (1), value (2) or mask (3),
## recomputed from the arguments: `softmax(scale q k^T + mask)` is `p`, the
## value gets `p^T ct`, and with `dp = ct v^T` the scores get `p (dp - sum(dp p))`,
## which `scale` carries on to the query (`. k`) and the key (`^T q`). The
## query heads that share a kv head are folded into the rows, `[B, H, Sq, D]`
## as `[B, Hkv, H / Hkv * Sq, D]`, so every product is rank 4 and the key and
## the value need no repeat: the sum over the group is the sum of the product.
## The mask gets `dscores`, summed over the batch and head axes it broadcast.
def attention_rule(n: Types.Node, ct: Types.Tensor, scale: Float, i: Int) -> Result[Optional[Types.Tensor]]
  q = n.args[0]
  k = n.args[1]
  v = n.args[2]
  kv_heads = k.shape[1]
  group = q.shape[1] / kv_heads
  rows = group * q.shape[2]
  grouped = try Tensor.reshape(q, [q.shape[0], kv_heads, rows, q.shape[3]])
  ct_grouped = try Tensor.reshape(ct, [q.shape[0], kv_heads, rows, v.shape[3]])
  swap = [0, 1, 3, 2]
  scores = try Tensor.batch_matmul(grouped, try Tensor.permute(k, swap))
  scaled = try Tensor.mul(scores, try const(scores, scale))
  biased = case n.args.size
    4 -> try Tensor.add(scaled, try grouped_mask(n.args[3], q, k, group))
    _ -> scaled
  end

  probs = try Tensor.softmax(biased, 3)
  return part(Tensor.batch_matmul(try Tensor.permute(probs, swap), ct_grouped)) if i == 2
  d_probs = try Tensor.batch_matmul(ct_grouped, try Tensor.permute(v, swap))
  dots = try Tensor.sum(try Tensor.mul(d_probs, probs), some([3]), true)
  d_scores = try Tensor.mul(probs, try Tensor.sub(d_probs, try Tensor.broadcast_to(dots, d_probs.shape)))
  if i == 3
    return part(
      unbroadcast(try Tensor.reshape(d_scores, [q.shape[0], q.shape[1], q.shape[2], k.shape[2]]), n.args[3].shape),
    )
  end

  d_scaled = try Tensor.mul(d_scores, try const(d_scores, scale))
  return part(Tensor.reshape(try Tensor.batch_matmul(d_scaled, k), q.shape)) if i == 0
  part(Tensor.batch_matmul(try Tensor.permute(d_scaled, swap), grouped))
end

## The additive mask of `attention` laid over the folded scores `[B, Hkv, H / Hkv
## * Sq, Sk]`: the mask repeated once per head of a group along the query rows,
## or, when it has a mask per head, reshaped to the folded heads.
def grouped_mask(mask: Types.Tensor, q: Types.Tensor, k: Types.Tensor, group: Int) -> Result[Types.Tensor]
  kv_heads = k.shape[1]
  rows = group * q.shape[2]
  target = [q.shape[0], kv_heads, rows, k.shape[2]]
  if mask.shape.size == 4 and mask.shape[1] == q.shape[1] and group > 1
    return Tensor.broadcast_to(try Tensor.reshape(mask, [mask.shape[0], kv_heads, rows, k.shape[2]]), target)
  end

  plain = mask
  if mask.shape.size == 2
    plain = try Tensor.reshape(mask, [1, 1, q.shape[2], k.shape[2]])
  end

  repeated = plain
  if group > 1
    repeated = try Tensor.concat(List.repeat(plain, group), 2)
  end

  Tensor.broadcast_to(repeated, target)
end

## The backward of `rope` for `x`: the opposite rotation. The positions have
## no gradient.
def rope_rule(
  n: Types.Node,
  ct: Types.Tensor,
  base: Float,
  dims: Int,
  neox: Bool,
  inverse: Bool,
  i: Int,
) -> Result[Optional[Types.Tensor]]
  return ok(none) if i == 1
  part(Tensor.rope(ct, n.args[1], base, some(dims), neox, !inverse))
end

## The value of `t` on `backend` from the values of its node's arguments, by
## the public API. A lifted constant must live on `backend`.
def apply(t: Types.Tensor, values: Map[Ref, Types.Tensor], backend: Types.Backend) -> Result[Types.Tensor]
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
  values: Map[Ref, Types.Tensor],
  backend: Types.Backend,
) -> Result[Types.Tensor]
  args = n.args.map { |a| values[Graph.id_of(a)] }
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
    Types.Op.LayerNorm(eps) -> Tensor.layer_norm(args[0], eps)
    Types.Op.RmsNorm(eps) -> Tensor.rms_norm(args[0], eps)
    Types.Op.AdamStep -> Tensor.stack(try Tensor.adam_step(args[0], args[1], args[2], args[3], args[4]), 0)
    Types.Op.AdamPart(i) ->
      Tensor.reshape(try Tensor.slice(args[0], [i] + t.shape.map { |_| 0 }, [i + 1] + t.shape), t.shape)
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
    Types.Op.SetRows -> Tensor.set_rows(args[0], args[1], args[2])
    Types.Op.TopK(k) -> Tensor.top_ids(args[0], k)
    Types.Op.ScatterAdd(rows) -> Tensor.scatter_add(args[0], args[1], rows)
    Types.Op.ToDtype -> Tensor.to_dtype(args[0], t.dtype)
    Types.Op.ToDevice(device) -> Tensor.to_device(args[0], device)
    Types.Op.Copy -> Tensor.copy(args[0])
    Types.Op.Contiguous -> Tensor.contiguous(args[0])
    Types.Op.Conv1d(stride, padding) -> Tensor.conv1d(args[0], args[1], stride, padding)
    Types.Op.AvgPool1d(kernel, stride, padding) -> Tensor.avg_pool1d(args[0], kernel, stride, padding)
    Types.Op.Conv2d(stride, padding, groups) -> Tensor.conv2d(args[0], args[1], stride, padding, groups)
    Types.Op.MaxPool2d(kernel, stride, padding) -> Tensor.max_pool2d(args[0], kernel, stride, padding)
    Types.Op.AvgPool2d(kernel, stride, padding) -> Tensor.avg_pool2d(args[0], kernel, stride, padding)
    Types.Op.Arange(start, stop, step) -> Tensor.arange(start, stop, step, t.dtype, backend)
    Types.Op.StopGrad -> Tensor.stop_grad(args[0])
    Types.Op.Permute(axes) -> Tensor.permute(args[0], axes)
    Types.Op.Attention(scale) ->
      mask: Optional[Types.Tensor] = none
      if args.size > 3
        mask = some(args[3])
      end

      Tensor.attention(args[0], args[1], args[2], mask, some(scale))
    Types.Op.Rope(base, dims, neox, inverse) -> Tensor.rope(args[0], args[1], base, some(dims), neox, inverse)
  end
end

## The most nodes `evaluate` asks of `backend` at once.
def slice_limit(backend: Types.Backend) -> Int
  case backend
    Types.Backend.Ggml(_) -> GGML_SLICE
    _ -> 1000000000
  end
end

## The value of every node of `computed` on `backend`, when `backend` runs them all at once while the other backend
## needs slices of `size`: one evaluation, so its slices do not run the ancestors again. Otherwise empty.
def at_once(
  computed: List[Types.Tensor],
  env: Map[Ref, Types.Tensor],
  backend: Types.Backend,
  size: Int,
) -> Result[List[Types.Tensor]]
  return ok([]) if size >= computed.size or slice_limit(backend) < computed.size

  evaluate(computed, env, backend, false)
end

## The `NodeDiff` of each node of `part`, which starts at `start` of the nodes. `wholes`, `envs` and `backends` hold
## the two sides. The tensors of the slice live in an arena of their own, which ends with the slice.
def slice_diffs(
  part: List[Types.Tensor],
  wholes: List[List[Types.Tensor]],
  start: Int,
  envs: List[Map[Ref, Types.Tensor]],
  backends: List[Types.Backend],
  rtol: Float,
  atol: Float,
) -> Result[List[NodeDiff]]
  Native.arena do
    got: List[List[Types.Tensor]] = []
    for side in 0...2
      if wholes[side].empty?
        got = got.push(try evaluate(part, envs[side], backends[side], false))
      else
        got = got.push(wholes[side].drop(start).take(part.size))
      end
    end

    found: List[NodeDiff] = []
    for i in 0...part.size
      found = found.push(try compare(part[i], got[0][i], got[1][i], rtol, atol))
    end

    ok(found)
  end
end

## `args` on `backend`, copied through the host when they live elsewhere.
def moved(args: List[Types.Tensor], backend: Types.Backend) -> Result[List[Types.Tensor]]
  out: List[Types.Tensor] = []
  for t in args
    if t.backend == backend
      out = out.push(t)
    else
      out = out.push(try Tensor.from_list(try Tensor.to_list(t), t.shape, t.dtype, backend))
    end
  end

  ok(out)
end

## The elements of `t` as floats.
def as_floats(t: Types.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Types.Values.Float(x) -> out = out.push(x)
      Types.Values.Int(i) -> out = out.push(i.to_f)
      Types.Values.Bool(flag) ->
        if flag
          out = out.push(1.0)
        else
          out = out.push(0.0)
        end
    end
  end

  ok(out)
end

## The `NodeDiff` of the traced tensor `t`, whose values on the two backends are `x` and `y`.
def compare(t: Types.Tensor, x: Types.Tensor, y: Types.Tensor, rtol: Float, atol: Float) -> Result[NodeDiff]
  n = try Graph.node_of(t)
  xs = try as_floats(x)
  ys = try as_floats(y)
  return error("node #{n.id}: #{xs.size} values on one backend, #{ys.size} on the other") if xs.size != ys.size
  exact = Types.is_int_dtype(t.dtype) or t.dtype == Types.Dtype.Bool
  infinity = 1.0e308 * 10.0
  max_abs = 0.0
  max_rel = 0.0
  within = true
  for i in 0...xs.size
    u = xs[i]
    v = ys[i]
    if Types.is_finite(u) and Types.is_finite(v)
      gap = (u - v).abs
      scale = bigger(u.abs, v.abs)
      rel = 0.0
      if scale > 0.0
        rel = gap / scale
      end

      max_abs = bigger(max_abs, gap)
      max_rel = bigger(max_rel, rel)
      if exact
        within = within and gap == 0.0
      else
        within = within and gap <= atol + rtol * scale
      end
    elsif !(u == v or (u != u and v != v))
      max_abs = infinity
      max_rel = infinity
      within = false
    end
  end

  ok(
    NodeDiff(
      node: n.id,
      op: Types.op_name(n.op),
      shape: t.shape,
      dtype: t.dtype,
      inputs: n.args.map { |a| a.shape },
      max_abs: max_abs,
      max_rel: max_rel,
      within: within,
    ),
  )
end

def bigger(x: Float, y: Float) -> Float
  return x if x > y
  y
end
