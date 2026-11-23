## `Trace.evaluate` on `Backend.Ggml`: the graph the outputs reach becomes ONE
## ggml graph (`native/ggml_graph.c`), compiled once per structure and run
## once per call (`tensor/DESIGN.md` §7, §10.3).
##
## Every node the outputs reach is sent in id order (a topological order,
## §10.2). The leaves are the bound inputs and the lifted constants, read in
## place on every call; a fill is a one-element leaf whose value is set on every
## call. So the graph depends on the ops, shapes, dtypes and integer
## parameters only, and that is the cache key: a retrace of the same
## function, with new data, new constants or new fill values, reuses the
## compiled graph. Whether a fill is zero is in the key too, because a `where`
## with a zero branch lowers without it.
##
## The C layer computes in F32. A node of another dtype is accepted only when
## F32 holds its values exactly (`exact`), and an op this lowering does not
## know is an error naming it, before anything is compiled.

import Ggml from "ggml"
import GgmlRaw from "ggml_native"
import Graph from "graph"
import Types from "types"

## The op codes of `native/ggml_graph.c`, in its order.
LEAF = 0
FULL = 1
ADD = 2
SUB = 3
MUL = 4
DIV = 5
NEG = 6
EXP = 7
LOG = 8
RELU = 9
SQRT = 10
SIGMOID = 11
TANH = 12
MATMUL = 13
TRANSPOSE = 14
RESHAPE = 15
BROADCAST = 16
SUM = 17
MEAN = 18
MAX = 19
IDENTITY = 20
ARANGE = 21
EQ = 22
LT = 23
WHERE = 24
SLICE = 25
GATHER = 26
GELU_ERF = 27
CONCAT = 28
CONV1D = 29

## One node as the C layer takes it.
struct Lowered
  op: Int
  args: List[Int]
  shape: List[Int]
  ints: List[Int]
end

## Where a leaf of the graph gets its tensor on every run: the input at
## position `input` of the run's inputs (`-1` for none), or `fixed`, a
## constant of the trace that every run binds again.
struct Leaf
  input: Int
  fixed: Optional[Types.Tensor]
end

## A traced graph lowered once: what a run sends to the C layer (the nodes,
## only on a cache miss), the structural key, where each leaf's tensor comes
## from, the fill values and the outputs, in the C layer's order. A program
## never changes, so any number of tasks may run it.
struct Program
  device: Types.Device
  nodes: List[Lowered]
  key: String
  leaves: List[Leaf]
  fills: List[Float]
  outputs: List[Int]
  shapes: List[List[Int]]
  dtypes: List[Types.Dtype]
  keep: Bool
end

## Runs the traced tensors `outs` on `Ggml(device)` as one graph. `env` holds
## the value of every input node, keyed by node id. With `keep` false a graph
## compiled here takes no slot of the structural cache.
def evaluate(
  outs: List[Types.Tensor],
  env: Map[Ref, Types.Tensor],
  device: Types.Device,
  keep: Bool = true,
) -> Result[List[Types.Tensor]]
  ids = env.keys
  program = try System.Profile.measure("ml.plan") do lower(outs, ids, device, keep) end
  inputs: List[Types.Tensor] = []
  for id in ids
    inputs = inputs.push(try bound(env[id], Types.Backend.Ggml(device), "an input"))
  end

  execute(program, inputs, "evaluate")
end

## The program of the traced tensors `outs` over the input nodes `inputs`
## (traced `Input` tensors, in the order a run takes its tensors).
def program(outs: List[Types.Tensor], inputs: List[Types.Tensor], device: Types.Device) -> Result[Program]
  System.Profile.measure("ml.plan") do lower(outs, inputs.map { |t| Graph.id_of(t) }, device, true) end
end

## The graph of `p` in the structural cache, built if it is not there, kept
## out of the cache's eviction for as long as the `Pin` lives (`ggml_native.agc`).
## The pin is owned by the task that makes it, and ends with it or with
## `Native.close`; the graph then stays in the cache as any other.
def pin(p: Program) -> Result[GgmlRaw.Pin]
  code = try Ggml.code_of(p.device, "pin")
  Native.arena do
    lease = try GgmlRaw.acquire(code, p.key, p.nodes.size + 1, true)
    if !GgmlRaw.ready(lease)
      for l in p.nodes
        _ = try GgmlRaw.node(lease, l.op, l.args, l.shape, l.ints)
      end

      try GgmlRaw.compile(lease, p.outputs)
    end

    held = try GgmlRaw.pin(lease)
    Native.close(lease)
    ok(held)
  end
end

## One run of `program` over `inputs`, one tensor per input node, already
## checked against the backend, on the program's device. Its outputs are new
## tensors, except an output that `into` gives a tensor for: that one is
## written there (a tensor of the output's shape, dtype and backend, which
## no input of the run may be). `op` names the call in an error. A tensor that is closed, an input
## or a constant of the trace, is an error naming the leaf; the constants must
## stay open while any task may run the program (nothing holds them between
## the bind and the run).
def execute(
  p: Program,
  inputs: List[Types.Tensor],
  op: String = "run",
  into: List[Optional[Types.Tensor]] = [],
) -> Result[List[Types.Tensor]]
  code = try Ggml.code_of(p.device, op)
  if !into.empty? and into.size != p.shapes.size
    return error("#{op}: into takes one entry per output, #{p.shapes.size}, got #{into.size}")
  end

  Native.arena do
    lease = try System.Profile.measure("ml.acquire") do GgmlRaw.acquire(code, p.key, p.nodes.size + 1, p.keep) end
    if !GgmlRaw.ready(lease)
      for l in p.nodes
        _ = try GgmlRaw.node(lease, l.op, l.args, l.shape, l.ints)
      end

      try System.Profile.measure("ml.compile") do GgmlRaw.compile(lease, p.outputs) end
    end

    bound: Result[Void] = System.Profile.measure("ml.bind") do
      for i in 0...p.leaves.size
        leaf = p.leaves[i]
        t = case leaf.fixed
          some(c) -> c
          none -> inputs[leaf.input]
        end

        h = try Ggml.handle_of(t)
        if Native.closed?(h)
          what = "an input"
          if leaf.input < 0
            what = "a constant"
          end

          return error(
            "#{op}: #{what} of the graph (leaf #{i}) is closed; constants must stay open while the graph can run",
          )
        end

        try GgmlRaw.bind(lease, i, h)
      end

      ok()
    end

    try bound
    out: List[Types.Tensor] = []
    for i in 0...p.shapes.size
      target: Optional[Types.Tensor] = none
      if !into.empty?
        target = into[i]
      end

      case target
        some(t) ->
          if t.shape != p.shapes[i] or t.dtype != p.dtypes[i] or t.backend != Types.Backend.Ggml(p.device)
            return error(
              "#{op}: output #{i} is #{Types.dtype_name(p.dtypes[i])} #{p.shapes[i]}, the tensor given is #{Types.dtype_name(t.dtype)} #{t.shape}",
            )
          end

          try GgmlRaw.output_into(lease, i, try Ggml.handle_of(t))
          out = out.push(t)
        none ->
          h = try GgmlRaw.output(lease, i, p.dtypes[i] == Types.Dtype.I32)
          out = out.push(Ggml.tensor_of(p.device, p.shapes[i], p.dtypes[i], h))
      end
    end

    try System.Profile.measure("ml.run") do GgmlRaw.run(lease, p.fills) end
    # A tensor made from the lease would keep it past the arena (one level
    # up), and the graph stays busy while leased; the outputs need nothing
    # of it after the run.
    Native.close(lease)
    ok(out)
  end
end

## How many graphs this process has compiled: a cache hit compiles none.
def compiles() -> Result[Int]
  _ = try Ggml.code_of(Types.Device.Cpu, "compiles")
  ok(GgmlRaw.compiles())
end

private

## The program of the graph `outs` reach; the input nodes are `ids`, in run order.
def lower(outs: List[Types.Tensor], ids: List[Ref], device: Types.Device, keep: Bool) -> Result[Program]
  positions: Map[Ref, Int] = {}
  for i in 0...ids.size
    positions = positions.put(ids[i], i)
  end

  backend = Types.Backend.Ggml(device)
  pos: Map[Ref, Int] = {}
  nodes: List[Lowered] = []
  parts: List[String] = []
  leaves: List[Leaf] = []
  fills: List[Float] = []
  for t in try walk(outs, positions)
    n = try Graph.node_of(t)
    if !exact(t.dtype)
      return error(
        "node #{n.id} (#{Types.op_name(n.op)}) has dtype #{Types.dtype_name(t.dtype)}, which F32 does not hold exactly",
      )
    end

    op = try code(t, n)
    ints: List[Int] = []
    case n.op
      Types.Op.Input ->
        leaves = leaves.push(Leaf(input: positions[n.id], fixed: none))
        op = LEAF
      Types.Op.Constant(c) -> leaves = leaves.push(Leaf(input: -1, fixed: some(try bound(c, backend, "a constant"))))
      Types.Op.Full(v) ->
        x = try Types.value_to_float(v, t.dtype)
        if Types.is_int_dtype(t.dtype) and x.abs > 16777216.0
          return error("node #{n.id}: a fill of #{x} is beyond 2^24, which F32 does not hold exactly")
        end

        fills = fills.push(x)
        zero = 0
        if x == 0.0
          zero = 1
        end

        ints = [zero]
      Types.Op.Sum(axes, _) -> ints = [n.args[0].shape.size] + axes
      Types.Op.Mean(axes, _) -> ints = [n.args[0].shape.size] + axes
      Types.Op.Max(axes, _) -> ints = [n.args[0].shape.size] + axes
      Types.Op.Arange(start, stop, step) -> ints = [start, stop, step]
      Types.Op.Slice(starts, _) -> ints = starts
      Types.Op.Gather -> ints = [n.args[0].shape.size]
      Types.Op.Concat(axis) -> ints = [axis, 0]
      Types.Op.Stack(axis) -> ints = [axis, 1]
      Types.Op.Conv1d(stride, padding) -> ints = [stride, padding]
      _ -> nil
    end

    l = Lowered(op: op, args: n.args.map { |a| pos[Graph.id_of(a)] }, shape: t.shape, ints: ints)
    pos = pos.put(n.id, nodes.size)
    nodes = nodes.push(l)
    parts = parts.push("#{l.op} #{l.args} #{l.shape} #{l.ints} #{Types.dtype_name(t.dtype)}")
  end

  outputs = outs.map { |o| pos[Graph.id_of(o)] }
  key = parts.join(";") + " -> #{outputs}"
  ok(
    Program(
      device: device,
      nodes: nodes,
      key: key,
      leaves: leaves,
      fills: fills,
      outputs: outputs,
      shapes: outs.map { |o| o.shape },
      dtypes: outs.map { |o| o.dtype },
      keep: keep,
    ),
  )
end

## A leaf's tensor, which must live on the backend the graph runs on.
def bound(t: Types.Tensor, backend: Types.Backend, what: String) -> Result[Types.Tensor]
  if t.backend != backend
    return error("#{what} on #{Types.backend_name(t.backend)} in a trace run on #{Types.backend_name(backend)}")
  end

  ok(t)
end

## Every traced tensor `roots` reach, each node once, in id order. The walk
## does not enter a node of `env`; an input that is not there belongs to
## another trace.
def walk(roots: List[Types.Tensor], env: Map[Ref, Int]) -> Result[List[Types.Tensor]]
  seen: Set[Ref] = Set.new([])
  found: List[Types.Tensor] = []
  stack = roots
  while !stack.empty?
    t = stack.last
    stack = stack.take(stack.size - 1)
    n = try Graph.node_of(t)
    if !seen.include?(n.id)
      seen = seen.add(n.id)
      found = found.push(t)
      case n.op
        Types.Op.Input -> return error("node #{n.id} is an input of another trace") if !env.has_key?(n.id)
        _ -> stack = stack + n.args
      end
    end
  end

  ok(found.sort_by { |t| Graph.id_of(t) })
end

## A dtype whose values F32 holds exactly here: F32 itself, `Bool` as 0/1,
## and integers of up to 32 bits (exact below 2^24, which labels and `arange`
## positions are). `I64`, `F64` and `F16` are refused.
def exact(dtype: Types.Dtype) -> Bool
  case dtype
    Types.Dtype.F64 -> false
    Types.Dtype.F16 -> false
    Types.Dtype.I64 -> false
    _ -> true
  end
end

## The ggml op code of a node, or an error naming the op this lowering does
## not have.
def code(t: Types.Tensor, n: Types.Node) -> Result[Int]
  case n.op
    Types.Op.Input -> ok(LEAF)
    Types.Op.Constant(_) -> ok(LEAF)
    Types.Op.Full(_) -> ok(FULL)
    Types.Op.Add -> ok(ADD)
    Types.Op.Sub -> ok(SUB)
    Types.Op.Mul -> ok(MUL)
    Types.Op.Div ->
      return error("node #{n.id}: an integer div truncates, which F32 does not") if t.dtype != Types.Dtype.F32
      ok(DIV)
    Types.Op.Neg -> ok(NEG)
    Types.Op.Exp -> ok(EXP)
    Types.Op.Log -> ok(LOG)
    Types.Op.Relu -> ok(RELU)
    Types.Op.Sqrt -> ok(SQRT)
    Types.Op.Sigmoid -> ok(SIGMOID)
    Types.Op.Tanh -> ok(TANH)
    Types.Op.Matmul -> ok(MATMUL)
    Types.Op.Transpose -> ok(TRANSPOSE)
    Types.Op.Reshape -> ok(RESHAPE)
    Types.Op.Squeeze(_) -> ok(RESHAPE)
    Types.Op.Unsqueeze(_) -> ok(RESHAPE)
    Types.Op.BroadcastTo -> ok(BROADCAST)
    Types.Op.Sum(_, _) -> ok(SUM)
    Types.Op.Mean(_, _) -> ok(MEAN)
    Types.Op.Max(_, _) -> ok(MAX)
    Types.Op.StopGrad -> ok(IDENTITY)
    Types.Op.Copy -> ok(IDENTITY)
    Types.Op.Contiguous -> ok(IDENTITY)
    # Every value is F32 on the ggml side: a mask or an integer turned into
    # a float is the same number; a float turned into an integer, an integer
    # into a mask (`!= 0`) or one integer width into another (a range check)
    # is not.
    Types.Op.ToDtype ->
      from = n.args[0].dtype
      return ok(IDENTITY) if t.dtype == from or t.dtype == Types.Dtype.F32
      return ok(IDENTITY) if from == Types.Dtype.Bool and Types.is_int_dtype(t.dtype)
      error(
        "node #{n.id}: to_dtype from #{Types.dtype_name(from)} to #{Types.dtype_name(t.dtype)} is not lowered to ggml",
      )
    Types.Op.Arange(_, _, _) -> ok(ARANGE)
    Types.Op.Eq -> ok(EQ)
    Types.Op.Lt -> ok(LT)
    Types.Op.Where -> ok(WHERE)
    Types.Op.Slice(_, _) -> ok(SLICE)
    Types.Op.Gather -> ok(GATHER)
    Types.Op.GeluErf -> ok(GELU_ERF)
    Types.Op.Concat(_) -> ok(CONCAT)
    Types.Op.Stack(_) -> ok(CONCAT)
    Types.Op.Conv1d(_, _) -> ok(CONV1D)
    _ -> error("node #{n.id}: #{Types.op_name(n.op)} is not lowered to ggml yet")
  end
end
