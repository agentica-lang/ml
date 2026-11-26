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

import Binary from "std/binary"
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
SCATTER_ADD = 30
PERMUTE = 31
ATTENTION = 32
ROPE = 33
BATCH_MATMUL = 34
NORM = 35
RMS_NORM = 36
SET_ROWS = 37
TOP_K = 38
CAST = 39
ADAM_STEP = 40
ADAM_PART = 41
CONV2D = 42
MAX_POOL2D = 43
AVG_POOL2D = 44
GATED_DELTA_RULE = 45
EXPERT_MATMUL = 46

## One node as the C layer takes it.
struct Lowered
  op: Int
  args: List[Int]
  shape: List[Int]
  ints: List[Int]
end

## Where a leaf of the graph gets its tensor on every run: the input at
## position `input` of the run's inputs (`-1` for none), or `fixed`, a
## constant of the trace.
struct Leaf
  input: Int
  fixed: Optional[Types.Tensor]
  held: Int
end

## A traced graph lowered once: what a run sends to the C layer (the nodes,
## only on a cache miss), the structural key, where each leaf's tensor comes
## from, the fill values and the outputs, in the C layer's order. `donates`
## names, per output, the input it is: a `set_rows` writes its input in place,
## and the run gives that very tensor back (`-1` for an output that is new).
## `kept` are the leaves that stay bound between the runs of a pinned program
## (the constants the graph reads in place) and `fed` the leaves every run
## binds (the inputs, and an `I32` constant, which is converted); `pin` is the
## id of the pin that keeps them (`pinned`), 0 for none.
## A program never changes, so any number of tasks may run it.
struct Program
  device: Types.Device
  nodes: List[Lowered]
  key: String
  leaves: List[Leaf]
  kept: List[Int]
  fed: List[Int]
  pin: Int
  fills: List[Float]
  outputs: List[Int]
  donates: List[Int]
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
  program = try System.Profile.measure("ml.plan") do lower(outs, ids, device, keep, false) end
  inputs: List[Types.Tensor] = []
  for id in ids
    inputs = inputs.push(try bound(env[id], Types.Backend.Ggml(device), "an input"))
  end

  execute(program, inputs, "evaluate")
end

## The program of the traced tensors `outs` over the input nodes `inputs`
## (traced `Input` tensors, in the order a run takes its tensors).
def program(outs: List[Types.Tensor], inputs: List[Types.Tensor], device: Types.Device) -> Result[Program]
  System.Profile.measure("ml.plan") do lower(outs, inputs.map { |t| Graph.id_of(t) }, device, true, true) end
end

## `p` as the program of the pin `held`: its runs leave its constants bound to
## the pinned graph, so a run binds only its inputs, for as long as the pin
## lives, the constants stay open and no other program runs on that graph.
def pinned(p: Program, held: GgmlRaw.Pin) -> Program = p.with(pin: GgmlRaw.pin_id(held))

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
## the bind and the run). A program of a pin (`pinned`) leaves its constants
## bound to the pinned graph, and its next run binds only `fed`.
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
      if !GgmlRaw.resume(lease, p.pin)
        try bind(p, p.kept, lease, inputs, op)
        GgmlRaw.keep(lease, p.pin, p.kept)
      end

      bind(p, p.fed, lease, inputs, op)
    end

    try bound
    out: List[Types.Tensor] = []
    for i in 0...p.shapes.size
      target: Optional[Types.Tensor] = none
      if !into.empty?
        target = into[i]
      end

      if p.donates[i] >= 0
        if target.some?
          return error("#{op}: output #{i} is the input it writes in place, so `into` cannot name a tensor for it")
        end

        out = out.push(inputs[p.donates[i]])
        next
      end

      case target
        some(t) ->
          if t.shape != p.shapes[i] or t.dtype != p.dtypes[i] or !Types.reads?(Types.Backend.Ggml(p.device), t.backend)
            return error(
              "#{op}: output #{i} is #{Types.dtype_name(p.dtypes[i])} #{p.shapes[i]}, the tensor given is #{Types.dtype_name(t.dtype)} #{t.shape}",
            )
          end

          held = try Ggml.handle_of(t)
          return error("#{op}: output #{i}, a tensor given in `into`, is closed") if Native.closed?(held)
          try GgmlRaw.output_into(lease, i, held)
          out = out.push(t)
        none ->
          h = try GgmlRaw.output(lease, i, p.dtypes[i] == Types.Dtype.I32)
          home = p.device
          if GgmlRaw.on_cpu(h)
            home = Types.Device.Cpu
          end

          out = out.push(Ggml.tensor_of(home, p.shapes[i], p.dtypes[i], h))
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

## The threads the ggml CPU backend runs a graph on: `TENSOR_THREADS` when it is set, else the physical cores.
def cpu_threads() -> Result[Int]
  _ = try Ggml.code_of(Types.Device.Cpu, "cpu_threads")
  GgmlRaw.cpu_threads()
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

## Per output of `outs`, the position of the input an `adam_step` updates in place when it is that output (its first
## or second moment, under the rules of `adam_donors`), else `-1`. `positions` holds the inputs' positions.
def adam_moments(outs: List[Types.Tensor], positions: Map[Ref, Int]) -> Result[List[Int]]
  donors = try adam_donors(outs, positions, try walk(outs, positions))
  per_output: List[Int] = []
  for o in outs
    per_output = per_output.push(try adam_input(o, positions, donors))
  end

  ok(per_output)
end

private

## Binds the leaves `slots` of `p` for the next run of `lease`: an input to its tensor of `inputs`, a constant to
## the tensor it was traced with.
def bind(p: Program, slots: List[Int], lease: GgmlRaw.Lease, inputs: List[Types.Tensor], op: String) -> Result[Void]
  for i in slots
    leaf = p.leaves[i]
    t = case leaf.fixed
      some(c) -> c
      none -> inputs[leaf.input]
    end

    if Types.held_code(t.dtype) != leaf.held
      return error(
        "#{op}: leaf #{i} was lowered as #{leaf_name(leaf.held)} and the tensor given is #{Types.dtype_name(t.dtype)}",
      )
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

## The program of the graph `outs` reach; the input nodes are `ids`, in run order.
def lower(outs: List[Types.Tensor], ids: List[Ref], device: Types.Device, keep: Bool, donate: Bool) -> Result[Program]
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
  walked = try walk(outs, positions)
  donors: List[Ref] = []
  if donate
    donors = try adam_donors(outs, positions, walked)
  end

  for t in walked
    n = try Graph.node_of(t)
    half_ok = try half_node(t, n)
    if !exact(t.dtype) and !half_ok
      return error(
        "node #{n.id} (#{Types.op_name(n.op)}) has dtype #{Types.dtype_name(t.dtype)}, which F32 does not hold exactly",
      )
    end

    _ = try half_operands(n)
    op = try code(t, n)
    ints: List[Int] = []
    held = Types.held_code(t.dtype)
    case n.op
      Types.Op.Input ->
        leaves = leaves.push(Leaf(input: positions[n.id], fixed: none, held: held))
        ints = [held]
        op = LEAF
      Types.Op.Constant(c) ->
        leaves = leaves.push(Leaf(input: -1, fixed: some(try bound(c, backend, "a constant")), held: held))
        ints = [held]
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
      Types.Op.ScatterAdd(rows) -> ints = [rows]
      Types.Op.Concat(axis) -> ints = [axis, 0]
      Types.Op.Stack(axis) -> ints = [axis, 1]
      Types.Op.Conv1d(stride, padding) -> ints = [stride, padding]
      Types.Op.Conv2d(stride, padding, groups) -> ints = [stride, padding, groups]
      Types.Op.MaxPool2d(kernel, stride, padding) -> ints = [kernel, stride, padding]
      Types.Op.AvgPool2d(kernel, stride, padding) -> ints = [kernel, stride, padding]
      Types.Op.Permute(axes) -> ints = axes
      Types.Op.Attention(scale) -> ints = [try bits(scale, n), flag(try flash_mask(n.args))]
      Types.Op.Rope(base, dims, neox, inverse) -> ints = [dims, flag(neox), try bits(base, n), flag(inverse)]
      Types.Op.LayerNorm(eps) -> ints = [Float32.to_bits(eps.to_float32).to_i]
      Types.Op.SetRows -> ints = [flag(!donate)]
      Types.Op.TopK(k) -> ints = [k]
      Types.Op.ToDtype ->
        if op == CAST
          ints = [flag(t.dtype == Types.Dtype.F16)]
        end

      Types.Op.RmsNorm(eps) -> ints = [Float32.to_bits(eps.to_float32).to_i]
      Types.Op.AdamStep -> ints = [flag(donors.include?(n.id))]
      Types.Op.AdamPart(i) -> ints = [i]
      _ -> nil
    end

    l = Lowered(op: op, args: n.args.map { |a| pos[Graph.id_of(a)] }, shape: t.shape, ints: ints)
    pos = pos.put(n.id, nodes.size)
    nodes = nodes.push(l)
    parts = parts.push("#{l.op} #{l.args} #{l.shape} #{l.ints} #{Types.dtype_name(t.dtype)}")
  end

  for i in 0...outs.size
    if Types.is_packed_dtype(outs[i].dtype)
      n = try Graph.node_of(outs[i])
      if !try cache_output(outs[i], n)
        return error(
          "output #{i} (node #{n.id}, #{Types.op_name(n.op)}) is #{Types.dtype_name(outs[i].dtype)}: ggml returns F32, except an F16 cache made by to_dtype or written by set_rows, so widen it with to_dtype first",
        )
      end
    end
  end

  outputs = outs.map { |o| pos[Graph.id_of(o)] }
  donates = outs.map { |_| -1 }
  if donate
    donates = try donations(outs, positions, walked, donors)
  end

  # ggml holds an I32 tensor as I32 and a leaf as F32: that one is converted at every bind.
  kept: List[Int] = []
  fed: List[Int] = []
  for i in 0...leaves.size
    case leaves[i].fixed
      some(c) ->
        if c.dtype == Types.Dtype.I32
          fed = fed.push(i)
        else
          kept = kept.push(i)
        end

      none -> fed = fed.push(i)
    end
  end

  key = parts.join(";") + " -> #{outputs}"
  ok(
    Program(
      device: device,
      nodes: nodes,
      key: key,
      leaves: leaves,
      kept: kept,
      fed: fed,
      pin: 0,
      fills: fills,
      outputs: outputs,
      donates: donates,
      shapes: outs.map { |o| o.shape },
      dtypes: outs.map { |o| o.dtype },
      keep: keep,
    ),
  )
end

## Per output, the position of the input a `set_rows` writes in place (`-1` for the others), when the writes are sound:
## the cache of a `set_rows` is an input read by no other node, the `set_rows` is an output (its result is the input
## itself, so it is not given twice), and an input is not also an output.
def donations(
  outs: List[Types.Tensor],
  positions: Map[Ref, Int],
  walked: List[Types.Tensor],
  donors: List[Ref],
) -> Result[List[Int]]
  output_ids = outs.map { |o| Graph.id_of(o) }
  written: Map[Ref, Int] = {}
  for t in walked
    n = try Graph.node_of(t)
    case n.op
      Types.Op.SetRows ->
        if !output_ids.include?(n.id)
          return error("node #{n.id} (set_rows) writes its input in place, so it must be an output")
        end

        dst = try Graph.node_of(n.args[0])
        case dst.op
          Types.Op.Input -> nil
          _ ->
            return error(
              "node #{n.id} (set_rows) writes its cache in place, which must be an input, got node #{dst.id} (#{Types.op_name(dst.op)})",
            )
        end

        if output_ids.include?(dst.id)
          return error("node #{n.id} (set_rows): its cache, node #{dst.id}, is also an output")
        end

        if output_ids.select { |id| id == n.id }.size > 1
          return error("node #{n.id} (set_rows) is listed twice as an output, and its result is the input itself")
        end

        for u in walked
          m = try Graph.node_of(u)
          # the rows and the positions of the set_rows itself are read too, the cache (its first argument) is written
          reads = m.args
          if m.id == n.id
            reads = m.args.drop(1)
          end

          if reads.any? { |a| Graph.id_of(a) == dst.id }
            return error(
              "node #{m.id} (#{Types.op_name(m.op)}) reads node #{dst.id}, which node #{n.id} (set_rows) writes in place",
            )
          end
        end

        written = written.put(n.id, positions[dst.id])
      _ -> nil
    end
  end

  per_output: List[Int] = []
  for o in outs
    id = Graph.id_of(o)
    if written.has_key?(id)
      per_output = per_output.push(written[id])
    else
      per_output = per_output.push(try adam_input(o, positions, donors))
    end
  end

  ok(per_output)
end

## The position of the input an output is when it is the first or second moment of an `adam_step` that updates its
## moments in place (`adam_donors`), else `-1`.
def adam_input(o: Types.Tensor, positions: Map[Ref, Int], donors: List[Ref]) -> Result[Int]
  n = try Graph.node_of(o)
  case n.op
    Types.Op.AdamPart(i) ->
      step = try Graph.node_of(n.args[0])
      return ok(positions[Graph.id_of(step.args[1 + i])]) if i > 0 and donors.include?(step.id)
      ok(-1)
    _ -> ok(-1)
  end
end

## The `adam_step` nodes that update their moments in place: the first and second moment are inputs (not constants) that
## the step alone reads, once each, and that are not outputs, and the step's two parts for them are outputs, one node
## each, listed once. A step that fails any of that works on copies, which is what any other caller gets. One walk counts
## the readers and the parts.
def adam_donors(outs: List[Types.Tensor], positions: Map[Ref, Int], walked: List[Types.Tensor]) -> Result[List[Ref]]
  listed: Map[Ref, Int] = {}
  for o in outs
    id = Graph.id_of(o)
    listed = listed.put(id, listed.get(id, 0) + 1)
  end

  reads: Map[Ref, Int] = {}
  parts: Map[Ref, List[Int]] = {}
  for t in walked
    n = try Graph.node_of(t)
    for a in n.args
      id = Graph.id_of(a)
      reads = reads.put(id, reads.get(id, 0) + 1)
    end

    case n.op
      Types.Op.AdamPart(i) ->
        # `Tensor.adam_step` makes one part node per index and step (`Graph.adam_part`), so a part listed once as
        # an output
        # is the only one; a part that is not an output leaves the list short and the step copies
        step = Graph.id_of(n.args[0])
        if i > 0 and listed.get(n.id, 0) == 1
          parts = parts.put(step, parts.get(step, []).push(i))
        end

      _ -> nil
    end
  end

  found: List[Ref] = []
  for t in walked
    n = try Graph.node_of(t)
    case n.op
      Types.Op.AdamStep ->
        m = Graph.id_of(n.args[2])
        v = Graph.id_of(n.args[3])
        sound = m != v and try an_input(n.args[2]) and try an_input(n.args[3])
        sound = sound and reads.get(m, 0) == 1 and reads.get(v, 0) == 1
        sound = sound and listed.get(m, 0) == 0 and listed.get(v, 0) == 0
        if sound and parts.get(n.id, []).sort == [1, 2]
          found = found.push(n.id)
        end

      _ -> nil
    end
  end

  ok(found)
end

## The bits of the double `x`, which the C layer reads back as `x`: a node takes integers only. A float
## that ggml holds in `F32` must fit it.
def bits(x: Float, n: Types.Node) -> Result[Int]
  if !Types.is_finite(x) or x.abs > 3.4028234663852886e38
    return error("node #{n.id} (#{Types.op_name(n.op)}): #{x} is not a float ggml holds")
  end

  # A negative double has its bits above 2^63, where `to_i` fails: the same eight bytes read as an `Int`.
  Binary.read_i64_le(Binary.write_u64_le(Float.to_bits(x)), 0)
end

## Whether flash attention may take the mask of an attention node: it reads the mask in F16, so the mask must be absent
## or hold only the fills 0 and a value from F32's lowest to `-65504` (what `where` of two `full`s, or a comparison times
## a `full`, gives: the padded mask of a decoder and the causal mask). A fill below F32's lowest is `-inf` in the fill
## store, and its product with 0 is NaN.
def flash_mask(args: List[Types.Tensor]) -> Result[Bool]
  return ok(true) if args.size == 3
  fills_only(args[3])
end

def fills_only(t: Types.Tensor) -> Result[Bool]
  n = try Graph.node_of(t)
  case n.op
    Types.Op.Full(v) ->
      x = try Types.value_to_float(v, t.dtype)
      ok(x == 0.0 or (x >= -3.4028234663852886e38 and x <= -65504.0))
    Types.Op.Where -> ok(try fills_only(n.args[1]) and try fills_only(n.args[2]))
    Types.Op.Mul -> ok(try indicator(n.args[0]) and try fills_only(n.args[1]))
    Types.Op.Reshape -> fills_only(n.args[0])
    Types.Op.BroadcastTo -> fills_only(n.args[0])
    _ -> ok(false)
  end
end

## A comparison, as the dtype of the mask: 0 and 1, so its product with a fill is 0 or the fill.
def indicator(t: Types.Tensor) -> Result[Bool]
  n = try Graph.node_of(t)
  case n.op
    Types.Op.Lt -> ok(true)
    Types.Op.Eq -> ok(true)
    Types.Op.ToDtype -> indicator(n.args[0])
    Types.Op.Reshape -> indicator(n.args[0])
    Types.Op.BroadcastTo -> indicator(n.args[0])
    _ -> ok(false)
  end
end

## `true` as `1`, `false` as `0`.
def flag(b: Bool) -> Int
  return 1 if b
  0
end

def leaf_name(held: Int) -> String
  return "F16" if held == 1
  return "Q8_0" if held == 2
  return "Q4_K" if held == 3
  return "Q6_K" if held == 4
  "F32"
end

## A leaf's tensor, which must live on the backend the graph runs on (or, on a Vulkan device, on the CPU: `Types.reads?`).
def bound(t: Types.Tensor, backend: Types.Backend, what: String) -> Result[Types.Tensor]
  if !Types.reads?(backend, t.backend)
    return error("#{what} on #{Types.backend_name(t.backend)} in a trace run on #{Types.backend_name(backend)}")
  end

  ok(t)
end

## A dtype whose values F32 holds exactly here: F32 itself, `Bool` as 0/1,
## and integers of up to 32 bits (exact below 2^24, which labels and `arange`
## positions are). `I64`, `F64`, `F16` and the quantized dtypes are refused.
def exact(dtype: Types.Dtype) -> Bool
  case dtype
    Types.Dtype.F64 -> false
    Types.Dtype.F16 -> false
    Types.Dtype.I64 -> false
    _ -> !Types.is_quantized_dtype(dtype)
  end
end

## An `F16` or quantized node ggml holds: a packed weight (an input or a constant), its
## transpose, which a `matmul` reads in place, or the rows a `gather` reads from
## one, which ggml has widened to F32 already and only `to_dtype` may take, or
## a slice of an `F16` cache, which an attention reads.
## Any other such node is an error: a packed weight is read-only, so a gradient
## taken for one (its fill of zeros included) ends here.
def half_node(t: Types.Tensor, n: Types.Node) -> Result[Bool]
  return ok(false) if !Types.is_packed_dtype(t.dtype)
  case n.op
    Types.Op.Input -> ok(true)
    Types.Op.Constant(_) -> ok(true)
    Types.Op.Transpose -> ok(true)
    Types.Op.Gather -> ok(true)
    Types.Op.SetRows -> ok(t.dtype == Types.Dtype.F16)
    Types.Op.ToDtype -> ok(t.dtype == Types.Dtype.F16)
    Types.Op.Slice(_, _) ->
      if t.dtype == Types.Dtype.F16
        ok(true)
      else
        unheld(t, n)
      end

    _ -> unheld(t, n)
  end
end

## The error of a packed node that ggml does not hold.
def unheld(t: Types.Tensor, n: Types.Node) -> Result[Bool]
  error(
    "node #{n.id} (#{Types.op_name(n.op)}) is #{Types.dtype_name(t.dtype)}, which ggml holds only for a packed weight read by matmul, linear, gather and embedding, or an F16 cache that set_rows writes, to_dtype makes and attention reads (whole, or a slice of it): " +
    "a packed weight is read-only, so no gradient is taken for it",
  )
end

## The `F16` and quantized operands of a node: the weight of a `matmul` (second operand) and
## the table of a `gather` (first), the argument of a transpose, or the rows of a
## `gather` that `to_dtype` widens, an `F16` input that `to_dtype` widens, the cache of a `set_rows` and the key and
## value of an attention, or a slice of a cache that an attention reads (F16 only). Anywhere else the packed weight would need an
## op that ggml does not run on it. A quantized weight goes into a `matmul` only as the
## transpose that `linear(..., out_in: true)` makes, which is read in place.
def half_operands(n: Types.Node) -> Result[Bool]
  for i in 0...n.args.size
    if Types.is_packed_dtype(n.args[i].dtype)
      fits = case n.op
        Types.Op.Matmul -> i == 1 and (!Types.is_quantized_dtype(n.args[i].dtype) or try transposed(n.args[i]))
        Types.Op.Gather -> i == 0 and (!Types.is_quantized_dtype(n.args[i].dtype) or try leaf(n.args[i]))
        Types.Op.Transpose -> !Types.is_quantized_dtype(n.args[i].dtype) or try leaf(n.args[i])
        Types.Op.ToDtype -> try gathered(n.args[0]) or (n.args[i].dtype == Types.Dtype.F16 and try an_input(n.args[0]))
        Types.Op.SetRows -> i == 0 and n.args[i].dtype == Types.Dtype.F16 and try an_input(n.args[0])
        Types.Op.Attention(_) -> (i == 1 or i == 2) and n.args[i].dtype == Types.Dtype.F16 and try attended(n.args[i])
        Types.Op.Slice(_, _) -> n.args[i].dtype == Types.Dtype.F16 and try cached(n.args[i])
        _ -> false
      end

      if !fits
        return error(
          "node #{n.id} (#{Types.op_name(n.op)}) takes an #{Types.dtype_name(n.args[i].dtype)} operand: a packed weight goes only into the weight of matmul and linear (a quantized one with out_in) and the table of gather and embedding",
        )
      end
    end
  end

  ok(true)
end

## The `F16` tensor `t` is an F16 cache: the result of `to_dtype` (from F32) or of `set_rows`.
def cache_output(t: Types.Tensor, n: Types.Node) -> Result[Bool]
  return ok(false) if t.dtype != Types.Dtype.F16
  case n.op
    Types.Op.ToDtype -> ok(true)
    Types.Op.SetRows -> ok(true)
    _ -> ok(false)
  end
end

## The tensor `t` is an input of the graph (a constant is a weight, which is not widened or written).
def an_input(t: Types.Tensor) -> Result[Bool]
  n = try Graph.node_of(t)
  case n.op
    Types.Op.Input -> ok(true)
    _ -> ok(false)
  end
end

## The `F16` tensor `t` is an input, the result of a `set_rows` or of a `to_dtype`: where a cache comes from.
def cached(t: Types.Tensor) -> Result[Bool]
  n = try Graph.node_of(t)
  case n.op
    Types.Op.Input -> ok(true)
    Types.Op.SetRows -> ok(true)
    Types.Op.ToDtype -> ok(true)
    _ -> ok(false)
  end
end

## The `F16` tensor `t` is a cache (`cached`), or a slice of one: what an attention reads.
def attended(t: Types.Tensor) -> Result[Bool]
  n = try Graph.node_of(t)
  case n.op
    Types.Op.Slice(_, _) -> cached(n.args[0])
    _ -> cached(t)
  end
end

## The tensor `t` is an input or a constant: a quantized weight is read only where it is made, never a view of it.
def leaf(t: Types.Tensor) -> Result[Bool]
  n = try Graph.node_of(t)
  case n.op
    Types.Op.Input -> ok(true)
    Types.Op.Constant(_) -> ok(true)
    _ -> ok(false)
  end
end

## The tensor `t` is the transpose of another.
def transposed(t: Types.Tensor) -> Result[Bool]
  n = try Graph.node_of(t)
  case n.op
    Types.Op.Transpose -> ok(true)
    _ -> ok(false)
  end
end

## The `F16` or quantized tensor `t` is the rows of a `gather`.
def gathered(t: Types.Tensor) -> Result[Bool]
  n = try Graph.node_of(t)
  case n.op
    Types.Op.Gather -> ok(true)
    _ -> ok(false)
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
      return ok(IDENTITY) if t.dtype == from
      return ok(CAST) if t.dtype == Types.Dtype.F32 and Types.is_packed_dtype(from)
      return ok(CAST) if t.dtype == Types.Dtype.F16 and from == Types.Dtype.F32
      return ok(IDENTITY) if t.dtype == Types.Dtype.F32
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
    Types.Op.ScatterAdd(_) -> ok(SCATTER_ADD)
    Types.Op.GeluErf -> ok(GELU_ERF)
    Types.Op.LayerNorm(_) -> ok(NORM)
    Types.Op.RmsNorm(_) -> ok(RMS_NORM)
    Types.Op.AdamStep -> ok(ADAM_STEP)
    Types.Op.AdamPart(_) -> ok(ADAM_PART)
    Types.Op.SetRows -> ok(SET_ROWS)
    Types.Op.TopK(_) -> ok(TOP_K)
    Types.Op.GatedDeltaRule -> ok(GATED_DELTA_RULE)
    Types.Op.ExpertMatmul -> ok(EXPERT_MATMUL)
    Types.Op.Concat(_) -> ok(CONCAT)
    Types.Op.Stack(_) -> ok(CONCAT)
    Types.Op.Conv1d(_, _) -> ok(CONV1D)
    Types.Op.Conv2d(_, _, _) -> ok(CONV2D)
    Types.Op.MaxPool2d(_, _, _) -> ok(MAX_POOL2D)
    Types.Op.AvgPool2d(_, _, _) -> ok(AVG_POOL2D)
    Types.Op.Permute(_) -> ok(PERMUTE)
    Types.Op.Attention(_) -> ok(ATTENTION)
    Types.Op.Rope(_, _, _, _) -> ok(ROPE)
    Types.Op.BatchMatmul -> ok(BATCH_MATMUL)
    _ -> error("node #{n.id}: #{Types.op_name(n.op)} is not lowered to ggml yet")
  end
end
