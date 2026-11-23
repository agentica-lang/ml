## Lowers a graph that `Trace` recorded (`tensor/DESIGN.md` §10) to ONE ggml
## program: every node the outputs and the writes reach becomes one node of
## `native/ggmlgraph.c`, in id order (a topological order, §10.2).
##
## The leaves are the traced constants. A `Step` says which of them are kept
## across steps (`states`: parameters, optimizer moments) and which are
## uploaded per step (`inputs`: the batch, per-step scalars); any other
## constant is set once from its values (`fixed`). A `Write` replaces a state
## with a node's value after each run, inside the same graph.

import Raw from "ggml_graph"
import Tensor from "pkg/tensor/tensor"
import Types from "pkg/tensor/src/types"

## The op codes of `native/ggmlgraph.c`, in its order.
INPUT = 0
STATE = 1
FIXED = 2
FULL = 3
ADD = 4
SUB = 5
MUL = 6
DIV = 7
NEG = 8
EXP = 9
LOG = 10
RELU = 11
SQRT = 12
SIGMOID = 13
TANH = 14
MATMUL = 15
TRANSPOSE = 16
RESHAPE = 17
BROADCAST = 18
SUM = 19
MEAN = 20
MAX = 21
IDENTITY = 22
ARANGE = 23
EQ = 24
LT = 25
WHERE = 26

## After each run, `state` takes the value of `value`.
struct Write
  state: Tensor.Tensor
  value: Tensor.Tensor
end

## One training step as traced tensors: `outputs[0]` is the loss.
struct Step
  outputs: List[Tensor.Tensor]
  states: List[Tensor.Tensor]
  inputs: List[Tensor.Tensor]
  writes: List[Write]
end

## A program whose nodes are sent but which is not compiled yet. `states` and
## `inputs` are the program indices of `Step.states` and `Step.inputs`.
struct Built
  program: Raw.Program
  step: Step
  index: Map[Int, Int]
  states: List[Int]
  inputs: List[Int]
  fixed: List[Fixed]
end

## A lifted constant that is neither a state nor an input: set once.
struct Fixed
  node: Int
  tensor: Tensor.Tensor
end

## Sends every node of `step` to a new program on `backend`.
def build(backend: Raw.Backend, step: Step) -> Result[Built]
  roots = step.outputs + step.writes.map { |w| w.value }
  nodes = try walk(roots)
  slots: Map[Int, Int] = {}
  for i in 0...step.states.size
    slots = slots.put(try id_of(step.states[i]), STATE)
  end

  for i in 0...step.inputs.size
    slots = slots.put(try id_of(step.inputs[i]), INPUT)
  end

  p = try Raw.program(backend, nodes.size + step.states.size + step.inputs.size + 8)
  index: Map[Int, Int] = {}
  fixed: List[Fixed] = []
  for t in nodes + step.states + step.inputs
    n = try node_of(t)
    if !index.has_key?(n.id)
      i = try emit(p, t, n, index, slots)
      index = index.put(n.id, i)
      case n.op
        Types.Op.Constant(c) ->
          if !slots.has_key?(n.id)
            fixed = fixed.push(Fixed(node: i, tensor: c))
          end

        _ -> nil
      end
    end
  end

  states = step.states.map { |s| index[id_unchecked(s)] }
  inputs = step.inputs.map { |s| index[id_unchecked(s)] }
  ok(Built(program: p, step: step, index: index, states: states, inputs: inputs, fixed: fixed))
end

## Compiles a built program and sets its fixed constants.
def compile(b: Built) -> Result[Built]
  outputs = b.step.outputs.map { |t| b.index[id_unchecked(t)] }
  writebacks: List[Int] = []
  for w in b.step.writes
    writebacks = writebacks + [b.index[try id_of(w.state)], b.index[try id_of(w.value)]]
  end

  try Raw.compile(b.program, outputs, writebacks)
  for f in b.fixed
    try Raw.set(b.program, f.node, try values_of(f.tensor))
  end

  ok(b)
end

## `build` and then `compile`.
def lower(backend: Raw.Backend, step: Step) -> Result[Built] = compile(try build(backend, step))

## The program index of a traced tensor of the step.
def index_of(b: Built, t: Tensor.Tensor) -> Result[Int]
  id = try id_of(t)
  case b.index.get(id)
    some(i) -> ok(i)
    none -> error("node #{id} is not in the program")
  end
end

## The row-major values of a reference tensor, as floats.
def values_of(t: Tensor.Tensor) -> Result[List[Float]]
  case t.handle
    Types.Handle.Plain(d) -> ok(d.values)
    _ -> error("a constant to upload must be a reference tensor")
  end
end

## Every op this lowering knows, by its `Types.op_name`; the others answer an
## error naming the op.
def lowered_ops() -> List[String]
  [
    "constant", "full", "add", "sub", "mul", "div", "neg", "exp", "log", "relu", "sqrt", "sigmoid", "tanh", "matmul",
    "transpose", "reshape", "squeeze", "unsqueeze", "broadcast_to", "sum", "mean", "max", "stop_grad", "copy",
    "contiguous", "to_dtype", "to_device", "arange", "eq", "lt", "where",
  ]
end

private

def node_of(t: Tensor.Tensor) -> Result[Types.Node]
  case t.handle
    Types.Handle.Traced(n) -> ok(n)
    _ -> error("a #{Types.backend_name(t.backend)} tensor is not traced")
  end
end

def id_of(t: Tensor.Tensor) -> Result[Int]
  n = try node_of(t)
  ok(n.id)
end

def id_unchecked(t: Tensor.Tensor) -> Int
  case t.handle
    Types.Handle.Traced(n) -> n.id
    _ -> -1
  end
end

## Every traced tensor `roots` reach, each node once, in id order.
def walk(roots: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  seen: Set[Int] = Set.new([])
  found: List[Tensor.Tensor] = []
  stack = roots
  while !stack.empty?
    t = stack.last
    stack = stack.take(stack.size - 1)
    n = try node_of(t)
    if !seen.include?(n.id)
      seen = seen.add(n.id)
      found = found.push(t)
      stack = stack + n.args
    end
  end

  ok(found.sort_by { |t| id_unchecked(t) })
end

## One node to the program; answers its index.
def emit(p: Raw.Program, t: Tensor.Tensor, n: Types.Node, index: Map[Int, Int], slots: Map[Int, Int]) -> Result[Int]
  if !Types.is_float_dtype(t.dtype) and !Types.is_int_dtype(t.dtype) and t.dtype != Types.Dtype.Bool
    return error("node #{n.id} has dtype #{Types.dtype_name(t.dtype)}")
  end

  args = n.args.map { |a| index[id_unchecked(a)] }
  op = try code(t, n)
  ints: List[Int] = []
  value = 0.0
  case n.op
    Types.Op.Constant(_) -> op = slots.get(n.id).unwrap_or(FIXED)
    Types.Op.Full(v) -> value = try Types.value_to_float(v, t.dtype)
    Types.Op.Sum(axes, _) -> ints = [n.args[0].shape.size] + axes
    Types.Op.Mean(axes, _) -> ints = [n.args[0].shape.size] + axes
    Types.Op.Max(axes, _) -> ints = [n.args[0].shape.size] + axes
    Types.Op.Arange(start, stop, step) -> ints = [start, stop, step]
    _ -> nil
  end

  Raw.node(p, op, args, t.shape, ints, value)
end

## The ggml op code of a node, or an error naming the op this lowering does
## not have.
def code(t: Tensor.Tensor, n: Types.Node) -> Result[Int]
  case n.op
    Types.Op.Constant(_) -> ok(FIXED)
    Types.Op.Full(_) -> ok(FULL)
    Types.Op.Add -> ok(ADD)
    Types.Op.Sub -> ok(SUB)
    Types.Op.Mul -> ok(MUL)
    Types.Op.Div -> ok(DIV)
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
    Types.Op.ToDevice(Types.Device.Cpu) -> ok(IDENTITY)
    # Every value is F32 on the ggml side: a mask or an integer turned into
    # a float is the same number, a float turned into an integer is not.
    Types.Op.ToDtype ->
      if !Types.is_float_dtype(t.dtype) and Types.is_float_dtype(n.args[0].dtype)
        return error("node #{n.id}: to_dtype from a float to #{Types.dtype_name(t.dtype)} is not lowered")
      end

      ok(IDENTITY)
    Types.Op.Arange(_, _, _) -> ok(ARANGE)
    Types.Op.Eq -> ok(EQ)
    Types.Op.Lt -> ok(LT)
    Types.Op.Where -> ok(WHERE)
    _ -> error("node #{n.id}: #{Types.op_name(n.op)} is not lowered to ggml")
  end
end
