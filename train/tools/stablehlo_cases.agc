## The StableHLO programs of the golden tests and of `stablehlo_emit.agc`: an MLP forward and one training step of
## `fit` (a relu MLP, softmax cross-entropy and Adam), with the arguments their runs start from. The text is checked
## into `test/fixtures/stablehlo/`; IREE's outputs for it are the goldens (`make iree-goldens`).

import Data from "../data"
import Loss from "../loss"
import Optim from "../optim"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "../train"

## The run of the fit step the goldens record: its steps, so the bias corrections of Adam change from one to the next.
STEPS = 3
CLASSES = 3

ROWS = [0.5, -1.2, 0.3, 1.1, 0.8, -0.4, -0.7, 0.2, 1.5, 0.9, -1.1, 0.6]
LABELS = [0, 1, 2, 1]
W1 = [0.2, -0.5, 0.8, 0.1, -0.3, 0.7, 0.4, -0.6, 0.3, 0.9, -0.2, 0.5, 0.6, -0.1, 0.4]
B1 = [0.1, 0.2, -0.1, 0.25, 0.05]
W2_FORWARD = [0.3, -0.4, 0.6, 0.2, -0.5, 0.7, 0.1, -0.3, 0.8, 0.4]
B2_FORWARD = [0.05, -0.1]
W2_FIT = [0.3, -0.4, 0.6, 0.2, -0.5, 0.7, 0.1, -0.3, 0.8, 0.4, 0.2, -0.6, -0.2, 0.5, 0.1]
B2_FIT = [0.05, -0.1, 0.0]

## One argument of a program, as the golden files hold it: `dtype` is the MLIR element type.
struct Arg
  dtype: String
  shape: List[Int]
  data: List[Float]
end

## How IREE runs a case: the arguments of the first run, how many outputs a run gives, which output feeds which
## argument of the next run (`carry`, as `[output, argument]`), and for each run the arguments it sets (by position)
## before it starts.
struct Plan
  args: List[Arg]
  outputs: Int
  carry: List[List[Int]]
  sets: List[Map[String, List[Float]]]
end

struct Case
  name: String
  text: String
  plan: Plan
end

## The loss of the fit step over the parameters `[w1, b1, w2, b2]`.
def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.cross_entropy(try logits(ps, b.inputs, &Tensor.relu), b.targets)
end

def optimizer() -> Optim.Optimizer = Optim.Optimizer.Adam(Optim.Adam(lr: 0.1))

def cases() -> Result[List[Case]] = ok([try forward(), try fit_step()])

## `x w1 + b1`, the activation, then `w2` and `b2`.
def logits(
  ps: List[Tensor.Tensor],
  x: Tensor.Tensor,
  activation: Fn[Tensor.Tensor, Result[Tensor.Tensor]],
) -> Result[Tensor.Tensor]
  hidden = try activation.call(try Tensor.linear(x, ps[0], some(ps[1])))
  Tensor.linear(hidden, ps[2], some(ps[3]))
end

private

def forward() -> Result[Case]
  args = [
    try Tensor.floats(ROWS, [4, 3]),
    try Tensor.floats(W1, [3, 5]),
    try Tensor.floats(B1, [5]),
    try Tensor.floats(W2_FORWARD, [5, 2]),
    try Tensor.floats(B2_FORWARD, [2]),
  ]

  m = try Trace.stablehlo(
    args.map { |t| Trace.spec_of(t) },
    ->(xs: List[Tensor.Tensor]) { ok([try logits(xs.drop(1), xs[0], &Tensor.gelu)]) },
  )

  ok(Case(name: "mlp_forward", text: m.text, plan: Plan(args: try wrap(args), outputs: 1, carry: [], sets: [{}])))
end

## Three steps of Adam on the relu MLP, over the batch `ROWS` and `LABELS`. The arguments are the parameters, the two
## moments, Adam's two scalars, then the batch.
def fit_step() -> Result[Case]
  params = [
    try Tensor.floats(W1, [3, 5]),
    try Tensor.floats(B1, [5]),
    try Tensor.floats(W2_FIT, [5, CLASSES]),
    try Tensor.floats(B2_FIT, [CLASSES]),
  ]

  moments: List[Tensor.Tensor] = []
  for _ in 0...2
    for p in params
      moments = moments.push(try Tensor.like(p))
    end
  end

  ids = LABELS.map { |c| Tensor.Values.Int(c) }
  batch = Data.Batch(
    inputs: try Tensor.floats(ROWS, [4, 3]),
    targets: try Tensor.from_list(ids, [4], Tensor.Dtype.I32, Tensor.Backend.Reference),
  )

  m = try Train.stablehlo(&loss, params, batch, optimizer())
  state = try Optim.init(optimizer(), params)
  base = params.size + moments.size
  sets: List[Map[String, List[Float]]] = []
  for step in 0...STEPS
    scalars = Optim.step_scalars(state.with(steps: step))
    set: Map[String, List[Float]] = {}
    sets = sets.push(set.put("#{base}", [scalars[0]]).put("#{base + 1}", [scalars[1]]))
  end

  carry: List[List[Int]] = []
  for i in 0...base
    carry = carry.push([i + 1, i])
  end

  scalars = Optim.step_scalars(state).map { |x| Arg(dtype: "f32", shape: [], data: [x]) }
  args = try wrap(params + moments) + scalars + try wrap(m.constants)
  ok(Case(name: "fit_step", text: m.text, plan: Plan(args: args, outputs: base + 1, carry: carry, sets: sets)))
end

def wrap(tensors: List[Tensor.Tensor]) -> Result[List[Arg]]
  out: List[Arg] = []
  for t in tensors
    data: List[Float] = []
    for v in try Tensor.to_list(t)
      case v
        Tensor.Values.Float(x) -> data = data.push(x)
        Tensor.Values.Int(i) -> data = data.push(i.to_f)
        Tensor.Values.Bool(_) -> return error("a Bool argument is not a golden input")
      end
    end

    out = out.push(Arg(dtype: try element(Tensor.dtype(t)), shape: Tensor.shape(t), data: data))
  end

  ok(out)
end

def element(dtype: Tensor.Dtype) -> Result[String]
  case dtype
    Tensor.Dtype.F32 -> ok("f32")
    Tensor.Dtype.I32 -> ok("i32")
    _ -> error("the golden files hold f32 and i32 arguments, not #{dtype}")
  end
end
