## Training steps on a ggml backend against the step `Train.fit` runs on the
## reference backend: the loss, every gradient and every new parameter.
##
## Two models: a small one (8 → 16 → 16 → 4, a batch of 6) for one SGD step
## in one graph, in two graphs, and two Adam steps; and the benchmark's model
## (784 → 512 → 512 → 10, a batch of 256) for two SGD steps.
##
## A gap is `max |ggml - reference| / max |reference|` over one tensor, the
## largest over the tensors of a kind. A non-finite value on either side is an
## error, never a gap.

import Lower from "lower"
import Optim from "pkg/train/optim"
import Raw from "ggml_graph"
import Step from "step"
import Tensor from "pkg/tensor/tensor"
import Types from "pkg/tensor/src/types"

SIZES = [8, 16, 16, 4]
BATCH = 6
LR = 0.1
BIG_SIZES = [784, 512, 512, 10]
BIG_BATCH = 256
BIG_STEPS = 2
BIG_LR = 0.01

struct Gaps
  loss: Float
  grads: Float
  params: Float
end

struct Model
  init: Step.Init
  params: List[Tensor.Tensor]
  x: Tensor.Tensor
  y: Tensor.Tensor
  xs: List[Float]
  ys: List[Int]
end

## The model, its data, and what the reference backend computed for some SGD
## steps on that one batch: the loss of each step, the gradients of the last,
## the parameters after it.
struct Run
  model: Model
  losses: List[Float]
  grads: List[Tensor.Tensor]
  params: List[Tensor.Tensor]
end

## The worst of the three.
def worst(g: Gaps) -> Float = [g.loss, g.grads, g.params].max.unwrap_or(0.0)

## One SGD step in one graph.
def sgd(backend: Raw.Backend) -> Result[Gaps]
  ref = try reference(try model(SIZES, BATCH), LR, 1)
  steps(backend, ref, LR)
end

## The reference for `BIG_STEPS` SGD steps of the benchmark's model: about
## 16 s per step on the reference backend, so it is made once and shared.
def big_reference() -> Result[Run] = reference(try model(BIG_SIZES, BIG_BATCH), BIG_LR, BIG_STEPS)

## `BIG_STEPS` SGD steps of the benchmark's model in one graph.
def big(backend: Raw.Backend, ref: Run) -> Result[Gaps] = steps(backend, ref, BIG_LR)

## One SGD step as two graphs sharing the parameter and gradient buffers.
def sgd_two(backend: Raw.Backend) -> Result[Gaps]
  ref = try reference(try model(SIZES, BATCH), LR, 1)
  m = ref.model
  two = try Step.sgd_two(m.params, m.x, m.y, LR)
  g = try ready(backend, two.grad, m)
  u = try Lower.compile(try Lower.share_states(try Lower.build(backend, two.update), g))
  loss = try Raw.run(g.program, 0)
  _ = try Raw.run(u.program, 0)
  grads = try read_states(g, m.params.size, m.params.size)
  params = try read_states(g, 0, m.params.size)
  gaps_of([loss], ref, grads, params)
end

## Two Adam steps in one graph (the per-step scalars are inputs), against
## two eager Adam steps of `train/optim.agc`.
def adam(backend: Raw.Backend) -> Result[Gaps]
  m = try model(SIZES, BATCH)
  o = Optim.Adam(lr: 0.01)
  state = try Optim.init(Optim.Optimizer.Adam(o), m.params)
  first = try Step.eager(m.params, m.x, m.y, state)
  after = try Optim.step(state, m.params, first.grads)
  second = try Step.eager(after.params, m.x, m.y, after.state)
  ref = Run(model: m, losses: [first.loss, second.loss], grads: second.grads, params: second.params)
  step = try Step.adam_step(m.params, m.x, m.y, o, true)
  b = try ready(backend, step, m)
  losses: List[Float] = []
  for t in 1..2
    scalars = Step.adam_scalars(o, t)
    try Raw.set(b.program, b.inputs[2], [scalars[0]])
    try Raw.set(b.program, b.inputs[3], [scalars[1]])
    losses = losses.push(try Raw.run(b.program, 0))
  end

  gaps_of(losses, ref, try read_all(b, step.outputs.drop(1)), try read_states(b, 0, m.params.size))
end

## `BIG_STEPS` SGD steps of the benchmark's model on `backend` against the
## same steps on `oracle`, compared in C. Parameters and the batch are made
## in C from fixed seeds, so the run costs no VM steps per value: it fits the
## test runner's budget, where a reference step of this size (16 s) does not.
def cross(backend: Raw.Backend, oracle: Raw.Backend) -> Result[Gaps]
  a = try seeded(backend)
  b = try seeded(oracle)
  la: List[Float] = []
  lb: List[Float] = []
  for i in 0...BIG_STEPS
    la = la.push(try Raw.run(a.program, i))
    lb = lb.push(try Raw.run(b.program, i))
  end

  grads = 0.0
  for i in 1...a.step.outputs.size
    ia = try Lower.index_of(a, a.step.outputs[i])
    ib = try Lower.index_of(b, b.step.outputs[i])
    grads = [grads, try Raw.gap(a.program, ia, b.program, ib)].max.unwrap_or(1.0)
  end

  params = 0.0
  for k in 0...a.states.size
    params = [params, try Raw.gap(a.program, a.states[k], b.program, b.states[k])].max.unwrap_or(1.0)
  end

  ok(Gaps(loss: try rel(la, lb), grads: grads, params: params))
end

## `max |got - want| / max |want|`; an error when either holds a NaN or an
## infinity, or when their sizes differ.
def rel(got: List[Float], want: List[Float]) -> Result[Float]
  return error("compared #{got.size} values with #{want.size}") if got.size != want.size
  scale = 0.000001
  diff = 0.0
  for i in 0...want.size
    if !Types.is_finite(got[i]) or !Types.is_finite(want[i])
      return error("value #{i} is not finite: ggml #{got[i]}, reference #{want[i]}")
    end

    scale = [scale, want[i].abs].max.unwrap_or(scale)
    diff = [diff, (got[i] - want[i]).abs].max.unwrap_or(diff)
  end

  ok(diff / scale)
end

private

def model(sizes: List[Int], batch: Int) -> Result[Model]
  init = Step.init(sizes, 7)
  xs = Step.uniform(batch * sizes[0], 11, 1.0)
  ys = Step.labels(batch, sizes[sizes.size - 1], 13)
  ok(
    Model(
      init: init,
      params: try Step.params_on(init, Tensor.Backend.Reference),
      x: try Step.floats(xs, [batch, sizes[0]], Tensor.Backend.Reference),
      y: try Step.ints(ys, Tensor.Backend.Reference),
      xs: xs,
      ys: ys,
    ),
  )
end

## `count` eager SGD steps on the reference backend, one batch.
def reference(m: Model, lr: Float, count: Int) -> Result[Run]
  state = try Optim.init(Optim.Optimizer.Sgd(Optim.Sgd(lr: lr)), m.params)
  params = m.params
  losses: List[Float] = []
  grads: List[Tensor.Tensor] = []
  for _ in 0...count
    e = try Step.eager(params, m.x, m.y, state)
    losses = losses.push(e.loss)
    grads = e.grads
    params = e.params
  end

  ok(Run(model: m, losses: losses, grads: grads, params: params))
end

## The reference's steps as one compiled SGD program, run once per step.
def steps(backend: Raw.Backend, ref: Run, lr: Float) -> Result[Gaps]
  m = ref.model
  step = try Step.sgd_step(m.params, m.x, m.y, lr, true)
  b = try ready(backend, step, m)
  losses: List[Float] = []
  for i in 0...ref.losses.size
    losses = losses.push(try Raw.run(b.program, i))
  end

  gaps_of(losses, ref, try read_all(b, step.outputs.drop(1)), try read_states(b, 0, m.params.size))
end

def gaps_of(losses: List[Float], ref: Run, grads: List[List[Float]], params: List[List[Float]]) -> Result[Gaps]
  ok(Gaps(loss: try rel(losses, ref.losses), grads: try gaps(grads, ref.grads), params: try gaps(params, ref.params)))
end

## The benchmark's SGD step (gradients as outputs) compiled on `backend`,
## its parameters and one batch made in C from fixed seeds.
def seeded(backend: Raw.Backend) -> Result[Lower.Built]
  shapes: List[Tensor.Tensor] = []
  for i in 0...(BIG_SIZES.size - 1)
    shapes = (
      shapes +
      [
        Step.placeholder([BIG_SIZES[i], BIG_SIZES[i + 1]], Tensor.Dtype.F32),
        Step.placeholder([BIG_SIZES[i + 1]], Tensor.Dtype.F32),
      ]
    )
  end

  x = Step.placeholder([BIG_BATCH, BIG_SIZES[0]], Tensor.Dtype.F32)
  y = Step.placeholder([BIG_BATCH], Tensor.Dtype.I32)
  b = try Lower.lower(backend, try Step.sgd_step(shapes, x, y, BIG_LR, true))
  for k in 0...b.states.size
    scale = 0.1
    if k % 2 == 0
      scale = 1.0 / Math.sqrt(shapes[k].shape[0].to_f)
    end

    try Raw.fill(b.program, b.states[k], 100 + k, scale)
  end

  try Raw.batches(b.program, b.inputs[0], b.inputs[1], BIG_SIZES[BIG_SIZES.size - 1], 1, 7)
  ok(b)
end

## The step lowered and compiled, its parameters and its batch uploaded.
def ready(backend: Raw.Backend, step: Lower.Step, m: Model) -> Result[Lower.Built]
  b = try Lower.lower(backend, step)
  for k in 0...m.params.size
    try Raw.set(b.program, b.states[k], m.init.values[k])
  end

  try Raw.set(b.program, b.inputs[0], m.xs)
  try Raw.set(b.program, b.inputs[1], m.ys.map { |v| v.to_f })
  ok(b)
end

def read_all(b: Lower.Built, ts: List[Tensor.Tensor]) -> Result[List[List[Float]]]
  out: List[List[Float]] = []
  for t in ts
    out = out.push(try Raw.read(b.program, try Lower.index_of(b, t)))
  end

  ok(out)
end

def read_states(b: Lower.Built, from: Int, count: Int) -> Result[List[List[Float]]]
  out: List[List[Float]] = []
  for k in from...(from + count)
    out = out.push(try Raw.read(b.program, b.states[k]))
  end

  ok(out)
end

def gaps(got: List[List[Float]], want: List[Tensor.Tensor]) -> Result[Float]
  return error("compared #{got.size} tensors with #{want.size}") if got.size != want.size
  worst = 0.0
  for k in 0...want.size
    gap = try rel(got[k], try Lower.values_of(want[k]))
    if gap > worst
      worst = gap
    end
  end

  ok(worst)
end
