## One step of a small MLP classifier (8 → 16 → 16 → 4, a batch of 6,
## cross-entropy) on a ggml backend against the step `Train.fit` runs on the
## reference backend: the loss, every gradient and every new parameter.
##
## A gap is `max |ggml - reference| / max |reference|` over one tensor, the
## largest over the tensors of a kind.

import Lower from "lower"
import Optim from "pkg/train/optim"
import Raw from "ggml_graph"
import Step from "step"
import Tensor from "pkg/tensor/tensor"

SIZES = [8, 16, 16, 4]
BATCH = 6
LR = 0.1

struct Gaps
  loss: Float
  grads: Float
  params: Float
end

## The worst of the three.
def worst(g: Gaps) -> Float = [g.loss, g.grads, g.params].max.unwrap_or(0.0)

## One SGD step in one graph.
def sgd(backend: Raw.Backend) -> Result[Gaps]
  m = try model()
  ref = try Step.eager(m.params, m.x, m.y, try Optim.init(Optim.Optimizer.Sgd(Optim.Sgd(lr: LR)), m.params))
  step = try Step.sgd_step(m.params, m.x, m.y, LR, true)
  b = try ready(backend, step, m)
  loss = try Raw.run(b.program, 0)
  grads = try read_all(b, step.outputs.drop(1))
  params = try read_states(b, 0, m.params.size)
  ok(Gaps(loss: rel([loss], [ref.loss]), grads: try gaps(grads, ref.grads), params: try gaps(params, ref.params)))
end

## One SGD step as two graphs sharing the parameter and gradient buffers.
def sgd_two(backend: Raw.Backend) -> Result[Gaps]
  m = try model()
  ref = try Step.eager(m.params, m.x, m.y, try Optim.init(Optim.Optimizer.Sgd(Optim.Sgd(lr: LR)), m.params))
  two = try Step.sgd_two(m.params, m.x, m.y, LR)
  g = try ready(backend, two.grad, m)
  u = try Lower.build(backend, two.update)
  for k in 0...u.states.size
    try Raw.share(u.program, u.states[k], g.program, g.states[k])
  end

  u = try Lower.compile(u)
  loss = try Raw.run(g.program, 0)
  _ = try Raw.run(u.program, 0)
  grads = try read_states(g, m.params.size, m.params.size)
  params = try read_states(g, 0, m.params.size)
  ok(Gaps(loss: rel([loss], [ref.loss]), grads: try gaps(grads, ref.grads), params: try gaps(params, ref.params)))
end

## Two Adam steps in one graph (the per-step scalars are inputs), against
## two eager Adam steps of `train/optim.agc`.
def adam(backend: Raw.Backend) -> Result[Gaps]
  m = try model()
  o = Optim.Adam(lr: 0.01)
  state = try Optim.init(Optim.Optimizer.Adam(o), m.params)
  first = try Step.eager(m.params, m.x, m.y, state)
  after = try Optim.step(state, m.params, first.grads)
  second = try Step.eager(after.params, m.x, m.y, after.state)
  step = try Step.adam_step(m.params, m.x, m.y, o, true)
  b = try ready(backend, step, m)
  loss = 0.0
  for t in 1..2
    scalars = Step.adam_scalars(o, t)
    try Raw.set(b.program, b.inputs[2], [scalars[0]])
    try Raw.set(b.program, b.inputs[3], [scalars[1]])
    loss = try Raw.run(b.program, 0)
  end

  grads = try read_all(b, step.outputs.drop(1))
  params = try read_states(b, 0, m.params.size)
  ok(
    Gaps(
      loss: rel([loss], [second.loss]),
      grads: try gaps(grads, second.grads),
      params: try gaps(params, second.params),
    ),
  )
end

private

struct Model
  init: Step.Init
  params: List[Tensor.Tensor]
  x: Tensor.Tensor
  y: Tensor.Tensor
  xs: List[Float]
  ys: List[Int]
end

def model() -> Result[Model]
  init = Step.init(SIZES, 7)
  xs = Step.uniform(BATCH * SIZES[0], 11, 1.0)
  ys = Step.labels(BATCH, SIZES[SIZES.size - 1], 13)
  ok(
    Model(
      init: init,
      params: try Step.params_on(init, Tensor.Backend.Reference),
      x: try Step.floats(xs, [BATCH, SIZES[0]], Tensor.Backend.Reference),
      y: try Step.ints(ys, Tensor.Backend.Reference),
      xs: xs,
      ys: ys,
    ),
  )
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
  worst = 0.0
  for k in 0...want.size
    expected = try Lower.values_of(want[k])
    gap = rel(got[k], expected)
    if gap > worst
      worst = gap
    end
  end

  ok(worst)
end

def rel(got: List[Float], want: List[Float]) -> Float
  return 1.0 if got.size != want.size
  scale = want.reduce(0.000001) do |acc, v| [acc, v.abs].max.unwrap_or(acc) end
  diff = (0...want.size).to_a.reduce(0.0) do |acc, i| [acc, (got[i] - want[i]).abs].max.unwrap_or(acc) end
  diff / scale
end
