## The MLP classifier this spike trains, its deterministic data, and its
## training step in three traced forms (`Lower.Step`): SGD in one graph, SGD
## as two graphs (gradients, then the update), and Adam in one graph. The
## eager step `Train.fit` runs today is here too, as the reference.
##
## The model and the loss are the ordinary `Tensor.*` and `Loss.*` calls; a
## traced step calls them on `Trace.constant` tensors and asks
## `Trace.gradients` for the backward, so the graph ggml runs is the one
## `Trace` records, not a second implementation.

import Lower from "lower"
import Loss from "pkg/train/loss"
import Optim from "pkg/train/optim"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Types from "pkg/tensor/src/types"

## The parameters of an MLP with layer sizes `sizes`, as row-major floats:
## `w` is `[in, out]` uniform in `±1/sqrt(in)`, `b` is `[out]` in `±0.1`.
struct Init
  shapes: List[List[Int]]
  values: List[List[Float]]
end

## The traced SGD step as two programs: `grad` writes the gradients into
## state buffers, `update` reads them and replaces the parameters. Their
## parameter and gradient states are one buffer each (`Raw.share`).
struct TwoSteps
  grad: Lower.Step
  update: Lower.Step
end

## The result of the step `Train.fit` runs today: the loss, the gradients and
## the new parameters, computed op by op on the parameters' backend.
struct Eager
  loss: Float
  grads: List[Tensor.Tensor]
  params: List[Tensor.Tensor]
end

## `relu` between layers, nothing after the last.
def logits(params: List[Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  h = x
  layers = params.size / 2
  for i in 0...layers
    h = try Tensor.linear(h, params[2 * i], some(params[2 * i + 1]))
    if i < layers - 1
      h = try Tensor.relu(h)
    end
  end

  ok(h)
end

def loss(params: List[Tensor.Tensor], x: Tensor.Tensor, labels: Tensor.Tensor) -> Result[Tensor.Tensor]
  Loss.cross_entropy(try logits(params, x), labels)
end

## `count` floats uniform in `[-scale, scale)` from `seed`.
def uniform(count: Int, seed: Int, scale: Float) -> List[Float]
  s = seed % 2147483648
  out: List[Float] = []
  for _ in 0...count
    s = (s * 1103515245 + 12345) % 2147483648
    out = out.push(((s.to_f / 2147483648.0) * 2.0 - 1.0) * scale)
  end

  out
end

## `count` labels below `classes` from `seed`.
def labels(count: Int, classes: Int, seed: Int) -> List[Int]
  uniform(count, seed, 1.0).map { |u| ((u + 1.0) * 0.5 * classes.to_f).floor % classes }
end

def init(sizes: List[Int], seed: Int) -> Init
  shapes: List[List[Int]] = []
  values: List[List[Float]] = []
  for i in 0...(sizes.size - 1)
    fan_in = sizes[i]
    shapes = shapes + [[fan_in, sizes[i + 1]], [sizes[i + 1]]]
    values = values.push(uniform(fan_in * sizes[i + 1], seed + 2 * i, 1.0 / Math.sqrt(fan_in.to_f)))
    values = values.push(uniform(sizes[i + 1], seed + 2 * i + 1, 0.1))
  end

  Init(shapes: shapes, values: values)
end

def floats(values: List[Float], shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(values.map { |v| Tensor.Values.Float(v) }, shape, Tensor.Dtype.F32, backend)
end

def ints(values: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(values.map { |v| Tensor.Values.Int(v) }, [values.size], Tensor.Dtype.I32, backend)
end

def params_on(i: Init, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for k in 0...i.shapes.size
    out = out.push(try floats(i.values[k], i.shapes[k], backend))
  end

  ok(out)
end

## A tensor with a shape and no values: what a traced input needs to be
## recorded, when its values only ever live on the ggml side.
def placeholder(shape: List[Int], dtype: Tensor.Dtype) -> Tensor.Tensor
  Types.Tensor(
    shape: shape,
    dtype: dtype,
    backend: Types.Backend.Reference,
    handle: Types.Handle.Plain(Types.Data(values: [])),
  )
end

## SGD in one graph: the loss (and the gradients when `with_grads`) are the
## outputs, every parameter is a state, and its update is a write.
def sgd_step(
  params: List[Tensor.Tensor],
  x: Tensor.Tensor,
  y: Tensor.Tensor,
  lr: Float,
  with_grads: Bool,
) -> Result[Lower.Step]
  ps = params.map { |p| Trace.constant(p) }
  xs = Trace.constant(x)
  ys = Trace.constant(y)
  l = try loss(ps, xs, ys)
  grads = try Trace.gradients(l, ps)
  next = try sgd(ps, grads, lr)
  outputs = [l]
  if with_grads
    outputs = outputs + grads
  end

  ok(Lower.Step(outputs: outputs, states: ps, inputs: [xs, ys], writes: writes(ps, next)))
end

## SGD as two graphs: the first writes the gradients into states, the second
## reads them and writes the parameters.
def sgd_two(params: List[Tensor.Tensor], x: Tensor.Tensor, y: Tensor.Tensor, lr: Float) -> Result[TwoSteps]
  ps = params.map { |p| Trace.constant(p) }
  buffers = params.map { |p| Trace.constant(placeholder(p.shape, p.dtype)) }
  xs = Trace.constant(x)
  ys = Trace.constant(y)
  l = try loss(ps, xs, ys)
  grads = try Trace.gradients(l, ps)
  grad = Lower.Step(outputs: [l], states: ps + buffers, inputs: [xs, ys], writes: writes(buffers, grads))
  ps2 = params.map { |p| Trace.constant(p) }
  buffers2 = params.map { |p| Trace.constant(placeholder(p.shape, p.dtype)) }
  next = try sgd(ps2, buffers2, lr)
  update = Lower.Step(outputs: [], states: ps2 + buffers2, inputs: [], writes: writes(ps2, next))
  ok(TwoSteps(grad: grad, update: update))
end

## Adam in one graph. The moments are states; the two per-step scalars of the
## bias correction (`lr / (1 - beta1^t)` and `1 / (1 - beta2^t)`, from
## `adam_scalars`) are inputs, so one graph serves every step. Outputs: the
## loss, then the gradients when `with_grads`. States: parameters, first
## moments, second moments.
def adam_step(
  params: List[Tensor.Tensor],
  x: Tensor.Tensor,
  y: Tensor.Tensor,
  o: Optim.Adam,
  with_grads: Bool,
) -> Result[Lower.Step]
  ps = params.map { |p| Trace.constant(p) }
  ms = params.map { |p| Trace.constant(placeholder(p.shape, p.dtype)) }
  vs = params.map { |p| Trace.constant(placeholder(p.shape, p.dtype)) }
  lr_t = Trace.constant(placeholder([], Tensor.Dtype.F32))
  c2 = Trace.constant(placeholder([], Tensor.Dtype.F32))
  xs = Trace.constant(x)
  ys = Trace.constant(y)
  l = try loss(ps, xs, ys)
  grads = try Trace.gradients(l, ps)
  ws: List[Lower.Write] = []
  for i in 0...ps.size
    g = grads[i]
    m = try Tensor.add(try scale(ms[i], o.beta1), try scale(g, 1.0 - o.beta1))
    v = try Tensor.add(try scale(vs[i], o.beta2), try scale(try Tensor.mul(g, g), 1.0 - o.beta2))
    denom = try Tensor.sqrt(try Tensor.mul(v, try Tensor.broadcast_to(c2, v.shape)))
    denom = try Tensor.add(denom, try Tensor.full(v.shape, Tensor.Values.Float(o.eps), v.dtype, Tensor.Backend.Trace))
    delta = try Tensor.div(try Tensor.mul(m, try Tensor.broadcast_to(lr_t, m.shape)), denom)
    ws = (
      ws +
      [
        Lower.Write(state: ps[i], value: try Tensor.sub(ps[i], delta)),
        Lower.Write(state: ms[i], value: m),
        Lower.Write(state: vs[i], value: v),
      ]
    )
  end

  outputs = [l]
  if with_grads
    outputs = outputs + grads
  end

  ok(Lower.Step(outputs: outputs, states: ps + ms + vs, inputs: [xs, ys, lr_t, c2], writes: ws))
end

## The bias-corrected learning rate and second-moment factor of Adam's step
## `t` (from 1), as `train/optim.agc` computes them.
def adam_scalars(o: Optim.Adam, t: Int) -> List[Float]
  [o.lr / (1.0 - power(o.beta1, t)), 1.0 / (1.0 - power(o.beta2, t))]
end

## The step `Train.fit` runs: trace, differentiate, evaluate op by op, update.
def eager(params: List[Tensor.Tensor], x: Tensor.Tensor, y: Tensor.Tensor, state: Optim.State) -> Result[Eager]
  f = ->(ps: List[Tensor.Tensor]) { loss(ps, Trace.constant(x), Trace.constant(y)) }
  both = try Trace.value_and_grad(params, f)
  next = try Optim.step(state, params, both.grads)
  values = try Tensor.to_list(both.value)
  case values[0]
    Tensor.Values.Float(v) -> ok(Eager(loss: v, grads: both.grads, params: next.params))
    _ -> error("the loss is not a float")
  end
end

private

def power(x: Float, n: Int) -> Float = (0...n).to_a.reduce(1.0) do |acc, _| acc * x end

def sgd(ps: List[Tensor.Tensor], grads: List[Tensor.Tensor], lr: Float) -> Result[List[Tensor.Tensor]]
  state = try Optim.init(Optim.Optimizer.Sgd(Optim.Sgd(lr: lr)), ps)
  next = try Optim.step(state, ps, grads)
  ok(next.params)
end

def writes(states: List[Tensor.Tensor], values: List[Tensor.Tensor]) -> List[Lower.Write]
  (0...states.size).to_a.map { |i| Lower.Write(state: states[i], value: values[i]) }
end

def scale(t: Tensor.Tensor, c: Float) -> Result[Tensor.Tensor]
  Tensor.mul(t, try Tensor.full(t.shape, Tensor.Values.Float(c), t.dtype, Tensor.Backend.Trace))
end
