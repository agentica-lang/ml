## Optimizers as pure functions: `init` makes the state for a list of
## parameters, `step` takes the state, the parameters and their gradients and
## answers the new state and the new parameters. Nothing mutates; every
## tensor is computed eagerly on the parameters' backend.

import Tensor from "pkg/tensor/tensor"

## `p - lr v`, with `v = momentum v + g` (PyTorch's form); `momentum = 0` is
## plain SGD and keeps no velocity.
struct Sgd
  lr: Float
  momentum: Float = 0.0
end

## Adam (Kingma and Ba), with bias correction.
struct Adam
  lr: Float = 0.001
  beta1: Float = 0.9
  beta2: Float = 0.999
  eps: Float = 0.00000001
end

enum Optimizer
  Sgd(Sgd)
  Adam(Adam)
end

## `first` is SGD's velocity or Adam's first moment, `second` Adam's second
## moment, one tensor per parameter; a list the optimizer does not use is
## empty.
struct State
  optimizer: Optimizer
  steps: Int
  first: List[Tensor.Tensor]
  second: List[Tensor.Tensor]
end

struct Step
  state: State
  params: List[Tensor.Tensor]
end

def init(optimizer: Optimizer, params: List[Tensor.Tensor]) -> Result[State]
  zeros: List[Tensor.Tensor] = []
  for p in params
    zeros = zeros.push(try Tensor.like(p))
  end

  unused: List[Tensor.Tensor] = []
  case optimizer
    Optimizer.Sgd(o) ->
      return ok(State(optimizer: optimizer, steps: 0, first: unused, second: unused)) if o.momentum == 0.0
      ok(State(optimizer: optimizer, steps: 0, first: zeros, second: unused))
    Optimizer.Adam(_) -> ok(State(optimizer: optimizer, steps: 0, first: zeros, second: zeros))
  end
end

def step(state: State, params: List[Tensor.Tensor], grads: List[Tensor.Tensor]) -> Result[Step]
  if grads.size != params.size
    return error("step needs one gradient per parameter, got #{grads.size} for #{params.size}")
  end

  case state.optimizer
    Optimizer.Sgd(o) -> sgd(o, state, params, grads)
    Optimizer.Adam(o) -> adam(o, state, params, grads)
  end
end

private

def sgd(o: Sgd, state: State, params: List[Tensor.Tensor], grads: List[Tensor.Tensor]) -> Result[Step]
  first: List[Tensor.Tensor] = []
  out: List[Tensor.Tensor] = []
  for i in 0...params.size
    v = grads[i]
    if o.momentum != 0.0
      v = try Tensor.add(try scale(state.first[i], o.momentum), grads[i])
      first = first.push(v)
    end

    out = out.push(try Tensor.sub(params[i], try scale(v, o.lr)))
  end

  ok(Step(state: state.with(steps: state.steps + 1, first: first), params: out))
end

def adam(o: Adam, state: State, params: List[Tensor.Tensor], grads: List[Tensor.Tensor]) -> Result[Step]
  t = state.steps + 1
  correct1 = 1.0 / (1.0 - power(o.beta1, t))
  correct2 = 1.0 / (1.0 - power(o.beta2, t))
  first: List[Tensor.Tensor] = []
  second: List[Tensor.Tensor] = []
  out: List[Tensor.Tensor] = []
  for i in 0...params.size
    g = grads[i]
    m = try Tensor.add(try scale(state.first[i], o.beta1), try scale(g, 1.0 - o.beta1))
    v = try Tensor.add(try scale(state.second[i], o.beta2), try scale(try Tensor.mul(g, g), 1.0 - o.beta2))
    denom = try shift(try Tensor.sqrt(try scale(v, correct2)), o.eps)
    delta = try Tensor.div(try scale(m, o.lr * correct1), denom)
    first = first.push(m)
    second = second.push(v)
    out = out.push(try Tensor.sub(params[i], delta))
  end

  ok(Step(state: state.with(steps: t, first: first, second: second), params: out))
end

def scale(t: Tensor.Tensor, c: Float) -> Result[Tensor.Tensor] = Tensor.mul(t, try constant(t, c))

def shift(t: Tensor.Tensor, c: Float) -> Result[Tensor.Tensor] = Tensor.add(t, try constant(t, c))

def constant(t: Tensor.Tensor, c: Float) -> Result[Tensor.Tensor]
  Tensor.full(Tensor.shape(t), Tensor.Values.Float(c), Tensor.dtype(t), Tensor.backend(t))
end

## `x^n` for `n >= 0` by squaring.
def power(x: Float, n: Int) -> Float
  result = 1.0
  base = x
  k = n
  while k > 0
    if k % 2 == 1
      result = result * base
    end

    base = base * base
    k = k / 2
  end

  result
end
