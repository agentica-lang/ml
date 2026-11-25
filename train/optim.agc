## Optimizers as pure functions: `init` makes the state for a list of
## parameters, `step` takes the state, the parameters and their gradients and
## answers the new state and the new parameters. Nothing mutates; every
## tensor is a `Tensor.*` call, so a step runs eagerly on the parameters'
## backend or traced (`fit` traces it).

import Tensor from "pkg/tensor/tensor"

PI = 3.141592653589793

## How the learning rate follows the step count `t`, from 0: the optimizer's `lr` times a factor. `WarmupLinear(warmup,
## total)` rises from 0 to 1 over `warmup` steps, then falls straight to 0 at step `total`; `WarmupCosine` falls along a
## half cosine. They are the schedules of Hugging Face's `Trainer` (`get_linear_schedule_with_warmup` and
## `get_cosine_schedule_with_warmup`) up to `total`; after it the factor stays 0, where Hugging Face's cosine rises
## again. `fit` starts every run at step 0, so a resumed run starts its warmup again.
enum Schedule
  Constant
  WarmupLinear(Int, Int)
  WarmupCosine(Int, Int)
end

## `p - lr v`, with `v = momentum v + g` (PyTorch's form); `momentum = 0` is
## plain SGD and keeps no velocity.
struct Sgd
  lr: Float
  momentum: Float = 0.0
  schedule: Schedule = Schedule.Constant
end

## Adam (Kingma and Ba), with bias correction.
struct Adam
  lr: Float = 0.001
  beta1: Float = 0.9
  beta2: Float = 0.999
  eps: Float = 0.00000001
  schedule: Schedule = Schedule.Constant
end

## Adam with decoupled weight decay (Loshchilov and Hutter), as PyTorch's `AdamW`: `p (1 - lr weight_decay)`, then
## Adam's step from it. `decay` says which parameters decay, one flag per parameter in the order of the parameters;
## empty, all of them do. A fine-tune leaves the biases and the layer-norm weights out; with the names of a
## `Params.split`, whose `trainable` list follows `names`:
##
##   norm = ->(name: String) { name.include?("LayerNorm") or name.include?("layer_norm") }
##   decay = split.names.map { |name: String| !(name.end_with?(".bias") or norm.call(name)) }
struct AdamW
  lr: Float = 0.001
  beta1: Float = 0.9
  beta2: Float = 0.999
  eps: Float = 0.00000001
  weight_decay: Float = 0.01
  decay: List[Bool] = []
  schedule: Schedule = Schedule.Constant
end

enum Optimizer
  Sgd(Sgd)
  Adam(Adam)
  AdamW(AdamW)
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
  unused: List[Tensor.Tensor] = []
  case optimizer
    Optimizer.Sgd(o) ->
      try check_schedule(o.schedule)
      return ok(State(optimizer: optimizer, steps: 0, first: unused, second: unused)) if o.momentum == 0.0
      ok(State(optimizer: optimizer, steps: 0, first: try zeros_like(params), second: unused))
    Optimizer.Adam(o) ->
      try check_schedule(o.schedule)
      ok(State(optimizer: optimizer, steps: 0, first: try zeros_like(params), second: try zeros_like(params)))
    Optimizer.AdamW(o) ->
      try check_schedule(o.schedule)
      try check_decay(o, params.size)
      ok(State(optimizer: optimizer, steps: 0, first: try zeros_like(params), second: try zeros_like(params)))
  end
end

## `scalars` are the inputs of a step that is one compiled program of every step count, where it otherwise holds the
## numbers as constants. `Train.fit` passes the tensors of `step_inputs` (Adam's and AdamW's are rank-1 tensors of 7
## numbers, which `Tensor.adam_step` takes), and `Train.stablehlo` the numbers of `step_scalars` as rank-0 tensors.
def step(
  state: State,
  params: List[Tensor.Tensor],
  grads: List[Tensor.Tensor],
  scalars: List[Tensor.Tensor] = [],
) -> Result[Step]
  if grads.size != params.size
    return error("step needs one gradient per parameter, got #{grads.size} for #{params.size}")
  end

  case state.optimizer
    Optimizer.Sgd(o) ->
      try check_scalars(scalars, 1, "an SGD step takes one scalar")
      sgd(o, state, params, grads, scalars)
    Optimizer.Adam(o) ->
      try check_adam_scalars(scalars, 1, 2, "an Adam step takes one input of 7 numbers, or two scalars")
      adam(o, none, state, params, grads, scalars)
    Optimizer.AdamW(o) ->
      try check_adam_scalars(scalars, 2, 3, "an AdamW step takes two inputs of 7 numbers, or three scalars")
      try check_decay(o, params.size)
      decays = o.decay
      if decays.empty?
        decays = List.repeat(true, params.size)
      end

      adam(plain(o), some(decays), state, params, grads, scalars)
  end
end

## The learning rate of the next step of `state`: the optimizer's `lr` times its schedule at the step count.
def learning_rate(state: State) -> Float
  case state.optimizer
    Optimizer.Sgd(o) -> o.lr * rate_factor(o.schedule, state.steps)
    Optimizer.Adam(o) -> o.lr * rate_factor(o.schedule, state.steps)
    Optimizer.AdamW(o) -> o.lr * rate_factor(o.schedule, state.steps)
  end
end

## The numbers the next step of `state` depends on, as `step` reads them from its `scalars`: SGD's learning rate, and
## Adam's `lr / (1 - beta1^t)` and `1 / (1 - beta2^t)` for the step count `t` (its learning rate, with the bias
## correction). AdamW has Adam's two, then `lr weight_decay`.
def step_scalars(state: State) -> List[Float]
  case state.optimizer
    Optimizer.Sgd(_) -> [learning_rate(state)]
    Optimizer.Adam(o) -> adam_scalars(o, state.steps)
    Optimizer.AdamW(o) -> adam_scalars(plain(o), state.steps).push(learning_rate(state) * o.weight_decay)
  end
end

## The numbers of each input of the step that `Train.fit` compiles, one list per input: SGD's learning rate, and for
## Adam the 7 numbers of `Tensor.adam_step` (the learning rate with its schedule, `beta1`, `beta2`, `eps`, the weight
## decay, then the two bias corrections); AdamW has two, the second without decay, for the parameters that do not
## decay. A one-number input is a rank-0 tensor, the other a rank-1 tensor.
def step_inputs(state: State) -> List[List[Float]]
  case state.optimizer
    Optimizer.Sgd(_) -> [[learning_rate(state)]]
    Optimizer.Adam(o) -> [adam_numbers(o, state.steps, 0.0)]
    Optimizer.AdamW(o) ->
      [adam_numbers(plain(o), state.steps, o.weight_decay), adam_numbers(plain(o), state.steps, 0.0)]
  end
end

## The shape of the input of `step_inputs` that holds `numbers`: one number is a rank-0 tensor, more are a rank-1 one.
def input_shape(numbers: List[Float]) -> List[Int]
  return [] if numbers.size == 1
  [numbers.size]
end

private

def zeros_like(params: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  zeros: List[Tensor.Tensor] = []
  for p in params
    zeros = zeros.push(try Tensor.like(p))
  end

  ok(zeros)
end

def check_decay(o: AdamW, n: Int) -> Result[Void]
  if !o.decay.empty? and o.decay.size != n
    return error("AdamW needs one decay flag per parameter, got #{o.decay.size} for #{n}")
  end

  ok()
end

def check_schedule(s: Schedule) -> Result[Void]
  case s
    Schedule.Constant -> ok()
    Schedule.WarmupLinear(warmup, total) -> check_steps(warmup, total)
    Schedule.WarmupCosine(warmup, total) -> check_steps(warmup, total)
  end
end

def check_steps(warmup: Int, total: Int) -> Result[Void]
  if warmup < 0 or total < warmup or total < 1
    return error("a schedule needs 0 <= warmup <= total and total >= 1, got #{warmup} and #{total}")
  end

  ok()
end

def rate_factor(s: Schedule, t: Int) -> Float
  case s
    Schedule.Constant -> 1.0
    Schedule.WarmupLinear(warmup, total) ->
      return t.to_f / [1, warmup].max.unwrap!.to_f if t < warmup
      [0.0, (total - t).to_f / [1, total - warmup].max.unwrap!.to_f].max.unwrap!
    Schedule.WarmupCosine(warmup, total) ->
      return t.to_f / [1, warmup].max.unwrap!.to_f if t < warmup
      progress = [1.0, (t - warmup).to_f / [1, total - warmup].max.unwrap!.to_f].min.unwrap!
      0.5 * (1.0 + Math.cos(PI * progress))
  end
end

def sgd(
  o: Sgd,
  state: State,
  params: List[Tensor.Tensor],
  grads: List[Tensor.Tensor],
  scalars: List[Tensor.Tensor],
) -> Result[Step]
  held = step_scalars(state)
  first: List[Tensor.Tensor] = []
  out: List[Tensor.Tensor] = []
  for i in 0...params.size
    v = grads[i]
    if o.momentum != 0.0
      v = try Tensor.add(try scale(state.first[i], o.momentum), grads[i])
      first = first.push(v)
    end

    out = out.push(try Tensor.sub(params[i], try Tensor.mul(v, try factor(v, held, scalars, 0))))
  end

  ok(Step(state: state.with(steps: state.steps + 1, first: first), params: out))
end

def adam_numbers(o: Adam, steps: Int, decay: Float) -> List[Float]
  lr = o.lr * rate_factor(o.schedule, steps)
  [
    lr,
    o.beta1,
    o.beta2,
    o.eps,
    decay,
    1.0 / (1.0 - power(o.beta1, steps + 1)),
    1.0 / (1.0 - power(o.beta2, steps + 1)),
  ]
end

def adam_scalars(o: Adam, steps: Int) -> List[Float]
  correct1 = 1.0 / (1.0 - power(o.beta1, steps + 1))
  correct2 = 1.0 / (1.0 - power(o.beta2, steps + 1))
  [o.lr * rate_factor(o.schedule, steps) * correct1, correct2]
end

## The Adam of the moments of `o`, without its decay.
def plain(o: AdamW) -> Adam = Adam(lr: o.lr, beta1: o.beta1, beta2: o.beta2, eps: o.eps, schedule: o.schedule)

def decays?(decays: Optional[List[Bool]], i: Int) -> Bool
  case decays
    some(flags) -> flags[i]
    none -> false
  end
end

## Adam's step, and AdamW's with `decays`: the flag of each parameter, which shrinks by `lr weight_decay` of itself.
## The update is `Tensor.adam_step` with the rows of `scalars` (`step_inputs`; without them, those of the optimizer's
## state as constants), the row of a parameter that decays first. Rank-0 `scalars` (`Train.stablehlo`) are the older
## composition of the same formula (`adam_of_scalars`), which the StableHLO goldens pin.
def adam(
  o: Adam,
  decays: Optional[List[Bool]],
  state: State,
  params: List[Tensor.Tensor],
  grads: List[Tensor.Tensor],
  scalars: List[Tensor.Tensor],
) -> Result[Step]
  if !scalars.empty? and Tensor.shape(scalars[0]).empty?
    return adam_of_scalars(o, decays, state, params, grads, scalars)
  end

  rows = scalars
  if rows.empty?
    rows = try number_rows(state, params)
  end

  first: List[Tensor.Tensor] = []
  second: List[Tensor.Tensor] = []
  out: List[Tensor.Tensor] = []
  for i in 0...params.size
    row = rows[0]
    if !decays?(decays, i) and rows.size > 1
      row = rows[1]
    end

    if Tensor.dtype(row) != Tensor.dtype(params[i])
      row = try Tensor.to_dtype(row, Tensor.dtype(params[i]))
    end

    moved = try Tensor.adam_step(params[i], grads[i], state.first[i], state.second[i], row)
    out = out.push(moved[0])
    first = first.push(moved[1])
    second = second.push(moved[2])
  end

  ok(Step(state: state.with(steps: state.steps + 1, first: first, second: second), params: out))
end

## `step_inputs` as constants on the backend and dtype of the first parameter.
def number_rows(state: State, params: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  rows: List[Tensor.Tensor] = []
  return ok(rows) if params.empty?
  for numbers in step_inputs(state)
    values = numbers.map { |x: Float| Tensor.Values.Float(x) }
    rows = rows.push(try Tensor.from_list(values, [numbers.size], Tensor.dtype(params[0]), Tensor.backend(params[0])))
  end

  ok(rows)
end

def adam_of_scalars(
  o: Adam,
  decays: Optional[List[Bool]],
  state: State,
  params: List[Tensor.Tensor],
  grads: List[Tensor.Tensor],
  scalars: List[Tensor.Tensor],
) -> Result[Step]
  t = state.steps + 1
  held = step_scalars(state)
  first: List[Tensor.Tensor] = []
  second: List[Tensor.Tensor] = []
  out: List[Tensor.Tensor] = []
  for i in 0...params.size
    g = grads[i]
    m = try Tensor.add(try scale(state.first[i], o.beta1), try scale(g, 1.0 - o.beta1))
    v = try Tensor.add(try scale(state.second[i], o.beta2), try scale(try Tensor.mul(g, g), 1.0 - o.beta2))
    denom = try shift(try Tensor.sqrt(try Tensor.mul(v, try factor(v, held, scalars, 1))), o.eps)
    delta = try Tensor.div(try Tensor.mul(m, try factor(m, held, scalars, 0)), denom)
    first = first.push(m)
    second = second.push(v)
    p = params[i]
    if decays?(decays, i)
      p = try Tensor.sub(p, try Tensor.mul(p, try factor(p, held, scalars, 2)))
    end

    out = out.push(try Tensor.sub(p, delta))
  end

  ok(Step(state: state.with(steps: t, first: first, second: second), params: out))
end

# An Adam step reads none, `rows` rank-1 tensors of 7 numbers, or `n` rank-0 tensors.
def check_adam_scalars(scalars: List[Tensor.Tensor], rows: Int, n: Int, rule: String) -> Result[Void]
  return ok() if scalars.empty?
  if Tensor.shape(scalars[0]) == [7]
    return error("#{rule}, got #{scalars.size}") if scalars.size != rows
    for s in scalars
      return error("an input of an Adam step has shape [7], got #{Tensor.shape(s)}") if Tensor.shape(s) != [7]
    end

    return ok()
  end

  check_scalars(scalars, n, rule)
end

# A step reads none, or `n` rank-0 tensors; `rule` says how many ("an SGD step takes one scalar").
def check_scalars(scalars: List[Tensor.Tensor], n: Int, rule: String) -> Result[Void]
  return ok() if scalars.empty?
  return error("#{rule}, got #{scalars.size}") if scalars.size != n
  for s in scalars
    return error("a scalar of a step is a rank-0 tensor, got shape #{Tensor.shape(s)}") if !Tensor.shape(s).empty?
  end

  ok()
end

# The scalar `i` of `held` as a tensor of `like`'s shape: a constant, or the input `scalars[i]` widened.
def factor(like: Tensor.Tensor, held: List[Float], scalars: List[Tensor.Tensor], i: Int) -> Result[Tensor.Tensor]
  return constant(like, held[i]) if scalars.empty?
  s = scalars[i]
  if Tensor.dtype(s) != Tensor.dtype(like)
    s = try Tensor.to_dtype(s, Tensor.dtype(like))
  end

  Tensor.broadcast_to(s, Tensor.shape(like))
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
