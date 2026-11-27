## Optimizers as pure functions: `init` makes the state for a list of
## parameters, `step` takes the state, the parameters and their gradients and
## answers the new state and the new parameters. Nothing mutates; every
## tensor is a `Tensor.*` call, so a step runs eagerly on the parameters'
## backend or traced (`fit` traces it).

import Data from "data"
import Hub from "pkg/hub/hub"
import SafetensorsWrite from "pkg/hub/safetensors_write"
import Tensor from "pkg/tensor/tensor"

PI = 3.141592653589793

## How the learning rate follows the step count `t`, from 0: the optimizer's `lr` times a factor. `WarmupLinear(warmup,
## total)` rises from 0 to 1 over `warmup` steps, then falls straight to 0 at step `total`; `WarmupCosine` falls along a
## half cosine. They are the schedules of Hugging Face's `Trainer` (`get_linear_schedule_with_warmup` and
## `get_cosine_schedule_with_warmup`) up to `total`; after it the factor stays 0, where Hugging Face's cosine rises
## again. `fit` starts a run at step 0, or at the step of the `start` state it is given (`save` and `load`), so a resumed
## run goes on along its schedule.
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

## The optimizers with their usual knobs, for a pipe: `Optim.adamw(lr: 0.00002, schedule: ...)` is
## `Optimizer.AdamW(AdamW(lr: 0.00002, schedule: ...))`. The knobs they leave out (the betas and `eps`) keep the defaults of their struct: build the struct to set them.
def sgd(lr: Float, momentum: Float = 0.0, schedule: Schedule = Schedule.Constant) -> Optimizer
  Optimizer.Sgd(Sgd(lr: lr, momentum: momentum, schedule: schedule))
end

def adam(lr: Float = 0.001, schedule: Schedule = Schedule.Constant) -> Optimizer
  Optimizer.Adam(Adam(lr: lr, schedule: schedule))
end

def adamw(
  lr: Float = 0.001,
  weight_decay: Float = 0.01,
  decay: List[Bool] = [],
  schedule: Schedule = Schedule.Constant,
) -> Optimizer
  Optimizer.AdamW(AdamW(lr: lr, weight_decay: weight_decay, decay: decay, schedule: schedule))
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

## Writes the step count and the moments of `state` to a safetensors file at `path`: the tensors `first.I` and `second.I`
## for parameter `I`, and the metadata `steps` and `optimizer` (`sgd`, `adam` or `adamw`). The knobs and the schedule are
## not saved: the code gives them again to `load`.
def save(state: State, path: String) -> Result[Void]
  tensors: Map[String, Tensor.Tensor] = {}
  for i in 0...state.first.size
    tensors = tensors.put("first.#{i}", state.first[i])
  end

  for i in 0...state.second.size
    tensors = tensors.put("second.#{i}", state.second[i])
  end

  SafetensorsWrite.write(path, tensors, {"steps": "#{state.steps}", "optimizer": kind_of(state.optimizer)})
end

## The state that `save` wrote to `path`, for `optimizer` and `params`: the step count, and the moments on the backend of
## each parameter, in its dtype and with its shape. The knobs and the schedule are the `optimizer`'s, so a state can be
## loaded with another learning rate or schedule (the step count is kept), and an AdamW `decay` list is given again. An
## error names what does not fit: another optimizer, a missing or foreign tensor, a moment of another shape or dtype, a
## step count that is not a number from 0, a header without its metadata. The tensors read before an error are freed.
def load(optimizer: Optimizer, params: List[Tensor.Tensor], path: String) -> Result[State]
  try check_knobs(optimizer, params.size)
  c = try Hub.open(path)
  files = Hub.metadata(c)
  return error("#{path} has no header") if files.empty?

  meta = files[files.keys[0]]
  saved = try meta.get("optimizer").to_result("#{path} has no optimizer in its metadata")
  return error("#{path} holds an #{saved} state, not #{kind_of(optimizer)}") if saved != kind_of(optimizer)

  text = try meta.get("steps").to_result("#{path} has no step count in its metadata")
  steps = try text.to_i
  return error("#{path} has the step count #{steps}") if steps < 0

  wanted = moment_names(optimizer, params.size)
  for name in Hub.names(c)
    if !wanted.include?(name)
      return error("#{path} has the tensor #{name}, which is no moment of #{params.size} parameters")
    end
  end

  first: List[Tensor.Tensor] = []
  second: List[Tensor.Tensor] = []
  for i in 0...params.size
    if uses_first?(optimizer)
      first = first.push(try loaded(c, "first.#{i}", params[i], first + second))
    end

    if uses_second?(optimizer)
      second = second.push(try loaded(c, "second.#{i}", params[i], first + second))
    end
  end

  ok(State(optimizer: optimizer, steps: steps, first: first, second: second))
end

## `state` for `optimizer` and `params`, checked as `init` and `load` check: the knobs of `optimizer`, the same kind of
## optimizer, and one moment of each parameter's shape and dtype. `fit` takes its `start` state through it. The
## optimizer of the result is `optimizer`: the learning rate and the schedule may change on a resume, the step count of
## `state` is kept, and the AdamW `decay` list is the one given here.
def resume(optimizer: Optimizer, state: State, params: List[Tensor.Tensor]) -> Result[State]
  try check_knobs(optimizer, params.size)
  if kind_of(state.optimizer) != kind_of(optimizer)
    return error("the start state holds an #{kind_of(state.optimizer)} state, not #{kind_of(optimizer)}")
  end

  return error("the start state has the step count #{state.steps}") if state.steps < 0
  try check_moments("first", uses_first?(optimizer), state.first, params)
  try check_moments("second", uses_second?(optimizer), state.second, params)
  ok(state.with(optimizer: optimizer))
end

## `scalars` are the inputs of a step that is one compiled program of every step count, where it otherwise holds the
## numbers as constants: the tensors of `step_inputs` (Adam's and AdamW's are rank-1 tensors of 7 numbers, which
## `Tensor.adam_step` takes).
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
      sgd_update(o, state, params, grads, scalars)
    Optimizer.Adam(o) ->
      try check_adam_rows(scalars, 1, "an Adam step takes one input of 7 numbers")
      adam_update(o, none, state, params, grads, scalars)
    Optimizer.AdamW(o) ->
      try check_adam_rows(scalars, 2, "an AdamW step takes two inputs of 7 numbers")
      try check_decay(o, params.size)
      decays = o.decay
      if decays.empty?
        decays = List.repeat(true, params.size)
      end

      adam_update(plain(o), some(decays), state, params, grads, scalars)
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

## The numbers of each input of the step that `Train.fit` and `Train.stablehlo` compile, one list per input: SGD's
## learning rate, and for Adam the 7 numbers of `Tensor.adam_step` (the learning rate with its schedule, `beta1`,
## `beta2`, `eps`, the weight decay, then the two bias corrections); AdamW has two, the second without decay, for the
## parameters that do not decay. A one-number input is a rank-0 tensor, the other a rank-1 tensor.
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

def kind_of(optimizer: Optimizer) -> String
  case optimizer
    Optimizer.Sgd(_) -> "sgd"
    Optimizer.Adam(_) -> "adam"
    Optimizer.AdamW(_) -> "adamw"
  end
end

def uses_first?(optimizer: Optimizer) -> Bool
  case optimizer
    Optimizer.Sgd(o) -> o.momentum != 0.0
    _ -> true
  end
end

def uses_second?(optimizer: Optimizer) -> Bool
  case optimizer
    Optimizer.Sgd(_) -> false
    _ -> true
  end
end

## What `init` checks: the schedule, and for AdamW one decay flag per parameter.
def check_knobs(optimizer: Optimizer, n: Int) -> Result[Void]
  case optimizer
    Optimizer.Sgd(o) -> check_schedule(o.schedule)
    Optimizer.Adam(o) -> check_schedule(o.schedule)
    Optimizer.AdamW(o) ->
      try check_schedule(o.schedule)
      check_decay(o, n)
  end
end

## The tensor names of a saved state of `n` parameters.
def moment_names(optimizer: Optimizer, n: Int) -> List[String]
  out: List[String] = []
  for i in 0...n
    if uses_first?(optimizer)
      out = out.push("first.#{i}")
    end

    if uses_second?(optimizer)
      out = out.push("second.#{i}")
    end
  end

  out
end

## The moment `name` of the file, in the dtype and on the backend of `param`; when it fails the moments of `read` are
## freed.
def loaded(c: Hub.Checkpoint, name: String, param: Tensor.Tensor, read: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  made = Hub.tensor(c, name, Tensor.backend(param), Tensor.dtype(param))
  if !made.ok?
    free_all(read)
    return error(made.message)
  end

  t = made.unwrap!
  fits = check_moment(name, t, param)
  if !fits.ok?
    free_all(read + [t])
    return error(fits.message)
  end

  ok(t)
end

def free_all(tensors: List[Tensor.Tensor]) -> Void
  for t in tensors
    Data.free(t)
  end
end

def check_moment(name: String, moment: Tensor.Tensor, param: Tensor.Tensor) -> Result[Void]
  if Tensor.shape(moment) != Tensor.shape(param)
    return error("#{name} has the shape #{Tensor.shape(moment)}, its parameter #{Tensor.shape(param)}")
  end

  if Tensor.dtype(moment) != Tensor.dtype(param)
    return error("#{name} has the dtype #{Tensor.dtype(moment)}, its parameter #{Tensor.dtype(param)}")
  end

  ok()
end

def check_moments(name: String, used: Bool, moments: List[Tensor.Tensor], params: List[Tensor.Tensor]) -> Result[Void]
  if !used
    return ok() if moments.empty?

    return error("the start state has #{name} moments that its optimizer does not use")
  end

  if moments.size != params.size
    return error("the start state has #{moments.size} #{name} moments for #{params.size} parameters")
  end

  for i in 0...params.size
    try check_moment("#{name}.#{i}", moments[i], params[i])
  end

  ok()
end

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

def sgd_update(
  o: Sgd,
  state: State,
  params: List[Tensor.Tensor],
  grads: List[Tensor.Tensor],
  scalars: List[Tensor.Tensor],
) -> Result[Step]
  first: List[Tensor.Tensor] = []
  out: List[Tensor.Tensor] = []
  for i in 0...params.size
    v = grads[i]
    if o.momentum != 0.0
      v = try Tensor.add(try scale(state.first[i], o.momentum), grads[i])
      first = first.push(v)
    end

    out = out.push(try Tensor.sub(params[i], try Tensor.mul(v, try rate(v, state, scalars))))
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
## state as constants), the row of a parameter that decays first.
def adam_update(
  o: Adam,
  decays: Optional[List[Bool]],
  state: State,
  params: List[Tensor.Tensor],
  grads: List[Tensor.Tensor],
  scalars: List[Tensor.Tensor],
) -> Result[Step]
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

# An Adam step reads none, or `rows` rank-1 tensors of 7 numbers; `rule` says how many.
def check_adam_rows(scalars: List[Tensor.Tensor], rows: Int, rule: String) -> Result[Void]
  return ok() if scalars.empty?
  return error("#{rule}, got #{scalars.size}") if scalars.size != rows
  for s in scalars
    return error("an input of an Adam step has shape [7], got #{Tensor.shape(s)}") if Tensor.shape(s) != [7]
  end

  ok()
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

# The learning rate of an SGD step as a tensor of `like`'s shape: a constant, or the input `scalars[0]` widened.
def rate(like: Tensor.Tensor, state: State, scalars: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  return constant(like, learning_rate(state)) if scalars.empty?
  s = scalars[0]
  if Tensor.dtype(s) != Tensor.dtype(like)
    s = try Tensor.to_dtype(s, Tensor.dtype(like))
  end

  Tensor.broadcast_to(s, Tensor.shape(like))
end

def scale(t: Tensor.Tensor, c: Float) -> Result[Tensor.Tensor] = Tensor.mul(t, try constant(t, c))

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
