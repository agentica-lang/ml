## The `train/` package: losses (`loss.agc`), optimizers (`optim.agc`) and
## `fit`, a minimal training loop over `tensor/`'s tracing
## (`train/DESIGN.md`).
##
## A model is a loss function of a parameter list and a batch. `fit` traces
## one training step per batch shape (`Trace.compile`), with the parameters,
## the optimizer state and the batch as inputs: the loss, its gradients and the
## optimizer's update, run as one evaluation on the parameters' backend (one
## compiled graph on `Ggml`) on every batch of that shape. The batches come
## from a `Data.Loader` (`data.agc`).

import Data from "data"
import Metrics from "metrics"
import Optim from "optim"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## What `on_epoch` gets after each epoch: its number from 1 and the mean loss
## of its batches.
struct Epoch
  epoch: Int
  loss: Float
end

## The trained parameters and the mean loss of every epoch, in order.
struct Fit
  params: List[Tensor.Tensor]
  history: List[Float]
end

## Trains `params` for `epochs` passes over the batches of `data`, one
## optimizer step per batch. `loss(params, batch)` must answer a float scalar;
## inside it the parameters and the batch are `Trace` tensors. The loader's
## prefetch task, if any, ends before `fit` returns, on every path. With `metrics`,
## every step writes `train/loss`, `train/lr` and `train/step_ms` to that run at
## its step count (from 1); `fit` does not close the run. A metric that
## `Metrics.scalar` refuses (a loss that is not finite, for one) stops `fit`
## with an error that names the epoch and the batch. The parameters and the optimizer state that a step replaces are
## freed at once on libtorch and ggml, except `params` themselves, which stay the caller's, as the trained ones of the
## result are.
def fit(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  data: Data.Loader,
  optimizer: Optim.Optimizer,
  epochs: Int,
  on_epoch: Optional[Fn[Epoch, Void]] = none,
  metrics: Optional[Metrics.Run] = none,
) -> Result[Fit]
  feed = try Data.feed(data, epochs)
  batches = ->(e: Int) {
    current = Data.epoch(feed, e)
    ->(i: Int) { Data.take(current, i) }
  }

  result = trained(
    Run(loss: loss, batches: batches, count: Data.count(data), epochs: epochs, on_epoch: on_epoch, metrics: metrics),
    params,
    optimizer,
  )

  Data.stop(feed)
  result
end

## `fit` over batches that a function makes. The function makes new tensors on every call: `fit_batches` frees a batch
## after its step (`Data.release`), so a batch kept and answered again (a cache, a second epoch) is an error that says
## so. `batch(epoch, i)` answers batch `i` (from 0) of `count` in epoch `epoch` (from 1), for data that is not two
## tensors of one shape, such as sentences padded to the longest of each batch. Each batch may have its own shape; the
## step is compiled once per shape, so few distinct shapes keep the compiles few.
def fit_batches(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  batch: Fn[Int, Int, Result[Data.Batch]],
  count: Int,
  optimizer: Optim.Optimizer,
  epochs: Int,
  on_epoch: Optional[Fn[Epoch, Void]] = none,
  metrics: Optional[Metrics.Run] = none,
) -> Result[Fit]
  return error("fit_batches needs at least one batch, got #{count}") if count < 1
  batches = ->(e: Int) { ->(i: Int) { fresh(batch.call(e, i), e, i) } }
  trained(
    Run(loss: loss, batches: batches, count: count, epochs: epochs, on_epoch: on_epoch, metrics: metrics),
    params,
    optimizer,
  )
end

## The training step of `fit` as one StableHLO program (`Trace.stablehlo`): its arguments are the parameters, the
## optimizer's first and second moments, then the inputs of its step (`Optim.step_inputs`: SGD's rate as a rank-0
## tensor, Adam's 7 numbers or AdamW's two rows of them as rank-1 tensors), so one program is every step. The batch is
## lifted as constants, which are the module's `constants`. It answers the loss, then the new parameters, first moments
## and second moments, as a step of `fit` does. The parameters share one dtype, which the inputs take.
def stablehlo(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  batch: Data.Batch,
  optimizer: Optim.Optimizer,
) -> Result[Trace.Module]
  return error("a step needs at least one parameter") if params.empty?
  state = try Optim.init(optimizer, params)
  n = params.size
  k = state.first.size
  dtype = Tensor.dtype(params[0])
  backend = Tensor.backend(params[0])
  scalars = Optim.step_inputs(state).map { |numbers|
    Trace.Spec(shape: Optim.input_shape(numbers), dtype: dtype, backend: backend)
  }

  specs = (params + state.first + state.second).map { |t| Trace.spec_of(t) } + scalars
  Trace.stablehlo(specs, ->(xs: List[Tensor.Tensor]) { traced_step(loss, state, batch, n, k, xs) })
end

private

## `batches` gives the batches of an epoch (from 1), each by its index below `count`.
struct Run
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]]
  batches: Fn[Int, Fn[Int, Result[Data.Batch]]]
  count: Int
  epochs: Int
  on_epoch: Optional[Fn[Epoch, Void]]
  metrics: Optional[Metrics.Run]
end

## `kept` says, for each moment (the first ones, then the second), whether the step updated it where it lay, so that
## the new one is the old tensor and must not be freed with it.
struct Stepped
  step: Optim.Step
  loss: Float
  kept: List[Bool]
end

## What `run` answers: the fit or its error, and the compiled steps to close, one per batch shape seen.
struct Ran
  result: Result[Fit]
  programs: Map[String, Trace.Compiled]
end

## What `take_step` answers: the step or its error, and the compiled steps, with this batch's shape added when it was new.
struct Taken
  stepped: Result[Stepped]
  programs: Map[String, Trace.Compiled]
end

## The batch a function made, or an error when its tensors are freed already: a batch answered twice.
def fresh(made: Result[Data.Batch], e: Int, i: Int) -> Result[Data.Batch]
  b = try made
  return error("batch(#{e}, #{i}) returned tensors that are already freed: make new ones") if Data.freed?(b)
  ok(b)
end

## The run, then its compiled steps closed.
def trained(r: Run, params: List[Tensor.Tensor], optimizer: Optim.Optimizer) -> Result[Fit]
  ran = run(r, try Optim.init(optimizer, params), params)
  for key in ran.programs.keys
    Trace.close(ran.programs[key])
  end

  ran.result
end

## The tensors of `held` are the run's own: the moments always, and the parameters once a step has made them
## (`owned`). They are freed when a step replaces them and when the run stops on an error.
def run(r: Run, start: Optim.State, params: List[Tensor.Tensor]) -> Ran
  state = start
  current = params
  owned = false
  history: List[Float] = []
  programs: Map[String, Trace.Compiled] = {}
  for e in 1..r.epochs
    take = r.batches.call(e)
    total = 0.0
    for i in 0...r.count
      started = Time.monotonic_ns()
      grown = take_step(r, take, i, state, current, programs)
      programs = grown.programs
      stepped = grown.stepped
      if !stepped.ok?
        release(state, current, owned)
        return Ran(result: error("epoch #{e}, batch #{i}: #{stepped.message}"), programs: programs)
      end

      done = stepped.unwrap!
      rate = Optim.learning_rate(state)
      release_replaced(state, done.kept, current, owned)
      state = done.step.state
      current = done.step.params
      owned = true
      case r.metrics
        some(m) ->
          ms = (Time.monotonic_ns() - started).to_f / 1000000.0
          logged = log_step(m, done, rate, ms)
          if !logged.ok?
            release(state, current, owned)
            return Ran(result: error("epoch #{e}, batch #{i}: #{logged.message}"), programs: programs)
          end

        none -> nil
      end

      total = total + done.loss
    end

    mean = total / r.count.to_f
    history = history.push(mean)
    case r.on_epoch
      some(f) -> f.call(Epoch(epoch: e, loss: mean))
      none -> nil
    end
  end

  release(state, [], false)
  Ran(result: ok(Fit(params: current, history: history)), programs: programs)
end

## Frees what a step replaced: the moments of `state` that the step did not update in place (`kept`), and `params`
## when the run owns them.
def release_replaced(state: Optim.State, kept: List[Bool], params: List[Tensor.Tensor], owned: Bool) -> Void
  moments = state.first + state.second
  for i in 0...moments.size
    if !kept[i]
      Data.free(moments[i])
    end
  end

  if owned
    for t in params
      Data.free(t)
    end
  end
end

## Frees the moments of `state`, and `params` when the run owns them.
def release(state: Optim.State, params: List[Tensor.Tensor], owned: Bool) -> Void
  for t in state.first + state.second
    Data.free(t)
  end

  if owned
    for t in params
      Data.free(t)
    end
  end
end

def log_step(m: Metrics.Run, done: Stepped, lr: Float, ms: Float) -> Result[Void]
  n = done.step.state.steps
  try Metrics.scalar(m, "train/loss", n, done.loss)
  try Metrics.scalar(m, "train/lr", n, lr)
  Metrics.scalar(m, "train/step_ms", n, ms)
end

def take_step(
  r: Run,
  take: Fn[Int, Result[Data.Batch]],
  i: Int,
  state: Optim.State,
  params: List[Tensor.Tensor],
  programs: Map[String, Trace.Compiled],
) -> Taken
  case take.call(i)
    error(e) -> Taken(stepped: error(e), programs: programs)
    ok(batch) ->
      key = shapes_of(batch)
      taken = case compiled_for(r.loss, state, params, batch, key, programs)
        ok(grown) -> Taken(stepped: step(grown[key], state, params, batch), programs: grown)
        error(e) -> Taken(stepped: error(e), programs: programs)
      end

      Data.release(batch)
      taken
  end
end

## `programs` with the step of `batch`'s shape (`key`), compiled now when no batch of that shape came before.
def compiled_for(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  state: Optim.State,
  params: List[Tensor.Tensor],
  batch: Data.Batch,
  key: String,
  programs: Map[String, Trace.Compiled],
) -> Result[Map[String, Trace.Compiled]]
  return ok(programs) if programs.key?(key)
  ok(programs.put(key, try compile_step(loss, state, params, batch)))
end

## The batch's shapes and dtypes: the steps of two batches that agree on them are one compiled graph.
def shapes_of(batch: Data.Batch) -> String = JSON.encode([Trace.spec_of(batch.inputs), Trace.spec_of(batch.targets)])

## The step of `fit` traced and lowered once for batches shaped like `batch`: its arguments are the parameters, the
## optimizer's first and second moments, the inputs of the step (`Optim.step_inputs`), then the batch's inputs and
## targets. The function keeps `state` for its optimizer and for how many moments it has, never for its tensors, which
## are arguments: `run` frees them after the step, except the moments the step updated in place.
def compile_step(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  state: Optim.State,
  params: List[Tensor.Tensor],
  batch: Data.Batch,
) -> Result[Trace.Compiled]
  return error("a step needs at least one parameter") if params.empty?
  n = params.size
  k = state.first.size
  dtype = Tensor.dtype(params[0])
  backend = Tensor.backend(params[0])
  scalars = Optim.step_inputs(state).map { |numbers|
    Trace.Spec(shape: Optim.input_shape(numbers), dtype: dtype, backend: backend)
  }

  moments = (state.first + state.second).map { |t| Trace.spec_of(t) }
  specs = (
    params.map { |t| Trace.spec_of(t) } +
    moments +
    scalars +
    [Trace.spec_of(batch.inputs), Trace.spec_of(batch.targets)]
  )

  Trace.compile(
    specs,
    ->(xs: List[Tensor.Tensor]) {
      data = xs.drop(xs.size - 2)
      traced_step(loss, state, Data.Batch(inputs: data[0], targets: data[1]), n, k, xs.take(xs.size - 2))
    },
  )
end

## One step in its own native arena: only the new state, parameters and the
## loss outlive it (`tensor/DESIGN.md` §8).
def step(
  program: Trace.Compiled,
  state: Optim.State,
  params: List[Tensor.Tensor],
  batch: Data.Batch,
) -> Result[Stepped]
  Native.arena do
    n = params.size
    k = state.first.size
    args = params + state.first + state.second + try scalar_inputs(state, params) + [batch.inputs, batch.targets]
    outs = try Trace.run(program, args)
    moved = state.with(steps: state.steps + 1, first: outs.drop(1 + n).take(k), second: outs.drop(1 + n + k))
    kept = Trace.donated(program).drop(1 + n)
    ok(Stepped(step: Optim.Step(state: moved, params: outs.drop(1).take(n)), loss: try scalar(outs[0]), kept: kept))
  end
end

## The traced step over `xs`, the `n` parameters then the optimizer's `k`
## first and its second moments, then the inputs of its step (`Optim.step_inputs`): the loss, then the new
## parameters, first and second moments.
def traced_step(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  state: Optim.State,
  batch: Data.Batch,
  n: Int,
  k: Int,
  xs: List[Tensor.Tensor],
) -> Result[List[Tensor.Tensor]]
  # `Trace.constant` matters to `stablehlo`, where the batch is a constant of the program; in `fit` it is an
  # input already.
  ps = xs.take(n)
  value = try loss.call(ps, Data.Batch(inputs: Trace.constant(batch.inputs), targets: Trace.constant(batch.targets)))
  grads = try Trace.gradients(value, ps)
  moments = n + k + state.second.size
  held = state.with(first: xs.drop(n).take(k), second: xs.drop(n + k).take(state.second.size))
  stepped = try Optim.step(held, ps, grads, xs.drop(moments))
  ok([value] + stepped.params + stepped.state.first + stepped.state.second)
end

## The numbers of `Optim.step_inputs` as tensors of the dtype and backend of the first parameter: inputs of the step,
## so that a learning rate that changes (a schedule) is not a new graph.
def scalar_inputs(state: Optim.State, params: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  return ok(out) if params.empty?
  for numbers in Optim.step_inputs(state)
    values = numbers.map { |x: Float| Tensor.Values.Float(x) }
    out = out.push(
      try Tensor.from_list(values, Optim.input_shape(numbers), Tensor.dtype(params[0]), Tensor.backend(params[0])),
    )
  end

  ok(out)
end

def scalar(t: Tensor.Tensor) -> Result[Float]
  values = try Tensor.to_list(t)
  return error("a loss must be a scalar, got #{values.size} values") if values.size != 1
  case values[0]
    Tensor.Values.Float(f) -> ok(f)
    _ -> error("a loss must be a float scalar")
  end
end
