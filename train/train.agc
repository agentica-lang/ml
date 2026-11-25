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
## with an error that names the epoch and the batch.
def fit(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  data: Data.Loader,
  optimizer: Optim.Optimizer,
  epochs: Int,
  on_epoch: Optional[Fn[Epoch, Void]] = none,
  metrics: Optional[Metrics.Run] = none,
) -> Result[Fit]
  state = try Optim.init(optimizer, params)
  feed = try Data.feed(data, epochs)
  ran = run(Run(loss: loss, feed: feed, epochs: epochs, on_epoch: on_epoch, metrics: metrics), state, params)
  for key in ran.programs.keys
    Trace.close(ran.programs[key])
  end

  Data.stop(feed)
  ran.result
end

## The training step of `fit` as one StableHLO program (`Trace.stablehlo`): its arguments are the parameters, the
## optimizer's first and second moments, then the `Optim.step_scalars` (SGD's rate, Adam's two numbers or AdamW's
## three) as rank-0 tensors, so one program is every step. The batch is lifted as constants, which are the module's
## `constants`. It answers the loss, then the new parameters, first moments and second moments, as a step of `fit`
## does. The parameters share one dtype, which the scalars take.
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
  scalar = Trace.Spec(shape: [], dtype: Tensor.dtype(params[0]), backend: Tensor.backend(params[0]))
  scalars = Optim.step_scalars(state).map { |_| scalar }
  specs = (params + state.first + state.second).map { |t| Trace.spec_of(t) } + scalars
  Trace.stablehlo(specs, ->(xs: List[Tensor.Tensor]) { traced_step(loss, state, batch, n, k, xs) })
end

private

struct Run
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]]
  feed: Data.Feed
  epochs: Int
  on_epoch: Optional[Fn[Epoch, Void]]
  metrics: Optional[Metrics.Run]
end

struct Stepped
  step: Optim.Step
  loss: Float
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

def run(r: Run, start: Optim.State, params: List[Tensor.Tensor]) -> Ran
  state = start
  current = params
  history: List[Float] = []
  programs: Map[String, Trace.Compiled] = {}
  batches = Data.count(r.feed.loader)
  for e in 1..r.epochs
    feed = Data.epoch(r.feed, e)
    total = 0.0
    for i in 0...batches
      started = Time.monotonic_ns()
      grown = take_step(r, feed, i, state, current, programs)
      programs = grown.programs
      stepped = grown.stepped
      return Ran(result: error("epoch #{e}, batch #{i}: #{stepped.message}"), programs: programs) if !stepped.ok?

      done = stepped.unwrap!
      case r.metrics
        some(m) ->
          ms = (Time.monotonic_ns() - started).to_f / 1000000.0
          logged = log_step(m, done, Optim.learning_rate(state), ms)
          return Ran(result: error("epoch #{e}, batch #{i}: #{logged.message}"), programs: programs) if !logged.ok?
        none -> nil
      end

      state = done.step.state
      current = done.step.params
      total = total + done.loss
    end

    mean = total / batches.to_f
    history = history.push(mean)
    case r.on_epoch
      some(f) -> f.call(Epoch(epoch: e, loss: mean))
      none -> nil
    end
  end

  Ran(result: ok(Fit(params: current, history: history)), programs: programs)
end

def log_step(m: Metrics.Run, done: Stepped, lr: Float, ms: Float) -> Result[Void]
  n = done.step.state.steps
  try Metrics.scalar(m, "train/loss", n, done.loss)
  try Metrics.scalar(m, "train/lr", n, lr)
  Metrics.scalar(m, "train/step_ms", n, ms)
end

def take_step(
  r: Run,
  feed: Data.Feed,
  i: Int,
  state: Optim.State,
  params: List[Tensor.Tensor],
  programs: Map[String, Trace.Compiled],
) -> Taken
  case Data.take(feed, i)
    error(e) -> Taken(stepped: error(e), programs: programs)
    ok(batch) ->
      key = shapes_of(batch)
      grown = programs
      if !programs.key?(key)
        case compile_step(r.loss, state, params, batch)
          ok(program) -> grown = programs.put(key, program)
          error(e) ->
            Data.release(batch)
            return Taken(stepped: error(e), programs: programs)
        end
      end

      done = step(grown[key], state, params, batch)
      Data.release(batch)
      Taken(stepped: done, programs: grown)
  end
end

## The batch's shapes and dtypes: the steps of two batches that agree on them are one compiled graph.
def shapes_of(batch: Data.Batch) -> String = JSON.encode([Trace.spec_of(batch.inputs), Trace.spec_of(batch.targets)])

## The step of `fit` traced and lowered once for batches shaped like `batch`: its arguments are the parameters, the
## optimizer's first and second moments, the scalars of the step, then the batch's inputs and targets.
def compile_step(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  state: Optim.State,
  params: List[Tensor.Tensor],
  batch: Data.Batch,
) -> Result[Trace.Compiled]
  return error("a step needs at least one parameter") if params.empty?
  n = params.size
  k = state.first.size
  scalar = Trace.Spec(shape: [], dtype: Tensor.dtype(params[0]), backend: Tensor.backend(params[0]))
  scalars = Optim.step_scalars(state).map { |_| scalar }
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
    ok(Stepped(step: Optim.Step(state: moved, params: outs.drop(1).take(n)), loss: try scalar(outs[0])))
  end
end

## The traced step over `xs`, the `n` parameters then the optimizer's `k`
## first and its second moments, then the scalars of its step when they are
## inputs (`fit` and `stablehlo`): the loss, then the new parameters, first and second
## moments.
def traced_step(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  state: Optim.State,
  batch: Data.Batch,
  n: Int,
  k: Int,
  xs: List[Tensor.Tensor],
) -> Result[List[Tensor.Tensor]]
  ps = xs.take(n)
  value = try loss.call(ps, Data.Batch(inputs: Trace.constant(batch.inputs), targets: Trace.constant(batch.targets)))
  grads = try Trace.gradients(value, ps)
  moments = n + k + state.second.size
  held = state.with(first: xs.drop(n).take(k), second: xs.drop(n + k).take(state.second.size))
  stepped = try Optim.step(held, ps, grads, xs.drop(moments))
  ok([value] + stepped.params + stepped.state.first + stepped.state.second)
end

## The numbers of `Optim.step_scalars` as rank-0 tensors of the dtype and backend of the first parameter: inputs of the
## step, so that a learning rate that changes (a schedule) is not a new graph.
def scalar_inputs(state: Optim.State, params: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  return ok(out) if params.empty?
  for v in Optim.step_scalars(state)
    out = out.push(
      try Tensor.from_list([Tensor.Values.Float(v)], [], Tensor.dtype(params[0]), Tensor.backend(params[0])),
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
