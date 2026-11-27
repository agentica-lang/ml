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
import Telemetry from "pkg/telemetry/telemetry"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## What `on_epoch` gets after each epoch: its number from 1, the mean loss of its batches, the step count of the run so
## far, and the parameters the epoch ended with. The parameters are the run's own: use them inside `on_epoch` (the next
## step frees them), and copy what must outlive it. `on_epoch` answers a `Control`: an `error` stops the run with an
## error that names the epoch.
struct Epoch
  epoch: Int
  loss: Float
  steps: Int
  params: List[Tensor.Tensor]
end

## What a hook answers: `Continue` goes on, `Stop` ends the run after the step (or the epoch) it follows, and `fit`
## answers its `Fit` as usual: the parameters so far and the history (the last epoch's mean is over the steps it ran).
enum Control
  Continue
  Stop
end

## What `on_step` gets after each step: the epoch (from 1) and the batch in it (from 0), the step count of the run (from
## 1), the loss of the step, and the parameters it made. The parameters are the run's own: use them inside `on_step` (the
## next step frees them), and copy what must outlive it.
struct StepInfo
  epoch: Int
  batch: Int
  step: Int
  loss: Float
  params: List[Tensor.Tensor]
end

## The trained parameters and the mean loss of every epoch, in order. `metrics` is what `Loop` measured: the value of
## each of its metrics at the end of every epoch, by name (empty for a plain `fit`).
struct Fit
  params: List[Tensor.Tensor]
  history: List[Float]
  metrics: Map[String, List[Float]] = {}
end

## Trains `params` for `epochs` passes over the batches of `data`, one
## optimizer step per batch. `loss(params, batch)` must answer a float scalar;
## inside it the parameters and the batch are `Trace` tensors. The loader's
## prefetch task, if any, ends before `fit` returns, on every path. With `on_step`, every step is followed by a call
## with its `StepInfo`, and `on_epoch` is called after every epoch: each answers a `Control`, and `Stop` ends the run
## there with its `Fit`; an `error` stops it with an error that names the epoch (and the batch, for `on_step`). With
## `metrics`, every step writes `train/loss`, `train/lr` and `train/step_ms` to that run at
## its step count (from 1); `fit` does not close the run. A metric that
## `Metrics.scalar` refuses (a loss that is not finite, for one) stops `fit`
## with an error that names the epoch and the batch. The parameters and the optimizer state that a step replaces are
## freed at once on libtorch, ggml and XLA, except `params` themselves, which stay the caller's, as the trained ones of the
## result are.
def fit(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  data: Data.Loader,
  optimizer: Optim.Optimizer,
  epochs: Int,
  on_epoch: Optional[Fn[Epoch, Result[Control]]] = none,
  metrics: Optional[Metrics.Run] = none,
  on_step: Optional[Fn[StepInfo, Result[Control]]] = none,
  run: String = "",
) -> Result[Fit]
  if gathers_in_graph?(data, params)
    return fit_indexed(loss, params, data, optimizer, epochs, on_epoch, metrics, on_step, run)
  end

  feed = try Data.feed(data, epochs)
  batches = ->(e: Int) {
    current = Data.epoch(feed, e)
    ->(i: Int) { Data.take(current, i) }
  }

  result = trained(
    Run(
      loss: loss,
      batches: batches,
      count: Data.count(data),
      epochs: epochs,
      on_epoch: on_epoch,
      metrics: metrics,
      on_step: on_step,
      name: run,
    ),
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
##
## With `prefetch > 0`, a task calls `batch` for every batch of every epoch, in order, building up to `prefetch + 1` ahead,
## while the steps run: the decoding of image files (`Vision.samples`) overlaps the device. `batch` then runs on that
## task, and `epochs * count` calls happen in order whatever the steps do; the task ends before `fit_batches` returns,
## on every path. A batch that failed reaches the fit as the error of its step.
def fit_batches(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  batch: Fn[Int, Int, Result[Data.Batch]],
  count: Int,
  optimizer: Optim.Optimizer,
  epochs: Int,
  on_epoch: Optional[Fn[Epoch, Result[Control]]] = none,
  metrics: Optional[Metrics.Run] = none,
  prefetch: Int = 0,
  on_step: Optional[Fn[StepInfo, Result[Control]]] = none,
  run: String = "",
) -> Result[Fit]
  return error("fit_batches needs at least one batch, got #{count}") if count < 1
  return error("prefetch must be 0 or more, got #{prefetch}") if prefetch < 0
  run_of = ->(batches: Fn[Int, Fn[Int, Result[Data.Batch]]]) {
    Run(
      loss: loss,
      batches: batches,
      count: count,
      epochs: epochs,
      on_epoch: on_epoch,
      metrics: metrics,
      on_step: on_step,
      name: run,
    )
  }

  if prefetch == 0
    return trained(run_of.call(->(e: Int) { ->(i: Int) { fresh(batch.call(e, i), e, i) } }), params, optimizer)
  end

  stream = try Data.ahead(batch, count, epochs, prefetch)
  result = trained(run_of.call(->(_e: Int) { ->(_i: Int) { Data.next(stream) } }), params, optimizer)
  Data.finish(stream)
  result
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

## `batches` gives the batches of an epoch (from 1), each by its index below `count`. With `indexed`, a batch is the
## dataset's own tensors and the ids of its rows (`index`): the step gathers them in its graph, and frees only the ids.
struct Run
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]]
  batches: Fn[Int, Fn[Int, Result[Data.Batch]]]
  count: Int
  epochs: Int
  on_epoch: Optional[Fn[Epoch, Result[Control]]]
  metrics: Optional[Metrics.Run]
  on_step: Optional[Fn[StepInfo, Result[Control]]]
  name: String = ""
  indexed: Bool = false
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

## The events of a run for `telemetry` handlers: `train.step` after each step and `train.epoch` after each epoch, with
## the run name as `run` in the metadata. Nothing is built when no handler hears them.
def announce_step(name: String, epoch: Int, step: Int, loss: Float, rate: Float) -> Void
  if Telemetry.enabled?("train.step")
    Telemetry.execute(
      "train.step",
      ints: {"step_count": step, "epoch_count": epoch},
      floats: {"loss": loss, "lr": rate},
      metadata: {"run": name},
    )
  end
end

def announce_epoch(name: String, epoch: Int, step: Int, loss: Float) -> Void
  if Telemetry.enabled?("train.epoch")
    Telemetry.execute(
      "train.epoch",
      ints: {"epoch_count": epoch, "step_count": step},
      floats: {"loss": loss},
      metadata: {"run": name},
    )
  end
end

## Whether `fit` gathers the rows of a batch inside the step: on ggml and XLA, where a gather in its own graph costs a
## launch (on XLA a compile per batch shape) and a readback of the batch, when the dataset lives where the parameters
## do.
def gathers_in_graph?(data: Data.Loader, params: List[Tensor.Tensor]) -> Bool
  return false if params.empty?
  backend = Tensor.backend(params[0])
  case backend
    Tensor.Backend.Ggml(_) ->
      Tensor.backend(data.dataset.inputs) == backend and Tensor.backend(data.dataset.targets) == backend
    Tensor.Backend.Xla(_) ->
      Tensor.backend(data.dataset.inputs) == backend and Tensor.backend(data.dataset.targets) == backend
    _ -> false
  end
end

## `fit` for a loader that gathers in the step. The batches are the same, in the same order; no prefetch task is
## needed, since a batch is only its row ids (the loader's checks run as for any `fit`).
def fit_indexed(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  data: Data.Loader,
  optimizer: Optim.Optimizer,
  epochs: Int,
  on_epoch: Optional[Fn[Epoch, Result[Control]]],
  metrics: Optional[Metrics.Run],
  on_step: Optional[Fn[StepInfo, Result[Control]]],
  run: String,
) -> Result[Fit]
  return error("prefetch must be 0 or more, got #{data.prefetch}") if data.prefetch < 0
  Data.stop(try Data.feed(data.with(prefetch: 0), epochs))
  batches = ->(e: Int) {
    order = Data.order(data, e)
    ->(i: Int) { row_ids(data, order, i) }
  }

  trained(
    Run(
      loss: loss,
      batches: batches,
      count: Data.count(data),
      epochs: epochs,
      on_epoch: on_epoch,
      metrics: metrics,
      on_step: on_step,
      name: run,
      indexed: true,
    ),
    params,
    optimizer,
  )
end

## The dataset's tensors and the ids of the rows of batch `i` of an epoch whose rows are `order`.
def row_ids(data: Data.Loader, order: List[Int], i: Int) -> Result[Data.Batch]
  return error("batch #{i} is past the #{Data.count(data)} of an epoch") if i < 0 or i >= Data.count(data)
  d = data.dataset
  from = i * data.batch_size
  to = [from + data.batch_size, Data.size(d)].min.unwrap!
  picked = (from...to).map { |k| Tensor.Values.Int(order[k]) }
  at = try Tensor.from_list(picked, [to - from], Tensor.Dtype.I32, Tensor.backend(d.inputs))
  ok(Data.Batch(inputs: d.inputs, targets: d.targets, index: some(at)))
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
      announce_step(r.name, e, state.steps, done.loss, rate)
      case r.on_step
        some(f) ->
          called = f.call(StepInfo(epoch: e, batch: i, step: state.steps, loss: done.loss, params: current))
          if !called.ok?
            release(state, current, owned)
            return Ran(result: error("epoch #{e}, batch #{i}: #{called.message}"), programs: programs)
          end

          if called.unwrap! == Control.Stop
            release(state, [], false)
            history = history.push(total / (i + 1).to_f)
            return Ran(result: ok(Fit(params: current, history: history)), programs: programs)
          end

        none -> nil
      end
    end

    mean = total / r.count.to_f
    history = history.push(mean)
    announce_epoch(r.name, e, state.steps, mean)
    case r.on_epoch
      some(f) ->
        called = f.call(Epoch(epoch: e, loss: mean, steps: state.steps, params: current))
        if !called.ok?
          release(state, current, owned)
          return Ran(result: error("epoch #{e}: #{called.message}"), programs: programs)
        end

        if called.unwrap! == Control.Stop
          release(state, [], false)
          return Ran(result: ok(Fit(params: current, history: history)), programs: programs)
        end

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
      key = shapes_of(batch, r.indexed)
      taken = case compiled_for(r.loss, state, params, batch, r.indexed, key, programs)
        ok(grown) -> Taken(stepped: step(grown[key], state, params, batch, r.indexed), programs: grown)
        error(e) -> Taken(stepped: error(e), programs: programs)
      end

      release_batch(batch, r.indexed)
      taken
  end
end

## `programs` with the step of `batch`'s shape (`key`), compiled now when no batch of that shape came before.
def compiled_for(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  state: Optim.State,
  params: List[Tensor.Tensor],
  batch: Data.Batch,
  indexed: Bool,
  key: String,
  programs: Map[String, Trace.Compiled],
) -> Result[Map[String, Trace.Compiled]]
  return ok(programs) if programs.key?(key)
  ok(programs.put(key, try compile_step(loss, state, params, batch, indexed)))
end

## Frees what a step is done with: the batch's tensors, or for an indexed one the row ids alone (the dataset is the
## caller's).
def release_batch(batch: Data.Batch, indexed: Bool) -> Void
  return Data.release(batch) if !indexed
  case batch.index
    some(at) -> Data.free(at)
    none -> nil
  end
end

## The batch's shapes and dtypes: the steps of two batches that agree on them are one compiled graph. An indexed batch
## is its row ids; the dataset is the same for the whole run.
def shapes_of(batch: Data.Batch, indexed: Bool) -> String
  case batch.index
    some(at) -> return JSON.encode([Trace.spec_of(at)]) if indexed
    none -> nil
  end

  JSON.encode([Trace.spec_of(batch.inputs), Trace.spec_of(batch.targets)])
end

## The step of `fit` traced and lowered once for batches shaped like `batch`: its arguments are the parameters, the
## optimizer's first and second moments, the inputs of the step (`Optim.step_inputs`), then the batch's inputs and
## targets. The function keeps `state` for its optimizer and for how many moments it has, never for its tensors, which
## are arguments: `run` frees them after the step, except the moments the step updated in place.
def compile_step(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  state: Optim.State,
  params: List[Tensor.Tensor],
  batch: Data.Batch,
  indexed: Bool,
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
  fed = try step_batch_specs(batch, indexed)
  specs = params.map { |t| Trace.spec_of(t) } + moments + scalars + fed
  Trace.compile(
    specs,
    ->(xs: List[Tensor.Tensor]) {
      data = xs.drop(xs.size - fed.size)
      made = try step_batch(batch, indexed, data)
      traced_step(loss, state, made, n, k, xs.take(xs.size - fed.size))
    },
  )
end

## The inputs a step takes for its batch: its two tensors, or the row ids alone for an indexed one.
def step_batch_specs(batch: Data.Batch, indexed: Bool) -> Result[List[Trace.Spec]]
  if indexed
    case batch.index
      some(at) -> return ok([Trace.spec_of(at)])
      none -> return error("an indexed batch has no row ids")
    end
  end

  ok([Trace.spec_of(batch.inputs), Trace.spec_of(batch.targets)])
end

## The batch the traced step gives its loss: its inputs, or the rows of the dataset (constants of the program, bound
## once) that the row ids `data[0]` name.
def step_batch(batch: Data.Batch, indexed: Bool, data: List[Tensor.Tensor]) -> Result[Data.Batch]
  return ok(Data.Batch(inputs: data[0], targets: data[1])) if !indexed
  ok(
    Data.Batch(
      inputs: try Tensor.gather(Trace.constant(batch.inputs), data[0]),
      targets: try Tensor.gather(Trace.constant(batch.targets), data[0]),
    ),
  )
end

## The tensors of `batch` that a step takes as inputs: its two tensors, or the row ids alone for an indexed one.
def step_batch_tensors(batch: Data.Batch, indexed: Bool) -> List[Tensor.Tensor]
  return batch.index.map { |at| [at] }.unwrap_or([]) if indexed

  [batch.inputs, batch.targets]
end

## One step in its own native arena: only the new state, parameters and the
## loss outlive it (`tensor/DESIGN.md` §8).
def step(
  program: Trace.Compiled,
  state: Optim.State,
  params: List[Tensor.Tensor],
  batch: Data.Batch,
  indexed: Bool,
) -> Result[Stepped]
  Native.arena do
    n = params.size
    k = state.first.size
    args = params + state.first + state.second + try scalar_inputs(state, params) + step_batch_tensors(batch, indexed)
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
