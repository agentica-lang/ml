## A training loop as a value, a thin layer over `Train.fit` and `Train.fit_batches` (`train/DESIGN.md`): build it with
## `trainer` and its steps, then `run` it, as a pipe.
##
## ```
## fit = try (
##   Loop.trainer(&loss, params, Optim.adamw(lr: 0.00002))
##   |> Loop.metric(spearman)
##   |> Loop.handle(Loop.Event.StepCompleted, report, every: 50)
##   |> Loop.run(data, epochs: 3)
## )
## ```
##
## `Train.fit` stays the engine: a run is one call of it, and answers its `Train.Fit`. `run` takes the batches of a
## `Data.Loader`; `run_batches` takes a function that makes them, for data that is not two tensors of one shape (the
## padded text of a sentence model, whose batches are built from lists and not cut from a loader).

import Data from "data"
import Hub from "pkg/hub/hub"
import Optim from "optim"
import SafetensorsWrite from "pkg/hub/safetensors_write"
import Telemetry from "pkg/telemetry/telemetry"
import Tensor from "pkg/tensor/tensor"
import Train from "train"

TEMPORARY = ".tmp-step-"
WEIGHTS = "model.safetensors"
OPTIM = "optim.safetensors"
META = "checkpoint.json"

## A number measured at the end of every epoch, on the batches of its own data (a validation set, or the training
## one). `batch(i)` makes batch `i` (from 0) of `count`; the loop frees it after `observe(params, batch)`, which answers
## one row of numbers for each example, with the parameters the epoch ended with: it is forward only, a `Trace.eval` of
## the model with no gradient. `reduce` turns the rows of all the batches into the value (`Loop.mean`,
## `Loop.spearman`, or your own).
struct Metric
  name: String
  count: Int
  batch: Fn[Int, Result[Data.Batch]]
  observe: Fn[List[Tensor.Tensor], Data.Batch, Result[List[List[Float]]]]
  reduce: Fn[List[List[Float]], Float]
end

## When a handler is called: once before the first step (`Started`), after each step (`StepCompleted`) or after each epoch
## (`EpochCompleted`, once its metrics are measured).
enum Event
  Started
  StepCompleted
  EpochCompleted
end

## What a handler gets. `epoch` counts from 1 (0 at `Started`), `batch` is the index in the epoch and `step` the count of
## the run, from 1 (the steps so far at `EpochCompleted`, 0 at `Started`); `batch` is 0 at `Started` and `EpochCompleted`.
## `loss` is the loss of the step, or the mean loss of the epoch at `EpochCompleted` (0 at `Started`).
## `params` are the parameters the step or the epoch made, the run's own: use them inside the handler (the next step
## frees them). `metrics` holds the values measured at the end of this epoch, by name, at `EpochCompleted`. `state` is the
## optimizer state the same way (`none` at `Started`), and `seed` the seed of the loader of `run` (0 for `run_batches`).
## `halt` is what `Loop.stop` writes.
struct Info
  event: Event
  epoch: Int
  batch: Int
  step: Int
  loss: Float
  params: List[Tensor.Tensor]
  metrics: Map[String, Float] = {}
  state: Optional[Optim.State] = none
  seed: Int = 0
  halt: Channel[Bool]
end

## A function called at an event, every `every` steps or epochs. An `error` from it ends the run.
struct Handler
  event: Event
  every: Int
  call: Fn[Info, Result[Void]]
end

## What `checkpoint` wrote: the weights and the optimizer state (`params` and `state`, to give back to `run`), and where
## the run was. `event` is `step` or `epoch`; `epoch` counts from 1, `batch` is the last batch run in that epoch (0 for
## an `epoch` checkpoint, which is its epoch's end), `step` the steps of the run, `seed` the seed of the loader and
## `loss` the loss of the step, or the mean of the epoch.
struct Checkpoint
  event: String
  epoch: Int
  batch: Int
  step: Int
  seed: Int
  loss: Float
  params: List[Tensor.Tensor]
  state: Optim.State
end

## What a run trains: the loss of a parameter list and a batch (inside it the parameters and the batch are `Trace`
## tensors), the parameters it starts from, and the optimizer.
struct Trainer
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]]
  params: List[Tensor.Tensor]
  optimizer: Optim.Optimizer
  metrics: List[Metric] = []
  handlers: List[Handler] = []
  run: String = ""
  resume: Optional[String] = none
end

def trainer(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  optimizer: Optim.Optimizer,
  run: String = "",
) -> Trainer
  Trainer(loss: loss, params: params, optimizer: optimizer, run: run)
end

## The trainer with `metric` measured at the end of every epoch of its run; the values are in `Fit.metrics` under the
## name of the metric, one per epoch. A metric that fails ends the run with its error at the end of its
## epoch. A run stopped from a step handler has no values for the epoch it cut short: `Fit.metrics` has one value per epoch
## that ended, possibly fewer than the history.
def metric(t: Trainer, m: Metric) -> Trainer = t.with(metrics: t.metrics.push(m))

## The trainer with `f` called at `event`: before the first step, after every `every`th step, or after every `every`th
## epoch (with the metrics of that epoch). Handlers of an event are called in the order they were added, and in the
## thread of the run. A handler that answers an `error` ends the run with it: a step handler's error is named by the
## epoch and the batch, an epoch handler's (and a metric's) by the epoch, and a `Started` handler's by "before the first
## step". An `every` below 1 is an error of the run.
def handle(t: Trainer, event: Event, f: Fn[Info, Result[Void]], every: Int = 1) -> Trainer
  t.with(handlers: t.handlers.push(Handler(event: event, every: every, call: f)))
end

## What a handler returns to end the run early, without an error: `return Loop.stop(info) if info.step == 100`. The run
## ends after the step or epoch of `info` (after the first step, for a `Started` handler), and answers its `Train.Fit`
## as usual: the parameters so far, and a history whose last epoch is the mean over the steps it ran. A stop from a step
## handler skips the metrics of the epoch it cut short; a stop from `EpochCompleted` keeps those of its epoch. An `error`
## wins over a stop: a handler that stops and then answers an `error` ends the run with the error.
def stop(info: Info) -> Result[Void]
  _ = info.halt.send(true, 0)
  ok()
end

## The trainer that writes a checkpoint to a new folder `dir/step-NNNNNNNNNN` (the step, to ten digits) every `every`
## steps (or every `every` epochs, with `event: Event.EpochCompleted`), and keeps the last `keep`. A checkpoint holds
## `model.safetensors` (the weights as `param.I`), `optim.safetensors` (`Optim.save`) and `checkpoint.json`. It is
## written in `dir/.tmp-step-NNNNNNNNNN` and renamed whole, so a run killed or failing while it writes leaves the earlier
## checkpoints as they were: a temporary folder is not a checkpoint. There is no `fsync`: a checkpoint survives a kill
## of the process, not a power loss.
##
## One `dir` serves one run. Before the first step the run checks `keep`, makes `dir`, and removes the folders
## `.tmp-step-*` that a run killed before it left (no other folder of `dir` is touched, and no folder that is not
## `step-` and ten digits is listed or pruned). A step that has a checkpoint already is an error, and ends the run. An
## error while writing ends the run too. The checkpoints beyond `keep` are moved to `.tmp-step-old-N` and deleted; a
## deletion that fails is not an error of the run, and what it left is swept at the next start.
def checkpoint(t: Trainer, dir: String, every: Int = 100, keep: Int = 3, event: Event = Event.StepCompleted) -> Trainer
  (
    t
    |> handle(Event.Started, ->(_) { prepare(dir, keep) })
    |> handle(event, ->(info) { write_checkpoint(dir, info, keep) }, every: every)
  )
end

## The names of the checkpoints of `dir`, oldest first: the folders `step-` and ten digits that hold a `checkpoint.json`;
## no other folder is listed, so none is pruned.
def checkpoints(dir: String) -> Result[List[String]]
  out: List[String] = []
  for name in try Dir.list(dir)
    if checkpoint_name?(name) and File.stat("#{dir}/#{name}/#{META}").ok?
      out = out.push(name)
    end
  end

  ok(out)
end

## The checkpoint `name` of `dir` for `optimizer` and the parameters `like`: the weights on the backend and in the dtype
## of each of `like`, which must have their shapes, and the optimizer state (`Optim.load`). What was read is freed when
## the checkpoint does not load.
def read_checkpoint(
  dir: String,
  name: String,
  optimizer: Optim.Optimizer,
  like: List[Tensor.Tensor],
) -> Result[Checkpoint]
  path = "#{dir}/#{name}"
  meta = try read_meta("#{path}/#{META}")
  c = try Hub.open("#{path}/#{WEIGHTS}")
  return error("#{path} has #{Hub.names(c).size} weights, #{like.size} are needed") if Hub.names(c).size != like.size

  params: List[Tensor.Tensor] = []
  for i in 0...like.size
    read = Hub.tensor(c, "param.#{i}", Tensor.backend(like[i]), Tensor.dtype(like[i]))
    if !read.ok?
      free_all(params)
      return error(read.message)
    end

    params = params.push(read.unwrap!)
    if Tensor.shape(params[i]) != Tensor.shape(like[i])
      free_all(params)
      return error(
        "param.#{i} of #{path} has the shape #{Tensor.shape(params[i])}, its parameter #{Tensor.shape(like[i])}",
      )
    end
  end

  state = Optim.load(optimizer, params, "#{path}/#{OPTIM}")
  if !state.ok?
    free_all(params)
    return error(state.message)
  end

  ok(
    Checkpoint(
      event: meta.event,
      epoch: meta.epoch,
      batch: meta.batch,
      step: meta.step,
      seed: meta.seed,
      loss: Float.from_bits(meta.loss_bits),
      params: params,
      state: state.unwrap!,
    ),
  )
end

## The trainer that goes on from the last checkpoint of `dir` (`checkpoint`), if it has one, when it runs: with the
## weights and the optimizer state of the checkpoint, at its step, from the batch after the one it was written at, with
## the order of the data that the whole run has (the same seed and epoch). Without a checkpoint it starts from the
## beginning. `run` and `run_batches` take the same `epochs` as the whole run: the checkpoint is `epoch`, `batch` and
## `step` of it, and the loader must make the same batches (an error says when its seed or its size differ). The
## `Fit` is of the steps that this run took: its history starts with the epoch the checkpoint was in, and for a
## checkpoint in the middle of an epoch the mean of that epoch is over the steps that are left; `Fit.metrics` likewise.
## The weights of the trainer are not changed. A resumed run that checkpoints into `dir` continues the same run.
def resume(t: Trainer, dir: String) -> Trainer = t.with(resume: some(dir))

## `Train.fit` of the trainer over the batches of `data`, for `epochs` passes (one by default).
def run(t: Trainer, data: Data.Loader, epochs: Int = 1) -> Result[Train.Fit]
  seen = measuring(t, epochs, data.seed)
  try started(t, seen)
  from = try resume_point(t, Data.count(data), epochs, data.seed)
  fit = Train.fit(
    t.loss,
    from.params,
    data,
    t.optimizer,
    epochs,
    seen.on_epoch,
    none,
    seen.on_step,
    start: from.start,
    run: t.run,
    skip: from.skip,
  )

  measured(try released(from, fit), t, seen)
end

## `Train.fit_batches` of the trainer: `batch(epoch, i)` makes batch `i` (from 0) of `count` in epoch `epoch` (from
## 1), with `prefetch` batches built ahead.
def run_batches(
  t: Trainer,
  batch: Fn[Int, Int, Result[Data.Batch]],
  count: Int,
  epochs: Int = 1,
  prefetch: Int = 0,
) -> Result[Train.Fit]
  seen = measuring(t, epochs, 0)
  try started(t, seen)
  from = try resume_point(t, count, epochs, 0)
  fit = Train.fit_batches(
    t.loss,
    from.params,
    batch,
    count,
    t.optimizer,
    epochs,
    seen.on_epoch,
    none,
    prefetch,
    seen.on_step,
    start: from.start,
    run: t.run,
    skip: from.skip,
  )

  measured(try released(from, fit), t, seen)
end

## The mean of the first number of every row.
def mean(rows: List[List[Float]]) -> Float
  return 0.0 if rows.empty?

  rows.map { |row: List[Float]| row[0] }.sum / rows.size.to_f
end

## Spearman's correlation between the first and the second number of the rows: the Pearson correlation of their ranks,
## equal values sharing the mean of their ranks. 0 when one of the two is constant.
def spearman(rows: List[List[Float]]) -> Float
  pearson(ranks(rows.map { |row: List[Float]| row[0] }), ranks(rows.map { |row: List[Float]| row[1] }))
end

## The value of `m` under `params`: its rows over all its batches, reduced.
def measure(m: Metric, params: List[Tensor.Tensor]) -> Result[Float]
  rows: List[List[Float]] = []
  for i in 0...m.count
    made = m.batch.call(i)
    return error("metric #{m.name}: #{made.message}") if !made.ok?

    batch = made.unwrap!
    observed = m.observe.call(params, batch)
    Data.release(batch)
    case observed
      ok(more) ->
        for row in more
          rows = rows.push(row)
        end

      error(e) -> return error("metric #{m.name}: #{e.message}")
    end
  end

  ok(m.reduce.call(rows))
end

private

## Where a run starts: the parameters (the trainer's, or a checkpoint's, `loaded`), the state it starts from, and the
## batches that are already run.
struct Point
  params: List[Tensor.Tensor]
  start: Optional[Optim.State]
  skip: Int
  loaded: Bool
  finished: Bool
end

## What `checkpoint.json` holds. The loss is its IEEE bits, which JSON can hold whatever the loss is (a NaN or an
## infinity has no JSON number) and which come back exact.
struct Meta
  event: String
  epoch: Int
  batch: Int
  step: Int
  seed: Int
  loss_bits: UInt64
end

## What a run hands to the engine: the closures of `on_epoch` and `on_step`, the channel that `on_epoch` sends the
## metric values of an epoch to, and the channel that `Loop.stop` writes to.
struct Seen
  on_epoch: Optional[Fn[Train.Epoch, Result[Train.Control]]]
  on_step: Optional[Fn[Train.StepInfo, Result[Train.Control]]]
  values: Channel[List[Float]]
  halt: Channel[Bool]
  seed: Int
end

## The point of the last checkpoint of the trainer's `resume` folder, or the beginning. `count` batches make an epoch.
def resume_point(t: Trainer, count: Int, epochs: Int, seed: Int) -> Result[Point]
  beginning = Point(params: t.params, start: none, skip: 0, loaded: false, finished: false)
  dir = case t.resume
    some(d) -> d
    none -> ""
  end

  return ok(beginning) if dir == ""

  names = try checkpoints(dir)
  return ok(beginning) if names.empty?

  name = names[names.size - 1]
  saved = try read_checkpoint(dir, name, t.optimizer, t.params)
  skip = saved.epoch * count
  if saved.event == "step"
    skip = (saved.epoch - 1) * count + saved.batch + 1
  end

  fits = fits_data(name, saved, skip, count, seed)
  if !fits.ok?
    free_all(saved.params)
    free_all(saved.state.first + saved.state.second)
    return error(fits.message)
  end

  ok(Point(params: saved.params, start: some(saved.state), skip: skip, loaded: true, finished: skip >= count * epochs))
end

## The checkpoint was written by a run of this data: its seed, and a step count that the batches add up to.
def fits_data(name: String, saved: Checkpoint, skip: Int, count: Int, seed: Int) -> Result[Void]
  if saved.seed != seed
    return error("#{name} was written with the loader seed #{saved.seed}, the loader has seed #{seed}")
  end

  if saved.step != skip
    return error(
      "#{name} is step #{saved.step}, but epoch #{saved.epoch} of #{count} batches ends at step #{skip}: the batches differ",
    )
  end

  ok()
end

## `fit`, and the parameters of a checkpoint freed: they were the run's start, and a run keeps none of them unless it ran
## nothing, when they are its answer.
def released(from: Point, fit: Result[Train.Fit]) -> Result[Train.Fit]
  if from.loaded and !(from.finished and fit.ok?)
    free_all(from.params)
  end

  fit
end

def measuring(t: Trainer, epochs: Int, seed: Int) -> Seen
  values: Channel[List[Float]] = Channel.new(epochs)
  halt: Channel[Bool] = Channel.new(1)
  steps = handlers_at(t, Event.StepCompleted)
  on_step: Optional[Fn[Train.StepInfo, Result[Train.Control]]] = none
  if !t.handlers.empty? or !t.metrics.empty?
    on_step = some(->(s) { after_step(steps, s, halt, seed) })
  end

  on_epoch: Optional[Fn[Train.Epoch, Result[Train.Control]]] = none
  if !t.metrics.empty? or !handlers_at(t, Event.EpochCompleted).empty?
    on_epoch = some(->(e) { after_epoch(t, e, values, halt, seed) })
  end

  Seen(on_epoch: on_epoch, on_step: on_step, values: values, halt: halt, seed: seed)
end

## `fit` with the metric values of every epoch that ended.
def measured(fit: Train.Fit, t: Trainer, seen: Seen) -> Result[Train.Fit]
  return ok(fit) if seen.on_epoch.none?

  columns: List[List[Float]] = t.metrics.map { |_m: Metric| [] }
  got = seen.values.recv(0)
  while got.ok?
    for k in 0...t.metrics.size
      columns = columns.set(k, columns[k].push(got.unwrap![k]))
    end

    got = seen.values.recv(0)
  end

  named: Map[String, List[Float]] = {}
  for k in 0...t.metrics.size
    named = named.put(t.metrics[k].name, columns[k])
  end

  ok(fit.with(metrics: named))
end

def handlers_at(t: Trainer, event: Event) -> List[Handler] = t.handlers.select { |h: Handler| h.event == event }

## `Stop` when a handler called `Loop.stop` since the last check.
def control(halt: Channel[Bool]) -> Train.Control
  return Train.Control.Stop if halt.recv(0).ok?

  Train.Control.Continue
end

## Checks the handlers, then calls those of `Started`.
def started(t: Trainer, seen: Seen) -> Result[Void]
  for h in t.handlers
    return error("handle: every is 1 or more, got #{h.every}") if h.every < 1
  end

  info = Info(
    event: Event.Started,
    epoch: 0,
    batch: 0,
    step: 0,
    loss: 0.0,
    params: t.params,
    seed: seen.seed,
    halt: seen.halt,
  )

  for h in handlers_at(t, Event.Started)
    called = h.call.call(info)
    return error("before the first step: #{called.message}") if !called.ok?
  end

  ok()
end

## What `on_step` does after a step: calls the step handlers that are due, then stops if one of them asked to.
def after_step(steps: List[Handler], s: Train.StepInfo, halt: Channel[Bool], seed: Int) -> Result[Train.Control]
  info = Info(
    event: Event.StepCompleted,
    epoch: s.epoch,
    batch: s.batch,
    step: s.step,
    loss: s.loss,
    params: s.params,
    state: some(s.state),
    seed: seed,
    halt: halt,
  )

  for h in steps
    if s.step % h.every == 0
      try h.call.call(info)
    end
  end

  ok(control(halt))
end

## The metrics of the epoch, then its handlers with them. The values go to `measured`.
def after_epoch(
  t: Trainer,
  e: Train.Epoch,
  values: Channel[List[Float]],
  halt: Channel[Bool],
  seed: Int,
) -> Result[Train.Control]
  measured = try measure_all(t.metrics, e.params)
  named: Map[String, Float] = {}
  for k in 0...t.metrics.size
    named = named.put(t.metrics[k].name, measured[k])
    announce_metric(t.run, t.metrics[k].name, e.epoch, measured[k])
  end

  info = Info(
    event: Event.EpochCompleted,
    epoch: e.epoch,
    batch: 0,
    step: e.steps,
    loss: e.loss,
    params: e.params,
    metrics: named,
    state: some(e.state),
    seed: seed,
    halt: halt,
  )

  for h in handlers_at(t, Event.EpochCompleted)
    if e.epoch % h.every == 0
      try h.call.call(info)
    end
  end

  _ = values.send(measured, 0)
  ok(control(halt))
end

## `train.metric` for `telemetry` handlers: the value of a metric at the end of an epoch, with the run name and the
## metric name in the metadata.
def announce_metric(run: String, name: String, epoch: Int, value: Float) -> Void
  if Telemetry.enabled?("train.metric")
    Telemetry.execute(
      "train.metric",
      ints: {"epoch_count": epoch},
      floats: {"value": value},
      metadata: {"run": run, "metric": name},
    )
  end
end

def measure_all(metrics: List[Metric], params: List[Tensor.Tensor]) -> Result[List[Float]]
  out: List[Float] = []
  for m in metrics
    out = out.push(try measure(m, params))
  end

  ok(out)
end

## Before the first step: `keep` is 1 or more, `dir` exists, and what an earlier run left half written (the folders
## `.tmp-step-*`, and nothing else) is gone.
def read_meta(file: String) -> Result[Meta]
  text = File.read_text(file)
  return error("#{file}: #{text.message}") if !text.ok?

  meta = JSON.decode(text.unwrap!, Meta)
  return error("#{file}: #{meta.message}") if !meta.ok?

  meta
end

def prepare(dir: String, keep: Int) -> Result[Void]
  return error("checkpoint: keep is 1 or more, got #{keep}") if keep < 1

  if !Dir.list(dir).ok?
    try Dir.mkdir(dir)
  end

  remove_unfinished(dir)
end

def remove_unfinished(dir: String) -> Result[Void]
  for name in try Dir.list(dir)
    if name.start_with?(TEMPORARY)
      try Path.remove("#{dir}/#{name}")
    end
  end

  ok()
end

## The checkpoint of the step of `info`: written whole in a temporary folder, then renamed, then the old ones beyond
## `keep` are removed. A checkpoint of that step that exists is an error.
def write_checkpoint(dir: String, info: Info, keep: Int) -> Result[Void]
  state = try info.state.to_result("checkpoint: no optimizer state at this event")
  name = format("step-%010d", info.step)
  return error("checkpoint #{name} exists in #{dir}") if (try checkpoints(dir)).include?(name)

  temporary = "#{dir}/#{TEMPORARY}#{info.step}"
  try Dir.mkdir(temporary)
  written = fill_checkpoint(temporary, info, state)
  if !written.ok?
    _ = Path.remove(temporary)
    return error("checkpoint #{name}: #{written.message}")
  end

  placed = Path.rename(temporary, "#{dir}/#{name}")
  if !placed.ok?
    _ = Path.remove(temporary)
    return error("checkpoint #{name}: #{placed.message}")
  end

  prune(dir, keep)
end

def fill_checkpoint(folder: String, info: Info, state: Optim.State) -> Result[Void]
  weights: Map[String, Tensor.Tensor] = {}
  for i in 0...info.params.size
    weights = weights.put("param.#{i}", info.params[i])
  end

  try SafetensorsWrite.write("#{folder}/#{WEIGHTS}", weights)
  try Optim.save(state, "#{folder}/#{OPTIM}")
  event = "step"
  if info.event == Event.EpochCompleted
    event = "epoch"
  end

  meta = Meta(
    event: event,
    epoch: info.epoch,
    batch: info.batch,
    step: info.step,
    seed: info.seed,
    loss_bits: Float.to_bits(info.loss),
  )

  File.write_text("#{folder}/#{META}", JSON.encode(meta))
end

## The oldest checkpoints beyond the last `keep` removed: each is renamed to `.tmp-step-old-N` first, which takes it out of
## the list in one step, and then deleted. A deletion that fails is not an error of the run (the checkpoint is already
## gone from the list): what is left is swept at the next `Started`.
def prune(dir: String, keep: Int) -> Result[Void]
  names = try checkpoints(dir)
  if names.size > keep
    for name in names.take(names.size - keep)
      gone = "#{dir}/#{TEMPORARY}old-#{name.slice(5, 10)}"
      try Path.rename("#{dir}/#{name}", gone)
      _ = Path.remove(gone)
    end
  end

  ok()
end

## `step-` and exactly ten digits.
def checkpoint_name?(name: String) -> Bool
  return false if name.length != 15 or !name.start_with?("step-")

  name.slice(5, 10).split("").all? { |c: String| "0123456789".include?(c) }
end

def free_all(tensors: List[Tensor.Tensor]) -> Void
  for t in tensors
    Data.free(t)
  end
end

## The rank of each value from 1, equal values sharing the mean of their ranks.
def ranks(values: List[Float]) -> List[Float]
  order = (0...values.size).to_a.sort_by { |i: Int| values[i] }
  out = List.repeat(0.0, values.size)
  at = 0
  while at < order.size
    stop = at
    while stop + 1 < order.size and values[order[stop + 1]] == values[order[at]]
      stop = stop + 1
    end

    shared = (at + stop).to_f / 2.0 + 1.0
    for k in at..stop
      out = out.set(order[k], shared)
    end

    at = stop + 1
  end

  out
end

## Pearson's correlation, 0 when a list is constant.
def pearson(a: List[Float], b: List[Float]) -> Float
  n = a.size.to_f
  mean_a = a.sum / n
  mean_b = b.sum / n
  both = 0.0
  aa = 0.0
  bb = 0.0
  for i in 0...a.size
    both = both + (a[i] - mean_a) * (b[i] - mean_b)
    aa = aa + (a[i] - mean_a) * (a[i] - mean_a)
    bb = bb + (b[i] - mean_b) * (b[i] - mean_b)
  end

  return 0.0 if aa == 0.0 or bb == 0.0

  both / Math.sqrt(aa * bb)
end
