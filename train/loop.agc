## A training loop as a value, a thin layer over `Train.fit` and `Train.fit_batches` (`train/DESIGN.md`): build it with
## `trainer` and its steps, then `run` it, as a pipe.
##
## ```
## fit = try (
##   Loop.trainer(&loss, params, Optim.adamw(lr: 0.00002))
##   |> Loop.metric(spearman)
##   |> Loop.run(data, epochs: 3)
## )
## ```
##
## `Train.fit` stays the engine: a run is one call of it, and answers its `Train.Fit`. `run` takes the batches of a
## `Data.Loader`; `run_batches` takes a function that makes them, for data that is not two tensors of one shape (the
## padded text of a sentence model, whose batches are built from lists and not cut from a loader).

import Data from "data"
import Optim from "optim"
import Tensor from "pkg/tensor/tensor"
import Train from "train"

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

## What a run trains: the loss of a parameter list and a batch (inside it the parameters and the batch are `Trace`
## tensors), the parameters it starts from, and the optimizer.
struct Trainer
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]]
  params: List[Tensor.Tensor]
  optimizer: Optim.Optimizer
  metrics: List[Metric] = []
end

def trainer(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  optimizer: Optim.Optimizer,
) -> Trainer
  Trainer(loss: loss, params: params, optimizer: optimizer)
end

## The trainer with `metric` measured at the end of every epoch of its run; the values are in `Fit.metrics` under the
## name of the metric, one per epoch. A metric that fails is not measured again: the run goes on, and ends with its
## error after the last epoch.
def metric(t: Trainer, m: Metric) -> Trainer = t.with(metrics: t.metrics.push(m))

## `Train.fit` of the trainer over the batches of `data`, for `epochs` passes (one by default). Until `Loop.handle`, a
## run takes no `on_epoch` or `metrics` run of `Train.fit`.
def run(t: Trainer, data: Data.Loader, epochs: Int = 1) -> Result[Train.Fit]
  seen = measuring(t, epochs)
  fit = try Train.fit(t.loss, t.params, data, t.optimizer, epochs, seen.on_epoch)
  measured(fit, t, seen)
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
  seen = measuring(t, epochs)
  fit = try Train.fit_batches(t.loss, t.params, batch, count, t.optimizer, epochs, seen.on_epoch, none, prefetch)
  measured(fit, t, seen)
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

## What a run collects: the closure of `on_epoch` and the channel that it sends the values of an epoch to, one
## `Result` for each epoch.
struct Seen
  on_epoch: Optional[Fn[Train.Epoch, Void]]
  values: Channel[Result[List[Float]]]
  failed: Channel[Bool]
end

def measuring(t: Trainer, epochs: Int) -> Seen
  values: Channel[Result[List[Float]]] = Channel.new(epochs)
  failed: Channel[Bool] = Channel.new(1)
  return Seen(on_epoch: none, values: values, failed: failed) if t.metrics.empty?

  on_epoch = ->(e: Train.Epoch) { _ = values.send(after_epoch(t.metrics, e.params, failed), 0) }
  Seen(on_epoch: some(on_epoch), values: values, failed: failed)
end

## `fit` with the metric values of every epoch, or the error of the first metric that failed.
def measured(fit: Train.Fit, t: Trainer, seen: Seen) -> Result[Train.Fit]
  return ok(fit) if t.metrics.empty?

  columns: List[List[Float]] = t.metrics.map { |_m: Metric| [] }
  for _ in fit.history
    values = next_values(seen)
    if !values.ok?
      for p in fit.params
        Data.free(p)
      end

      return error(values.message)
    end

    for k in 0...t.metrics.size
      columns = columns.set(k, columns[k].push(values.unwrap![k]))
    end
  end

  named: Map[String, List[Float]] = {}
  for k in 0...t.metrics.size
    named = named.put(t.metrics[k].name, columns[k])
  end

  ok(fit.with(metrics: named))
end

## The values of the metrics for one epoch, or the error of the first that fails; once one has failed, later epochs
## measure nothing.
def after_epoch(metrics: List[Metric], params: List[Tensor.Tensor], failed: Channel[Bool]) -> Result[List[Float]]
  if failed.recv(0).ok?
    _ = failed.send(true, 0)
    return error("an earlier epoch's metric failed")
  end

  measured = measure_all(metrics, params)
  if !measured.ok?
    _ = failed.send(true, 0)
  end

  measured
end

## What the next epoch's `on_epoch` sent: its values or its error.
def next_values(seen: Seen) -> Result[List[Float]]
  case seen.values.recv(0)
    ok(values) -> values
    error(e) -> error(e.message)
  end
end

def measure_all(metrics: List[Metric], params: List[Tensor.Tensor]) -> Result[List[Float]]
  out: List[Float] = []
  for m in metrics
    out = out.push(try measure(m, params))
  end

  ok(out)
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
