## `train.step`, `train.epoch` and `train.metric` events of a run, seen by a handler attached to `telemetry`, on the
## reference: one `train.step` per step with its loss and learning rate, one `train.epoch` per epoch with its mean loss,
## and the run name in the metadata. A run with no handler gives what it gave before. Run from `train/`.

import Data from "../data"
import Loop from "../loop"
import Loss from "../loss"
import Optim from "../optim"
import S from "support"
import Telemetry from "pkg/telemetry/telemetry"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"

INPUTS = [1.0, 2.0, -1.0, 0.5, 0.25, -2.0, 1.5, 1.0]
TARGETS = [0.5, -1.0, 2.0, 1.0]
LR = 0.05
EPOCHS = 3

## What a handler keeps, by step or epoch number: the events come from the task of the run, in order, and a table takes
## them from the handler's own task.
struct Seen
  steps: ATS[Int, Float]
  rates: ATS[Int, Float]
  epochs: ATS[Int, Float]
  metrics: ATS[String, Float]
  runs: ATS[String, Int]
end

private

def adam() -> Optim.Optimizer = Optim.adam(lr: LR)

def weights() -> Result[List[Tensor.Tensor]]
  ok([try S.tensor([0.3, -0.2], [2, 1], Tensor.Dtype.F32, S.REF), try S.tensor([0.1], [1], Tensor.Dtype.F32, S.REF)])
end

def linear(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.mse(try Tensor.linear(b.inputs, ps[0], some(ps[1])), b.targets)
end

def pair(i: Int) -> Result[Data.Batch]
  ok(
    Data.Batch(
      inputs: try S.tensor(INPUTS.drop(4 * i).take(4), [2, 2], Tensor.Dtype.F32, S.REF),
      targets: try S.tensor(TARGETS.drop(2 * i).take(2), [2, 1], Tensor.Dtype.F32, S.REF),
    ),
  )
end

def fit(name: String) -> Result[Train.Fit]
  Train.fit_batches(
    &linear,
    try weights(),
    ->(_e: Int, i: Int) { pair(i) },
    2,
    adam(),
    EPOCHS,
    none,
    none,
    0,
    none,
    name,
  )
end

def squared_errors(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[List[List[Float]]]
  gap = try Tensor.sub(try Tensor.linear(b.inputs, ps[0], some(ps[1])), b.targets)
  rows = try S.floats_of(try Tensor.mul(gap, gap))
  ok(rows.map { |x: Float| [x] })
end

def remember(seen: Seen, event: Telemetry.Event) -> Result[Void]
  seen.runs.add(event.metadata.get("run", ""), 1)
  case event.name
    "train.step" ->
      seen.steps.put(event.ints["step_count"], event.floats["loss"])
      seen.rates.put(event.ints["step_count"], event.floats["lr"])
    "train.epoch" -> seen.epochs.put(event.ints["epoch_count"], event.floats["loss"])
    "train.metric" ->
      seen.metrics.put("#{event.ints["epoch_count"]} #{event.metadata["metric"]}", event.floats["value"])
    _ -> nil
  end

  ok()
end

def attached() -> Result[Seen]
  seen = Seen(steps: ATS.new(), rates: ATS.new(), epochs: ATS.new(), metrics: ATS.new(), runs: ATS.new())
  try Telemetry.start()
  try Telemetry.attach("seen", ["train"], ->(event) { remember(seen, event) })
  ok(seen)
end

def mean(xs: List[Float]) -> Float = xs.sum / xs.size.to_f

## One `train.step` per step, with the loss of the step and the rate it used, and one `train.epoch` per epoch with the
## mean of its steps, which is the history of the fit.
def test_a_run_emits_a_step_event_per_step_and_an_epoch_event_per_epoch() -> Result[Void]
  seen = try attached()
  done = try fit("xor")
  try Telemetry.flush()
  try Telemetry.stop()

  try Test.eq(seen.steps.size, 2 * EPOCHS)
  try Test.eq(seen.epochs.size, EPOCHS)
  for e in 1..EPOCHS
    losses = [seen.steps.get(2 * e - 1).unwrap!, seen.steps.get(2 * e).unwrap!]
    try S.expect_close([mean(losses)], [done.history[e - 1]], 0.0000001, 0.0, "mean of the steps of epoch #{e}")
    try S.expect_close([seen.epochs.get(e).unwrap!], [done.history[e - 1]], 0.0, 0.0, "epoch event #{e}")
  end

  for s in 1..(2 * EPOCHS)
    try S.expect_close([seen.rates.get(s).unwrap!], [LR], 0.0, 0.0, "rate of step #{s}")
  end

  try Test.eq(seen.runs.to_map(), {"xor": 2 * EPOCHS + EPOCHS})
  ok()
end

## A metric of `Loop` is one `train.metric` event per epoch, with the run name `Loop.trainer` was given.
def test_a_metric_emits_an_event_per_epoch() -> Result[Void]
  seen = try attached()
  loader = Data.Loader(
    dataset: try Data.dataset(
      try S.tensor(INPUTS, [4, 2], Tensor.Dtype.F32, S.REF),
      try S.tensor(TARGETS, [4, 1], Tensor.Dtype.F32, S.REF),
    ),
    batch_size: 2,
    shuffle: false,
  )

  errors = Loop.Metric(name: "mse", count: 2, batch: ->(i) { pair(i) }, observe: &squared_errors, reduce: &Loop.mean)
  done = (
    Loop.trainer(&linear, try weights(), adam(), run: "loop")
    |> Loop.metric(errors)
    |> try Loop.run(loader, epochs: EPOCHS)
  )

  try Telemetry.flush()
  try Telemetry.stop()
  for e in 1..EPOCHS
    try S.expect_close(
      [seen.metrics.get("#{e} mse").unwrap!],
      [done.metrics["mse"][e - 1]],
      0.0,
      0.0,
      "mse of epoch #{e}",
    )
  end

  try Test.eq(seen.metrics.size, EPOCHS)
  try Test.eq(seen.runs.get("loop").unwrap!, 2 * EPOCHS + EPOCHS + EPOCHS)
  ok()
end

## With no hub or no handler a run is the run it was: the same losses and weights as with one attached.
def test_a_run_with_no_handler_gives_the_same_fit() -> Result[Void]
  plain = try fit("")
  seen = try attached()
  watched = try fit("xor")
  try Telemetry.flush()
  try Telemetry.stop()
  try Test.assert(seen.steps.size > 0, "the handler saw nothing")
  try S.expect_close(watched.history, plain.history, 0.0, 0.0, "history")
  for i in 0...plain.params.size
    try S.expect_close(try S.floats_of(watched.params[i]), try S.floats_of(plain.params[i]), 0.0, 0.0, "weight #{i}")
  end

  ok()
end
