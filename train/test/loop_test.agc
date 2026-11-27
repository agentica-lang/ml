## `Loop`: a trainer run as a pipe gives what `Train.fit` and `Train.fit_batches` give with the same arguments, on the
## reference and on every ggml backend, and the short optimizer constructors are the structs they name. Run from `train/`.

import Data from "../data"
import Loop from "../loop"
import Loss from "../loss"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "../train"

INPUTS = [1.0, 2.0, -1.0, 0.5, 0.25, -2.0, 1.5, 1.0]
TARGETS = [0.5, -1.0, 2.0, 1.0]

private

def adam() -> Optim.Optimizer = Optim.adam(lr: 0.05)

def weights(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([
    try S.tensor([0.3, -0.2], [2, 1], Tensor.Dtype.F32, backend),
    try S.tensor([0.1], [1], Tensor.Dtype.F32, backend),
  ])
end

def linear(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.mse(try Tensor.linear(b.inputs, ps[0], some(ps[1])), b.targets)
end

def rows(backend: Tensor.Backend) -> Result[Data.Dataset]
  Data.dataset(
    try S.tensor(INPUTS, [4, 2], Tensor.Dtype.F32, backend),
    try S.tensor(TARGETS, [4, 1], Tensor.Dtype.F32, backend),
  )
end

def pair(backend: Tensor.Backend, i: Int) -> Result[Data.Batch]
  ok(
    Data.Batch(
      inputs: try S.tensor(INPUTS.drop(4 * i).take(4), [2, 2], Tensor.Dtype.F32, backend),
      targets: try S.tensor(TARGETS.drop(2 * i).take(2), [2, 1], Tensor.Dtype.F32, backend),
    ),
  )
end

def same(got: Train.Fit, want: Train.Fit, what: String) -> Result[Void]
  try S.expect_close(got.history, want.history, 0.000001, 0.00001, "#{what}: history")
  for i in 0...want.params.size
    try S.expect_close(
      try S.floats_of(got.params[i]),
      try S.floats_of(want.params[i]),
      0.000001,
      0.00001,
      "#{what}: parameter #{i}",
    )
  end

  ok()
end

def backends() -> Result[List[Tensor.Backend]] = ok([S.REF] + try S.compiled_backends())

## `Loop.run` is `Train.fit`: the same loss history and the same parameters.
def test_run_is_fit() -> Result[Void]
  for backend in try backends()
    loader = Data.Loader(dataset: try rows(backend), batch_size: 2, shuffle: false)
    want = try Train.fit(&linear, try weights(backend), loader, adam(), 3)
    got = Loop.trainer(&linear, try weights(backend), adam()) |> try Loop.run(loader, epochs: 3)
    try same(got, want, "run on #{backend}")
  end

  ok()
end

## `Loop.run_batches` is `Train.fit_batches`, with the batches a function makes.
def test_run_batches_is_fit_batches() -> Result[Void]
  for backend in try backends()
    batch = ->(_epoch: Int, i: Int) { pair(backend, i) }
    want = try Train.fit_batches(&linear, try weights(backend), batch, 2, adam(), 2)
    got = Loop.trainer(&linear, try weights(backend), adam()) |> try Loop.run_batches(batch, 2, epochs: 2)
    try same(got, want, "run_batches on #{backend}")
  end

  ok()
end

## `epochs` left out is one epoch, and `prefetch` builds batches ahead without changing the run.
def test_one_epoch_by_default_and_prefetch_changes_nothing() -> Result[Void]
  for backend in try backends()
    loader = Data.Loader(dataset: try rows(backend), batch_size: 2, shuffle: false)
    want = try Train.fit(&linear, try weights(backend), loader, adam(), 1)
    got = Loop.trainer(&linear, try weights(backend), adam()) |> try Loop.run(loader)

    try same(got, want, "default epochs on #{backend}")
    batch = ->(_epoch: Int, i: Int) { pair(backend, i) }
    ahead = Loop.trainer(&linear, try weights(backend), adam()) |> try Loop.run_batches(batch, 2, prefetch: 2)

    try same(ahead, want, "prefetch on #{backend}")
  end

  ok()
end

## A metric is measured at the end of every epoch on the batches of its own data, with the parameters of that epoch:
## its values are in `Fit.metrics`, one per epoch, and the last one is the metric of the trained parameters.
def test_a_metric_is_measured_at_the_end_of_every_epoch() -> Result[Void]
  for backend in try backends()
    loader = Data.Loader(dataset: try rows(backend), batch_size: 2, shuffle: false)
    errors = Loop.Metric(
      name: "mse",
      count: 2,
      batch: ->(i) { pair(backend, i) },
      observe: &squared_errors,
      reduce: &Loop.mean,
    )

    fit = (
      Loop.trainer(&linear, try weights(backend), adam())
      |> Loop.metric(errors)
      |> try Loop.run(loader, epochs: 3)
    )

    values = fit.metrics["mse"]
    try Test.eq(values.size, 3)
    try Test.eq(fit.metrics.keys, ["mse"])
    try Test.assert(values[2] < values[0], "the metric did not fall on #{backend}")
    again = try Loop.measure(errors, fit.params)
    try S.expect_close([again], [values[2]], 0.000001, 0.00001, "the last value on #{backend}")
  end

  ok()
end

## A metric that fails ends the run with an error that names it.
def test_a_failing_metric_is_an_error_that_names_it() -> Result[Void]
  failing = Loop.Metric(
    name: "broken",
    count: 1,
    batch: ->(i) { pair(S.REF, i) },
    observe: ->(_, _) { error("no luck") },
    reduce: &Loop.mean,
  )

  loader = Data.Loader(dataset: try rows(S.REF), batch_size: 2, shuffle: false)
  r = (
    Loop.trainer(&linear, try weights(S.REF), adam())
    |> Loop.metric(failing)
    |> Loop.run(loader)
  )

  S.expect_error(r, "metric broken: no luck")
end

## The parameters of each epoch are the ones that epoch ended with: a metric that reads a weight sees it change, and
## `Loop.spearman` through a run is the correlation of the trained parameters' predictions with the targets.
def test_a_metric_sees_the_parameters_of_its_epoch() -> Result[Void]
  for backend in try backends()
    loader = Data.Loader(dataset: try rows(backend), batch_size: 2, shuffle: false)
    weight = Loop.Metric(
      name: "weight",
      count: 1,
      batch: ->(i) { pair(backend, i) },
      observe: ->(ps, _) { first_weight(ps) },
      reduce: &Loop.mean,
    )

    order = Loop.Metric(
      name: "spearman",
      count: 2,
      batch: ->(i) { pair(backend, i) },
      observe: &predictions_and_targets,
      reduce: &Loop.spearman,
    )

    fit = (
      Loop.trainer(&linear, try weights(backend), adam())
      |> Loop.metric(weight)
      |> Loop.metric(order)
      |> try Loop.run(loader, epochs: 3)
    )

    seen = fit.metrics["weight"]
    try Test.assert(seen[0] != seen[1] and seen[1] != seen[2], "the weight did not change on #{backend}: #{seen}")
    try S.expect_close([seen[2]], (try S.floats_of(fit.params[0])).take(1), 0.000001, 0.00001, "the last weight")
    last = try Loop.measure(order, fit.params)
    try S.expect_close([last], [fit.metrics["spearman"][2]], 0.000001, 0.00001, "spearman on #{backend}")
  end

  ok()
end

def first_weight(ps: List[Tensor.Tensor]) -> Result[List[List[Float]]] = ok([[(try S.floats_of(ps[0]))[0]]])

## The prediction of the linear model and the target of every row of a batch.
def predictions_and_targets(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[List[List[Float]]]
  predicted = try Trace.eval(ps + [b.inputs], ->(xs: List[Tensor.Tensor]) { Tensor.linear(xs[2], xs[0], some(xs[1])) })
  got = try S.floats_of(predicted)
  want = try S.floats_of(b.targets)
  ok((0...want.size).map { |i: Int| [got[i], want[i]] })
end

def labelled(log: Channel[String], tag: String) -> Fn[Loop.Info, Result[Void]]
  ->(i) {
    _ = log.send("#{tag}#{i.step}", 0)
    ok()
  }
end

def failing_at(log: Channel[String], step: Int) -> Fn[Loop.Info, Result[Void]]
  ->(i) {
    return error("stop at #{i.step}") if i.step == step

    _ = log.send("f#{i.step}", 0)
    ok()
  }
end

def drained(log: Channel[String], count: Int) -> Result[List[String]]
  out: List[String] = []
  for _ in 0...count
    out = out.push(try log.recv(0))
  end

  try Test.error?(log.recv(0))
  ok(out)
end

## Handlers are called in the order they were added, at their events, every `every` steps or epochs; an epoch handler
## sees the metrics of its epoch. Two epochs of two steps and an epoch handler every 2: only epoch 2 fires.
def test_handlers_run_in_order_at_their_events() -> Result[Void]
  for backend in try backends()
    log: Channel[String] = Channel.new(32)
    loader = Data.Loader(dataset: try rows(backend), batch_size: 2, shuffle: false)
    on_epoch = ->(i: Loop.Info) {
      _ = log.send("e#{i.epoch}:#{i.metrics.size}", 0)
      ok()
    }

    mse = Loop.Metric(
      name: "mse",
      count: 2,
      batch: ->(i) { pair(backend, i) },
      observe: &squared_errors,
      reduce: &Loop.mean,
    )

    fit = (
      Loop.trainer(&linear, try weights(backend), adam())
      |> Loop.metric(mse)
      |> Loop.handle(
        Loop.Event.Started,
        ->(_) {
          _ = log.send("start", 0)
          ok()
        },
      )
      |> Loop.handle(Loop.Event.StepCompleted, labelled(log, "a"))
      |> Loop.handle(Loop.Event.StepCompleted, labelled(log, "b"), every: 3)
      |> Loop.handle(Loop.Event.EpochCompleted, on_epoch, every: 2)
      |> try Loop.run(loader, epochs: 3)
    )

    try Test.eq(fit.history.size, 3)
    got = try drained(log, 10)
    try Test.eq(got, ["start", "a1", "a2", "a3", "b3", "a4", "e2:1", "a5", "a6", "b6"])
  end

  ok()
end

## An error from a step handler stops the run at that step: no later step, no later handler.
def test_a_step_handler_error_stops_the_run_at_once() -> Result[Void]
  for backend in try backends()
    log: Channel[String] = Channel.new(32)
    loader = Data.Loader(dataset: try rows(backend), batch_size: 2, shuffle: false)
    r = (
      Loop.trainer(&linear, try weights(backend), adam())
      |> Loop.handle(Loop.Event.StepCompleted, failing_at(log, 3))
      |> Loop.handle(Loop.Event.StepCompleted, labelled(log, "late"))
      |> Loop.run(loader, epochs: 3)
    )

    try S.expect_error(r, "epoch 2, batch 0: stop at 3")
    try Test.eq(try drained(log, 4), ["f1", "late1", "f2", "late2"])
  end

  ok()
end

## A metric that fails stops the run at the next step, not after the last epoch: the step handlers of that step do not
## run, and the metric is called once.
def test_a_failed_metric_stops_the_run_at_the_next_step() -> Result[Void]
  log: Channel[String] = Channel.new(32)
  calls: Channel[Int] = Channel.new(3)
  failing = Loop.Metric(
    name: "broken",
    count: 1,
    batch: ->(i) { pair(S.REF, i) },
    observe: ->(_, _) {
      _ = calls.send(1, 0)
      error("no luck")
    },
    reduce: &Loop.mean,
  )

  loader = Data.Loader(dataset: try rows(S.REF), batch_size: 2, shuffle: false)
  r = (
    Loop.trainer(&linear, try weights(S.REF), adam())
    |> Loop.metric(failing)
    |> Loop.handle(Loop.Event.StepCompleted, labelled(log, "s"))
    |> Loop.run(loader, epochs: 3)
  )

  try S.expect_error(r, "metric broken: no luck")
  try Test.eq(try drained(log, 2), ["s1", "s2"])
  try Test.eq(try calls.recv(0), 1)
  Test.error?(calls.recv(0))
end

def test_every_is_one_or_more() -> Result[Void]
  loader = Data.Loader(dataset: try rows(S.REF), batch_size: 2, shuffle: false)
  r = (
    Loop.trainer(&linear, try weights(S.REF), adam())
    |> Loop.handle(Loop.Event.StepCompleted, labelled(Channel.new(1), "x"), every: 0)
    |> Loop.run(loader)
  )

  S.expect_error(r, "every is 1 or more")
end

## The reducers: the mean of the first number, Spearman's correlation of the first two (ties share their ranks, a
## constant column is 0).
def test_the_reducers() -> Result[Void]
  try Test.eq(Loop.mean([[1.0], [2.0], [6.0]]), 3.0)
  try Test.eq(Loop.mean([]), 0.0)
  try Test.eq(Loop.spearman([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]), 1.0)
  try Test.eq(Loop.spearman([[1.0, 30.0], [2.0, 20.0], [3.0, 10.0]]), -1.0)
  try Test.eq(Loop.spearman([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]]), 0.0)
  both = Loop.spearman([[1.0, 1.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0]])
  S.expect_close([both], [0.9486832980505138], 0.0000001, 0.0, "ties")
end

## The squared error of every row of a batch under the parameters of the linear model.
def squared_errors(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[List[List[Float]]]
  predicted = try Trace.eval(ps + [b.inputs], ->(xs: List[Tensor.Tensor]) { Tensor.linear(xs[2], xs[0], some(xs[1])) })
  got = try S.floats_of(predicted)
  want = try S.floats_of(b.targets)
  ok((0...want.size).map { |i: Int| [(got[i] - want[i]) * (got[i] - want[i])] })
end

## The short constructors are the structs with the same knobs, and the knobs they leave out keep their defaults.
def test_the_short_optimizer_constructors_are_the_structs() -> Result[Void]
  decay = [true, false]
  try Test.eq(Optim.sgd(0.1), Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1)))
  try Test.eq(Optim.sgd(0.1, momentum: 0.9), Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1, momentum: 0.9)))
  try Test.eq(Optim.adam(), Optim.Optimizer.Adam(Optim.Adam()))
  try Test.eq(Optim.adam(lr: 0.05), Optim.Optimizer.Adam(Optim.Adam(lr: 0.05)))
  warm = Optim.Schedule.WarmupLinear(2, 10)
  try Test.eq(
    Optim.adamw(lr: 0.0001, weight_decay: 0.0, decay: decay, schedule: warm),
    Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.0001, weight_decay: 0.0, decay: decay, schedule: warm)),
  )

  Test.eq(Optim.adamw(), Optim.Optimizer.AdamW(Optim.AdamW()))
end
