## `Train.fit_batches`: a fit over batches that a function makes. With the batches of a loader it is `fit`; with rows of
## token ids padded to the longest of each batch, every batch has its own shape and the mask comes from the ids inside
## the loss, as a sentence model's does. On ggml a shape compiles once. Run from `train/`.

import Data from "../data"
import Loss from "../loss"
import Lower from "pkg/tensor/src/lower"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"

INPUTS = [1.0, 2.0, -1.0, 0.5, 0.25, -2.0, 1.5, 1.0]
TARGETS = [0.5, -1.0, 2.0, 1.0]

## Rows of token ids with a target each; 0 is padding. Batches of two rows, each as long as its longest row: 3, 5, 3.
ROWS = [[4, 2, 7], [5, 1], [3, 3, 9, 2, 6], [8], [2, 6, 1], [7, 7]]
SCORES = [1.0, 0.5, 2.0, 1.5, 0.75, 1.25]

## `F32` on libtorch and ggml against the doubles of the reference.
ABS = 0.00001
REL = 0.0001

private

def adam() -> Optim.Optimizer = Optim.Optimizer.Adam(Optim.Adam(lr: 0.05))

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

## Batch `i` of the four rows, two rows each, built anew.
def pair(backend: Tensor.Backend, i: Int) -> Result[Data.Batch]
  ok(
    Data.Batch(
      inputs: try S.tensor(INPUTS.drop(4 * i).take(4), [2, 2], Tensor.Dtype.F32, backend),
      targets: try S.tensor(TARGETS.drop(2 * i).take(2), [2, 1], Tensor.Dtype.F32, backend),
    ),
  )
end

def same(a: Train.Fit, b: Train.Fit, what: String) -> Result[Void]
  try S.expect_close(a.history, b.history, 0.0, 0.0, "history #{what}")
  for i in 0...a.params.size
    try S.expect_close(try S.floats_of(a.params[i]), try S.floats_of(b.params[i]), 0.0, 0.0, "weight #{i} #{what}")
  end

  ok()
end

def backends() -> Result[List[Tensor.Backend]]
  out = [S.REF].concat(try S.ggml_backends())
  return ok(out.push(S.TORCH)) if Tensor.available(S.TORCH)
  ok(out)
end

## The batches of an unshuffled loader, made by a function, train as the loader does: the same losses and weights, bit
## for bit. The weights the caller gave are still its own after the fit.
def test_the_batches_of_a_loader_from_a_function_are_fit() -> Result[Void]
  for backend in try backends()
    start = try weights(backend)
    loader = Data.Loader(dataset: try rows(backend), batch_size: 2, shuffle: false)
    want = try Train.fit(&linear, start, loader, adam(), 3)
    got = try Train.fit_batches(&linear, start, ->(_e: Int, i: Int) { pair(backend, i) }, 2, adam(), 3)
    try same(got, want, "on #{backend}")
    try Test.eq(want.history.size, 3)
    try Test.assert(want.history[2] < want.history[0], "the loss did not fall on #{backend}")
    try S.expect_close(try S.floats_of(start[0]), [0.3, -0.2], 0.0000001, 0.0, "the caller's weights on #{backend}")
  end

  ok()
end

## The rows `fit` takes for each batch of a loader are the rows `Data.batch` cuts (shuffled with a last partial batch,
## with `drop_last`, unshuffled): training over batches that `Data.batch` made, through `fit_batches`, ends where `fit`
## ends, bit for bit. On ggml `fit` makes the ids and gathers in its step, so this is what keeps the two cuts together.
def test_fit_takes_the_rows_data_batch_cuts() -> Result[Void]
  for backend in try backends()
    d = try rows(backend)
    for loader in [
      Data.Loader(dataset: d, batch_size: 3, seed: 4),
      Data.Loader(dataset: d, batch_size: 3, seed: 9, drop_last: true),
      Data.Loader(dataset: d, batch_size: 3, shuffle: false),
    ]
      start = try weights(backend)
      want = try Train.fit_batches(
        &linear,
        start,
        ->(e: Int, i: Int) { cut(loader, e, i) },
        Data.count(loader),
        adam(),
        4,
      )

      got = try Train.fit(&linear, start, loader, adam(), 4)
      try same(got, want, "on #{backend}, batch size #{loader.batch_size}, drop_last #{loader.drop_last}")
    end
  end

  ok()
end

## Batch `i` of epoch `e` of `loader`, as `Data.batch` cuts it.
def cut(loader: Data.Loader, e: Int, i: Int) -> Result[Data.Batch] = Data.batch(loader, Data.order(loader, e), i)

## Batch `i` of `ROWS`: the ids padded with 0 to the longest row of the batch, and the scores.
def padded(backend: Tensor.Backend, i: Int) -> Result[Data.Batch]
  picked = ROWS.drop(2 * i).take(2)
  seq = picked.map { |r| r.size }.max.unwrap!
  ids: List[Tensor.Values] = []
  for r in picked
    for k in 0...seq
      if k < r.size
        ids = ids.push(Tensor.Values.Int(r[k]))
      else
        ids = ids.push(Tensor.Values.Int(0))
      end
    end
  end

  ok(
    Data.Batch(
      inputs: try Tensor.from_list(ids, [2, seq], Tensor.Dtype.I32, backend),
      targets: try S.tensor(SCORES.drop(2 * i).take(2), [2, 1], Tensor.Dtype.F32, backend),
    ),
  )
end

## `w` times the mean of the ids of each row, padding left out, plus `b`: the mask is `ids != 0`, taken in the loss.
def pooled(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  shape = Tensor.shape(b.inputs)
  zeros = try Tensor.zeros(shape, Tensor.Dtype.I32, Tensor.backend(b.inputs))
  ones = try Tensor.full(shape, Tensor.Values.Float(1.0), Tensor.Dtype.F32, Tensor.backend(b.inputs))
  pad = try Tensor.to_dtype(try Tensor.eq(b.inputs, zeros), Tensor.Dtype.F32)
  mask = try Tensor.sub(ones, pad)
  values = try Tensor.mul(try Tensor.to_dtype(b.inputs, Tensor.Dtype.F32), mask)
  mean = try Tensor.div(try Tensor.sum(values, some([1]), true), try Tensor.sum(mask, some([1]), true))
  w = try Tensor.broadcast_to(try Tensor.reshape(ps[0], [1, 1]), [shape[0], 1])
  bias = try Tensor.broadcast_to(try Tensor.reshape(ps[1], [1, 1]), [shape[0], 1])
  Loss.mse(try Tensor.add(try Tensor.mul(mean, w), bias), b.targets)
end

def scalars(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([try S.tensor([0.1], [1], Tensor.Dtype.F32, backend), try S.tensor([0.0], [1], Tensor.Dtype.F32, backend)])
end

def varied(backend: Tensor.Backend) -> Result[Train.Fit]
  Train.fit_batches(&pooled, try scalars(backend), ->(_e: Int, i: Int) { padded(backend, i) }, 3, adam(), 4)
end

## Batches of lengths 3, 5 and 3 over four epochs: the other backends follow the reference, and ggml compiles the step
## of each of the two shapes once, not once per batch.
def test_batches_of_their_own_shape_train_with_a_mask_from_the_ids() -> Result[Void]
  want = try varied(S.REF)
  try Test.assert(want.history[3] < want.history[0], "the loss did not fall")
  for backend in (try backends()).drop(1)
    before = try Lower.compiles()
    got = try varied(backend)
    try S.expect_close(got.history, want.history, ABS, REL, "history on #{backend}")
    for i in 0...want.params.size
      try S.expect_close(try S.floats_of(got.params[i]), try S.floats_of(want.params[i]), ABS, REL, "#{i}")
    end

    if backend != S.TORCH
      try Test.assert(try Lower.compiles() - before <= 2, "the steps of two shapes compiled more than two graphs")
    end
  end

  ok()
end

## No batch at all is an error, and the error of a batch says which one it was.
def test_a_batch_that_fails_names_its_epoch_and_index() -> Result[Void]
  start = try weights(S.REF)
  none_made = ->(_e: Int, i: Int) { pair(S.REF, i) }
  try Test.error?(Train.fit_batches(&linear, start, none_made, 0, adam(), 1))
  broken = ->(e: Int, i: Int) {
    return error("no rows") if e == 2 and i == 1
    pair(S.REF, i)
  }

  case Train.fit_batches(&linear, start, broken, 2, adam(), 3)
    ok(_) -> return error("a failed batch did not stop the fit")
    error(e) -> try Test.eq(e.message, "epoch 2, batch 1: no rows")
  end

  S.expect_close(try S.floats_of(start[0]), [0.3, -0.2], 0.0000001, 0.0, "the caller's weights after the error")
end

## `fit_batches` frees a batch after its step, so a function that answers the batches it kept fails at the second
## epoch, with an error that says what to do. The run's own tensors are freed, and the caller's weights are not: on
## libtorch and ggml, where a tensor has native memory to free.
def test_a_batch_answered_twice_is_an_error_that_says_so() -> Result[Void]
  for backend in (try backends()).drop(1)
    start = try weights(backend)
    kept = [try pair(backend, 0), try pair(backend, 1)]
    case Train.fit_batches(&linear, start, ->(_e: Int, i: Int) { ok(kept[i]) }, 2, adam(), 2)
      ok(_) -> return error("a freed batch trained on #{backend}")
      error(e) ->
        try Test.eq(e.message, "epoch 2, batch 0: batch(2, 0) returned tensors that are already freed: make new ones")
    end

    try S.expect_close(try S.floats_of(start[0]), [0.3, -0.2], 0.0000001, 0.0, "the caller's weights on #{backend}")
  end

  ok()
end

## `prefetch` builds the batches on a task that stays ahead of the steps: the same losses and weights, bit for bit, for
## batches of two shapes too, and a batch that fails is the same error of the same step, with the task ended.
def test_prefetch_builds_the_batches_on_a_task_and_trains_the_same() -> Result[Void]
  for backend in try backends()
    start = try weights(backend)
    make = ->(_e: Int, i: Int) { padded(backend, i) }
    want = try Train.fit_batches(&pooled, try scalars(backend), make, 3, adam(), 4)
    for ahead in [1, 3]
      got = try Train.fit_batches(&pooled, try scalars(backend), make, 3, adam(), 4, prefetch: ahead)
      try same(got, want, "on #{backend}, prefetch #{ahead}")
    end

    broken = ->(e: Int, i: Int) {
      return error("no rows") if e == 2 and i == 1
      pair(backend, i)
    }

    case Train.fit_batches(&linear, start, broken, 2, adam(), 3, prefetch: 2)
      ok(_) -> return error("a failed batch did not stop the fit on #{backend}")
      error(e) -> try Test.eq(e.message, "epoch 2, batch 1: no rows")
    end
  end

  Test.error?(
    Train.fit_batches(&linear, try weights(S.REF), ->(_e: Int, i: Int) { pair(S.REF, i) }, 2, adam(), 1, prefetch: -1),
  )
end

## `on_step` after each step with prefetch on: the epoch from 1, the batch from 0, the step counted across epochs. An
## error from it at step 3 stops the fit at once with the epoch and the batch, the task ends, the run's own tensors are
## freed (on ggml and libtorch) and the caller's weights are not.
def test_on_step_stops_a_prefetching_fit_at_once() -> Result[Void]
  for backend in try backends()
    start = try weights(backend)
    seen: Channel[String] = Channel.new(16)
    last: Channel[Tensor.Tensor] = Channel.new(1)
    on_step = ->(s) {
      _ = seen.send("#{s.epoch}/#{s.batch}/#{s.step}", 0)
      return error("stop") if s.step == 3 and last.send(s.params[0], 0).ok?

      ok(Train.Control.Continue)
    }

    r = Train.fit_batches(
      &linear,
      start,
      ->(_e: Int, i: Int) { pair(backend, i) },
      2,
      adam(),
      3,
      prefetch: 2,
      on_step: some(on_step),
    )

    try S.expect_error(r, "epoch 2, batch 0: stop")
    try Test.eq([try seen.recv(0), try seen.recv(0), try seen.recv(0)], ["1/0/1", "1/1/2", "2/0/3"])
    try Test.error?(seen.recv(0))
    try S.expect_close(try S.floats_of(start[0]), [0.3, -0.2], 0.0000001, 0.0, "the caller's weights on #{backend}")
    if backend != S.REF
      kept = try last.recv(0)
      try Test.assert(
        Data.freed?(Data.Batch(inputs: kept, targets: kept)),
        "the run's own weights are freed on #{backend}",
      )
    end
  end

  ok()
end

## `Stop` from `on_step` mid-epoch ends a prefetching fit with its `Fit`: the weights so far, usable, and a history
## whose last epoch is the mean over the steps it ran (one here).
def test_stop_from_on_step_answers_the_fit_so_far() -> Result[Void]
  for backend in try backends()
    stop = ->(s: Train.StepInfo) {
      return ok(Train.Control.Stop) if s.step == 3

      ok(Train.Control.Continue)
    }

    fit = try Train.fit_batches(
      &linear,
      try weights(backend),
      ->(_e: Int, i: Int) { pair(backend, i) },
      2,
      adam(),
      3,
      prefetch: 2,
      on_step: some(stop),
    )

    try Test.eq(fit.history.size, 2)
    try Test.eq((try S.floats_of(fit.params[0])).size, 2)
  end

  ok()
end

## The calls the prefetch task made, in order, as `epoch * 10 + i`.
def calls(log: Channel[Int]) -> List[Int]
  out: List[Int] = []
  got = log.recv(0)
  while got.ok?
    out = out.push(got.unwrap!)
    got = log.recv(0)
  end

  out
end

def failing(_ps: List[Tensor.Tensor], _b: Data.Batch) -> Result[Tensor.Tensor] = error("boom")

## The batch function runs `epochs * count` times, in order, on the task; a step that fails stops the fit after a few
## calls only, with the task ended (`fit_batches` awaits it before it returns, and the runner fails a test that leaves
## one running).
def test_prefetch_calls_the_batch_function_in_order_and_stops_with_the_fit() -> Result[Void]
  log: Channel[Int] = Channel.new(100)
  record = ->(e: Int, i: Int) {
    _ = log.send(e * 10 + i, 1000)
    pair(S.REF, i)
  }

  try Train.fit_batches(&linear, try weights(S.REF), record, 2, adam(), 3, prefetch: 2)
  try Test.eq(calls(log), [10, 11, 20, 21, 30, 31])
  case Train.fit_batches(&failing, try weights(S.REF), record, 2, adam(), 50, prefetch: 2)
    ok(_) -> return error("a failing step did not stop the fit")
    error(e) -> try Test.eq(e.message, "epoch 1, batch 0: boom")
  end

  made = calls(log)
  try Test.assert(made.size >= 1 and made.size <= 4, "the task made #{made.size} batches of the 100 for a failed step")
  Test.eq(made.take(2), [10, 11])
end
