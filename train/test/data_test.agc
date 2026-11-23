## The loader: every row once per epoch, a seeded order, the last partial
## batch, the prefetch task giving what the synchronous path builds, and the
## task ending on every path (the runner fails a test that leaves one running).

import Data from "../data"
import Loss from "../loss"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "../train"

ROWS = 10

private

def test_every_row_comes_once_per_epoch() -> Result[Void]
  loader = Data.Loader(dataset: try rows(S.REF), batch_size: 3, seed: 7)
  try Test.eq(Data.count(loader), 4)
  for e in 1..3
    seen = try ids(try epoch(loader, e))
    try Test.eq(seen.size, ROWS)
    try Test.eq(seen.sort, (0...ROWS).to_a)
  end

  ok()
end

def test_drop_last_leaves_out_the_partial_batch() -> Result[Void]
  loader = Data.Loader(dataset: try rows(S.REF), batch_size: 3, seed: 7, drop_last: true)
  try Test.eq(Data.count(loader), 3)
  batches = try epoch(loader, 1)
  try Test.eq(batches.map { |b| Tensor.shape(b.inputs) }, [[3, 2], [3, 2], [3, 2]])
  seen = try ids(batches)
  try Test.eq(seen.uniq.size, 9)
  Test.eq(seen, Data.order(loader, 1).take(9))
end

## Unshuffled, the last batch is the one row left, and every target stays
## with its input.
def test_the_last_partial_batch_keeps_its_targets() -> Result[Void]
  loader = Data.Loader(dataset: try rows(S.REF), batch_size: 3, shuffle: false)
  batches = try epoch(loader, 1)
  last = batches[3]
  try Test.eq(Tensor.shape(last.inputs), [1, 2])
  try Test.eq(try S.floats_of(last.inputs), [9.0, 9.5])
  try Test.eq(try S.floats_of(last.targets), [90.0])
  for b in try epoch(Data.Loader(dataset: try rows(S.REF), batch_size: 4, seed: 3), 2)
    try Test.eq(try S.floats_of(b.targets), (try firsts(b)).map { |x| x * 10.0 })
  end

  ok()
end

def test_the_seed_fixes_the_order_and_epochs_differ() -> Result[Void]
  a = Data.Loader(dataset: try rows(S.REF), batch_size: 3, seed: 11)
  b = Data.Loader(dataset: try rows(S.REF), batch_size: 3, seed: 12)
  try Test.eq(Data.order(a, 1), Data.order(a, 1))
  try Test.assert(Data.order(a, 1) != Data.order(a, 2), "epochs 1 and 2 share an order")
  try Test.assert(Data.order(a, 2) != Data.order(a, 3), "epochs 2 and 3 share an order")
  try Test.assert(Data.order(a, 1) != Data.order(b, 1), "seeds 11 and 12 share an order")
  Test.eq(Data.order(Data.Loader(dataset: a.dataset, batch_size: 3, shuffle: false), 2), (0...ROWS).to_a)
end

def test_prefetch_gives_the_synchronous_batches() -> Result[Void]
  for backend in backends()
    sync = Data.Loader(dataset: try rows(backend), batch_size: 3, seed: 5, prefetch: 0)
    ahead = Data.Loader(dataset: sync.dataset, batch_size: 3, seed: 5, prefetch: 2)
    try Test.eq(try values(sync, 3), try values(ahead, 3))
  end

  ok()
end

## Stopped after one batch of three epochs, the prefetch task ends.
def test_stopping_early_ends_the_prefetch_task() -> Result[Void]
  feed = try Data.feed(Data.Loader(dataset: try rows(S.REF), batch_size: 1, prefetch: 1), 3)
  b = try Data.take(Data.epoch(feed, 1), 0)
  try Test.eq(Tensor.shape(b.inputs), [1, 2])
  Data.stop(feed)
  ok()
end

## A batch the task cannot build (`from_list` makes no traced tensor) reaches
## `fit` as the error of its step, and the task ends.
def test_a_failed_batch_is_an_error_of_fit() -> Result[Void]
  d = try rows(S.REF)
  traced = try Data.dataset(Trace.constant(d.inputs), Trace.constant(d.targets))
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.01))
  got = Train.fit(&linear, [try S.tensor([0.0, 0.0], [2, 1])], Data.Loader(dataset: traced, batch_size: 3), sgd, 2)
  try Test.assert(!got.ok?, "a failed batch is an error")
  Test.assert(got.message.start_with?("epoch 1, batch 0: "), "got #{got.message}")
end

## A loss that fails mid-run stops `fit` with the task still building ahead.
def test_a_failed_step_ends_the_prefetch_task() -> Result[Void]
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.01))
  loader = Data.Loader(dataset: try rows(S.REF), batch_size: 1, prefetch: 3)
  got = Train.fit(&wrong_shape, [try S.tensor([0.0, 0.0, 0.0], [3, 1])], loader, sgd, 5)
  Test.assert(got.message.start_with?("epoch 1, batch 0: "), "got #{got.message}")
end

## Shuffled minibatches train the same with and without prefetch.
def test_fit_through_prefetch_matches_the_synchronous_run() -> Result[Void]
  for backend in backends()
    d = try rows(backend)
    sync = try linear_fit(Data.Loader(dataset: d, batch_size: 3, seed: 2, prefetch: 0), backend)
    ahead = try linear_fit(Data.Loader(dataset: d, batch_size: 3, seed: 2, prefetch: 2), backend)
    try Test.eq(ahead.history, sync.history)
    try Test.assert(sync.history[3] < sync.history[0], "the loss falls: #{sync.history}")
  end

  ok()
end

## A dataset read from a safetensors file by its two tensor names.
def test_a_dataset_loads_from_safetensors() -> Result[Void]
  path = "/tmp/agentica-train-data-#{System.pid}.safetensors"
  f = try File.open(path, "write")
  try File.write_bytes(f, file())
  try File.close(f)
  loaded = Data.load(path, "x", "y", S.REF)
  _ = Path.remove(path)
  d = try loaded
  try Test.eq([Tensor.shape(d.inputs), Tensor.shape(d.targets)], [[3, 2], [3]])
  try Test.eq(Data.size(d), 3)
  try Test.eq(try S.floats_of(d.targets), [0.5, 1.5, 2.5])
  Test.assert(!Data.load(path, "x", "y", S.REF).ok?, "a missing file is an error")
end

def test_a_dataset_needs_one_row_count() -> Result[Void]
  got = Data.dataset(try S.tensor([1.0, 2.0], [2]), try S.tensor([1.0], [1]))
  Test.assert(!got.ok?, "two row counts are refused")
end

## Row `i` is `[i, i + 0.5]`, its target `10 i`.
def rows(backend: Tensor.Backend) -> Result[Data.Dataset]
  inputs = (0...ROWS).to_a.flat_map { |i| [i.to_f, i.to_f + 0.5] }
  targets = (0...ROWS).to_a.map { |i| 10.0 * i.to_f }
  Data.dataset(
    try S.tensor(inputs, [ROWS, 2], Tensor.Dtype.F32, backend),
    try S.tensor(targets, [ROWS], Tensor.Dtype.F32, backend),
  )
end

def backends() -> List[Tensor.Backend]
  return [S.REF, S.TORCH] if Tensor.available(S.TORCH)
  [S.REF]
end

def epoch(loader: Data.Loader, e: Int) -> Result[List[Data.Batch]]
  order = Data.order(loader, e)
  out: List[Data.Batch] = []
  for i in 0...Data.count(loader)
    out = out.push(try Data.batch(loader, order, i))
  end

  ok(out)
end

## The row ids of a list of batches, in order.
def ids(batches: List[Data.Batch]) -> Result[List[Int]]
  out: List[Int] = []
  for b in batches
    for x in try firsts(b)
      out = out.push(try x.to_i)
    end
  end

  ok(out)
end

def firsts(b: Data.Batch) -> Result[List[Float]]
  xs = try S.floats_of(b.inputs)
  ok((0...(xs.size / 2)).to_a.map { |k| xs[2 * k] })
end

## Every input value of every batch of `epochs` epochs, through `feed`.
def values(loader: Data.Loader, epochs: Int) -> Result[List[List[Float]]]
  feed = try Data.feed(loader, epochs)
  got = taken(feed, epochs)
  Data.stop(feed)
  got
end

def taken(feed: Data.Feed, epochs: Int) -> Result[List[List[Float]]]
  out: List[List[Float]] = []
  for e in 1..epochs
    f = Data.epoch(feed, e)
    for i in 0...Data.count(feed.loader)
      b = try Data.take(f, i)
      out = out.push(try S.floats_of(b.inputs) + try S.floats_of(b.targets))
    end
  end

  ok(out)
end

def linear_fit(loader: Data.Loader, backend: Tensor.Backend) -> Result[Train.Fit]
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.002))
  Train.fit(&linear, [try S.tensor([0.0, 0.0], [2, 1], Tensor.Dtype.F32, backend)], loader, sgd, 4)
end

def linear(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  rows = Tensor.shape(b.inputs)[0]
  Loss.mse(try Tensor.reshape(try Tensor.matmul(b.inputs, ps[0]), [rows]), b.targets)
end

def wrong_shape(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor] = Tensor.matmul(b.inputs, ps[0])

## A safetensors file: `x` `[3, 2]` and `y` `[3]`, both `F32`.
def file() -> Bytes
  header = (
    "{\"x\":{\"dtype\":\"F32\",\"shape\":[3,2],\"data_offsets\":[0,24]},\"y\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[24,36]}}"
  )

  padded = header + " " * ((8 - header.bytesize % 8) % 8)
  floats = [0.0, 0.25, 1.0, 1.25, 2.0, 2.25, 0.5, 1.5, 2.5]
  data = floats.flat_map { |f| le(Float.to_bits32(f), 4) }
  Bytes.from_list(le(padded.bytesize, 8) + padded.bytes.to_a + data).unwrap!
end

def le(v: Int, count: Int) -> List[Int] = (0...count).to_a.map { |k| v.shift_right(8 * k).bit_and(255) }
