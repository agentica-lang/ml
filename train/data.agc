## Datasets and loaders (`train/DESIGN.md`, "API"): `N` examples held as two
## tensors, `inputs` `[N, ...]` and `targets` `[N, ...]`, cut into batches of
## `batch_size` rows in a seeded order per epoch. A batch is one `gather` of
## each tensor, never a loop over examples, both run as one traced
## evaluation, so a backend that computes only whole graphs (`Ggml`) makes
## batches too.
##
## With `prefetch > 0`, `feed` starts a task that builds the next batches while
## the caller trains and hands them over through a bounded channel. Its native
## tensors belong to that task, so they stay usable until `stop` ends it.
##
## The shuffle runs in VM steps (`O(N)` per epoch), so it counts against
## `--max-steps` like any Agentica code.

import Hub from "pkg/hub/hub"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Table from "table"

## The Park-Miller generator of `nn/init.agc`: reproducible, not statistical.
## It draws row ids below `MODULUS - 1`, so a dataset holds fewer rows.
MODULUS = 2147483647
MULTIPLIER = 48271

## How long a wait on the prefetch channel lasts before it looks again.
POLL_MS = 1000

## `index` is the row ids the batch was gathered with, kept so that
## `release` frees it too.
struct Batch
  inputs: Tensor.Tensor
  targets: Tensor.Tensor
  index: Optional[Tensor.Tensor] = none
end

struct Dataset
  inputs: Tensor.Tensor
  targets: Tensor.Tensor
end

## `prefetch` is how many batches the background task may build ahead; `0`
## builds each one when it is taken.
struct Loader
  dataset: Dataset
  batch_size: Int
  shuffle: Bool = true
  seed: Int = 0
  drop_last: Bool = false
  prefetch: Int = 2
end

## The batches of a run, a sequential stream: built on demand, or received
## from the prefetch task in order.
struct Feed
  loader: Loader
  order: List[Int]
  ahead: Optional[Ahead]
end

struct Ahead
  batches: Channel[Result[Batch]]
  producer: Task[Result[Void]]
end

## A dataset of two tensors with one first dimension, on one backend.
def dataset(inputs: Tensor.Tensor, targets: Tensor.Tensor) -> Result[Dataset]
  a = Tensor.shape(inputs)
  b = Tensor.shape(targets)
  return error("a dataset needs rank >= 1 tensors, got #{a} and #{b}") if a.empty? or b.empty?
  return error("inputs have #{a[0]} rows and targets #{b[0]}") if a[0] != b[0]
  return error("inputs and targets are on different backends") if inputs.backend != targets.backend
  ok(Dataset(inputs: inputs, targets: targets))
end

## A dataset of the tensors named `inputs` and `targets` of a safetensors
## file (or the index of a sharded one), read whole onto `backend`.
def load(path: String, inputs: String, targets: String, backend: Tensor.Backend) -> Result[Dataset]
  c = try Hub.open(path)
  dataset(try Hub.tensor(c, inputs, backend), try Hub.tensor(c, targets, backend))
end

## A dataset of the columns of `table`: the numeric columns `inputs` are the features, `[rows, features]` row by row, and
## the numeric column `target` the targets `[rows]`, as `F32` or, with `target_dtype: I32`, whole numbers (class ids for
## `Loss.cross_entropy`; a target that is not whole is an error). A column that is text, or holds a null, is an error that
## names it. The tensors are made on `backend`.
def from_table(
  table: Table.Table,
  inputs: List[String],
  target: String,
  backend: Tensor.Backend,
  target_dtype: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[Dataset]
  return error("a dataset needs at least one input column") if inputs.empty?
  if target_dtype != Tensor.Dtype.F32 and target_dtype != Tensor.Dtype.I32
    return error("target_dtype is F32 or I32, got #{target_dtype}")
  end

  columns: List[List[Float]] = []
  for name in inputs
    columns = columns.push(try Table.floats(table, name))
  end

  values: List[Tensor.Values] = []
  for row in 0...table.rows
    for column in columns
      values = values.push(Tensor.Values.Float(column[row]))
    end
  end

  goals = try Table.floats(table, target)
  aimed: List[Tensor.Values] = []
  for goal in goals
    if target_dtype == Tensor.Dtype.I32
      return error("column #{target} holds #{goal}, not a whole number") if goal != goal.round.to_f
      aimed = aimed.push(Tensor.Values.Int(goal.round))
    else
      aimed = aimed.push(Tensor.Values.Float(goal))
    end
  end

  features = try Tensor.from_list(values, [table.rows, inputs.size], Tensor.Dtype.F32, backend)
  dataset(features, try Tensor.from_list(aimed, [table.rows], target_dtype, backend))
end

def size(d: Dataset) -> Int = Tensor.shape(d.inputs)[0]

## Batches per epoch: a last partial batch counts unless `drop_last`.
def count(loader: Loader) -> Int
  return 0 if loader.batch_size < 1
  n = size(loader.dataset)
  return n / loader.batch_size if loader.drop_last
  (n + loader.batch_size - 1) / loader.batch_size
end

## The rows of epoch `epoch` (from 1), in order: a permutation of `0...N` drawn
## from the seed and the epoch, or `0...N` when not shuffled. `seed.abs` is
## used, so `s` and `-s` are one seed (see `start` for the other limits).
def order(loader: Loader, epoch: Int) -> List[Int]
  n = size(loader.dataset)
  rows = (0...n).to_a
  return rows if !loader.shuffle
  state = start(loader.seed, epoch)
  for i in (1...n).to_a.reverse
    state = (state * MULTIPLIER) % MODULUS
    j = state % (i + 1)
    swap = rows[i]
    rows = rows.set(i, rows[j]).set(j, swap)
  end

  rows
end

## Batch `i` of an epoch whose rows are `rows`.
def batch(loader: Loader, rows: List[Int], i: Int) -> Result[Batch]
  n = size(loader.dataset)
  from = i * loader.batch_size
  to = [from + loader.batch_size, n].min.unwrap!
  return error("batch #{i} is past the #{count(loader)} of an epoch") if i < 0 or i >= count(loader)
  d = loader.dataset
  picked = (from...to).map { |k| Tensor.Values.Int(rows[k]) }
  at = try Tensor.from_list(picked, [to - from], Tensor.Dtype.I32, d.inputs.backend)
  both = try Trace.eval_many([d.inputs, d.targets, at], &gathered)
  ok(Batch(inputs: both[0], targets: both[1], index: some(at)))
end

## Starts the batches of `epochs` epochs; with `prefetch > 0` a task builds
## them ahead. `stop` must follow, on every path.
def feed(loader: Loader, epochs: Int) -> Result[Feed]
  return error("batch_size must be at least 1, got #{loader.batch_size}") if loader.batch_size < 1
  return error("prefetch must be 0 or more, got #{loader.prefetch}") if loader.prefetch < 0
  return error("a loader takes fewer than #{MODULUS - 1} rows") if size(loader.dataset) >= MODULUS - 1
  return error("the loader makes no batch from #{size(loader.dataset)} rows") if count(loader) == 0
  return ok(Feed(loader: loader, order: [], ahead: none)) if loader.prefetch == 0
  batches: Channel[Result[Batch]] = Channel.new(loader.prefetch)
  producer = Task.spawn(&produce, Job(loader: loader, epochs: epochs, batches: batches))
  Task.monitor(producer, batches) do |down| error("the prefetch task ended: #{down.code} #{down.message}") end
  ok(Feed(loader: loader, order: [], ahead: some(Ahead(batches: batches, producer: producer))))
end

## The feed at the start of `epoch`: a synchronous one draws its order here.
def epoch(f: Feed, epoch: Int) -> Feed
  case f.ahead
    some(_) -> f
    none -> Feed(loader: f.loader, order: order(f.loader, epoch), ahead: none)
  end
end

## The batches that `batch(epoch, i)` makes, `count` for each of `epochs` epochs in order, built by a task that stays at
## most `prefetch + 1` ahead (`prefetch` at least 1): what a feed does for a loader, for a batch that a function makes, such as the
## decoding of image files. The task runs `batch`, so it overlaps whatever the caller does between `next` calls.
## `finish` must follow, on every path.
def ahead(batch: Fn[Int, Int, Result[Batch]], count: Int, epochs: Int, prefetch: Int) -> Result[Ahead]
  return error("prefetch must be at least 1, got #{prefetch}") if prefetch < 1
  batches: Channel[Result[Batch]] = Channel.new(prefetch)
  producer = Task.spawn(&produce_made, Made(batch: batch, count: count, epochs: epochs, batches: batches))
  Task.monitor(producer, batches) do |down| error("the prefetch task ended: #{down.code} #{down.message}") end
  ok(Ahead(batches: batches, producer: producer))
end

## The next batch of `a`, in order. A batch whose tensors were freed (the task ended without `finish`) is an error.
def next(a: Ahead) -> Result[Batch] = alive(try receive(a))

## Ends the task of `a`, which frees the native tensors it made.
def finish(a: Ahead) -> Void
  a.batches.close
  Task.cancel(a.producer)
  _ = Task.await(a.producer)
end

## Batch `i` of the current epoch. Under prefetch `i` is not read: the task
## gives the next batch of the stream, so take them in order. A batch whose
## tensors were freed (the task ended without `stop`) is an error.
def take(f: Feed, i: Int) -> Result[Batch]
  case f.ahead
    none -> batch(f.loader, f.order, i)
    some(a) -> next(a)
  end
end

## Ends the prefetch task, which frees the native tensors it made.
def stop(f: Feed) -> Void
  case f.ahead
    some(a) -> finish(a)
    none -> nil
  end
end

## Frees a batch's native tensors once a step is done with them.
def release(b: Batch) -> Void
  free(b.inputs)
  free(b.targets)
  case b.index
    some(t) -> free(t)
    none -> nil
  end
end

## True when a tensor of `b` was freed (`release`, `free`): the batch can no longer be read.
def freed?(b: Batch) -> Bool = closed?(b.inputs) or closed?(b.targets)

## Frees the native memory of `t` now on libtorch, ggml and XLA; the reference has none. Nothing may read `t` afterwards.
def free(t: Tensor.Tensor) -> Void
  case t.handle
    Tensor.Handle.Native(h) -> Native.close(h)
    Tensor.Handle.Ggml(h) -> Native.close(h)
    Tensor.Handle.Xla(h) -> Native.close(h)
    _ -> nil
  end
end

private

struct Job
  loader: Loader
  epochs: Int
  batches: Channel[Result[Batch]]
end

struct Made
  batch: Fn[Int, Int, Result[Batch]]
  count: Int
  epochs: Int
  batches: Channel[Result[Batch]]
end

## Waits for the next item; an error once the channel is closed and drained.
## A wait that timed out may have raced with the producer, which sends a batch
## and closes with its reason between the wait and the `closed?` check: the
## last `recv(0)` then gives that batch, or the reason, and not the timeout.
def receive(a: Ahead) -> Result[Result[Batch]]
  got = a.batches.recv(POLL_MS)
  while !got.ok? and !a.batches.closed?
    got = a.batches.recv(POLL_MS)
  end

  return a.batches.recv(0) if !got.ok?
  got
end

def alive(got: Result[Batch]) -> Result[Batch]
  b = try got
  return error("the batch was freed: its prefetch task ended") if freed?(b)
  ok(b)
end

## Builds every batch in order and sends it. A failure closes the channel with
## it: the consumer takes the batches built before, then that error. Then the
## task stays until `stop` cancels it, whether it failed or not: its tensors
## must outlive their use.
def produce(job: Job) -> Result[Void] = linger(send_all(job), job.batches)

def produce_made(job: Made) -> Result[Void] = linger(send_made(job), job.batches)

def linger(sent: Result[Void], batches: Channel[Result[Batch]]) -> Result[Void]
  case sent
    error(e) -> batches.close(e)
    ok(_) -> nil
  end

  while true
    Task.sleep(POLL_MS)
  end

  ok()
end

def send_made(job: Made) -> Result[Void]
  for e in 1..job.epochs
    for i in 0...job.count
      b = try job.batch.call(e, i)
      return ok() if !hand(job.batches, ok(b))
    end
  end

  ok()
end

def send_all(job: Job) -> Result[Void]
  for e in 1..job.epochs
    rows = order(job.loader, e)
    for i in 0...count(job.loader)
      b = try batch(job.loader, rows, i)
      return ok() if !hand(job.batches, ok(b))
    end
  end

  ok()
end

## Sends `item`, waiting for room; false once the consumer closed the channel.
def hand(batches: Channel[Result[Batch]], item: Result[Batch]) -> Bool
  sent = batches.send(item, POLL_MS)
  while !sent.ok? and !batches.closed?
    sent = batches.send(item, POLL_MS)
  end

  sent.ok?
end

## The rows `xs[2]` names of the inputs and the targets `xs[0]`, `xs[1]`.
def gathered(xs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  ok([try Tensor.gather(xs[0], xs[2]), try Tensor.gather(xs[1], xs[2])])
end

def closed?(t: Tensor.Tensor) -> Bool
  case t.handle
    Tensor.Handle.Native(h) -> Native.closed?(h)
    Tensor.Handle.Ggml(h) -> Native.closed?(h)
    Tensor.Handle.Xla(h) -> Native.closed?(h)
    _ -> false
  end
end

## A state in `1...MODULUS` for a seed and an epoch, warmed past the first
## draws, which follow the seed linearly. Seeds below `2^20` and epochs below
## `2048` give distinct states (the key `seed * 2048 + epoch` stays below
## `MODULUS - 1` and the multiply is a bijection); larger ones wrap and may
## repeat an order.
def start(seed: Int, epoch: Int) -> Int
  key = (seed.abs % 1048576) * 2048 + epoch % 2048
  state = key % (MODULUS - 1) + 1
  for _ in 0...4
    state = (state * MULTIPLIER) % MODULUS
  end

  state
end
