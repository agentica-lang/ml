## Datasets and loaders (`train/DESIGN.md`, "Data"): `N` examples held as two
## tensors, `inputs` `[N, ...]` and `targets` `[N, ...]`, cut into batches of
## `batch_size` rows in a seeded order per epoch. A batch is one `gather` of
## each tensor (one `slice` when not shuffled), never a loop over examples.
##
## With `prefetch > 0`, `feed` starts a task that builds the next batches while
## the caller trains and hands them over through a bounded channel. Its native
## tensors belong to that task, so they stay usable until `stop` ends it.

import Hub from "pkg/hub/hub"
import Tensor from "pkg/tensor/tensor"

## The Park-Miller generator of `nn/init.agc`: reproducible, not statistical.
MODULUS = 2147483647
MULTIPLIER = 48271

## How long a wait on the prefetch channel lasts before it looks again.
POLL_MS = 1000

struct Batch
  inputs: Tensor.Tensor
  targets: Tensor.Tensor
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

## The batches of a run: built on demand, or received from the prefetch task.
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

def size(d: Dataset) -> Int = Tensor.shape(d.inputs)[0]

## Batches per epoch: a last partial batch counts unless `drop_last`.
def count(loader: Loader) -> Int
  return 0 if loader.batch_size < 1
  n = size(loader.dataset)
  return n / loader.batch_size if loader.drop_last
  (n + loader.batch_size - 1) / loader.batch_size
end

## The rows of epoch `epoch` (from 1), in order: a permutation of `0...N` drawn
## from the seed and the epoch, or `0...N` when not shuffled.
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
  return ok(Batch(inputs: try window(d.inputs, from, to), targets: try window(d.targets, from, to))) if !loader.shuffle

  picked = rows.drop(from).take(to - from).map { |r| Tensor.Values.Int(r) }
  at = try Tensor.from_list(picked, [to - from], Tensor.Dtype.I64, d.inputs.backend)
  ok(Batch(inputs: try Tensor.gather(d.inputs, at), targets: try Tensor.gather(d.targets, at)))
end

## Starts the batches of `epochs` epochs; with `prefetch > 0` a task builds
## them ahead. `stop` must follow, on every path.
def feed(loader: Loader, epochs: Int) -> Result[Feed]
  return error("batch_size must be at least 1, got #{loader.batch_size}") if loader.batch_size < 1
  return error("prefetch must be 0 or more, got #{loader.prefetch}") if loader.prefetch < 0
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

## Batch `i` of the current epoch; the prefetch task gives them in order.
def take(f: Feed, i: Int) -> Result[Batch]
  case f.ahead
    none -> batch(f.loader, f.order, i)
    some(a) -> receive(a)
  end
end

## Ends the prefetch task, which frees the native tensors it made.
def stop(f: Feed) -> Void
  case f.ahead
    some(a) ->
      a.batches.close
      Task.cancel(a.producer)
      _ = Task.await(a.producer)
    none -> nil
  end
end

## Frees a batch's native tensors once a step is done with them.
def release(b: Batch) -> Void
  free(b.inputs)
  free(b.targets)
end

private

struct Job
  loader: Loader
  epochs: Int
  batches: Channel[Result[Batch]]
end

def receive(a: Ahead) -> Result[Batch]
  while true
    case a.batches.recv(POLL_MS)
      ok(b) -> return b
      error(_) -> return error("the prefetch task ended early") if a.batches.closed? and a.batches.empty?
    end
  end

  error("unreachable")
end

## Builds every batch in order and sends it, an error included, then waits for
## `stop`: its tensors must outlive their use.
def produce(job: Job) -> Result[Void]
  outcome = send_all(job)
  if !outcome.ok?
    _ = hand(job.batches, error(outcome.message))
  end

  while !job.batches.closed?
    Task.sleep(POLL_MS)
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
  while true
    case batches.send(item, POLL_MS)
      ok(_) -> return true
      error(_) -> return false if batches.closed?
    end
  end

  false
end

## Rows `from...to` of `t`, every other axis whole.
def window(t: Tensor.Tensor, from: Int, to: Int) -> Result[Tensor.Tensor]
  shape = Tensor.shape(t)
  starts = shape.map { |_| 0 }.set(0, from)
  Tensor.slice(t, starts, shape.set(0, to))
end

def free(t: Tensor.Tensor) -> Void
  case t.handle
    Tensor.Handle.Native(h) -> Native.close(h)
    _ -> nil
  end
end

## A state in `1...MODULUS` for a seed and an epoch, warmed past the first
## draws, which follow the seed linearly.
def start(seed: Int, epoch: Int) -> Int
  state = (seed.abs % (MODULUS - 1) + epoch * 7919) % (MODULUS - 1) + 1
  for _ in 0...4
    state = (state * MULTIPLIER) % MODULUS
  end

  state
end
