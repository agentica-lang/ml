---
title: data loader
description: Datasets of two tensors, cut into seeded, shuffled batches, with prefetch on a task and the feed API that fit uses.
weight: 1
---

`Data` holds a dataset of two tensors, `inputs` and `targets`, which share their first dimension. It cuts them into batches of rows, in a seeded order for each epoch. `Train.fit` takes a `Data.Loader` and uses it for every epoch.

## Example

Batches of two rows, in order, with the last partial batch left out. `feed` starts the batches, `take` gives one, and `release` and `stop` end them.

```agentica
import Data from "pkg/train/data"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  inputs = Tensor.floats(xs, [5, 2]).unwrap!
  targets = Tensor.floats([0.0, 1.0, 2.0, 3.0, 4.0], [5]).unwrap!
  dataset = Data.dataset(inputs, targets).unwrap!
  loader = Data.Loader(
    dataset: dataset,
    batch_size: 2,
    shuffle: false,
    drop_last: true,
  )
  puts(Data.count(loader))
  feed = Data.feed(loader, 2).unwrap!
  batch = Data.take(Data.epoch(feed, 1), 0).unwrap!
  puts(Tensor.shape(batch.inputs))
  puts(Tensor.to_list(batch.targets).unwrap!)
  Data.release(batch)
  Data.stop(feed)
end
```

```output
2
[2, 2]
[Values.Float(0.0), Values.Float(1.0)]
```

## Datasets

- `Data.dataset(inputs, targets)` checks that both tensors have rank 1 or more, the same number of rows, and the same backend.
- `Data.load(path, inputs, targets, backend)` reads two named tensors of a safetensors file, or of a sharded checkpoint through its index, onto `backend`. The whole file is read.
- `Data.size(dataset)` is the number of rows.

## Loader

`Data.Loader` has these fields:

- `dataset`, the dataset.
- `batch_size`, at least 1.
- `shuffle`, default `true`.
- `seed`, default `0`.
- `drop_last`, default `false`.
- `prefetch`, default `2`; `0` builds each batch when it is taken.

`Data.count(loader)` is the number of batches per epoch. The last partial batch counts unless `drop_last` is set.

## Epochs and order

- Each epoch is a permutation of the rows. It comes from the seed and the epoch number, so a seed gives the same order each time, and two epochs give different orders.
- `Data.order(loader, epoch)` gives the rows of an epoch. With `shuffle: false` the rows come in order.
- Each batch is one `from_list` of its row ids, `I32`, in order when not shuffled, then one `gather` per tensor.
- The `from_list` and both gathers run as one traced evaluation, so a backend that computes only whole graphs (`Ggml`) makes batches too.
- `s` and `-s` are the same seed. Seeds below `2^20` and epochs below `2048` give distinct starting states, so their orders differ.

## Feeding by hand

`fit` does this for you. To read batches yourself:

- `Data.feed(loader, epochs)` starts the feed. With `prefetch > 0` it also starts a task.
- `Data.epoch(feed, e)` gives the feed for epoch `e`, counting from 1.
- `Data.take(feed, i)` gives batch `i` of the epoch. A `Data.Batch` has `inputs`, `targets` and `index`, the row ids of the batch.
- `Data.release(batch)` frees the batch's native tensors. Call it when the step is done.
- `Data.stop(feed)` ends the prefetch task. Call it when you are done, including after an error.

## Prefetch

- With `prefetch = K > 0`, a task builds the batches of every epoch in order. It sends them through a channel of capacity `K`, so it stays at most `K` batches ahead while the caller steps.
- Under prefetch, `take` gives the next batch of the stream. The `i` argument is not read, so take the batches in order.
- A batch that fails to build reaches the caller as an error in its place. A crash of the task also reaches the caller as an error.
- Native tensors made by the task belong to it. They stay usable until `stop`.
- Prefetch hides the time it takes to build a batch, and nothing more. The measured gain is on libtorch on the CPU, with 16 384 rows and batches of 2 048: 154 steps per second without prefetch and 232 with it. The step's own compute is not overlapped.

## Limits

- On `Ggml`, the source of a batch's `gather` holds at most 2^24 rows, so a dataset of more rows must be on `Reference` or `Torch`.
- A batch is one compiled graph per batch shape on `Ggml`, so a last partial batch costs a second graph.
- A dataset must have fewer than 2 147 483 646 rows.
- The shuffle runs as VM steps, `O(N)` for each epoch, and counts against `--max-steps` like any Agentica code.
- `fit` retraces the loss on every step, so each step pays the trace again.
