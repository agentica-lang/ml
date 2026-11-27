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
- `Data.from_table(table, inputs, target, backend, target_dtype)` makes a dataset of the columns of a table read from a file, [below](#tables).

## Tables

`Table.read(path)` (`import Table from "pkg/train/table"`) reads a CSV file (the `csv` package) whose first row names the columns: every row must have as many cells as the header and a name may be used once. A byte order mark at the start is dropped, a line with nothing on it is skipped when there is more than one column, and a quoted cell may hold a comma, quotes and line breaks. The file is read whole, and a column is decoded when it is asked for. A column is numbers when every cell is a finite number, else text (so one empty cell makes a column text). `Table.floats(table, name)` and `Table.texts(table, name)` give a column; asking for the other kind is an error that names the column and the first cell that does not fit (`column gap has a non-number cell (row 3 is "")`). Rows are counted from 1: row 1 is the first one after the header. `table.names` are the columns in file order and `table.rows` the number of rows.

`Data.from_table(table, inputs, target, backend, target_dtype)` takes the numeric columns `inputs` as the features, `[rows, features]` row by row, and the numeric column `target` as the targets `[rows]`: `F32` by default, or with `target_dtype: Tensor.Dtype.I32` whole numbers in the I32 range, the class ids that `Loss.cross_entropy` takes (a target that is not whole is an error that names the column and the row). A column that is text is an error that names it, so the text columns of a file are left out of `inputs`, and a table with no rows is an error. The result is a `Dataset` for a `Loader`.

```agentica
import Data from "pkg/train/data"
import Table from "pkg/train/table"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  text = "x1,x2,label\n0.5,1.0,0\n-1.25,0.25,1\n2.0,-0.5,1\n3.5,2.0,0\n0.0,4.0,2\n1.0,8.0,2\n"
  File.write_text("scores.csv", text).unwrap!
  table = Table.read("scores.csv").unwrap!
  puts(table.names)
  dataset = Data.from_table(table, ["x1", "x2"], "label", Tensor.Backend.Reference, Tensor.Dtype.I32).unwrap!
  loader = Data.Loader(dataset: dataset, batch_size: 4, shuffle: false, prefetch: 0)
  puts(Data.count(loader))
  batch = Data.batch(loader, Data.order(loader, 1), 0).unwrap!
  puts(Tensor.shape(batch.inputs))
  puts(Tensor.to_list(batch.targets).unwrap!)
end
```

```output
["x1", "x2", "label"]
2
[4, 2]
[Values.Int(0), Values.Int(1), Values.Int(1), Values.Int(0)]
```

Parquet files are not read yet.

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
- `Data.batch(loader, rows, i)` gives batch `i` of an epoch whose rows are `rows`, from `Data.order`, without a feed. `take` uses it when there is no prefetch.
- `Data.release(batch)` frees the batch's native tensors. Call it when the step is done.
- `Data.free(tensor)` frees the native memory of one tensor now, on libtorch and ggml; the reference has none. Nothing may read the tensor afterwards. `Data.freed?(batch)` says whether a tensor of a batch was freed.
- `Data.stop(feed)` ends the prefetch task. Call it when you are done, including after an error.

For batches that a function makes (image files to decode, text to tokenize), without a loader, `Train.fit_batches(..., prefetch: K)` does the following for you. By hand:

- `Data.ahead(batch, count, epochs, prefetch)` starts a task that calls `batch(epoch, i)` for every batch of every epoch, in order, and answers the stream. `prefetch` is at least 1.
- `Data.next(stream)` gives the next batch, in order. A batch that failed is an `error`, as is one whose tensors were freed.
- `Data.finish(stream)` ends the task and frees the native tensors it made. Call it when you are done, including after an error: the task lives until then.

## Prefetch

- With `prefetch = K > 0`, a task builds the batches of every epoch in order. It sends them through a channel of capacity `K`, so up to `K + 1` batches are built ahead while the caller steps (the `K` in the channel and the one the task holds).
- Under prefetch, `take` gives the next batch of the stream. The `i` argument is not read, so take the batches in order.
- A batch that fails to build reaches the caller as an error in its place. A crash of the task also reaches the caller as an error.
- Native tensors made by the task belong to it. They stay usable until `stop`.
- `fit` on `Ggml` and `Xla` does not use `prefetch`: a batch there is its row ids, and the gather runs inside the step. `Data.feed` does, and `fit_batches` takes its own `prefetch` for a batch function.
- Prefetch hides the time it takes to build a batch, and nothing more. The step's own compute is not overlapped.

## Limits

- On `Ggml`, the source of a batch's `gather` holds at most 2^24 rows, so a dataset of more rows must be on `Reference` or `Torch`.
- A batch is one compiled graph per batch shape on `Ggml`, so a last partial batch costs a second graph.
- A dataset must have fewer than 2 147 483 646 rows.
- The shuffle runs as VM steps, `O(N)` for each epoch, and counts against `--max-steps` like any Agentica code.
- `fit` traces the loss once for each batch shape, not on every step. On `Ggml` and `Xla` the compiled step takes the row ids of the batch as its input and gathers the dataset's tensors, which it holds as constants; elsewhere the batch's two tensors are the inputs.
