## A training loop as a value, a thin layer over `Train.fit` and `Train.fit_batches` (`train/DESIGN.md`): build it with
## `trainer`, then `run` it, as a pipe.
##
## ```
## fit = (
##   Loop.trainer(loss, params, Optim.adamw(lr: 0.00002))
##   |> Loop.run(data, epochs: 3)
## )
## ```
##
## `Train.fit` stays the engine: a run is one call of it, with the same arguments, and answers its `Train.Fit`.

import Data from "data"
import Optim from "optim"
import Tensor from "pkg/tensor/tensor"
import Train from "train"

## What a run trains: the loss of a parameter list and a batch (inside it the parameters and the batch are `Trace`
## tensors), the parameters it starts from, and the optimizer.
struct Trainer
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]]
  params: List[Tensor.Tensor]
  optimizer: Optim.Optimizer
end

def trainer(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  optimizer: Optim.Optimizer,
) -> Trainer
  Trainer(loss: loss, params: params, optimizer: optimizer)
end

## `Train.fit` of the trainer over the batches of `data`, for `epochs` passes.
def run(t: Trainer, data: Data.Loader, epochs: Int = 1) -> Result[Train.Fit]
  Train.fit(t.loss, t.params, data, t.optimizer, epochs)
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
  Train.fit_batches(t.loss, t.params, batch, count, t.optimizer, epochs, none, none, prefetch)
end
