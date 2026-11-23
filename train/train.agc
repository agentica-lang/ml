## The `train/` package: losses (`loss.agc`), optimizers (`optim.agc`) and
## `fit`, a minimal training loop over `Trace.value_and_grad`
## (`train/DESIGN.md`).
##
## A model is a loss function of a parameter list and a batch. `fit` traces it
## on every step with the parameters as inputs and the batch as constants,
## evaluates the loss and its gradients on the parameters' backend, and hands
## them to the optimizer.

import Optim from "optim"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

struct Batch
  inputs: Tensor.Tensor
  targets: Tensor.Tensor
end

## What `on_epoch` gets after each epoch: its number from 1 and the mean loss
## of its batches.
struct Epoch
  epoch: Int
  loss: Float
end

## The trained parameters and the mean loss of every epoch, in order.
struct Fit
  params: List[Tensor.Tensor]
  history: List[Float]
end

struct Stepped
  step: Optim.Step
  loss: Float
end

## Trains `params` for `epochs` passes over `batches`, one optimizer step per
## batch. `loss(params, batch)` must answer a float scalar; inside it the
## parameters and the batch are `Trace` tensors.
def fit(
  loss: Fn[List[Tensor.Tensor], Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  batches: List[Batch],
  optimizer: Optim.Optimizer,
  epochs: Int,
  on_epoch: Optional[Fn[Epoch, Void]] = none,
) -> Result[Fit]
  return error("fit needs at least one batch") if batches.empty?
  state = try Optim.init(optimizer, params)
  current = params
  history: List[Float] = []
  for e in 1..epochs
    total = 0.0
    for b in batches
      done = try step(loss, state, current, b)
      state = done.step.state
      current = done.step.params
      total = total + done.loss
    end

    mean = total / batches.size.to_f
    history = history.push(mean)
    case on_epoch
      some(f) -> f.call(Epoch(epoch: e, loss: mean))
      none -> nil
    end
  end

  ok(Fit(params: current, history: history))
end

private

## One step in its own native arena: only the new state, parameters and the
## loss outlive it (`tensor/DESIGN.md` §8).
def step(
  loss: Fn[List[Tensor.Tensor], Batch, Result[Tensor.Tensor]],
  state: Optim.State,
  params: List[Tensor.Tensor],
  batch: Batch,
) -> Result[Stepped]
  Native.arena do
    traced = ->(ps: List[Tensor.Tensor]) {
      loss.call(ps, Batch(inputs: Trace.constant(batch.inputs), targets: Trace.constant(batch.targets)))
    }

    both = try Trace.value_and_grad(params, traced)
    value = try scalar(both.value)
    ok(Stepped(step: try Optim.step(state, params, both.grads), loss: value))
  end
end

def scalar(t: Tensor.Tensor) -> Result[Float]
  values = try Tensor.to_list(t)
  return error("a loss must be a scalar, got #{values.size} values") if values.size != 1
  case values[0]
    Tensor.Values.Float(f) -> ok(f)
    _ -> error("a loss must be a float scalar")
  end
end
