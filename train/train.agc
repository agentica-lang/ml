## The `train/` package: losses (`loss.agc`), optimizers (`optim.agc`) and
## `fit`, a minimal training loop over `Trace.value_and_grad`
## (`train/DESIGN.md`).
##
## A model is a loss function of a parameter list and a batch. `fit` traces it
## on every step with the parameters as inputs and the batch as constants,
## evaluates the loss and its gradients on the parameters' backend, and hands
## them to the optimizer. The batches come from a `Data.Loader` (`data.agc`).

import Data from "data"
import Optim from "optim"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

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

## Trains `params` for `epochs` passes over the batches of `data`, one
## optimizer step per batch. `loss(params, batch)` must answer a float scalar;
## inside it the parameters and the batch are `Trace` tensors. The loader's
## prefetch task, if any, ends before `fit` returns, on every path.
def fit(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  data: Data.Loader,
  optimizer: Optim.Optimizer,
  epochs: Int,
  on_epoch: Optional[Fn[Epoch, Void]] = none,
) -> Result[Fit]
  state = try Optim.init(optimizer, params)
  feed = try Data.feed(data, epochs)
  result = run(Run(loss: loss, feed: feed, epochs: epochs, on_epoch: on_epoch), state, params)
  Data.stop(feed)
  result
end

private

struct Run
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]]
  feed: Data.Feed
  epochs: Int
  on_epoch: Optional[Fn[Epoch, Void]]
end

struct Stepped
  step: Optim.Step
  loss: Float
end

def run(r: Run, start: Optim.State, params: List[Tensor.Tensor]) -> Result[Fit]
  state = start
  current = params
  history: List[Float] = []
  batches = Data.count(r.feed.loader)
  for e in 1..r.epochs
    feed = Data.epoch(r.feed, e)
    total = 0.0
    for i in 0...batches
      stepped = take_step(r.loss, feed, i, state, current)
      return error("epoch #{e}, batch #{i}: #{stepped.message}") if !stepped.ok?
      done = stepped.unwrap!
      state = done.step.state
      current = done.step.params
      total = total + done.loss
    end

    mean = total / batches.to_f
    history = history.push(mean)
    case r.on_epoch
      some(f) -> f.call(Epoch(epoch: e, loss: mean))
      none -> nil
    end
  end

  ok(Fit(params: current, history: history))
end

def take_step(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  feed: Data.Feed,
  i: Int,
  state: Optim.State,
  params: List[Tensor.Tensor],
) -> Result[Stepped]
  batch = try Data.take(feed, i)
  done = step(loss, state, params, batch)
  Data.release(batch)
  done
end

## One step in its own native arena: only the new state, parameters and the
## loss outlive it (`tensor/DESIGN.md` §8).
def step(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  state: Optim.State,
  params: List[Tensor.Tensor],
  batch: Data.Batch,
) -> Result[Stepped]
  Native.arena do
    traced = ->(ps: List[Tensor.Tensor]) {
      loss.call(ps, Data.Batch(inputs: Trace.constant(batch.inputs), targets: Trace.constant(batch.targets)))
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
