## The `train/` package: losses (`loss.agc`), optimizers (`optim.agc`) and
## `fit`, a minimal training loop over `tensor/`'s tracing
## (`train/DESIGN.md`).
##
## A model is a loss function of a parameter list and a batch. `fit` traces
## one training step on every batch, with the parameters and the optimizer
## state as inputs and the batch as constants: the loss, its gradients and the
## optimizer's update, run as one evaluation on the parameters' backend (one
## compiled graph on `Ggml`). The batches come from a `Data.Loader`
## (`data.agc`).

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

## The training step of `fit` as one StableHLO program (`Trace.stablehlo`): its arguments are the parameters, the
## optimizer's first and second moments, then the two `Optim.step_scalars` of Adam as rank-0 tensors, so one program is
## every step. The batch is lifted as constants, which are the module's `constants`. It answers the loss, then the new
## parameters, first moments and second moments, as a step of `fit` does.
def stablehlo(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  params: List[Tensor.Tensor],
  batch: Data.Batch,
  optimizer: Optim.Optimizer,
) -> Result[Trace.Module]
  return error("a step needs at least one parameter") if params.empty?
  state = try Optim.init(optimizer, params)
  n = params.size
  k = state.first.size
  scalar = Trace.Spec(shape: [], dtype: Tensor.dtype(params[0]), backend: Tensor.backend(params[0]))
  scalars = Optim.step_scalars(state).map { |_| scalar }
  specs = (params + state.first + state.second).map { |t| Trace.spec_of(t) } + scalars
  Trace.stablehlo(specs, ->(xs: List[Tensor.Tensor]) { traced_step(loss, state, batch, n, k, xs) })
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
    n = params.size
    k = state.first.size
    outs = try Trace.eval_many(
      params + state.first + state.second,
      ->(xs: List[Tensor.Tensor]) { traced_step(loss, state, batch, n, k, xs) },
    )

    moved = state.with(steps: state.steps + 1, first: outs.drop(1 + n).take(k), second: outs.drop(1 + n + k))
    ok(Stepped(step: Optim.Step(state: moved, params: outs.drop(1).take(n)), loss: try scalar(outs[0])))
  end
end

## The traced step over `xs`, the `n` parameters then the optimizer's `k`
## first and its second moments, then the scalars of its step when they are
## inputs (`stablehlo`): the loss, then the new parameters, first and second
## moments.
def traced_step(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  state: Optim.State,
  batch: Data.Batch,
  n: Int,
  k: Int,
  xs: List[Tensor.Tensor],
) -> Result[List[Tensor.Tensor]]
  ps = xs.take(n)
  value = try loss.call(ps, Data.Batch(inputs: Trace.constant(batch.inputs), targets: Trace.constant(batch.targets)))
  grads = try Trace.gradients(value, ps)
  moments = n + k + state.second.size
  held = state.with(first: xs.drop(n).take(k), second: xs.drop(n + k).take(state.second.size))
  stepped = try Optim.step(held, ps, grads, xs.drop(moments))
  ok([value] + stepped.params + stepped.state.first + stepped.state.second)
end

def scalar(t: Tensor.Tensor) -> Result[Float]
  values = try Tensor.to_list(t)
  return error("a loss must be a scalar, got #{values.size} values") if values.size != 1
  case values[0]
    Tensor.Values.Float(f) -> ok(f)
    _ -> error("a loss must be a float scalar")
  end
end
