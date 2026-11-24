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
  plans: Map[String, Trace.Compiled]
end

def run(r: Run, start: Optim.State, params: List[Tensor.Tensor]) -> Result[Fit]
  state = start
  current = params
  history: List[Float] = []
  plans: Map[String, Trace.Compiled] = {}
  batches = Data.count(r.feed.loader)
  for e in 1..r.epochs
    feed = Data.epoch(r.feed, e)
    total = 0.0
    for i in 0...batches
      stepped = take_step(r.loss, feed, i, plans, state, current)
      if !stepped.ok?
        close(plans)
        return error("epoch #{e}, batch #{i}: #{stepped.message}")
      end

      done = stepped.unwrap!
      state = done.step.state
      current = done.step.params
      plans = done.plans
      total = total + done.loss
    end

    mean = total / batches.to_f
    history = history.push(mean)
    case r.on_epoch
      some(f) -> f.call(Epoch(epoch: e, loss: mean))
      none -> nil
    end
  end

  close(plans)
  ok(Fit(params: current, history: history))
end

## Ends the pins of the compiled steps; their graphs stay in the cache.
def close(plans: Map[String, Trace.Compiled]) -> Void
  for key in plans.keys
    Trace.close(plans[key])
  end
end

def take_step(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  feed: Data.Feed,
  i: Int,
  plans: Map[String, Trace.Compiled],
  state: Optim.State,
  params: List[Tensor.Tensor],
) -> Result[Stepped]
  batch = try Data.take(feed, i)
  done = step(loss, plans, state, params, batch)
  Data.release(batch)
  done
end

## One step, run from a function compiled once per batch shape: the parameters, the optimizer state, the numbers of the
## step that change (`Optim.scalars`) and the batch are its inputs, so nothing is traced or planned after the first
## batch of a shape. Only the new state, parameters and the loss outlive the step's native arena (`tensor/DESIGN.md` §8).
def step(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  plans: Map[String, Trace.Compiled],
  state: Optim.State,
  params: List[Tensor.Tensor],
  batch: Data.Batch,
) -> Result[Stepped]
  n = params.size
  k = state.first.size
  m = state.second.size
  key = "#{Tensor.shape(batch.inputs)} #{Tensor.shape(batch.targets)}"
  known = plans.get(key)
  fresh = known.none?
  held = plans
  plan = case known
    some(p) -> p
    none ->
      try Trace.compile(
        (params + state.first + state.second + try Optim.scalars(state, params[0])).map { |t| Trace.spec_of(t) } +
        [Trace.spec_of(batch.inputs), Trace.spec_of(batch.targets)],
        ->(xs: List[Tensor.Tensor]) { traced_step(loss, state, n, k, m, xs) },
      )
  end

  inputs = params + state.first + state.second + try Optim.scalars(state, params[0]) + [batch.inputs, batch.targets]
  stepped = Native.arena do ran(plan, inputs, state, params.size, held, fresh, key) end

  if !stepped.ok? and fresh
    Trace.close(plan)
  end

  stepped
end

def ran(
  plan: Trace.Compiled,
  inputs: List[Tensor.Tensor],
  state: Optim.State,
  n: Int,
  held: Map[String, Trace.Compiled],
  fresh: Bool,
  key: String,
) -> Result[Stepped]
  k = state.first.size
  values = try Trace.run(plan, inputs)
  moved = state.with(steps: state.steps + 1, first: values.drop(1 + n).take(k), second: values.drop(1 + n + k))
  kept = held
  if fresh
    kept = held.put(key, plan)
  end

  ok(Stepped(step: Optim.Step(state: moved, params: values.drop(1).take(n)), loss: try scalar(values[0]), plans: kept))
end

## The traced step over `xs`: the `n` parameters, the optimizer's `k` first and its second moments, its scalars, then the
## batch. It answers the loss, then the new parameters, first and second moments.
def traced_step(
  loss: Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]],
  state: Optim.State,
  n: Int,
  k: Int,
  m: Int,
  xs: List[Tensor.Tensor],
) -> Result[List[Tensor.Tensor]]
  ps = xs.take(n)
  extra = xs.size - n - k - m - 2
  value = try loss.call(ps, Data.Batch(inputs: xs[xs.size - 2], targets: xs[xs.size - 1]))
  grads = try Trace.gradients(value, ps)
  moments = state.with(first: xs.drop(n).take(k), second: xs.drop(n + k).take(m))
  stepped = try Optim.step(moments, ps, grads, xs.drop(n + k + m).take(extra))
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
