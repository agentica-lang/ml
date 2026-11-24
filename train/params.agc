## Training part of a model (`train/DESIGN.md`): the weights of a model by name, split into the ones `fit` trains and
## the frozen rest. `fit` takes the trainable list as its parameters, so the frozen tensors have no gradient and no
## optimizer state; the loss gets every weight back by name with `join`.

import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## `trainable` holds the tensors of `names`, in that order (sorted); `frozen` the others.
struct Split
  names: List[String]
  trainable: List[Tensor.Tensor]
  frozen: Map[String, Tensor.Tensor]
end

## Splits `weights` by name: the ones `trainable` says yes to train. An error when it selects none.
def split(weights: Map[String, Tensor.Tensor], trainable: Fn[String, Bool]) -> Result[Split]
  names = weights.keys.sort.select { |name: String| trainable.call(name) }
  return error("no weight is trainable: the selection matches none of the #{weights.size} names") if names.empty?
  frozen = weights
  for name in names
    frozen = frozen.delete(name)
  end

  ok(Split(names: names, trainable: names.map { |name: String| weights[name] }, frozen: frozen))
end

## The trainable weights by name, with `params` as their tensors: what `fit` got or answered.
def named(s: Split, params: List[Tensor.Tensor]) -> Result[Map[String, Tensor.Tensor]]
  if params.size != s.names.size
    return error("the split has #{s.names.size} trainable weights, got #{params.size} tensors")
  end

  out: Map[String, Tensor.Tensor] = {}
  for i in 0...params.size
    out = out.put(s.names[i], params[i])
  end

  ok(out)
end

## Every weight by name again: `params` for the trainable ones and the frozen ones as they are. Inside a loss, where
## `params` are traced, the frozen ones are constants of the trace (`Trace.constant`).
def join(s: Split, params: List[Tensor.Tensor]) -> Result[Map[String, Tensor.Tensor]]
  out = try named(s, params)
  traced = params[0].backend == Tensor.Backend.Trace
  for name in s.frozen.keys
    frozen = s.frozen[name]
    if traced
      frozen = Trace.constant(frozen)
    end

    out = out.put(name, frozen)
  end

  ok(out)
end

## `weights` with the tensors of `changed` in place of the ones of the same name, for the trainable part read from a
## file. A name `weights` lacks or a tensor of another shape is an error.
def update(
  weights: Map[String, Tensor.Tensor],
  changed: Map[String, Tensor.Tensor],
) -> Result[Map[String, Tensor.Tensor]]
  out = weights
  for name in changed.keys.sort
    old = try weights.get(name).to_result("the model has no weight #{name}")
    if Tensor.shape(changed[name]) != Tensor.shape(old)
      return error("#{name}: the model holds #{Tensor.shape(old)}, got #{Tensor.shape(changed[name])}")
    end

    out = out.put(name, changed[name])
  end

  ok(out)
end
