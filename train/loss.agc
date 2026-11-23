## Losses: scalar functions written over `Tensor.*` alone, so they run eagerly
## and record a graph on `Trace` tensors that `Trace.value_and_grad`
## differentiates (`train/DESIGN.md`). Each answers a rank-0 tensor, the mean
## over every element.

import Tensor from "pkg/tensor/tensor"

## The mean of `(prediction - target)^2`.
def mse(prediction: Tensor.Tensor, target: Tensor.Tensor) -> Result[Tensor.Tensor]
  diff = try Tensor.sub(prediction, target)
  Tensor.mean(try Tensor.mul(diff, diff))
end

## Binary cross-entropy of `logits` against `targets` in `0..1`:
## `softplus(z) - t z`, with the stable
## `softplus(z) = relu(z) + log(1 + exp(-|z|))`. `|z|` is `2 relu(z) - z`,
## so that at `z = 0`, where `relu`'s derivative is taken as `0`, the
## gradient is still `sigmoid(0) - t`.
def binary_cross_entropy(logits: Tensor.Tensor, targets: Tensor.Tensor) -> Result[Tensor.Tensor]
  pos = try Tensor.relu(logits)
  abs = try Tensor.sub(try Tensor.add(pos, pos), logits)
  one = try Tensor.full(Tensor.shape(logits), Tensor.Values.Float(1.0), Tensor.dtype(logits), Tensor.backend(logits))
  soft = try Tensor.add(pos, try Tensor.log(try Tensor.add(one, try Tensor.exp(try Tensor.neg(abs)))))
  Tensor.mean(try Tensor.sub(soft, try Tensor.mul(logits, targets)))
end
