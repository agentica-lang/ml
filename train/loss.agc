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

## Softmax cross-entropy of `logits` (`[batch, classes]`) against the integer
## class `labels` (`[batch]`): the mean of `-log softmax(z)[label]`. The
## log-softmax subtracts the row maximum as a `stop_grad` constant, so it is
## stable and its gradient is `(softmax(z) - onehot) / batch`. A label outside
## `0...classes` matches no class and adds `0`: inside a trace the labels have
## no values to check.
def cross_entropy(logits: Tensor.Tensor, labels: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(logits)
  return error("cross_entropy needs [batch, classes] logits, got #{shape}") if shape.size != 2
  if Tensor.shape(labels) != [shape[0]]
    return error("cross_entropy needs [#{shape[0]}] labels, got #{Tensor.shape(labels)}")
  end

  top = try Tensor.stop_grad(try Tensor.max(logits, some([1]), true))
  shifted = try Tensor.sub(logits, try Tensor.broadcast_to(top, shape))
  total = try Tensor.sum(try Tensor.exp(shifted), some([1]), true)
  log_probs = try Tensor.sub(shifted, try Tensor.broadcast_to(try Tensor.log(total), shape))
  classes = try Tensor.arange(0, shape[1], 1, Tensor.dtype(labels), Tensor.backend(logits))
  hit = try Tensor.eq(
    try Tensor.broadcast_to(try Tensor.reshape(classes, [1, shape[1]]), shape),
    try Tensor.broadcast_to(try Tensor.reshape(labels, [shape[0], 1]), shape),
  )

  zeros = try Tensor.full(shape, Tensor.Values.Float(0.0), Tensor.dtype(logits), Tensor.backend(logits))
  picked = try Tensor.where(hit, log_probs, zeros)
  Tensor.neg(try Tensor.mean(try Tensor.sum(picked, some([1]))))
end
