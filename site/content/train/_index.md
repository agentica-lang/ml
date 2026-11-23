---
title: train
description: Losses, SGD and Adam, and Train.fit, a training loop over the gradients of tensor, with a seeded data loader.
weight: 3
---

`train` trains a model. A model is a loss function of a parameter list and a batch. `Train.fit` traces that function, and an optimizer updates the parameters from its gradients. Each step is one traced evaluation: `fit` traces the loss, its gradients and the optimizer update, and runs them together, so the value, the gradients and the new parameters come from one run of the graph.

The batches come from a data loader, which has [its own page](data.md).

## Example

Linear regression on eight rows, with SGD and the loader's default shuffle. The loss is the mean squared error of `x w`.

```agentica
import Data from "pkg/train/data"
import Loss from "pkg/train/loss"
import Optim from "pkg/train/optim"
import Tensor from "pkg/tensor/tensor"
import Train from "pkg/train/train"

def linear(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  rows = Tensor.shape(b.inputs)[0]
  Loss.mse(try Tensor.reshape(try Tensor.matmul(b.inputs, ps[0]), [rows]), b.targets)
end

def main() -> Void
  ref = Tensor.Backend.Reference
  xs = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 3.0, 2.0, 2.0]
  ys = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  inputs = Tensor.from_list(xs.map { |x: Float| Tensor.Values.Float(x) }, [8, 2], Tensor.Dtype.F32, ref).unwrap!
  targets = Tensor.from_list(ys.map { |y: Float| Tensor.Values.Float(y) }, [8], Tensor.Dtype.F32, ref).unwrap!
  dataset = Data.dataset(inputs, targets).unwrap!
  loader = Data.Loader(dataset: dataset, batch_size: 4, seed: 1)
  w = Tensor.zeros([2, 1], Tensor.Dtype.F32, ref).unwrap!
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.05))
  fit = Train.fit(&linear, [w], loader, sgd, 30).unwrap!
  puts(fit.history.size)
  puts(fit.history[0] > fit.history[29])
  puts(Tensor.to_list(fit.params[0]).unwrap!)
end
```

```output
30
true
[Values.Float(1.3264669610609146), Values.Float(1.980815866879087)]
```

## Losses

Each loss takes a prediction and a target and gives a rank-0 tensor, the mean over every element.

- `Loss.mse(prediction, target)` is the mean of `(prediction - target)^2`.
- `Loss.binary_cross_entropy(logits, targets)` takes logits and targets in `0...1`. It is stable for any logit.
- `Loss.cross_entropy(logits, labels)` takes `[batch, classes]` logits and `[batch]` integer labels. A label outside `0...classes` is an error in an eager call.

## Optimizers

`Optim.Optimizer` is `Sgd(Optim.Sgd(lr:, momentum: 0.0))` or `Adam(Optim.Adam(lr:, beta1:, beta2:, eps:))`. Adam's defaults are `lr` 0.001, `beta1` 0.9, `beta2` 0.999 and `eps` 1e-8.

The optimizers are pure. `Optim.init(optimizer, params)` makes the state, and `Optim.step(state, params, grads)` returns a new state and new parameters. Nothing is updated in place. `fit` does this for you.

## Train.fit

`Train.fit(loss, params, data, optimizer, epochs, on_epoch)` runs `epochs` passes over the batches of `data`, one optimizer step per batch. It returns a `Result` of `Fit`, with `params` (the trained list) and `history` (the mean loss of each epoch).

- `loss` has the type `Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]]`. It gets the parameters and a batch, and it returns a float scalar.
- Inside `loss`, the batch `inputs` and `targets` are traced constants, so the loss is written once, as it would be eagerly.
- `on_epoch` is optional. It gets an `Epoch` with `epoch` (from 1) and `loss`. `fit` prints nothing.
- A failed step stops `fit` with an error that names the epoch and the batch, for example `epoch 1, batch 0: ...`.

Parameters are a plain `List[Tensor.Tensor]`, the list that `Trace` takes. The loss rebuilds the layer structs from that list, for example `Nn.Dense(w: ps[0], b: some(ps[1]))`.

## Limits

- Every step retraces the loss, so the trace cost is paid on each step. On the reference backend, one epoch of the 2-8-1 MLP of the XOR test, with four rows, costs about 75 000 VM steps.
- The loss must return a float scalar. An op with no backward rule, listed on the [tensor page](../tensor/_index.md), makes `fit` fail.
- On `Ggml`, the step is one compiled graph with at most 256 outputs: the loss, the new parameters and the optimizer's moments. That allows about 85 parameters with Adam, 127 with momentum SGD and 255 with plain SGD.
- The losses and the optimizers use only ops that ggml lowers, so a batch and its step run on `Ggml` when the dataset is on `Ggml`.
