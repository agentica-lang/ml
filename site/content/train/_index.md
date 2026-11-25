---
title: train
description: Losses, SGD and Adam, and Train.fit, a training loop over the gradients of tensor, with a seeded data loader.
weight: 3
---

`train` fits a model to data. You give it a loss function, starting weights, a data loader and an optimizer, and `Train.fit` runs the training epochs for you. An epoch is one pass over all the batches.

The batches come from a data loader, which has [its own page](data.md). `fit` can report its loss, learning rate and step time to [metrics](metrics.md).

## Example

Linear regression on eight rows, with SGD (stochastic gradient descent, a simple optimizer) and the loader's default shuffle. The loss is the mean squared error of `x w`:

```agentica
import Data from "pkg/train/data"
import Loss from "pkg/train/loss"
import Optim from "pkg/train/optim"
import Tensor from "pkg/tensor/tensor"
import Train from "pkg/train/train"

def linear(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  rows = Tensor.shape(b.inputs)[0]
  pred = try Tensor.matmul(b.inputs, ps[0])
  Loss.mse(try Tensor.reshape(pred, [rows]), b.targets)
end

def main() -> Void
  xs = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0,
    1.0, 2.0, 3.0, 1.0, 1.0, 3.0, 2.0, 2.0]
  ys = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  inputs = Tensor.floats(xs, [8, 2]).unwrap!
  targets = Tensor.floats(ys, [8]).unwrap!
  dataset = Data.dataset(inputs, targets).unwrap!
  loader = Data.Loader(dataset: dataset, batch_size: 4, seed: 1)
  w = Tensor.zeros([2, 1]).unwrap!
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

Each loss takes a prediction and a target. It gives a rank-0 tensor (a single number), the mean over every element.

- `Loss.mse(prediction, target)` is the mean squared error: the mean of `(prediction - target)^2`.
- `Loss.binary_cross_entropy(logits, targets)` takes logits (raw scores) and targets in `0...1`. It is stable for any logit.
- `Loss.cross_entropy(logits, labels)` takes `[batch, classes]` logits and `[batch]` integer labels. A label outside `0...classes` is an error in an eager call (a direct call, outside a trace).

## Optimizers

`Optim.Optimizer` is `Sgd(Optim.Sgd(lr:, momentum: 0.0))`, `Adam(Optim.Adam(lr:, beta1:, beta2:, eps:))` or `AdamW(Optim.AdamW(...))`. Adam's defaults are `lr` 0.001, `beta1` 0.9, `beta2` 0.999 and `eps` 1e-8.

`AdamW(Optim.AdamW(lr:, weight_decay: 0.01, decay: []))` is Adam with decoupled weight decay, as PyTorch's `AdamW`: each step first shrinks a parameter by `lr * weight_decay` of itself, then takes Adam's step. It has Adam's fields and defaults. `decay` has one flag per parameter and says which ones decay; empty, all of them do. A fine-tune leaves the biases and the layer-norm weights out, which in PyTorch is two parameter groups.

All three take a `schedule:` that moves the learning rate with the step count `t` (from 0). `lr` is the peak, and the rate of a step is `lr` times a factor:

- `Optim.Schedule.Constant` (the default) is a factor of 1.
- `Optim.Schedule.WarmupLinear(warmup, total)` rises from 0 to 1 over `warmup` steps, then falls in a straight line to 0 at step `total`.
- `Optim.Schedule.WarmupCosine(warmup, total)` rises the same way and falls along a half cosine.

They are the schedules of Hugging Face's `Trainer` up to `total`; after it the factor stays 0, where Hugging Face's cosine rises again. `fit` starts every run at step 0, so a resumed run starts its warmup again. `Optim.init` refuses a `warmup` below 0 or above `total`, and a `total` below 1. `Optim.learning_rate(state)` is the rate of the next step, and it is the value that `fit` logs as `train/lr`.

```agentica
adam = Optim.Optimizer.Adam(Optim.Adam(lr: 0.0001, schedule: Optim.Schedule.WarmupLinear(100, 1000)))
```

The optimizers are pure. Nothing is updated in place. `Optim.init(optimizer, params)` makes the state, and `Optim.step(state, params, grads, scalars = [])` returns a new state and new parameters. `fit` does this for you.

`Optim.step_scalars(state)` gives the numbers that the next step depends on: SGD's learning rate, or Adam's `lr / (1 - beta1^t)` and `1 / (1 - beta2^t)` (its learning rate with the bias correction); AdamW adds `lr * weight_decay`. `fit` gives them to the step as inputs, so a learning rate that changes at every step is not a new graph on ggml. `Train.stablehlo(loss, params, batch, optimizer)` writes the step of `fit` as one StableHLO program for every step, with those numbers as inputs after the moments (`Trace.stablehlo` in the tensor page).

`Optim.step` answers an `Optim.Step`. It is an error unless there is one gradient per parameter. The `State` has `optimizer`, `steps`, and `first` and `second`: the SGD velocity or the Adam moments, one tensor per parameter. A list the optimizer does not use is empty.

## Train.fit

`Train.fit(loss, params, data, optimizer, epochs, on_epoch, metrics)` runs `epochs` passes over the batches of `data`, with one optimizer step per batch. It returns a `Result` of `Fit`, with `params` (the trained list) and `history` (the mean loss of each epoch).

- `loss` has the type `Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]]`. It gets the parameters and a batch, and it returns a float scalar.
- Inside `loss`, the batch `inputs` and `targets` are traced inputs, so the loss is written once, as plain code. `fit` traces and compiles the step once for each batch shape (a smaller last batch is a second one) and runs it on every batch of that shape.
- `on_epoch` is optional. It gets an `Epoch` with `epoch` (from 1) and `loss`. `fit` prints nothing.
- `metrics` is optional. It is an open [`Metrics.Run`](metrics.md) that gets `train/loss`, `train/lr` and `train/step_ms` for every step. A value it refuses, such as a loss that is not finite, stops `fit` too.
- A failed step stops `fit` with an error that names the epoch and the batch, for example `epoch 1, batch 0: ...`.

Parameters are a plain `List[Tensor.Tensor]`, the list that `Trace` takes. The loss rebuilds the layer structs from that list, for example `Nn.Dense(w: ps[0], b: some(ps[1]))`.

## Training part of a model

`Params` (`import Params from "pkg/train/params"`) trains some weights of a model and keeps the rest frozen. The weights are a `Map[String, Tensor.Tensor]` by name, such as `Qwen3.weights(m)` gives.

```agentica
weights = Qwen3.weights(m)
split = try Params.split(weights, ->(name: String) { name.start_with?("model.layers.27.") })
fit = try Train.fit(
  ->(ps: List[Tensor.Tensor], b: Data.Batch) { loss(try Qwen3.with_weights(m, try Params.join(split, ps)), b) },
  split.trainable,
  loader,
  Optim.Optimizer.Adam(Optim.Adam(lr: 0.0001)),
  3,
)

try SafetensorsWrite.write("last_block.safetensors", try Params.named(split, fit.params))
```

With `AdamW`, the `decay` flags follow `split.names`, the order of `split.trainable`. A fine-tune decays every weight but the biases and the layer-norm weights, which is the rule of Hugging Face's `Trainer`:

```agentica
split = try Params.split(DistilBert.weights(m), ->(name: String) { true })
norm = ->(name: String) { name.include?("LayerNorm") or name.include?("layer_norm") }
decay = split.names.map { |name: String| !(name.end_with?(".bias") or norm.call(name)) }
optimizer = Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.00002, weight_decay: 0.01, decay: decay))
```

A layer norm is named by its model: `LayerNorm` in BERT, and `embeddings.LayerNorm`, `sa_layer_norm` and `output_layer_norm` in DistilBERT.

- `Params.split(weights, trainable)` answers a `Split`: `names` (the names `trainable` says yes to, sorted), `trainable` (their tensors, the list `fit` takes) and `frozen` (the other weights by name). A selection that matches no name is an error.
- `Params.join(split, ps)` gives every weight by name again, with `ps` for the trainable ones. Inside the loss `ps` are traced and the frozen weights enter the trace as constants, so they get no gradient, and the optimizer keeps no state for them.
- `Params.named(split, ps)` gives the trainable weights alone by name: what to save. `SafetensorsWrite.write` of the hub package writes them.
- `Params.update(weights, changed)` puts the tensors of `changed` in place of the ones of the same name. To load a saved part: `Params.update(Qwen3.weights(m), try Hub.tensors(try Hub.open(path), backend))`, then `Qwen3.with_weights`. A name the model lacks or another shape is an error.

## Limits

- The loss is traced once for each batch shape, not on every step. On the reference backend, which evaluates the traced graph node by node on every run, one epoch of the 2-8-1 MLP of the XOR test, with four rows, costs about 75 000 VM steps.
- The loss must return a float scalar. An op with no backward rule, listed on the [tensor page](../tensor/_index.md), makes `fit` fail.
- On `Ggml`, the step is one compiled graph with at most 512 outputs: the loss, the new parameters and the optimizer's moments. That allows 170 parameters with Adam or AdamW, 255 with momentum SGD and 511 with plain SGD.
- The losses and the optimizers use only ops that ggml supports, so a batch and its step run on `Ggml` when the dataset is on `Ggml`.

## How it works

A model here is a loss function of a parameter list and a batch. `Train.fit` traces that function, and an optimizer updates the parameters from its gradients.

Each step is one traced evaluation. `fit` traces the loss, its gradients and the optimizer update, and runs them together. So the value, the gradients and the new parameters come from one run of the graph.
