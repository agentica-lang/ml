---
title: train
description: Losses, SGD and Adam, and Train.fit, a training loop over the gradients of tensor, with a seeded data loader.
weight: 3
---

`train` fits a model to data. You give it a loss function, starting weights, a data loader and an optimizer, and `Train.fit` runs the training epochs for you. An epoch is one pass over all the batches.

The batches come from a data loader, which has [its own page](data.md). `fit` can report its loss, learning rate and step time to [metrics](metrics.md).

A whole fine-tune of a sentence-embedding model on a Hugging Face dataset, on any backend, is in [fine-tune all-MiniLM-L6-v2](finetune-minilm.md).

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

They are the schedules of Hugging Face's `Trainer` up to `total`; after it the factor stays 0, where Hugging Face's cosine rises again. `fit` starts a run at step 0, or at the step of the `start` state it is given (below), so a resumed run goes on along its schedule. `Optim.init` refuses a `warmup` below 0 or above `total`, and a `total` below 1. `Optim.learning_rate(state)` is the rate of the next step, and it is the value that `fit` logs as `train/lr`.

```agentica
adam = Optim.Optimizer.Adam(Optim.Adam(lr: 0.0001, schedule: Optim.Schedule.WarmupLinear(100, 1000)))
```

The optimizers are pure: `Optim.step(state, params, grads, scalars = [])` returns a new state and new parameters, and changes none of its arguments. `Optim.init(optimizer, params)` makes the state. `fit` does this for you, and compiles the step so that the moments are updated where they lie: the new state's moments are the old ones' tensors, which `fit` owns. The same holds for a step you compile with `Trace.compile` (the tensors of the moments you pass are consumed).

`Optim.save(state, path)` writes the step count and the moments of a state to a safetensors file (the tensors `first.I` and `second.I` of parameter `I`, the metadata `steps` and `optimizer`), and `Optim.load(optimizer, params, path)` reads them back on the backend of each parameter: the knobs and the schedule are not in the file, the code gives them again (so a resume may change the learning rate or the schedule: the step count is kept and the new rate is read at it), and so is an AdamW `decay` list. The moments take the dtype and the backend of their parameter. An error names a file of another optimizer, a tensor that is no moment, a missing moment, a moment of another shape or dtype, a missing step count, or a schedule or decay list that `Optim.init` would refuse. `Train.fit(..., start: state)` (and `fit_batches`) begins from that state, so the schedule goes on from its step and the moments are not zero. The run takes the moments of `start` and frees them, on every path, so do not use or free `start` after `fit`. Inside `on_epoch` and `on_step` the `state` field gives the state of the run to save; it is the run's own, valid only inside the hook, so save it before the hook returns.

`Optim.step_inputs(state)` gives the numbers that the next step depends on, one list per input of the step: SGD's learning rate, or for Adam the 7 numbers of `Tensor.adam_step` (the learning rate with its schedule, `beta1`, `beta2`, `eps`, the weight decay, and the two bias corrections); AdamW has two such lists, the second without decay for the parameters that do not decay. `fit` gives them to the step as inputs, so a learning rate that changes at every step is not a new graph on ggml, and on ggml the Adam update of each parameter is one `ggml_opt_step_adamw` kernel. `Train.stablehlo(loss, params, batch, optimizer)` writes the step of `fit` as one StableHLO program for every step, with those inputs after the moments (`Trace.stablehlo` in the tensor page).

`Optim.step` answers an `Optim.Step`. It is an error unless there is one gradient per parameter. The `State` has `optimizer`, `steps`, and `first` and `second`: the SGD velocity or the Adam moments, one tensor per parameter. A list the optimizer does not use is empty.

`Optim.sgd(lr, momentum: 0.0, schedule:)`, `Optim.adam(lr:, schedule:)` and `Optim.adamw(lr:, weight_decay:, decay:, schedule:)` are the same optimizers with their usual knobs, as one call: `Optim.adamw(lr: 0.00002)` is `Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.00002))`. The betas and `eps` keep the defaults of their struct; build the struct to set them.

## Loop

`Loop` builds a training run as a value and runs it, as a pipe over `Train.fit`: `Loop.trainer(loss, params, optimizer)` makes a `Loop.Trainer`, and `Loop.run(trainer, data, epochs: 1)` is `Train.fit` of it. It answers the same `Fit`.

```agentica
fit = try (
  Loop.trainer(&loss, params, Optim.adamw(lr: 0.0001))
  |> Loop.run(data, epochs: 3)
)
```

`Loop.run_batches(trainer, batch, count, epochs: 1, prefetch: 0)` is `Train.fit_batches`: `batch(epoch, i)` makes batch `i` of `count`.

`Loop.metric(trainer, metric)` adds a number that the loop measures at the end of every epoch, on the batches of its own data (a validation set, say), with the parameters of that epoch. A `Loop.Metric` has a `name`, the `count` and the function `batch(i)` of its batches, `observe(params, batch)`, which answers one row of numbers for each example, and `reduce`, which turns all the rows into the value: `&Loop.mean` is the mean of the first number of each row, and `&Loop.spearman` Spearman's correlation of the first two. The values are in `fit.metrics[name]`, one per epoch; a metric that fails ends the run, at the end of its epoch, with an error that names it. A run stopped from a step handler has no values for the epoch it cut short, so `fit.metrics[name]` has one value per epoch that ended, possibly fewer than `fit.history`. `observe` is forward only: a `Trace.eval` of the model, with no gradient.

```agentica
fit = try (
  Loop.trainer(&loss, params, Optim.adamw(lr: 0.0001))
  |> Loop.metric(Loop.Metric(name: "mse", count: 4, batch: ->(i) { held_out(i) }, observe: &squared_errors, reduce: &Loop.mean))
  |> Loop.run(data, epochs: 3)
)
```

`Loop.handle(trainer, event, f, every: 1)` adds a function that the loop calls at an event, in the order the handlers were added and in the thread of the run: `Loop.Event.Started` (before the first step), `Loop.Event.StepCompleted` (after each step) or `Loop.Event.EpochCompleted` (after each epoch, once its metrics are measured). `every: n` calls it on every `n`th step or epoch. It gets a `Loop.Info` with the `epoch` (from 1), the `batch` and the `step` count of the run (at `EpochCompleted` the steps so far, with `batch` 0), the `loss` (of the step, or the mean of the epoch), the `params` (valid only inside the handler: the next step frees them) and, at `EpochCompleted`, the `metrics` of that epoch by name. A handler answers a `Result[Void]`: an `error` ends the run, and `return Loop.stop(info)` ends it without one, after that step or epoch (after the first step, from `Started`): `run` answers the `Fit` as usual (a stop from a step handler skips the metrics of the epoch it cut short; one from `EpochCompleted` keeps its epoch's; an `error` wins over a stop), with the parameters so far and a history whose last epoch is the mean over the steps it ran. A step handler's error is named by the epoch and the batch (`epoch 2, batch 0: ...`); an epoch handler's, like a metric's, by the epoch (`epoch 1: ...`), and it stops the run at the end of that epoch; a `Started` handler's is named `before the first step`.

```agentica
fit = try (
  Loop.trainer(&loss, params, Optim.adamw(lr: 0.0001))
  |> Loop.handle(Loop.Event.StepCompleted, ->(_) { ok() }, every: 50)
  |> Loop.handle(
    Loop.Event.StepCompleted,
    ->(info) {
      puts("step #{info.step} loss #{info.loss}")
      return Loop.stop(info) if info.step == 1000

      ok()
    },
    every: 50,
  )
  |> Loop.run(data, epochs: 3)
)
```

`Loop.checkpoint(trainer, dir, every: 100, keep: 3)` is a handler that is ready made: every `every` steps (or every `every` epochs with `event: Loop.Event.EpochCompleted`) it writes a checkpoint to the new folder `dir/step-NNNNNNNNNN` (the step, to ten digits) and keeps the last `keep`. A checkpoint holds `model.safetensors` (the weights as `param.I`), `optim.safetensors` (`Optim.save`) and `checkpoint.json` (the event, the epoch, the batch, the step, the seed of the loader and the loss). It is written in `dir/.tmp-step-NNNNNNNNNN` and renamed whole, so a run killed or failing while it writes leaves the earlier checkpoints as they were: a temporary folder is not a checkpoint. There is no `fsync`: a checkpoint survives a kill of the process, not a power loss.

One `dir` serves one run. Before its first step a run checks `keep`, makes `dir` and removes the folders `.tmp-step-*` that a killed run left; no other folder of `dir` is touched, and a folder that is not `step-` and ten digits is never listed or pruned. A step that has a checkpoint already is an error that ends the run (`checkpoint step-N exists in DIR`), and so is an error while writing. The checkpoints beyond `keep` are moved to `.tmp-step-old-N` and deleted; a deletion that fails is not an error of the run, and what it left is swept at the next start.

`Loop.checkpoints(dir)` lists the names, oldest first. `Loop.read_checkpoint(dir, name, optimizer, like)` reads one on the backend and in the dtype of the tensors `like` (the weights the run has now, which give the shapes), answering a `Loop.Checkpoint` with `params`, `state` (give it to `Train.fit` as `start`) and where the run was (`event`, `epoch`, `batch`, `step`, `seed`, `loss`, all exact). An error names the file or the weight that does not fit, and what was read is freed.

```agentica
fit = try (
  Loop.trainer(&loss, params, Optim.adamw(lr: 0.0001))
  |> Loop.checkpoint("runs/minilm", every: 100, keep: 3)
  |> Loop.run(data, epochs: 3)
)
```

## Train.fit

`Train.fit(loss, params, data, optimizer, epochs, on_epoch, metrics, on_step, run)` runs `epochs` passes over the batches of `data`, with one optimizer step per batch. It returns a `Result` of `Fit`, with `params` (the trained list) and `history` (the mean loss of each epoch).

- `loss` has the type `Fn[List[Tensor.Tensor], Data.Batch, Result[Tensor.Tensor]]`. It gets the parameters and a batch, and it returns a float scalar.
- Inside `loss`, the batch `inputs` and `targets` are traced inputs, so the loss is written once, as plain code. `fit` traces and compiles the step once for each batch shape (a smaller last batch is a second one) and runs it on every batch of that shape.
- `on_epoch` is optional. It gets an `Epoch` with `epoch` (from 1), `loss`, `steps` (the step count of the run so far), `state` (the optimizer state) and `params`, the parameters the epoch ended with (use them inside `on_epoch`: the next step frees them), and answers a `Result[Train.Control]`: `ok(Train.Control.Continue)`, `ok(Train.Control.Stop)` (ends `fit` after the epoch, answering its `Fit`) or an `error` (stops `fit` with an error that names the epoch). `fit` prints nothing.
- `on_step` is optional. It gets a `StepInfo` after each step, with `epoch` (from 1), `batch` (from 0), `step` (the count of the run, from 1), `loss`, `state` and `params` (valid only inside `on_step`), and answers a `Result[Train.Control]` as `on_epoch` does: `Stop` ends `fit` after the step, answering its `Fit` (the history's last epoch is the mean over the steps it ran), and an `error` stops it with an error that names the epoch and the batch.
- `run` is the name of the run (empty by default): the `run` of the metadata of the telemetry events below. Give it a short fixed name, not an id or a path, since a reporter may make a label of each value. `Loop.trainer(loss, params, optimizer, run: "name")` sets it.
- `metrics` is optional. It is an open [`Metrics.Run`](metrics.md) that gets `train/loss`, `train/lr` and `train/step_ms` for every step. A value it refuses, such as a loss that is not finite, stops `fit` too.
- A failed step stops `fit` with an error that names the epoch and the batch, for example `epoch 1, batch 0: ...`.

A run emits [telemetry](https://github.com/agentica-lang/packages/tree/main/telemetry) events, which a program attaches handlers to (`Telemetry.start()`, then `Telemetry.attach(id, ["train"], handler)`) to log, count or export them without touching the loop. They are delivered in handlers' own tasks and may be dropped when a handler is slow, so they observe and do not control: stopping a run stays with `on_step`, `on_epoch` and `Loop.handle`. With no handler attached a step builds nothing. Every event has the name of the run as `run` in its metadata, and its measurements are plain numbers: the loss and the learning rate have no unit, so they are outside the unit suffixes of the telemetry conventions (a rule for unit-less floats is [packages#292](https://github.com/agentica-lang/packages/issues/292)).

| Event | `ints` | `floats` | `metadata` |
|---|---|---|---|
| `train.step`, after each step | `step_count` (from 1), `epoch_count` | `loss`, `lr` (the rate of the step) | `run` |
| `train.epoch`, after each epoch | `epoch_count`, `step_count` (so far) | `loss` (the mean of the epoch) | `run` |
| `train.metric`, after each `Loop.metric` of an epoch | `epoch_count` | `value` | `run`, `metric` (its name) |

`Train.fit_batches(loss, params, batch, count, optimizer, epochs, on_epoch, metrics, prefetch, on_step, run)` is `fit` over batches that a function makes, for data that is not two tensors of one shape: `batch(epoch, i)` answers batch `i` (from 0) of `count` in epoch `epoch` (from 1) as a `Data.Batch(inputs:, targets:)`.

```agentica
fit = try Train.fit_batches(&loss, params, ->(_epoch: Int, i: Int) { padded(sentences, i) }, 63, optimizer, 1)
```

- The function makes new tensors on every call. `fit_batches` frees a batch after its step, so a batch that is kept and answered again (a cache, a second epoch) is an error: `batch(2, 0) returned tensors that are already freed: make new ones`.
- Each batch may have its own shape, such as token ids padded to the longest sentence of the batch. On ggml the step is compiled once per shape, so pad to a few lengths (a multiple of 8, for one) when the lengths vary a lot.
- A batch is two tensors. What else the loss needs it derives from them: a padding mask is `1 - to_dtype(eq(ids, zeros), F32)`.
- An error of the function stops the run and names the epoch and the batch.
- With `prefetch: K` (0 by default), a task calls `batch(epoch, i)` for every batch of every epoch, in order, and builds up to `K + 1` batches ahead of the steps, so decoding image files or tokenizing text overlaps the device. The function runs on that task, and the task ends before `fit_batches` returns, on every path. An error of the function reaches the run at the step that would have used the batch, as it does without `prefetch`.

`fit` and `fit_batches` free the parameters and the optimizer state that a step replaces, at once, on libtorch, ggml and XLA. The `params` given stay the caller's and are never freed; the trained ones of the result are the caller's too.

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
split = try Params.split(DistilBert.weights(m), ->(_name: String) { true })
norm = ->(name: String) { name.include?("LayerNorm") or name.include?("layer_norm") }
decay = split.names.map { |name: String| !(name.end_with?(".bias") or norm.call(name)) }
optimizer = Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.00002, weight_decay: 0.01, decay: decay))
```

A layer norm is named by its model: `LayerNorm` in BERT, and `embeddings.LayerNorm`, `sa_layer_norm` and `output_layer_norm` in DistilBERT.

- `Params.split(weights, trainable)` answers a `Split`: `names` (the names `trainable` says yes to, sorted), `trainable` (their tensors, the list `fit` takes) and `frozen` (the other weights by name). A selection that matches no name is an error.
- `Params.join(split, ps)` gives every weight by name again, with `ps` for the trainable ones. Inside the loss `ps` are traced and the frozen weights enter the trace as constants, so they get no gradient, and the optimizer keeps no state for them.
- `Params.named(split, ps)` gives the trainable weights alone by name: what to save. `SafetensorsWrite.write` of the hub package writes them.
- `Params.update(weights, changed)` puts the tensors of `changed` in place of the ones of the same name. To load a saved part: `Params.update(Qwen3.weights(m), try Hub.tensors(try Hub.open(path), backend))`, then `Qwen3.with_weights`. A name the model lacks or another shape is an error.

## LoRA

`Lora` (`import Lora from "pkg/train/lora"`) fine-tunes a model with small adapters. A weight `W` of a linear layer, `[out, in]`, stays frozen, and two small matrices are trained beside it: `A` of `[r, in]` and `B` of `[out, r]`. The layer then computes with `W + (alpha / r) · B A`. `B` starts at zero, so a new adapter changes nothing.

```agentica
weights = Qwen3.weights(m)
config = Lora.Config(targets: ["q_proj", "v_proj"], r: 8, alpha: 16.0)
adapter = try Lora.init(weights, config, 1, backend)
fit = try Train.fit(
  ->(ps: List[Tensor.Tensor], b: Data.Batch) {
    with = try Lora.applied(weights, try Lora.with_params(adapter, ps))
    loss(try Qwen3.with_weights(m, with), b)
  },
  try Lora.params(adapter),
  loader,
  Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.0001, weight_decay: 0.0)),
  3,
)

trained = try Lora.with_params(adapter, fit.params)
```

- `Lora.Config(targets, r = 8, alpha = 16.0, dropout = 0.0)`: `targets` name the layers as PEFT's `target_modules` do. A target may be an embedding table, which trains and runs here; `Lora.save` cannot write that one for PEFT. A layer is adapted when its name (the name of its weight without `.weight`) is a target or ends with `.` and a target, so `q_proj` takes every `model.layers.N.self_attn.q_proj`.
- `Lora.init(weights, config, seed, backend)` makes a new `Adapter`: every `A` uniform in `±1 / sqrt(in)` from `seed`, every `B` zero. `adapter.names` are the adapted weights, sorted. A target that names no layer, or a weight that is not a matrix, is an error.
- `Lora.params(adapter)` gives the list `fit` trains (`A` then `B` of each adapted weight), and `Lora.with_params(adapter, ps)` the adapter with those tensors, inside the loss and after `fit`.
- `Lora.applied(weights, adapter)` gives every weight by name, the adapted ones as `W + (alpha / r) · B A`. Inside the loss the weights of the model enter the trace as constants, so only the adapter gets a gradient.
- `Lora.merged(weights, adapter)` gives the weights with the adapter merged in, for inference: the adapted ones are new tensors on their backend (ggml included) and the others are those of `weights`, shared and not copied. The weights and the adapter must be on one backend. `Qwen3.with_weights(m, try Lora.merged(weights, trained))` is the fine-tuned model, and it runs at the cost of the model alone.

An adapter is kept in PEFT's files, so one trained with PEFT runs here and one trained here loads in PEFT:

```agentica
try Lora.save("adapter", trained)
adapter = try Lora.open("adapter", backend)
```

- `Lora.save(dir, adapter, embeddings = [])` writes `adapter_config.json` and `adapter_model.safetensors` into the folder `dir`, as `PeftModel.save_pretrained` does. Every adapted weight is written as a linear layer's. PEFT keeps the adapter of an `Embedding` in another layout, and an adapter here does not know which of its weights is one: pass the names of the model's embedding tables as `embeddings` (`Qwen3.embeddings(m)`, `DistilBert.embeddings(m)`), and an adapter of one of them is an error instead of a file PEFT would misread. A tensor whose rank is not `config.r`, or a weight no target takes, is an error too. The files do not say which model the adapter is for.
- `Lora.open(dir, backend)` reads such a folder, as `F32`. `adapter.names` come from the names of its tensors. An error names what is not computed here: another `peft_type`; DoRA, rsLoRA, aLoRA, QALoRA, BD-LoRA or another variant (Arrow, KaSA, MonteCLoRA, VeLoRA); `fan_in_fan_out`; a trained bias; `modules_to_save`, `target_parameters` or `trainable_token_indices`; `layer_replication`; a rank or alpha pattern; `target_modules` as a regular expression; and the adapter of an `Embedding`. An adapter trained with `lora_dropout` is read and runs.

On a model whose output head is tied to its embedding table (Qwen3), an adapter on the table also changes the head in `applied` and `merged`, since they are one weight; PEFT adapts the table alone. `Lora.save` with the model's `embeddings` refuses that adapter.

LoRA trains without dropout: `Lora.params` refuses an adapter whose `dropout` is above 0 ([ml#209](https://github.com/agentica-lang/ml/issues/209)).

## Limits

- A backend that does not compile (the reference, libtorch) still evaluates the traced graph node by node on every step.
- The loss must return a float scalar. An op with no backward rule, listed on the [tensor page](../tensor/_index.md), makes `fit` fail.
- On `Ggml`, the step is one compiled graph with at most 512 outputs: the loss, the new parameters and the optimizer's moments. That allows 170 parameters with Adam or AdamW, 255 with momentum SGD and 511 with plain SGD.
- The losses and the optimizers use only ops that ggml supports, so a batch and its step run on `Ggml` when the dataset is on `Ggml`.
- On `Xla` (`Backend.Xla(Device.Cpu)`, see the tensor page for the plugin) the step is one XLA program, compiled once per batch shape, and the batch is gathered in it when the dataset is on `Xla` too.

## How it works

A model here is a loss function of a parameter list and a batch. `Train.fit` traces that function, and an optimizer updates the parameters from its gradients.

Each step is one compiled evaluation: the loss, its gradients and the optimizer update are one graph, so the value, the gradients and the new parameters come from one run.
