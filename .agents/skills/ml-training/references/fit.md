# Losses, optimizers, `fit`, validation, checkpoints

## Pieces

| Piece | Call |
|---|---|
| Loss | `Loss.mse(pred, target)`, `Loss.binary_cross_entropy(logits, targets)` (targets in 0..1), `Loss.cross_entropy(logits, labels)` (`[batch, classes]` logits, `[batch]` integer labels). Each gives a rank-0 mean. |
| Optimizer | `Optim.Optimizer.Sgd(Optim.Sgd(lr:, momentum: 0.0))`, `Adam(Optim.Adam(lr:, beta1:, beta2:, eps:))`, `AdamW(Optim.AdamW(lr:, weight_decay: 0.01, decay: []))` |
| Schedule | `schedule:` on any optimizer: `Optim.Schedule.Constant`, `WarmupLinear(warmup, total)`, `WarmupCosine(warmup, total)`. `lr` is the peak. |
| Data | `Data.dataset(inputs, targets)` (the first dimension is the rows), `Data.Loader(dataset:, batch_size:, shuffle: true, seed: 0, drop_last: false, prefetch: 2)` |
| Run | `Train.fit(loss, params, loader, optimizer, epochs, on_epoch, metrics, on_step, run)` and `Train.fit_batches(loss, params, batch, count, optimizer, epochs, on_epoch, metrics, prefetch: 0, on_step, run)` |

`Adam` defaults: `lr` 0.001, betas 0.9 and 0.999, `eps` 1e-8. `AdamW`'s `decay` has one flag per parameter (empty: all
decay); a fine-tune leaves out biases and layer-norm weights. `fit` starts every run at step 0, so a resumed run starts
its warmup again and its moments from zero.

## Fit and validate

The loader shuffles with a seed, so a seed gives the same run. `fit` returns `params` (the trained list) and `history`
(the mean loss of each epoch). To validate, run the loss on held-out rows with `Trace.eval`: the trained parameters and
the rows are its arguments.

```agentica
import Data from "pkg/train/data"
import Loss from "pkg/train/loss"
import Optim from "pkg/train/optim"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "pkg/train/train"

def predict(ps: List[Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  rows = Tensor.shape(x)[0]
  Tensor.reshape(try Tensor.matmul(x, ps[0]), [rows])
end

def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.mse(try predict(ps, b.inputs), b.targets)
end

def validate(ps: List[Tensor.Tensor], x: Tensor.Tensor, y: Tensor.Tensor) -> Result[Float]
  scalar = try Trace.eval(ps + [x, y]) do |xs| loss(xs.take(1), Data.Batch(inputs: xs[1], targets: xs[2])) end
  case (try Tensor.to_list(scalar))[0]
    Tensor.Values.Float(f) -> ok(f)
    _ -> error("a loss is a float")
  end
end

def main() -> Void
  xs = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 3.0, 2.0, 2.0, 1.0, 4.0, 1.0, 5.0]
  ys = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  inputs = Tensor.floats(xs.take(16), [8, 2]).unwrap!
  targets = Tensor.floats(ys.take(8), [8]).unwrap!
  held_x = Tensor.floats(xs.drop(16), [2, 2]).unwrap!
  held_y = Tensor.floats(ys.drop(8), [2]).unwrap!
  loader = Data.Loader(dataset: Data.dataset(inputs, targets).unwrap!, batch_size: 4, seed: 1)
  w = Tensor.zeros([2, 1]).unwrap!
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.05))
  fit = Train.fit(&loss, [w], loader, sgd, 30).unwrap!
  puts(fit.history[29] < fit.history[0])
  puts(validate(fit.params, held_x, held_y).unwrap! < 1.0)
end
```

```output
true
true
```

## Progress, metrics, checkpoints

- `on_epoch` gets `Train.Epoch(epoch:, loss:, steps:, params:)` (the epoch counts from 1; `params` are the parameters the epoch
  ended with, valid only inside `on_epoch`: the next step frees them). It answers `Result[Train.Control]`:
  `ok(Train.Control.Continue)`, `ok(Train.Control.Stop)` (ends the fit after the epoch, with its `Fit`) or an `error`.
- `on_step` gets `Train.StepInfo(epoch:, batch:, step:, loss:, params:)` after each step and answers `Result[Train.Control]`;
  `Stop` ends the fit after the step, with its `Fit`, and an `error` stops it, naming the epoch and the batch.
- `run` names the run (`Loop.trainer(..., run:)` too): a run emits `telemetry` events `train.step` (`ints` `step_count`, `epoch_count`; `floats` `loss`, `lr`), `train.epoch` (`loss`) and `train.metric` (`value`, metadata `metric`) for handlers an app attaches, with `run` in the metadata; they observe only.
- `metrics` is an open `Metrics.Run`: every step writes `train/loss`, `train/lr` and `train/step_ms` to
  `runs/NAME/metrics.jsonl` and a TensorBoard event file (`tensorboard --logdir runs`). `fit` does not open or close
  the run. A loss that is not finite stops the fit with an error naming the epoch and the batch.
- A checkpoint is a safetensors file of the parameters by name. `fit` has no checkpoint of its own: save what it
  returns, and give the loaded tensors to the next `fit` as `params`. The optimizer state is not kept.

```agentica
import Data from "pkg/train/data"
import Hub from "pkg/hub/hub"
import Loss from "pkg/train/loss"
import Metrics from "pkg/train/metrics"
import Optim from "pkg/train/optim"
import SafetensorsWrite from "pkg/hub/safetensors_write"
import Tensor from "pkg/tensor/tensor"
import Train from "pkg/train/train"

def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  rows = Tensor.shape(b.inputs)[0]
  Loss.mse(try Tensor.reshape(try Tensor.matmul(b.inputs, ps[0]), [rows]), b.targets)
end

def main() -> Void
  dir = "/tmp/ml-skill-fit-#{Random.int(0, 999999999)}"
  Dir.mkdir_private(dir, 5000).unwrap!
  inputs = Tensor.floats([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0], [4, 2]).unwrap!
  targets = Tensor.floats([1.0, 2.0, 3.0, 4.0], [4]).unwrap!
  loader = Data.Loader(dataset: Data.dataset(inputs, targets).unwrap!, batch_size: 2, seed: 1)
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.05))
  run = Metrics.open(dir, "demo").unwrap!
  seen = Train.fit(
    &loss,
    [Tensor.zeros([2, 1]).unwrap!],
    loader,
    sgd,
    5,
    some(
      ->(e) {
        puts("epoch #{e.epoch}")
        ok(Train.Control.Continue)
      },
    ),
    some(run),
  )

  Metrics.close(run).unwrap!
  first = seen.unwrap!

  path = "#{dir}/w.safetensors"
  SafetensorsWrite.write(path, {"w": first.params[0]}).unwrap!
  w = Hub.tensor(Hub.open(path).unwrap!, "w", Tensor.Backend.Reference).unwrap!
  more = Train.fit(&loss, [w], loader, sgd, 5).unwrap!
  puts(more.history[4] < first.history[4])
  Path.remove(dir, 5000).unwrap!
end
```

```output
epoch 1
epoch 2
epoch 3
epoch 4
epoch 5
true
```

## Batches that a function makes

`fit_batches` is `fit` for data that is not two tensors of one shape, such as token ids padded to the longest sentence of
each batch. `batch(epoch, i)` answers batch `i` (from 0) of `count` in epoch `epoch` (from 1) as a `Data.Batch`.

- Make new tensors on every call. `fit_batches` frees a batch after its step, so answering a kept batch again is an
  error: `batch(2, 0) returned tensors that are already freed: make new ones`.
- Each batch may have its own shape. On ggml and XLA the step is compiled once per shape, so pad to a few lengths (a
  multiple of 8, for one) when the lengths vary a lot.
- The batch is two tensors. Derive the rest in the loss: a padding mask is `1 - to_dtype(eq(ids, pads), F32)`.
- `prefetch: K` runs `batch` on a task that builds up to `K + 1` batches ahead, so decoding image files (`Vision.samples`) overlaps the
  step. The batches come in order, the same ones as without it.

```agentica
import Data from "pkg/train/data"
import Loss from "pkg/train/loss"
import Optim from "pkg/train/optim"
import Tensor from "pkg/tensor/tensor"
import Train from "pkg/train/train"

def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  rows = Tensor.shape(b.inputs)[0]
  Loss.mse(try Tensor.reshape(try Tensor.matmul(b.inputs, ps[0]), [rows]), b.targets)
end

def batch(_epoch: Int, i: Int) -> Result[Data.Batch]
  xs = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 3.0, 2.0, 2.0]
  ys = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  inputs = try Tensor.floats(xs.drop(i * 8).take(8), [4, 2])
  targets = try Tensor.floats(ys.drop(i * 4).take(4), [4])
  ok(Data.Batch(inputs: inputs, targets: targets))
end

def main() -> Void
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.05))
  fit = Train.fit_batches(
    &loss,
    [Tensor.zeros([2, 1]).unwrap!],
    ->(e: Int, i: Int) { batch(e, i) },
    2,
    sgd,
    20,
  ).unwrap!

  puts(fit.history[19] < fit.history[0])
end
```

```output
true
```

## Notes

- A dataset on ggml holds at most 2^24 rows. `Data.load(path, "inputs", "targets", backend)` reads two named tensors of a
  safetensors file onto the backend.
- Check the loss of every step with `metrics`, not by printing: a loss that goes `NaN` stops the run and says where.
- Class labels for `cross_entropy` are `I32` or `I64` tensors; build them with `Tensor.from_list(values, shape,
  Tensor.Dtype.I32, backend)`.

Pages: `https://www.agentica-lang.com/ml/train/`, `.../train/data/`, `.../train/metrics/`.
