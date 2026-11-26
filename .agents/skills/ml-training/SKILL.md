---
name: ml-training
description: >-
  Train and fine-tune models from an Agentica app with the agentica-lang/ml
  `train` package: a loss function, SGD, Adam or AdamW, Data.dataset and
  Train.fit or fit_batches, metrics, validation and checkpoints, freezing
  layers, LoRA adapters saved in PEFT's files, fine-tuning a Hugging Face model
  such as DistilBERT, speed (ggml Vulkan, XLA on the CPU, batch size,
  Trace.compile), StableHLO export, and checking a run against PyTorch. Use
  when an Agentica task trains, fine-tunes or exports a model. Not for running
  a model without training (use ml-inference) and not for work on the ml
  repository itself.
---

# ml training

`train` fits a model to data. A model is a loss function of a parameter list and a batch: you write it as plain tensor
code, `Train.fit` traces it, builds the gradients, compiles one step and runs it on every batch. Values never change, so
`fit` returns new parameters. Install and backends are in `ml-inference` (read its `SKILL.md` and
`references/backends.md` first when the packages are not installed yet). Copy both skills together.

## Install

Use the install block of `ml-inference` (its `SKILL.md`: `tensor`, `nn`, `hub`, and `transformers` to fine-tune a
Hugging Face model) and add `train`:

```toml
train = { git = "https://github.com/agentica-lang/ml", tag = "train-0.7.0", subdir = "train" }
```

## First fit

A 2-8-1 MLP learns XOR on the reference backend (about a second), then classifies the four rows. The parameters are a
plain list, the loss is a function, and `fit` gives back the trained list and the mean loss of each epoch:

```agentica
import Data from "pkg/train/data"
import Loss from "pkg/train/loss"
import Nn from "pkg/nn/nn"
import Optim from "pkg/train/optim"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "pkg/train/train"

def predict(ps: List[Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  up = Nn.Dense(w: ps[0], b: some(ps[1]))
  down = Nn.Dense(w: ps[2], b: some(ps[3]))
  Nn.mlp(Nn.Mlp(up: up, down: down, activation: Nn.Activation.Relu), x)
end

def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.binary_cross_entropy(try predict(ps, b.inputs), b.targets)
end

def classes(ps: List[Tensor.Tensor], x: Tensor.Tensor) -> Result[List[Bool]]
  logits = try Trace.eval(ps + [x]) do |xs| predict(xs.take(4), xs[4]) end
  out: List[Bool] = []
  for v in try Tensor.to_list(logits)
    case v
      Tensor.Values.Float(f) -> out = out.push(f > 0.0)
      _ -> return error("logits are floats")
    end
  end

  ok(out)
end

def main() -> Void
  backend = Tensor.Backend.Reference
  inputs = Tensor.floats([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2], backend).unwrap!
  targets = Tensor.floats([0.0, 1.0, 1.0, 0.0], [4, 1], backend).unwrap!
  loader = Data.Loader(dataset: Data.dataset(inputs, targets).unwrap!, batch_size: 4, shuffle: false)
  up = Nn.init_dense(2, 8, 1, backend).unwrap!
  down = Nn.init_dense(8, 1, 2, backend).unwrap!
  params = [up.w, up.b.unwrap!, down.w, down.b.unwrap!]
  adam = Optim.Optimizer.Adam(Optim.Adam(lr: 0.2))
  fit = Train.fit(&loss, params, loader, adam, 20).unwrap!
  puts(fit.history.size)
  puts(fit.history[19] < fit.history[0])
  puts(classes(fit.params, inputs).unwrap!)
end
```

```output
20
true
[false, true, true, false]
```

## Rules

- `loss(params, batch)` answers a float scalar `Result[Tensor]`. Inside it, `params` and the batch are traced; the loss is
  traced once for each batch shape, not on every step. Rebuild layers from the list: `Nn.Dense(w: ps[0], b: some(ps[1]))`.
- An op with no backward rule on the path from the parameters to the loss makes `fit` fail with an error naming the op:
  `pow`, `clamp`, `to_dtype`, `batch_matmul` (on libtorch), `conv1d`, `avg_pool1d`. Use the derived ops (`softmax`,
  `gelu`, `layer_norm`, `linear`). An op off that path is fine, such as `to_dtype` on a padding mask.
- Train a model loaded as `F32`. Half (`F16`) and quantized (`Q8_0`, `Q4_K`, `Q6_K`) weights have no gradient.
- On ggml, one step is one graph with at most 512 outputs: the loss, the new parameters and the moments. That is 170
  parameters with Adam or AdamW, 255 with momentum SGD, 511 with plain SGD.
- `fit` frees the parameters and moments that a step replaces. The `params` you give stay yours, and so do the trained
  ones it returns.
- `fit` prints nothing. Use `on_epoch`, `metrics` or the returned `history`.
- A test that trains can run out of steps (`RUNTIME-STEP-LIMIT`): see `ml-inference/references/memory-and-errors.md`.

## References

Read only what the task needs.

| File in `references/` | Covers |
|---|---|
| `fit.md` | losses, optimizers and schedules, datasets and loaders, `fit`, `fit_batches`, `on_epoch`, metrics, validation, checkpoints |
| `finetune.md` | freezing layers (`Params`), LoRA (`Lora.init`, `save`, `open`, `merged`), fine-tuning DistilBERT and Qwen3, PEFT files |
| `speed.md` | which backend and batch size, `Trace.compile`, benchmark numbers, `Trace.stablehlo` export |
| `verify.md` | checking a run against a PyTorch peer, and an adapter against PEFT |

The full API: `https://www.agentica-lang.com/ml/train/` (and `.../train/data/`, `.../train/metrics/`), with
`.../tensor/` and `.../transformers/`.
