---
title: fine-tune DistilBERT
description: A whole program that fine-tunes DistilBERT for sentiment on SST-2 with a classification head and AdamW, then saves it, loads it back and predicts.
weight: 3
---

This page is one program, `transformers/tools/finetune_sst2.agc` of the ml repository, that fine-tunes `distilbert-base-uncased` to tell a positive movie sentence from a negative one. It downloads the data, reads and tokenizes it, cuts it into padded batches, puts a linear classification head on DistilBERT's mean-pooled output, trains with AdamW, measures the loss and the accuracy on the validation split before and after, saves the weights, loads them back and predicts two sentences with the loaded model. It runs on the CPU with the `Ggml` backend by default, and an argument picks ggml on Vulkan, XLA or libtorch (see [Benchmark](#benchmark)). It uses the pieces of the [train](_index.md), [data loader](data.md) and [transformers](../transformers/models.md) pages.

The data is SST-2 (`stanfordnlp/sst2`): 67,349 training sentences and 872 validation sentences, each with a label, 0 for negative and 1 for positive. The model is `distilbert/distilbert-base-uncased` (268 MB of weights).

## Fetch

`tools/fetch_sst2.sh DIR` downloads the config, the tokenizer and the weights of the checkpoint and the two splits of the dataset into `DIR`, at pinned Hugging Face revisions, and checks the SHA-256 of every file; a file with another hash is refused. The splits come as Parquet. `Table.read` reads Parquet, but a file of at most 1 MiB, and the training split is 3.1 MB, so `tools/sst2_csv.py` writes the `sentence` and `label` columns of each split as `validation.csv` and `train.csv`, the latter with the first 10,000 of the 67,349 training rows (about 0.6 MB). That step needs Python with `pyarrow`, given as `PYTHON` (default `python3`).

```sh
cd transformers
sh tools/fetch_sst2.sh data
```

## The program

The imports and the knobs. The program trains on the first `TRAIN_ROWS` sentences of the training split (or the number given as its last argument), for one epoch, in batches of `BATCH`. The validation split is read whole.

```agentica
import Data from "pkg/train/data"
import DistilBert from "../distilbert"
import Hub from "pkg/hub/hub"
import Loss from "pkg/train/loss"
import Nn from "pkg/nn/nn"
import Optim from "pkg/train/optim"
import Params from "pkg/train/params"
import SafetensorsWrite from "pkg/hub/safetensors_write"
import Table from "pkg/train/table"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "pkg/train/train"
import WordPiece from "../tokenizer/wordpiece"

TRAIN_ROWS = 2000
BATCH = 16
EVAL_BATCH = 64
LR = 0.00005
WEIGHT_DECAY = 0.01
LENGTH = 56
SEED = 1
LABELS = ["negative", "positive"]
DEMO = ["a gripping, funny movie", "a dull and lifeless mess"]
```

`main` runs `run` and turns an `error` into a message and an exit status. Everything below `private` is the program.

```agentica
def main() -> Void
  case run(System.args())
    ok(_) -> nil
    error(e) ->
      puts("finetune_sst2: #{e.message}")
      System.exit(1)
  end
end

private

# A sentence as token ids, with its label: 0 is negative, 1 is positive.
struct Example
  ids: List[Int]
  label: Int
end

struct Score
  loss: Float
  accuracy: Float
end

# DistilBERT and how its weights are split for training (`Params`).
struct Net
  base: DistilBert.DistilBert
  split: Params.Split
end

# `params` are what `Train.fit_batches` trains: the weights of DistilBERT, then the head's weight and bias.
struct Model
  net: Net
  params: List[Tensor.Tensor]
end
```

`Model` holds the parameters that `Train.fit_batches` trains as one list: the weights of DistilBERT in the order of `Params.split`, then the head's weight and bias. `run` reads the data, scores the model before training, trains for one epoch, saves, loads and compares, and then predicts.

```agentica
def run(args: List[String]) -> Result[Void]
  return error("usage: MODEL_DIR DATA_DIR OUT_DIR [cpu|vulkan|xla|torch [ROWS]]") if args.size < 3 or args.size > 5

  started = Time.monotonic_ns()
  backend = try backend_of(args)
  base = try DistilBert.open(args[0], backend)
  tok = try WordPiece.open("#{args[0]}/tokenizer.json")
  train = try examples(tok, "#{args[1]}/train.csv", some(try train_rows(args)))
  validation = try examples(tok, "#{args[1]}/validation.csv", none)

  start = try initial(base, backend)
  before = try score(start, validation)
  puts("before: validation loss #{before.loss}, accuracy #{before.accuracy}")

  trained = try tuned(start, train, validation)
  try round_trip(args[2], trained, validation)
  puts("seconds: #{(Time.monotonic_ns() - started).to_f / 1000000000.0}")

  for text in DEMO
    label = try predict(trained, tok, text)
    puts("#{text}: #{label}")
  end

  ok()
end

def backend_of(args: List[String]) -> Result[Tensor.Backend]
  return ok(Tensor.Backend.Ggml(Tensor.Device.Cpu)) if args.size == 3

  backend = try named(args[3])
  return error("the backend #{args[3]} is not available") if not Tensor.available(backend)

  ok(backend)
end

def named(name: String) -> Result[Tensor.Backend]
  case name
    "cpu" -> ok(Tensor.Backend.Ggml(Tensor.Device.Cpu))
    "vulkan" ->
      try Tensor.vulkan_fp16(false)
      ok(Tensor.Backend.Ggml(Tensor.Device.Vulkan(0)))
    "xla" -> ok(Tensor.Backend.Xla(Tensor.Device.Cpu))
    "torch" -> ok(Tensor.Backend.Torch(Tensor.Device.Cpu))
    _ -> error("the backend is cpu, vulkan, xla or torch, not #{name}")
  end
end

def train_rows(args: List[String]) -> Result[Int]
  return ok(TRAIN_ROWS) if args.size < 5

  args[4].to_i
end

def initial(base: DistilBert.DistilBert, backend: Tensor.Backend) -> Result[Model]
  split = try Params.split(DistilBert.weights(base), ->(_name: String) { true })
  head = try Nn.init_dense(base.config.dim, 2, SEED, backend)
  bias = try head.b.to_result("the head has no bias")
  ok(Model(net: Net(base: base, split: split), params: split.trainable + [head.w, bias]))
end

def tuned(start: Model, train: List[Example], validation: List[Example]) -> Result[Model]
  began = Time.monotonic_ns()
  fit = try train_epoch(start, train)
  seconds = (Time.monotonic_ns() - began).to_f / 1000000000.0
  puts("epoch 1: training loss #{fit.history[0]}, #{seconds} seconds")
  trained = Model(net: start.net, params: fit.params)
  after = try score(trained, validation)
  puts("after: validation loss #{after.loss}, accuracy #{after.accuracy}")
  ok(trained)
end

# The saved model must score exactly as the one in memory.
def round_trip(dir: String, m: Model, validation: List[Example]) -> Result[Void]
  try save(dir, m)
  loaded = try load(dir, m.net.base)
  expected = try score(m, validation)
  again = try score(loaded, validation)
  return error("the loaded model scores differently") if again.loss != expected.loss

  puts("loaded: validation loss #{again.loss}, accuracy #{again.accuracy}")
  ok()
end
```

## Save and load

`save` writes two files with `SafetensorsWrite`: `model.safetensors`, whose tensors have DistilBERT's checkpoint names, so it opens like the original, and `classifier.safetensors`, with the head's `weight` and `bias`. `load` reads both back with `Hub` and builds the same `Model`. `round_trip` scores the in-memory and the loaded model on the validation split and fails unless the two losses are equal.

```agentica
# `model.safetensors` has DistilBERT's weights by their checkpoint names; `classifier.safetensors` the head's.
def save(dir: String, m: Model) -> Result[Void]
  n = m.net.split.names.size
  weights = try Params.join(m.net.split, m.params.take(n))
  try SafetensorsWrite.write("#{dir}/model.safetensors", weights)
  try SafetensorsWrite.write("#{dir}/classifier.safetensors", {"weight": m.params[n], "bias": m.params[n + 1]})
  ok()
end

def load(dir: String, base: DistilBert.DistilBert) -> Result[Model]
  backend = Tensor.backend(base.words.table)
  weights = try Hub.tensors(try Hub.open("#{dir}/model.safetensors"), backend)
  head = try Hub.tensors(try Hub.open("#{dir}/classifier.safetensors"), backend)
  split = try Params.split(weights, ->(_name: String) { true })
  ok(Model(net: Net(base: base, split: split), params: split.trainable + [head["weight"], head["bias"]]))
end
```

## Read and tokenize

`Table.read` reads a CSV file, and `Table.texts` and `Table.floats` give its `sentence` and `label` columns (the labels are numbers in the file, so `floats`; `to_i` makes them class ids). The first `limit` rows go through the checkpoint's WordPiece tokenizer, with the special tokens and a limit of `LENGTH` ids.

```agentica
# The first `limit` rows (all when none) of a CSV split (`sentence`, `label`), tokenized.
def examples(tok: WordPiece.WordPiece, path: String, limit: Optional[Int]) -> Result[List[Example]]
  table = try Table.read(path)
  sentences = try Table.texts(table, "sentence")
  labels = try Table.floats(table, "label")
  count = case limit
    some(n) -> n.clamp(0, table.rows)
    none -> table.rows
  end

  out: List[Example] = []
  for i in 0...count
    encoded = try WordPiece.encode(tok, sentences[i], none, true, some(LENGTH))
    out = out.push(Example(ids: encoded.ids, label: try labels[i].to_i))
  end

  ok(out)
end
```

## Train

`train_epoch` calls `Train.fit_batches` with a loss, the parameters, a function that builds batch `i`, the number of steps, the optimizer and the number of epochs. The optimizer is AdamW with a learning rate of 5e-5 and a weight decay of 0.01, which applies to every weight but the biases and the layer norms, as in Hugging Face's `Trainer`. The loss is the cross entropy of the head's logits against the labels. The head's input is DistilBERT's output averaged over the real tokens, with the padding masked out (`pad_mask`).

```agentica
# One epoch with AdamW: a step per whole batch of BATCH rows (a last partial batch is left out), the head's weight
# decayed, its bias not.
def train_epoch(m: Model, rows: List[Example]) -> Result[Train.Fit]
  pad = m.net.base.config.pad_token_id
  backend = Tensor.backend(m.net.base.words.table)
  decay = m.net.split.names.map { |name: String| decays?(name) } + [true, false]
  optimizer = Optim.Optimizer.AdamW(Optim.AdamW(lr: LR, weight_decay: WEIGHT_DECAY, decay: decay))
  steps = rows.size / BATCH
  Train.fit_batches(
    ->(params: List[Tensor.Tensor], b: Data.Batch) { loss(m.net, params, b) },
    m.params,
    ->(_epoch: Int, i: Int) { padded(rows.drop(i * BATCH).take(BATCH), pad, backend) },
    steps,
    optimizer,
    1,
  )
end

# Hugging Face's rule: no decay on a bias or a layer norm.
def decays?(name: String) -> Bool
  !(name.end_with?(".bias") or name.include?("LayerNorm") or name.include?("layer_norm"))
end

def loss(net: Net, ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.cross_entropy(try classify(net, ps, b.inputs), b.targets)
end

# DistilBERT's mean-pooled output through the head: the logits `[rows, 2]` of the ids `[rows, length]`.
def classify(net: Net, ps: List[Tensor.Tensor], ids: Tensor.Tensor) -> Result[Tensor.Tensor]
  n = ps.size - 2
  model = try DistilBert.with_weights(net.base, try Params.join(net.split, ps.take(n)))
  mask = try pad_mask(ids, net.base.config.pad_token_id)
  pooled = try DistilBert.pooled(model, ids, mask)
  Nn.dense(Nn.Dense(w: ps[n], b: some(ps[n + 1])), pooled)
end

# 1.0 for a token and 0.0 for padding, as `F32`.
def pad_mask(ids: Tensor.Tensor, pad: Int) -> Result[Tensor.Tensor]
  backend = Tensor.backend(ids)
  shape = Tensor.shape(ids)
  pads = try Tensor.full(shape, Tensor.Values.Int(pad), Tensor.Dtype.I32, backend)
  zeros = try Tensor.full(shape, Tensor.Values.Float(0.0), Tensor.Dtype.F32, backend)
  ones = try Tensor.full(shape, Tensor.Values.Float(1.0), Tensor.Dtype.F32, backend)
  Tensor.where(try Tensor.eq(ids, pads), zeros, ones)
end
```

## Batches

`padded` turns some examples into the `I32` ids `[rows, LENGTH]`, padded with the pad id, and the `I32` labels `[rows]`. `LENGTH` is 56, the longest sentence of SST-2 in tokens, so every batch has one shape and XLA compiles the step once. A last partial batch of the training rows is left out, and `score` fills the last batch of the validation split up to `EVAL_BATCH` rows with copies of its first row, which it does not count.

```agentica
# Ids `[rows, LENGTH]` padded with `pad`, and labels `[rows]`, both `I32`. LENGTH is the longest sentence of SST-2 in
# tokens, so every batch has one shape and XLA compiles the step once.
def padded(rows: List[Example], pad: Int, backend: Tensor.Backend) -> Result[Data.Batch]
  ids: List[Tensor.Values] = []
  for r in rows
    for k in 0...LENGTH
      ids = ids.push(Tensor.Values.Int(token_at(r.ids, k, pad)))
    end
  end

  inputs = try Tensor.from_list(ids, [rows.size, LENGTH], Tensor.Dtype.I32, backend)
  labels = rows.map { |r: Example| Tensor.Values.Int(r.label) }
  targets = try Tensor.from_list(labels, [rows.size], Tensor.Dtype.I32, backend)
  ok(Data.Batch(inputs: inputs, targets: targets))
end

def token_at(ids: List[Int], k: Int, pad: Int) -> Int
  return ids[k] if k < ids.size

  pad
end
```

## Evaluate and predict

`score` runs the validation split in batches of `EVAL_BATCH` through `Trace.eval`, which is how a `Ggml` model runs outside a training loss, and returns the mean cross entropy and the accuracy. `predict` is the same for one sentence, and gives the name of the larger logit.

```agentica
# Batches of EVAL_BATCH rows run through a trace, which is how a ggml model runs outside a loss.
def score(m: Model, rows: List[Example]) -> Result[Score]
  pad = m.net.base.config.pad_token_id
  backend = Tensor.backend(m.net.base.words.table)
  losses: List[Float] = []
  hits: List[Bool] = []
  for i in 0...((rows.size + EVAL_BATCH - 1) / EVAL_BATCH)
    part = rows.drop(i * EVAL_BATCH).take(EVAL_BATCH)
    batch = try padded(part + filler(part, EVAL_BATCH), pad, backend)
    values = try logits(m, batch.inputs)
    for j in 0...part.size
      z0 = values[2 * j]
      z1 = values[2 * j + 1]
      losses = losses.push(row_loss(z0, z1, part[j].label))
      hits = hits.push(predicted(z0, z1) == part[j].label)
    end
  end

  ok(Score(loss: losses.sum / rows.size.to_f, accuracy: hits.select { |hit: Bool| hit }.size.to_f / rows.size.to_f))
end

# Copies of the first row that fill a last partial batch up to `size`, so that it has the shape of the others.
def filler(part: List[Example], size: Int) -> List[Example] = (0...(size - part.size)).map { |_i: Int| part[0] }

def logits(m: Model, ids: Tensor.Tensor) -> Result[List[Float]]
  out = try Trace.eval(m.params + [ids]) do |xs| classify(m.net, xs.take(m.params.size), xs[m.params.size]) end
  floats(out)
end

def predict(m: Model, tok: WordPiece.WordPiece, text: String) -> Result[String]
  encoded = try WordPiece.encode(tok, text, none, true, some(LENGTH))
  backend = Tensor.backend(m.net.base.words.table)
  batch = try padded([Example(ids: encoded.ids, label: 0)], m.net.base.config.pad_token_id, backend)
  values = try logits(m, batch.inputs)
  ok(LABELS[predicted(values[0], values[1])])
end

def predicted(z0: Float, z1: Float) -> Int
  return 1 if z1 > z0

  0
end

# The cross entropy of one row with two logits: log(1 + exp(other - label)).
def row_loss(z0: Float, z1: Float, label: Int) -> Float
  return Math.log(1.0 + Math.exp(z1 - z0)) if label == 0

  Math.log(1.0 + Math.exp(z0 - z1))
end

def floats(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("the logits are not floats")
    end
  end

  ok(out)
end
```

## Run it

From `transformers/`, with `data` from the fetch step and `out` as an empty folder for the saved model:

```sh
agentica tools/finetune_sst2.agc data/distilbert-base-uncased data/sst2 out [cpu|vulkan|xla|torch [ROWS]]
```

The fourth argument picks the backend (`cpu`, the default, is ggml on the CPU; `vulkan` is ggml on the first Vulkan device; `xla` and `torch` are XLA and libtorch on the CPU) and the fifth the number of training rows. A backend that is not available on the machine is an `error` that names it. `xla` needs the plugin that `tensor/tools/install_xla.agc` installs, and `torch` needs the native libtorch module of the `tensor` package.

It prints the loss and the accuracy on the validation split before the training, the mean training loss of the epoch, the same two numbers after, those of the model loaded back from `out`, the seconds of the whole run, and the prediction of two sentences. This is a run on the first 2,000 training rows, with a batch of 16 (125 steps), on an AMD Ryzen 5 3550H (4 cores, 8 threads, 12 GB) that other jobs share, on the ggml CPU backend, so the time is rough:

```text
before: validation loss 0.7770871823894626, accuracy 0.48853211009174313
epoch 1: training loss 0.38535212469100955
after: validation loss 0.5363854705634189, accuracy 0.7672018348623854
loaded: validation loss 0.5363854705634189, accuracy 0.7672018348623854
seconds: 523.475574741
a gripping, funny movie: positive
a dull and lifeless mess: negative
```

The loss before is that of a head with random weights, and the accuracy is about that of a coin on this balanced split. After the 125 steps it is 0.5364 and 76.7%. The line `loaded` equals the line `after` because the program fails when they differ, and `out` then holds `model.safetensors` and `classifier.safetensors`.

## Benchmark

The same program on the backends, on the first 400 training rows (25 steps of 16), one epoch, the same seed and settings, and the whole validation split of 872 rows. The times are rough: the machine is shared, and each backend ran once, one after another. Machine: AMD Ryzen 5 3550H (4 cores, 8 threads, 12 GB) and its Radeon Vega iGPU (RADV RAVEN, shared memory), for Vulkan.

| Backend | Whole run (s) | Train loop (s) | First step (s) | Validation before (loss, accuracy) | Validation after (loss, accuracy) | Reloaded accuracy |
|---|---|---|---|---|---|---|
| ggml CPU | | | | | | |
| ggml Vulkan | | | | | | |
| XLA CPU | does not run | | | | | |
| libtorch CPU | does not run | | | | | |

The train loop is the `epoch 1` line of the output, and the first step is the train loop of a run on 16 rows (one step) with a validation split of 64 rows, which holds the compiles of the backend.

The other two backends stop before they finish an epoch. `xla` stops at its first scoring with `StableHLO has no emitter for layer_norm yet`: XLA compiles a traced function to StableHLO, and the emitter of `tensor/` has no `layer_norm` yet. `torch` scores the validation split, with the same loss as ggml to about 1e-7, and stops at the first training step with `batch_matmul is not implemented on the torch backend yet`. It needs libtorch at `build-torch.sh`'s paths (`eval "$(sh ../tensor/tools/build-torch.sh)"` from `transformers/`).

## Next steps

- More rows or epochs. `TRAIN_ROWS` is 2,000, and `train.csv` has 10,000 rows (the whole split needs a reader that does not take the file whole); a run on the full split and with a few epochs trains the head and the model further. `Train.fit_batches` takes the number of epochs, and the loss it returns has one entry per epoch.
- A GPU. The fourth argument `vulkan` runs the same program on the first Vulkan device; the [tensor](../tensor/_index.md) page lists the backends.
- Fewer trained weights. [LoRA](_index.md#lora) trains an adapter on the attention projections instead of every weight.
