---
title: fine-tune DistilBERT
description: A whole program that fine-tunes DistilBERT for sentiment on SST-2 with a classification head and AdamW, then saves it, loads it back and predicts.
weight: 4
---

This page is one program, `transformers/tools/finetune_sst2.agc` of the ml repository, that fine-tunes `distilbert-base-uncased` to tell a positive movie sentence from a negative one. It reads SST-2 files, tokenizes them, cuts them into padded batches, puts a linear classification head on DistilBERT's mean-pooled output, trains with AdamW, measures the loss and the accuracy on the validation split before and after, saves the weights, loads them back and predicts two sentences with the loaded model. It runs on the CPU with the `Ggml` backend by default, and an argument picks ggml on Vulkan (see [Benchmark](#benchmark)). It uses the pieces of the [train](_index.md), [data loader](data.md) and [transformers](../transformers/_index.md) pages.

The data is SST-2 (`stanfordnlp/sst2`): 67,349 training sentences and 872 validation sentences, each with a label, 0 for negative and 1 for positive. The model is `distilbert/distilbert-base-uncased` (268 MB of weights).

## What it is not

This is the loop, not a recipe for the best score. The head is a linear layer on the mean of the token vectors, not the head of `DistilBertForSequenceClassification`; there is no dropout, no shuffle (the rows are read in file order), no warmup and no gradient clipping. Its accuracy is therefore not comparable with the usual figure of about 91% for DistilBERT on SST-2. A run is reproducible from the seed that starts the head and from the order of the files, but not bit for bit: ggml sums in several threads, so the last digits of a loss change between runs.

## Fetch

`tools/fetch_sst2.sh DIR` needs the network and `curl`. It downloads the config, the tokenizer and the weights of the checkpoint and the two Parquet splits of the dataset into `DIR`, at pinned Hugging Face revisions, and checks the SHA-256 of every file with `sha256sum`; a file that is there with the right hash is not fetched again, and a file with another hash is refused. `Table.read` reads the Parquet files as they come.

```sh
cd transformers
sh tools/fetch_sst2.sh data
```

## The program

The imports and the knobs. The program trains on the first `TRAIN_ROWS` sentences of the training split (or the number given as its last argument, at least `BATCH`), for one epoch, in batches of `BATCH`. The validation split is read whole.

```agentica
import Data from "pkg/train/data"
import DistilBert from "../distilbert"
import Hub from "pkg/hub/hub"
import Loss from "pkg/train/loss"
import Metrics from "pkg/train/metrics"
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
TOLERANCE = 0.000001
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

NS = 1000000000.0
HEAD_TENSORS = 2

# A sentence as token ids with its label (0 negative, 1 positive); `cut` when LENGTH dropped some of its tokens.
struct Example
  ids: List[Int]
  label: Int
  cut: Bool
end

struct Splits
  train: List[Example]
  validation: List[Example]
end

struct Score
  loss: Float
  accuracy: Float
end

struct Net
  base: DistilBert.DistilBert
  split: Params.Split
end

# `weights` are DistilBERT's, in the order of `net.split`; `head` maps the pooled output to the two labels.
struct Model
  net: Net
  weights: List[Tensor.Tensor]
  head: Nn.Dense
end

def run(args: List[String]) -> Result[Void]
  return error("usage: MODEL_DIR DATA_DIR OUT_DIR [cpu|vulkan|xla|torch [ROWS]]") if args.size < 3 or args.size > 5

  rows = try train_rows(args)
  backend = try backend_of(args)
  started = Time.monotonic_ns()
  clock = started
  base = try DistilBert.open(args[0], backend)
  tok = try WordPiece.open("#{args[0]}/tokenizer.json")
  clock = lap("load", clock)

  data = try splits(tok, args[1], rows)
  clock = lap("tokenize", clock)

  start = try initial(base, backend)
  before = try score(start, data.validation)
  puts("before: validation loss #{before.loss}, accuracy #{before.accuracy}")
  clock = lap("before", clock)

  trained = try tuned(start, data.train, args[2])
  clock = lap("train", clock)

  after = try score(trained, data.validation)
  puts("after: validation loss #{after.loss}, accuracy #{after.accuracy}")
  clock = lap("after", clock)

  try round_trip(args[2], trained, after, data.validation)
  clock = lap("round trip", clock)
  puts("seconds: #{(clock - started).to_f / NS}")

  for text in DEMO
    puts("#{text}: #{try predict(trained, tok, text)}")
  end

  ok()
end
```

`Model` holds what `Train.fit_batches` trains: the weights of DistilBERT in the order of `Params.split`, and the head as an `Nn.Dense`. `run` checks its arguments, loads the model and the data, scores the model before the training, trains for one epoch, scores it again, saves it, loads it back and compares, and then predicts. It prints the seconds of each phase (`lap`).

The backend is chosen before anything is loaded. `xla` and `torch` are refused with the op that they lack and its issue: XLA compiles a traced function to StableHLO, whose emitter has no `layer_norm` or `batch_matmul` yet (ml#243), and libtorch has no `batch_matmul` (ml#203). `vulkan` asks for `F32` products, which is what the [Benchmark](#benchmark) runs.

```agentica
def lap(name: String, since: Int) -> Int
  now = Time.monotonic_ns()
  puts("time #{name}: #{(now - since).to_f / NS} s")
  now
end

def train_rows(args: List[String]) -> Result[Int]
  return ok(TRAIN_ROWS) if args.size < 5

  rows = try args[4].to_i
  return error("ROWS is at least BATCH (#{BATCH}), got #{rows}") if rows < BATCH

  ok(rows)
end

def backend_of(args: List[String]) -> Result[Tensor.Backend]
  return ok(Tensor.Backend.Ggml(Tensor.Device.Cpu)) if args.size == 3

  case args[3]
    "cpu" -> ok(Tensor.Backend.Ggml(Tensor.Device.Cpu))
    "vulkan" ->
      try Tensor.vulkan_fp16(false)
      available(Tensor.Backend.Ggml(Tensor.Device.Vulkan(0)), "vulkan")
    "xla" -> error("xla cannot run DistilBERT yet: the StableHLO emitter has no layer_norm or batch_matmul (ml#243)")
    "torch" -> error("torch cannot train DistilBERT yet: it has no batch_matmul (ml#203)")
    _ -> error("the backend is cpu, vulkan, xla or torch, not #{args[3]}")
  end
end

def available(backend: Tensor.Backend, name: String) -> Result[Tensor.Backend]
  return error("the backend #{name} is not available on this machine") if not Tensor.available(backend)

  ok(backend)
end

def splits(tok: WordPiece.WordPiece, dir: String, rows: Int) -> Result[Splits]
  train = try examples(tok, "#{dir}/train.parquet", some(rows))
  validation = try examples(tok, "#{dir}/validation.parquet", none)
  cut = train.select { |e: Example| e.cut }.size + validation.select { |e: Example| e.cut }.size
  puts("rows: #{train.size} train, #{validation.size} validation, truncated: #{cut}")
  ok(Splits(train: train, validation: validation))
end
```

## Train, save and load

`tuned` opens a [metrics](metrics.md) run in the output folder, passes it to `Train.fit_batches` and reads back the `train/step_ms` of every step, to print the first step (which holds the compile of the step) and the median of the others. `round_trip` saves the model, loads it, scores it on the validation split and fails unless its loss and accuracy are those of the trained model to 1e-6.

```agentica
def initial(base: DistilBert.DistilBert, backend: Tensor.Backend) -> Result[Model]
  split = try Params.split(DistilBert.weights(base), ->(_name: String) { true })
  head = try Nn.init_dense(base.config.dim, LABELS.size, SEED, backend)
  ok(Model(net: Net(base: base, split: split), weights: split.trainable, head: head))
end

def tuned(start: Model, train: List[Example], dir: String) -> Result[Model]
  name = "sst2-#{Time.now_ms()}"
  run = try Metrics.open(dir, name, Metrics.Sinks.Jsonl)
  fit = try train_epoch(start, train, run)
  try Metrics.close(run)
  puts("epoch 1: training loss #{fit.history[0]}")
  steps = try step_ms("#{dir}/#{name}/metrics.jsonl")
  puts(steps_line(steps))
  ok(assemble(start.net, fit.params))
end

def steps_line(ms: List[Float]) -> String
  rest = ms.drop(1).sort
  return "steps: first #{ms[0]} ms" if rest.empty?

  "steps: first #{ms[0]} ms, median of the other #{rest.size} #{rest[rest.size / 2]} ms"
end

def round_trip(dir: String, m: Model, after: Score, validation: List[Example]) -> Result[Void]
  try save(dir, m)
  loaded = try load(dir, m.net.base)
  again = try score(loaded, validation)
  if (again.loss - after.loss).abs > TOLERANCE or (again.accuracy - after.accuracy).abs > TOLERANCE
    return error("loaded #{again.loss} and #{again.accuracy}, trained #{after.loss} and #{after.accuracy}")
  end

  puts("loaded: validation loss #{again.loss}, accuracy #{again.accuracy}")
  ok()
end
```

`save` writes two files with `SafetensorsWrite`: `model.safetensors`, whose tensors have DistilBERT's checkpoint names, so it opens like the original, and `classifier.safetensors`, with the head's `weight` and `bias`. `load` reads both back with `Hub` and builds the same `Model`; a missing head tensor is an `error` that names it.

```agentica
# `model.safetensors` has DistilBERT's weights under their checkpoint names.
def save(dir: String, m: Model) -> Result[Void]
  weights = try Params.join(m.net.split, m.weights)
  try SafetensorsWrite.write("#{dir}/model.safetensors", weights)
  bias = try m.head.b.to_result("the head has no bias")
  SafetensorsWrite.write("#{dir}/classifier.safetensors", {"weight": m.head.w, "bias": bias})
end

def load(dir: String, base: DistilBert.DistilBert) -> Result[Model]
  backend = Tensor.backend(base.words.table)
  weights = try Hub.tensors(try Hub.open("#{dir}/model.safetensors"), backend)
  head = try Hub.tensors(try Hub.open("#{dir}/classifier.safetensors"), backend)
  w = try head.get("weight").to_result("#{dir}/classifier.safetensors has no weight")
  b = try head.get("bias").to_result("#{dir}/classifier.safetensors has no bias")
  split = try Params.split(weights, ->(_name: String) { true })
  ok(Model(net: Net(base: base, split: split), weights: split.trainable, head: Nn.Dense(w: w, b: some(b))))
end
```

## Read and tokenize

`Table.read` reads a Parquet (or CSV) file, and `Table.texts` and `Table.floats` give its `sentence` and `label` columns (the labels are numbers, so `floats`; `to_i` makes them class ids). A sentence goes through the checkpoint's WordPiece tokenizer, with the special tokens and a limit of `LENGTH` ids. `example` asks for one id more than the limit, to know whether the sentence was cut, and `run` prints how many were (`truncated`).

```agentica
# The first `limit` rows (all when none) of a table with `sentence` and `label` columns, tokenized.
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
    out = out.push(try example(tok, sentences[i], try labels[i].to_i))
  end

  ok(out)
end

def example(tok: WordPiece.WordPiece, sentence: String, label: Int) -> Result[Example]
  longer = try WordPiece.encode(tok, sentence, none, true, some(LENGTH + 1))
  return ok(Example(ids: longer.ids, label: label, cut: false)) if longer.ids.size <= LENGTH

  encoded = try WordPiece.encode(tok, sentence, none, true, some(LENGTH))
  ok(Example(ids: encoded.ids, label: label, cut: true))
end
```

## Loss, AdamW and the batch

`train_epoch` calls `Train.fit_batches` with a loss, the parameters, a function that builds batch `i`, the number of steps, the optimizer, the number of epochs and the metrics run. The optimizer is AdamW with a learning rate of 5e-5 and a weight decay of 0.01, which applies to every weight but the biases and the layer norms, as in Hugging Face's `Trainer`. The loss is the cross entropy of the head's logits against the labels. The head's input is DistilBERT's output averaged over the real tokens, with the padding masked out (`pad_mask`). A last partial batch of the training rows is left out.

```agentica
def train_epoch(m: Model, rows: List[Example], run: Metrics.Run) -> Result[Train.Fit]
  pad = m.net.base.config.pad_token_id
  backend = Tensor.backend(m.net.base.words.table)
  decay = m.net.split.names.map { |name: String| decays?(name) } + [true, false]
  optimizer = Optim.Optimizer.AdamW(Optim.AdamW(lr: LR, weight_decay: WEIGHT_DECAY, decay: decay))
  Train.fit_batches(
    ->(params: List[Tensor.Tensor], b: Data.Batch) { loss(m.net, params, b) },
    try params(m),
    ->(_epoch: Int, i: Int) { padded(rows.drop(i * BATCH).take(BATCH), pad, backend) },
    rows.size / BATCH,
    optimizer,
    1,
    none,
    some(run),
  )
end

# Hugging Face's rule: no decay on a bias or a layer norm.
def decays?(name: String) -> Bool
  !(name.end_with?(".bias") or name.include?("LayerNorm") or name.include?("layer_norm"))
end

# The `train/step_ms` of each step, in order.
def step_ms(path: String) -> Result[List[Float]]
  out: List[Float] = []
  for line in (try File.read_text(path)).split("\n")
    next if line.empty?

    entry = try JSON.parse(line)
    if try JSON.string(try JSON.get(entry, "name")) == "train/step_ms"
      out = out.push(try JSON.float(try JSON.get(entry, "value")))
    end
  end

  ok(out)
end

def params(m: Model) -> Result[List[Tensor.Tensor]]
  bias = try m.head.b.to_result("the head has no bias")
  ok(m.weights + [m.head.w, bias])
end

def assemble(net: Net, ps: List[Tensor.Tensor]) -> Model
  n = ps.size - HEAD_TENSORS
  Model(net: net, weights: ps.take(n), head: Nn.Dense(w: ps[n], b: some(ps[n + 1])))
end

def loss(net: Net, ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.cross_entropy(try classify(assemble(net, ps), b.inputs), b.targets)
end

# DistilBERT's output averaged over the tokens, through the head: logits `[rows, 2]` of the ids `[rows, LENGTH]`.
def classify(m: Model, ids: Tensor.Tensor) -> Result[Tensor.Tensor]
  base = try DistilBert.with_weights(m.net.base, try Params.join(m.net.split, m.weights))
  mask = try pad_mask(ids, m.net.base.config.pad_token_id)
  Nn.dense(m.head, try DistilBert.pooled(base, ids, mask))
end

# 1.0 for a token and 0.0 for padding.
def pad_mask(ids: Tensor.Tensor, pad: Int) -> Result[Tensor.Tensor]
  backend = Tensor.backend(ids)
  shape = Tensor.shape(ids)
  pads = try Tensor.full(shape, Tensor.Values.Int(pad), Tensor.Dtype.I32, backend)
  zeros = try Tensor.full(shape, Tensor.Values.Float(0.0), Tensor.Dtype.F32, backend)
  ones = try Tensor.full(shape, Tensor.Values.Float(1.0), Tensor.Dtype.F32, backend)
  Tensor.where(try Tensor.eq(ids, pads), zeros, ones)
end
```

`padded` turns some examples into the `I32` ids `[rows, LENGTH]`, padded with the pad id, and the `I32` labels `[rows]`. Every batch has that one shape, so XLA, which compiles once for each shape, would compile the step once. The cost is that a batch is always `LENGTH` tokens wide, where padding to the longest sentence of each batch would make most of the training batches narrower; the same width is used on every backend.

`LENGTH` is 56. The longest sentence in the first 2,000 training rows and in the validation split is 56 and 55 tokens, so nothing is cut there, but 49 of the 67,349 training sentences are longer (up to 66 tokens) and are cut.

```agentica
# Every batch has the shape `[rows, LENGTH]`, so that XLA compiles the step once.
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

`score` runs the validation split in batches of `EVAL_BATCH` through `Trace.eval`, which is how a `Ggml` model runs outside a training loss, and returns the mean cross entropy and the accuracy. The last batch is filled up to `EVAL_BATCH` rows with copies of its first row, which are not counted. `predict` is the same for one sentence, and gives the name of the larger logit.

```agentica
def score(m: Model, rows: List[Example]) -> Result[Score]
  pad = m.net.base.config.pad_token_id
  backend = Tensor.backend(m.net.base.words.table)
  losses: List[Float] = []
  hits: List[Bool] = []
  for i in 0...((rows.size + EVAL_BATCH - 1) / EVAL_BATCH)
    part = rows.drop(i * EVAL_BATCH).take(EVAL_BATCH)
    batch = try padded(part + filler(part), pad, backend)
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

# Copies of the first row that fill a last partial batch to EVAL_BATCH, so that it has the shape of the others.
def filler(part: List[Example]) -> List[Example] = (0...(EVAL_BATCH - part.size)).map { |_i: Int| part[0] }

# Outside a loss a ggml model runs through a trace.
def logits(m: Model, ids: Tensor.Tensor) -> Result[List[Float]]
  ps = try params(m)
  out = try Trace.eval(ps + [ids]) do |xs| classify(assemble(m.net, xs.take(ps.size)), xs[ps.size]) end
  floats(out)
end

def predict(m: Model, tok: WordPiece.WordPiece, text: String) -> Result[String]
  one = try example(tok, text, 0)
  backend = Tensor.backend(m.net.base.words.table)
  batch = try padded([one], m.net.base.config.pad_token_id, backend)
  values = try logits(m, batch.inputs)
  ok(LABELS[predicted(values[0], values[1])])
end

def predicted(z0: Float, z1: Float) -> Int
  return 1 if z1 > z0

  0
end

# The cross entropy of a row with two logits is softplus(wrong - right).
def row_loss(z0: Float, z1: Float, label: Int) -> Float
  return softplus(z1 - z0) if label == 0

  softplus(z0 - z1)
end

def softplus(x: Float) -> Float
  return x + Math.log(1.0 + Math.exp(-x)) if x > 0.0

  Math.log(1.0 + Math.exp(x))
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

From `transformers/`, with `data` from the fetch step and `out` as a new folder for the saved model and the metrics:

```sh
mkdir -p out
agentica tools/finetune_sst2.agc data/distilbert-base-uncased data/sst2 out
```

The optional fourth and fifth arguments are the backend (`cpu`, the default, is ggml on the CPU; `vulkan` is ggml on the first Vulkan device) and the number of training rows. For example, 400 rows on Vulkan:

```sh
mkdir -p out-vulkan
agentica tools/finetune_sst2.agc data/distilbert-base-uncased data/sst2 out-vulkan vulkan 400
```

A backend that is not available on the machine is an `error` that names it. The program prints the number of rows and of truncated ones, the loss and the accuracy on the validation split before the training, the mean training loss of the epoch and the milliseconds of the first step and the median of the others, the same two numbers after, those of the model loaded back from `out`, the seconds of each phase and of the whole run, and the prediction of two sentences. The seconds of the train phase include building every batch on the host (the ids go from lists into tensors in the step loop), not only the steps.

This is a run on the first 2,000 training rows, with a batch of 16 (125 steps), on an AMD Ryzen 5 3550H (4 cores, 8 threads, 12 GB) that other jobs share, with the ggml CPU backend; the times are rough:

```text
before: validation loss 0.7770871823894626, accuracy 0.48853211009174313
epoch 1: training loss 0.38535212469100955
after: validation loss 0.5363854705634189, accuracy 0.7672018348623854
loaded: validation loss 0.5363854705634189, accuracy 0.7672018348623854
seconds: 523.475574741
a gripping, funny movie: positive
a dull and lifeless mess: negative
```

The loss before is that of a head with random weights, and the accuracy is about that of a coin on this balanced split. After the 125 steps it is 0.5364 and 76.7%. The line `loaded` equals the line `after` to 1e-6 because the program fails when they differ, and `out` then holds `model.safetensors`, `classifier.safetensors` and the metrics of the run.

## Benchmark

This is not the run above. It is the same program on the first 400 training rows (25 steps of 16), one epoch, the same seed and settings, and the whole validation split of 872 rows, once for each backend, one after another. The times are rough: the machine, an AMD Ryzen 5 3550H (4 cores, 8 threads, 12 GB) and its Radeon Vega iGPU (RADV RAVEN, memory shared with the CPU), is shared with other jobs. Vulkan multiplies in `F32` (`Tensor.vulkan_fp16(false)`, as the checks of `transformers/` do), not in fp16.

| Backend | Whole run (s) | Train phase (s) | First step (ms) | Median of the other steps (ms) | Before (loss, accuracy) | After (loss, accuracy) | Reloaded (loss, accuracy) |
|---|---|---|---|---|---|---|---|
| ggml CPU | | | | | | | |
| ggml Vulkan | | | | | | | |
| XLA CPU | does not run | | | | | | |
| libtorch CPU | does not run | | | | | | |

Commit SHA of the program: SHA. Date: DATE. Truncated rows: none of the 400 training rows and none of the 872 validation rows. The train phase includes building the batches on the host. The first step holds the compile or the setup of the backend.

`xla` and `torch` do not run the loop, and the program says so before it loads anything: the StableHLO emitter of `tensor/` has no `layer_norm` or `batch_matmul` yet (ml#243), and the libtorch backend has no `batch_matmul` (ml#203).

## Next steps

- More rows or epochs. `TRAIN_ROWS` is 2,000 of the 67,349 rows; a run on the whole split and with a few epochs trains the head and the model further. `Train.fit_batches` takes the number of epochs, and the loss it returns has one entry per epoch. A shuffle, a warmup and dropout are what the usual recipe adds.
- A GPU. The argument `vulkan` runs the same program on the first Vulkan device; the [tensor](../tensor/_index.md) page lists the backends, and the XLA and libtorch ones join the choice when they have the ops of this model.
- Fewer trained weights. [LoRA](_index.md#lora) trains an adapter on the attention projections instead of every weight.
