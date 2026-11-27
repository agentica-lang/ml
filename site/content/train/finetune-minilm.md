---
title: fine-tune all-MiniLM-L6-v2
description: A whole program that fine-tunes a sentence-embedding model on scored sentence pairs from a Hugging Face dataset, measures it before and after, and saves it, on any backend.
weight: 4
---

This page is one program, `transformers/tools/finetune_stsb.agc` of the ml repository, that fine-tunes `sentence-transformers/all-MiniLM-L6-v2` on STS-B. STS-B is a set of sentence pairs, each with a score from 0 (unrelated) to 1 (the same meaning), and the model is a sentence-embedding model: it turns a sentence into a vector, and two sentences that mean the same should have vectors with a high cosine. The program reads the pairs, tokenizes them, cuts them into batches, trains with AdamW and the cosine loss of sentence-transformers, measures Spearman's correlation of the cosines with the scores on a validation sample before, after every epoch and after, saves the model, loads it back and measures again. The training is a [`Loop`](_index.md#loop): a trainer, two metrics and two handlers, run over batches. The backend, the number of training pairs, the epochs, the size of the validation sample and a time budget are arguments; the backend is `cpu`, `vulkan`, `xla` or `torch`. It uses the pieces of the [train](_index.md), [data loader](data.md) and [transformers](../transformers/_index.md) pages.

The code on this page is the program: CI joins the code blocks of the page in order and fails when they differ from the program, ignoring blank lines.

The data is `sentence-transformers/stsb`: 5,749 training pairs and 1,500 validation pairs, with columns `sentence1`, `sentence2` and `score`. The model is `sentence-transformers/all-MiniLM-L6-v2` (6 layers, width 384, 91 MB of weights).

## What it is not

This is the loop, not a recipe for the best score. The program trains on the first rows of a seeded shuffle of the training split, in a new seeded order each epoch, with no gradient clipping and no dropout. The learning rate is 2e-5 with a linear warmup over the first tenth of all the steps and a linear decay to 0 at the last (`Optim.Schedule.WarmupLinear`, the schedule of Hugging Face's `get_linear_schedule_with_warmup`), and the weight decay is 0.01. It measures on a sample of the validation split: the first rows of a seeded shuffle of it, 1,500 (all of it) by default, the same sample before, after every epoch, after and on reload. A run is reproducible from the files and the seed, but not bit for bit: the last digits of a loss change between backends and, on ggml, between runs.

## Fetch

`tools/fetch_stsb.sh DIR` needs the network and `curl`. It downloads the config, the tokenizer and the weights of the model (the pinned files of `tools/fetch_minilm.sh`) and the two Parquet splits of the dataset into `DIR`, at pinned Hugging Face revisions, and checks the SHA-256 of every file with `sha256sum`: a file that is there with the right hash is not fetched again, and a file with another hash is refused. `Table.read` reads the Parquet files as they come.

```sh
cd transformers
sh tools/fetch_stsb.sh data
```

## The program

The imports and the knobs. The program trains on `TRAIN_ROWS` pairs of the training split for `EPOCHS` epochs, in batches of `BATCH` pairs, and measures on `VALIDATION_ROWS` validation pairs (each can be given as an argument, below); it prints the step and its loss every `LOG_EVERY` steps. Every sentence is cut or padded to `LENGTH` ids, the same width for every batch, so that a backend that compiles a step for each shape (XLA) compiles it once. In STS-B the longest sentence is 70 tokens in the training split and 53 in the validation split, so 64 cuts only the longest few of the training split; 32 would cut about 4% of its sentences.

```agentica
import Bert from "../bert"
import Data from "pkg/train/data"
import Loop from "pkg/train/loop"
import Optim from "pkg/train/optim"
import Params from "pkg/train/params"
import SafetensorsWrite from "pkg/hub/safetensors_write"
import Sentence from "../sentence"
import Table from "pkg/train/table"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "pkg/train/train"
import WordPiece from "../tokenizer/wordpiece"

TRAIN_ROWS = 1000
EPOCHS = 1
VALIDATION_ROWS = 1500
LOG_EVERY = 10
SEED = 1
BATCH = 16
EVAL_BATCH = 32
LENGTH = 64
LR = 0.00002
WEIGHT_DECAY = 0.01
TOLERANCE = 0.000001
```

`main` runs `run` and turns an `error` into a message and an exit status. Everything below `private` is the program. `run` reads the arguments (`options_of`), opens the model and the data, measures the model, trains, measures it again, saves it, loads it back and fails unless the loaded model scores what the trained one did. The backend is chosen by its name before anything is loaded, and a backend that is not available on the machine is an `error` that names it. `vulkan` asks for `F32` products (`Tensor.vulkan_fp16(false)`), the exact arithmetic that the checks against Hugging Face run.

`Options` holds the arguments after the three folders: the backend, the number of training pairs, the epochs, the size of the validation sample and, when given, a time budget in seconds. `trainable` says which weights the optimizer changes: all of them, on every backend.

```agentica
def main() -> Void
  case run(System.args())
    ok(_) -> nil
    error(e) ->
      puts("finetune_stsb: #{e.message}")
      System.exit(1)
  end
end

private

# Two sentences as token ids, and how alike they are, from 0 to 1.
struct Pair
  left: List[Int]
  right: List[Int]
  score: Float
end

struct Net
  base: Bert.Bert
  split: Params.Split
end

# On the validation sample: Spearman's correlation of the cosines with the scores, and the mean squared gap between
# them.
struct Score
  spearman: Float
  loss: Float
end

# What the arguments after the three folders say.
struct Options
  backend: String
  rows: Int
  epochs: Int
  validation: Int
  seconds: Optional[Int]
end

def run(args: List[String]) -> Result[Void]
  if args.size < 3 or args.size > 8
    return error("usage: MODEL_DIR DATA_DIR OUT_DIR [cpu|vulkan|xla|torch [ROWS [EPOCHS [VALIDATION [SECONDS]]]]]")
  end

  options = try options_of(args)
  backend = try backend_of(options.backend)
  base = try Bert.open(args[0], backend)
  tok = try WordPiece.open("#{args[0]}/tokenizer.json")
  train = try pairs(tok, "#{args[1]}/train.parquet", options.rows)
  validation = try pairs(tok, "#{args[1]}/validation.parquet", options.validation)
  puts("#{train.size} training pairs, #{validation.size} validation pairs, #{options.epochs} epochs, on #{backend}")

  split = try trainable(base)
  net = Net(base: base, split: split)
  before = try evaluate(net, split.trainable, validation)
  puts("before: Spearman #{before.spearman}, cosine loss #{before.loss}")

  fit = try train_epochs(net, train, validation, options)
  puts("training loss of each epoch: #{fit.history}")
  after = try evaluate(net, fit.params, validation)
  puts("after: Spearman #{after.spearman}, cosine loss #{after.loss}")

  try save(args[2], args[0], net, fit.params)
  again = try reloaded(args[2], backend, validation)
  puts("loaded: Spearman #{again.spearman}, cosine loss #{again.loss}")
  if (again.spearman - after.spearman).abs > TOLERANCE or (again.loss - after.loss).abs > TOLERANCE
    return error("the loaded model gives #{again}, the trained one #{after}")
  end

  ok()
end

def options_of(args: List[String]) -> Result[Options]
  options = Options(
    backend: text_at(args, 3, "cpu"),
    rows: try int_at(args, 4, TRAIN_ROWS),
    epochs: try int_at(args, 5, EPOCHS),
    validation: try int_at(args, 6, VALIDATION_ROWS),
    seconds: none,
  )

  return error("ROWS is at least BATCH (#{BATCH}), got #{options.rows}") if options.rows < BATCH
  return error("EPOCHS is at least 1, got #{options.epochs}") if options.epochs < 1
  return error("VALIDATION is at least 1, got #{options.validation}") if options.validation < 1

  return ok(options.with(seconds: some(try args[7].to_i))) if args.size > 7

  ok(options)
end

def text_at(args: List[String], at: Int, fallback: String) -> String
  return args[at] if args.size > at

  fallback
end

def int_at(args: List[String], at: Int, fallback: Int) -> Result[Int]
  return args[at].to_i if args.size > at

  ok(fallback)
end

# Every weight.
def trainable(base: Bert.Bert) -> Result[Params.Split] = Params.split(Bert.weights(base), ->(_name: String) { true })

def backend_of(name: String) -> Result[Tensor.Backend]
  case name
    "cpu" -> available(Tensor.Backend.Ggml(Tensor.Device.Cpu), name)
    "vulkan" ->
      try Tensor.vulkan_fp16(false)
      available(Tensor.Backend.Ggml(Tensor.Device.Vulkan(0)), name)
    "xla" -> available(Tensor.Backend.Xla(Tensor.Device.Cpu), name)
    "torch" -> available(Tensor.Backend.Torch(Tensor.Device.Cpu), name)
    _ -> error("the backend is cpu, vulkan, xla or torch, not #{name}")
  end
end

def available(backend: Tensor.Backend, name: String) -> Result[Tensor.Backend]
  return error("the backend #{name} is not available on this machine") if !Tensor.available(backend)

  ok(backend)
end
```

## Read and tokenize

`Table.read` reads a Parquet (or CSV) file, and `Table.texts` and `Table.floats` give its columns. The training pairs and the validation sample are the first rows of a seeded shuffle of the whole split (`shuffled` uses the order that a `Data.Loader` of the same seed gives), so they are not the head of a file that may be ordered by genre, and the same number of rows always gives the same pairs. A sentence goes through the checkpoint's WordPiece tokenizer, with the special tokens and a limit of `LENGTH` ids. `padded` fills the rest of a sentence with the model's pad id.

```agentica
# The pairs of a Parquet file with the columns `sentence1`, `sentence2` and `score`: the first `limit` rows of a seeded
# shuffle of the file (all of them when the file has fewer), so the same `limit` always gives the same pairs.
def pairs(tok: WordPiece.WordPiece, path: String, limit: Int) -> Result[List[Pair]]
  table = try Table.read(path)
  left = try Table.texts(table, "sentence1")
  right = try Table.texts(table, "sentence2")
  scores = try Table.floats(table, "score")
  out: List[Pair] = []
  for i in (try shuffled(table.rows, 1)).take(limit)
    out = out.push(Pair(left: try ids(tok, left[i]), right: try ids(tok, right[i]), score: scores[i]))
  end

  ok(out)
end

# The rows `0...count` in the seeded order that a `Data.Loader` of this seed gives its epoch `epoch`.
def shuffled(count: Int, epoch: Int) -> Result[List[Int]]
  rows = try Tensor.zeros([count], Tensor.Dtype.F32)
  loader = Data.Loader(dataset: try Data.dataset(rows, rows), batch_size: BATCH, seed: SEED)
  ok(Data.order(loader, epoch))
end

def ids(tok: WordPiece.WordPiece, text: String) -> Result[List[Int]]
  ok((try WordPiece.encode(tok, text, none, true, some(LENGTH))).ids)
end

# Every sentence is `LENGTH` ids wide, padded with the pad id, so that XLA compiles a step once.
def padded(sentences: List[List[Int]], pad: Int) -> List[Tensor.Values]
  out: List[Tensor.Values] = []
  for s in sentences
    for k in 0...LENGTH
      out = out.push(Tensor.Values.Int(token_at(s, k, pad)))
    end
  end

  out
end

def token_at(ids: List[Int], k: Int, pad: Int) -> Int
  return ids[k] if k < ids.size

  pad
end
```

## Batches, loss and AdamW

A batch is the ids of `n` pairs as an `I32` tensor `[n, 2, LENGTH]` and the `n` scores. `train_epochs` is the whole training as one pipe. `Loop.trainer` takes the loss, the parameters and the optimizer: AdamW (`Optim.adamw`) with the schedule above and a weight decay that skips the biases and the layer norms. `Loop.metric` adds the two numbers that the loop measures at the end of every epoch on the validation sample (`gauge`, below). The first `Loop.handle` prints the step and its loss every `LOG_EVERY` steps, and the second prints the epoch with its metrics: each is a function of a `Loop.Info`, with the types of its arguments inferred. `within` adds a handler that stops the run, with `Loop.stop`, at the first step past the time budget, if there is one: `run_batches` then answers what the run trained so far, without an error. `Loop.run_batches` takes the function that makes batch `i` of epoch `epoch` (`picked` takes the pairs of that batch from the order of the epoch), the number of batches and the epochs. A last partial batch is left out.

The loss sends both sentences of every pair through the model as one batch of `2 n` rows. `embeddings` is the sentence embedding of sentence-transformers: the mean of the token vectors, the padding masked out, then each vector divided by its length. The mask is a constant of the step, made from the ids, and the model is rebuilt from the parameters (`Bert.with_weights`), so that the gradient reaches every weight. The loss is the mean of the squared gap between the cosine of a pair and its score (`Sentence.cosine_similarity_loss`).

```agentica
def batch(m: Bert.Bert, rows: List[Pair]) -> Result[Data.Batch]
  backend = Tensor.backend(m.words.table)
  sentences = rows.flat_map { |p: Pair| [p.left, p.right] }
  inputs = try Tensor.from_list(
    padded(sentences, m.config.pad_token_id),
    [rows.size, 2, LENGTH],
    Tensor.Dtype.I32,
    backend,
  )

  scores = rows.map { |p: Pair| Tensor.Values.Float(p.score) }
  ok(Data.Batch(inputs: inputs, targets: try Tensor.from_list(scores, [rows.size], Tensor.Dtype.F32, backend)))
end

# The training: `Loop` runs `epochs` passes of batches of `BATCH` pairs, in a new seeded order each epoch (a
# last partial
# batch is left out). The schedule warms up over the first tenth of all the steps and decays to 0 at the last. At the
# end of each epoch the validation sample is measured, and every `LOG_EVERY` steps the step and its loss are printed.
def train_epochs(net: Net, rows: List[Pair], validation: List[Pair], options: Options) -> Result[Train.Fit]
  steps = rows.size / BATCH
  total = steps * options.epochs
  orders: List[List[Int]] = []
  for e in 1..options.epochs
    orders = orders.push(try shuffled(rows.size, e))
  end

  decay = net.split.names.map { |name: String| !(name.end_with?(".bias") or name.include?("LayerNorm")) }
  optimizer = Optim.adamw(
    lr: LR,
    weight_decay: WEIGHT_DECAY,
    decay: decay,
    schedule: Optim.Schedule.WarmupLinear(total / 10, total),
  )

  (
    Loop.trainer(->(ps, b) { loss(net, ps, b) }, net.split.trainable, optimizer)
    |> Loop.metric(gauge("spearman", &Loop.spearman, net, validation))
    |> Loop.metric(gauge("cosine loss", &squared_gap, net, validation))
    |> Loop.handle(
      Loop.Event.StepCompleted,
      ->(info) {
        puts("step #{info.step} loss #{info.loss}")
        ok()
      },
      every: LOG_EVERY,
    )
    |> Loop.handle(
      Loop.Event.EpochCompleted,
      ->(info) {
        puts("epoch #{info.epoch}: Spearman #{info.metrics["spearman"]}, cosine loss #{info.metrics["cosine loss"]}")
        ok()
      },
    )
    |> within(options.seconds, Time.monotonic_ms())
    |> Loop.run_batches(
      ->(epoch, i) { batch(net.base, picked(rows, orders[epoch - 1], i)) },
      steps,
      epochs: options.epochs,
    )
  )
end

# With a budget of `seconds`, the trainer that stops after the first step past it, answering what it has trained.
def within(t: Loop.Trainer, seconds: Optional[Int], started: Int) -> Loop.Trainer
  case seconds
    some(s) ->
      (
        t
        |> Loop.handle(
          Loop.Event.StepCompleted,
          ->(info) {
            return Loop.stop(info) if Time.monotonic_ms() - started > s * 1000

            ok()
          },
        )
      )
    none -> t
  end
end

# The pairs of batch `i` of an epoch whose rows are in `order`.
def picked(rows: List[Pair], order: List[Int], i: Int) -> List[Pair]
  order.drop(i * BATCH).take(BATCH).map { |k: Int| rows[k] }
end

# The mean-pooled, normalized embeddings of `ids` `[rows, LENGTH]` under the weights `ps`: padding is masked out.
def embeddings(net: Net, ps: List[Tensor.Tensor], ids: Tensor.Tensor) -> Result[Tensor.Tensor]
  model = try Bert.with_weights(net.base, try Params.join(net.split, ps))
  backend = Tensor.backend(ids)
  shape = Tensor.shape(ids)
  pads = try Tensor.full(shape, Tensor.Values.Int(net.base.config.pad_token_id), Tensor.Dtype.I32, backend)
  ones = try Tensor.full(shape, Tensor.Values.Float(1.0), Tensor.Dtype.F32, backend)
  mask = try Tensor.sub(ones, try Tensor.to_dtype(try Tensor.eq(ids, pads), Tensor.Dtype.F32))
  types = try Tensor.full(shape, Tensor.Values.Int(0), Tensor.Dtype.I32, backend)
  Bert.pooled(model, ids, types, mask, true)
end

# The cosine loss of a batch `[pairs, 2, LENGTH]`: both sentences of every pair go through the model as one batch.
def loss(net: Net, ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  n = Tensor.shape(b.inputs)[0]
  width = net.base.config.hidden_size
  flat = try Tensor.reshape(b.inputs, [2 * n, LENGTH])
  sides = try Tensor.reshape(try embeddings(net, ps, flat), [n, 2, width])
  left = try Tensor.reshape(try Tensor.slice(sides, [0, 0, 0], [n, 1, width]), [n, width])
  right = try Tensor.reshape(try Tensor.slice(sides, [0, 1, 0], [n, 2, width]), [n, width])
  Sentence.cosine_similarity_loss(left, right, b.targets)
end
```

## Measure

A `Loop.Metric` is a number measured on batches of its own: `gauge` makes one from the validation sample, with a `reduce` function. For each batch `cosines` gives one row per pair, the cosine of its two sentences and its score, and `reduce` turns all the rows into the number: `Loop.spearman` is Spearman's correlation of the cosines with the scores (the Pearson correlation of their ranks), and `squared_gap` is the mean squared gap between them, the loss. The embeddings go through `Trace.eval` with the parameters as inputs, so the loop measures the model with the weights of each epoch, and `evaluate` (`Loop.measure`) the model before the training (with its own weights), after it and on reload, on every backend. The last batch of the sample may be shorter than the others.

```agentica
# A number measured on the validation pairs at the end of each epoch: `observe` gives the cosine of each pair and its
# score, in batches of `EVAL_BATCH` pairs, and `reduce` turns all those rows into the number.
def gauge(name: String, reduce: Fn[List[List[Float]], Float], net: Net, rows: List[Pair]) -> Loop.Metric
  Loop.Metric(
    name: name,
    count: (rows.size + EVAL_BATCH - 1) / EVAL_BATCH,
    batch: ->(i) { batch(net.base, rows.drop(i * EVAL_BATCH).take(EVAL_BATCH)) },
    observe: ->(ps, b) { cosines(net, ps, b) },
    reduce: reduce,
  )
end

# The cosine of the two sentences of each pair of a batch under the weights `ps`, with its score: `[cosine, score]`.
def cosines(net: Net, ps: List[Tensor.Tensor], b: Data.Batch) -> Result[List[List[Float]]]
  n = Tensor.shape(b.inputs)[0]
  width = net.base.config.hidden_size
  unit = try Trace.eval(ps + [b.inputs]) do |xs| both_sides(net, xs.take(ps.size), xs[ps.size], n) end
  vectors = try floats(unit)
  scores = try floats(b.targets)
  ok(
    (0...n).map { |i: Int|
      [dot(vectors.drop(2 * i * width).take(width), vectors.drop((2 * i + 1) * width).take(width)), scores[i]]
    },
  )
end

# The embeddings of both sentences of the `n` pairs of `ids` `[n, 2, LENGTH]`, one row each: `[2 n, width]`.
def both_sides(net: Net, ps: List[Tensor.Tensor], ids: Tensor.Tensor, n: Int) -> Result[Tensor.Tensor]
  embeddings(net, ps, try Tensor.reshape(ids, [2 * n, LENGTH]))
end

# The mean squared gap between the cosine and the score.
def squared_gap(rows: List[List[Float]]) -> Float
  rows.map { |row: List[Float]| (row[0] - row[1]) * (row[0] - row[1]) }.sum / rows.size.to_f
end

# The validation measures of the weights `ps`.
def evaluate(net: Net, ps: List[Tensor.Tensor], rows: List[Pair]) -> Result[Score]
  ok(
    Score(
      spearman: try Loop.measure(gauge("spearman", &Loop.spearman, net, rows), ps),
      loss: try Loop.measure(gauge("cosine loss", &squared_gap, net, rows), ps),
    ),
  )
end
```

## Save and load back

`save` writes `model.safetensors` with the tensors under the names of the checkpoint, and copies `config.json` and `tokenizer.json` of the model next to it, so the folder opens like the original: `reloaded` opens it again with `Bert.open` and measures it.

```agentica
# `dir` gets the weights, and the config and the tokenizer of `model_dir`, so that it loads as the original does.
def save(dir: String, model_dir: String, net: Net, ps: List[Tensor.Tensor]) -> Result[Void]
  try SafetensorsWrite.write("#{dir}/model.safetensors", try Params.join(net.split, ps))
  for name in ["config.json", "tokenizer.json"]
    try copy("#{model_dir}/#{name}", "#{dir}/#{name}")
  end

  ok()
end

def copy(from: String, to: String) -> Result[Void]
  text = try File.read_text(from)
  File.open(to, "write") do |f| File.write(f, text) end
end

# The score of the model that `save` wrote, opened from its folder as the original is.
def reloaded(dir: String, backend: Tensor.Backend, rows: List[Pair]) -> Result[Score]
  base = try Bert.open(dir, backend)
  split = try trainable(base)
  evaluate(Net(base: base, split: split), split.trainable, rows)
end
```

The rest is the dot product and the floats of a tensor.

```agentica
def dot(a: List[Float], b: List[Float]) -> Float
  total = 0.0
  for i in 0...a.size
    total = total + a[i] * b[i]
  end

  total
end

def floats(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end
```

## Run it

From `transformers/`, with `data` from the fetch step and `out` as a new folder for the saved model:

```sh
mkdir -p out
agentica --max-steps 1000000000000 tools/finetune_stsb.agc data/all-minilm-l6-v2 data/stsb out
```

The optional arguments after the model, the data and the output folders are, in order, the backend, the number of training pairs, the epochs, the size of the validation sample and a training time budget in seconds. The backend is the only thing that changes between machines:

```sh
agentica --max-steps 1000000000000 tools/finetune_stsb.agc data/all-minilm-l6-v2 data/stsb out cpu 2000 2 1500
agentica --max-steps 1000000000000 tools/finetune_stsb.agc data/all-minilm-l6-v2 data/stsb out vulkan 2000 2 1500
agentica --max-steps 1000000000000 tools/finetune_stsb.agc data/all-minilm-l6-v2 data/stsb out xla 2000 2 1500
agentica --max-steps 1000000000000 tools/finetune_stsb.agc data/all-minilm-l6-v2 data/stsb out torch 2000 2 1500
```

With a last argument the run stops after the first step past that many seconds and goes on to measure, save and reload what it has trained: `... out cpu 2000 2 1500 90` trains for 90 seconds at most. Without it, it runs all the epochs.

`cpu` and `vulkan` are ggml on the CPU and on the first Vulkan device, `xla` is XLA on the CPU (`tensor/tools/install_xla.agc` installs its plugin) and `torch` is libtorch on the CPU (`tensor/tools/build-torch.sh`). A backend that is not on the machine is an `error` that names it.

## What it prints

`N training pairs, M validation pairs, E epochs, on BACKEND`, then `before:` with Spearman's correlation and the cosine loss of the model as it comes, a line `step S loss L` every `LOG_EVERY` steps, a line `epoch E: Spearman ..., cosine loss ...` at the end of every epoch, `training loss of each epoch: [...]`, and `after:` and `loaded:` with the same two numbers. `loaded` equals `after` to 1e-6 because the program fails when they differ, and `out` then holds `model.safetensors`, `config.json` and `tokenizer.json`. The numbers depend on the arguments, the backend and the machine, so this page gives none: they are the output of your run.

## Next steps

- More rows or epochs: the second and third arguments. The loss of the run has one entry per epoch, and with more steps the schedule has more to follow.
- Another dataset. Any table with two text columns and a score from 0 to 1 fits the loop; `Table.read` reads a CSV file too.
- Fewer trained weights. [LoRA](_index.md#lora) trains an adapter on the attention projections instead of every weight.
