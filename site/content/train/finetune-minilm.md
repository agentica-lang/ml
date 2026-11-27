---
title: fine-tune all-MiniLM-L6-v2
description: A whole program that fine-tunes a sentence-embedding model on scored sentence pairs from a Hugging Face dataset, measures it before and after, and saves it, on any backend.
weight: 4
---

This page is one program, `transformers/tools/finetune_stsb.agc` of the ml repository, that fine-tunes `sentence-transformers/all-MiniLM-L6-v2` on STS-B. STS-B is a set of sentence pairs, each with a score from 0 (unrelated) to 1 (the same meaning), and the model is a sentence-embedding model: it turns a sentence into a vector, and two sentences that mean the same should have vectors with a high cosine. The program reads the pairs, tokenizes them, cuts them into batches, trains with AdamW and the cosine loss of sentence-transformers, measures Spearman's correlation of the cosines with the scores before and after, saves the weights, loads them back and measures again. The backend is one argument: `cpu`, `vulkan`, `xla` or `torch`. It uses the pieces of the [train](_index.md), [data loader](data.md) and [transformers](../transformers/_index.md) pages.

The data is `sentence-transformers/stsb`: 5,749 training pairs and 1,500 validation pairs, with columns `sentence1`, `sentence2` and `score`. The model is `sentence-transformers/all-MiniLM-L6-v2` (6 layers, width 384, 91 MB of weights).

## What it is not

This is the loop, not a recipe for the best score. There is no shuffle (the rows are read in file order), no warmup, no gradient clipping and no dropout, the program trains one epoch on the first rows of the training split, and it measures on the first 500 validation pairs. A run is reproducible from the files, but not bit for bit: the last digits of a loss change between backends and, on ggml, between runs.

## Fetch

`tools/fetch_stsb.sh DIR` needs the network and `curl`. It downloads the config, the tokenizer and the weights of the model (the pinned files of `tools/fetch_minilm.sh`) and the two Parquet splits of the dataset into `DIR`, at pinned Hugging Face revisions, and checks the SHA-256 of every file with `sha256sum`: a file that is there with the right hash is not fetched again, and a file with another hash is refused. `Table.read` reads the Parquet files as they come.

```sh
cd transformers
sh tools/fetch_stsb.sh data
```

## The program

The imports and the knobs. The program trains on the first `TRAIN_ROWS` pairs of the training split (or the number given as its last argument, at least `BATCH`), for one epoch, in batches of `BATCH` pairs. Every sentence is cut or padded to `LENGTH` ids, the same width for every batch, so that a backend that compiles a step for each shape (XLA) compiles it once.

```agentica
import Bert from "../bert"
import Data from "pkg/train/data"
import Hub from "pkg/hub/hub"
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
VALID_ROWS = 500
BATCH = 16
EVAL_BATCH = 32
LENGTH = 64
LR = 0.00002
WEIGHT_DECAY = 0.01
TOLERANCE = 0.000001
```

`main` runs `run` and turns an `error` into a message and an exit status. Everything below `private` is the program.

```agentica
def main() -> Void
  case run(System.args())
    ok(_) -> nil
    error(e) ->
      puts("finetune_stsb: #{e.message}")
      System.exit(1)
  end
end
```

`run` reads the arguments, opens the model and the data, measures the model, trains for one epoch, measures it again, saves it, loads it back and fails unless the loaded model scores what the trained one did. The backend is chosen by its name before anything is loaded, and a backend that is not available on the machine is an `error` that names it. `vulkan` asks for `F32` products (`Tensor.vulkan_fp16(false)`), the exact arithmetic that the checks against Hugging Face run.

```agentica
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

# On the validation pairs: Spearman's correlation of the cosines with the scores, and the mean squared gap
# between them.
struct Score
  spearman: Float
  loss: Float
end

def run(args: List[String]) -> Result[Void]
  return error("usage: MODEL_DIR DATA_DIR OUT_DIR [cpu|vulkan|xla|torch [ROWS]]") if args.size < 3 or args.size > 5

  name = "cpu"
  if args.size > 3
    name = args[3]
  end

  rows = TRAIN_ROWS
  if args.size > 4
    rows = try args[4].to_i
  end

  backend = try backend_of(name)
  return error("ROWS is at least BATCH (#{BATCH}), got #{rows}") if rows < BATCH

  base = try Bert.open(args[0], backend)
  tok = try WordPiece.open("#{args[0]}/tokenizer.json")
  train = try pairs(tok, "#{args[1]}/train.parquet", rows)
  validation = try pairs(tok, "#{args[1]}/validation.parquet", VALID_ROWS)
  puts("#{train.size} training pairs, #{validation.size} validation pairs, on #{backend}")

  split = try trainable(base, backend)
  net = Net(base: base, split: split)
  before = try evaluate(net, split.trainable, validation)
  puts("before: Spearman #{before.spearman}, cosine loss #{before.loss}")

  fit = try train_epoch(net, train)
  puts("training loss #{fit.history[0]}")
  after = try evaluate(net, fit.params, validation)
  puts("after: Spearman #{after.spearman}, cosine loss #{after.loss}")

  try save(args[2], net, fit.params)
  again = try reloaded(args[2], base.config, backend, validation)
  puts("loaded: Spearman #{again.spearman}, cosine loss #{again.loss}")
  if (again.spearman - after.spearman).abs > TOLERANCE or (again.loss - after.loss).abs > TOLERANCE
    return error("the loaded model gives #{again}, the trained one #{after}")
  end

  ok()
end

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

`trainable` says which weights the optimizer changes: all of them, except on libtorch the word table, which holds 12 of the 22 million weights; a step that trains it needs more than 2.5 GB there ([ml#394](https://github.com/agentica-lang/ml/issues/394)).

```agentica
# Every weight, but on libtorch the word table: a step that trains it holds more than 2.5 GB there (ml#394).
def trainable(base: Bert.Bert, backend: Tensor.Backend) -> Result[Params.Split]
  frozen = backend == Tensor.Backend.Torch(Tensor.Device.Cpu)
  Params.split(Bert.weights(base), ->(name: String) { !(frozen and name.end_with?("word_embeddings.weight")) })
end
```

## Read and tokenize

`Table.read` reads a Parquet (or CSV) file, and `Table.texts` and `Table.floats` give its columns. A sentence goes through the checkpoint's WordPiece tokenizer, with the special tokens and a limit of `LENGTH` ids (the longest sentences are cut). `padded` fills the rest of a sentence with the model's pad id.

```agentica
# The first `limit` rows of a Parquet file with the columns `sentence1`, `sentence2` and `score`.
def pairs(tok: WordPiece.WordPiece, path: String, limit: Int) -> Result[List[Pair]]
  table = try Table.read(path)
  left = try Table.texts(table, "sentence1")
  right = try Table.texts(table, "sentence2")
  scores = try Table.floats(table, "score")
  out: List[Pair] = []
  for i in 0...[limit, table.rows].min.unwrap!
    out = out.push(Pair(left: try ids(tok, left[i]), right: try ids(tok, right[i]), score: scores[i]))
  end

  ok(out)
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

A batch is the ids of `n` pairs as an `I32` tensor `[n, 2, LENGTH]` and the `n` scores. `Train.fit_batches` takes a loss, the parameters, a function that makes batch `i`, the number of batches, the optimizer and the number of epochs. The optimizer is AdamW with a learning rate of 2e-5 and a weight decay of 0.01 on every weight but the biases and the layer norms, as in Hugging Face's `Trainer`. A last partial batch is left out.

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

def train_epoch(net: Net, rows: List[Pair]) -> Result[Train.Fit]
  decay = net.split.names.map { |name: String| !(name.end_with?(".bias") or name.include?("LayerNorm")) }
  Train.fit_batches(
    ->(ps: List[Tensor.Tensor], b: Data.Batch) { loss(net, ps, b) },
    net.split.trainable,
    ->(_epoch: Int, i: Int) { batch(net.base, rows.drop(i * BATCH).take(BATCH)) },
    rows.size / BATCH,
    Optim.Optimizer.AdamW(Optim.AdamW(lr: LR, weight_decay: WEIGHT_DECAY, decay: decay)),
    1,
  )
end
```

The loss sends both sentences of every pair through the model as one batch of `2 n` rows. `embeddings` is the sentence embedding of sentence-transformers: the mean of the token vectors, the padding masked out, then each vector divided by its length. The mask is a constant of the step, made from the ids, and the model is rebuilt from the parameters (`Bert.with_weights`), so that the gradient reaches every weight. The loss is the mean of the squared gap between the cosine of a pair and its score (`Sentence.cosine_similarity_loss`).

```agentica
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

`evaluate` embeds the sentences of the validation pairs, a fixed number at a time, takes the cosine of each pair, and returns Spearman's correlation of the cosines with the scores (the Pearson correlation of their ranks) and the mean squared gap between the cosines and the scores, which is the loss. The embeddings go through `Trace.eval` with the parameters as inputs, so the same function measures the model before the training (with its own weights) and after it (with the trained ones), on every backend. A last short batch repeats its first sentence to keep the shape.

```agentica
# The cosine of the two sentences of each pair against its score.
def evaluate(net: Net, ps: List[Tensor.Tensor], rows: List[Pair]) -> Result[Score]
  vectors = try vectors_of(net, ps, rows.flat_map { |p: Pair| [p.left, p.right] })
  cosines = (0...rows.size).map { |i: Int| dot(vectors[2 * i], vectors[2 * i + 1]) }
  scores = rows.map { |p: Pair| p.score }
  gaps = (0...rows.size).map { |i: Int| (cosines[i] - scores[i]) * (cosines[i] - scores[i]) }
  ok(Score(spearman: pearson(ranks(cosines), ranks(scores)), loss: gaps.sum / rows.size.to_f))
end

# The unit embedding of each sentence, `EVAL_BATCH` pairs at a time; a last short batch repeats its first sentence.
def vectors_of(net: Net, ps: List[Tensor.Tensor], sentences: List[List[Int]]) -> Result[List[List[Float]]]
  width = net.base.config.hidden_size
  size = 2 * EVAL_BATCH
  backend = Tensor.backend(net.base.words.table)
  out: List[List[Float]] = []
  at = 0
  while at < sentences.size
    part = sentences.drop(at).take(size)
    full = part + List.repeat(part[0], size - part.size)
    ids = try Tensor.from_list(padded(full, net.base.config.pad_token_id), [size, LENGTH], Tensor.Dtype.I32, backend)
    unit = try Trace.eval(ps + [ids]) do |xs| embeddings(net, xs.take(ps.size), xs[ps.size]) end
    flat = try floats(unit)
    for i in 0...part.size
      out = out.push(flat.drop(i * width).take(width))
    end

    at = at + size
  end

  ok(out)
end

def dot(a: List[Float], b: List[Float]) -> Float
  total = 0.0
  for i in 0...a.size
    total = total + a[i] * b[i]
  end

  total
end

# The rank of each value from 1, equal values sharing the mean of their ranks.
def ranks(values: List[Float]) -> List[Float]
  order = (0...values.size).to_a.sort_by { |i: Int| values[i] }
  out = List.repeat(0.0, values.size)
  at = 0
  while at < order.size
    stop = at
    while stop + 1 < order.size and values[order[stop + 1]] == values[order[at]]
      stop = stop + 1
    end

    shared = (at + stop).to_f / 2.0 + 1.0
    for k in at..stop
      out = out.set(order[k], shared)
    end

    at = stop + 1
  end

  out
end

def pearson(a: List[Float], b: List[Float]) -> Float
  n = a.size.to_f
  mean_a = a.sum / n
  mean_b = b.sum / n
  both = 0.0
  aa = 0.0
  bb = 0.0
  for i in 0...a.size
    both = both + (a[i] - mean_a) * (b[i] - mean_b)
    aa = aa + (a[i] - mean_a) * (a[i] - mean_a)
    bb = bb + (b[i] - mean_b) * (b[i] - mean_b)
  end

  both / Math.sqrt(aa * bb)
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

## Save and load back

`save` writes `model.safetensors` with the tensors under the names of the checkpoint, so the file opens like the original. `reloaded` opens it again with `Bert.load`, with the config of the model, and measures it.

```agentica
def save(dir: String, net: Net, ps: List[Tensor.Tensor]) -> Result[Void]
  SafetensorsWrite.write("#{dir}/model.safetensors", try Params.join(net.split, ps))
end

# The score of the model that `save` wrote.
def reloaded(dir: String, config: Bert.Config, backend: Tensor.Backend, rows: List[Pair]) -> Result[Score]
  base = try Bert.load(try Hub.open("#{dir}/model.safetensors"), config, backend)
  split = try trainable(base, backend)
  evaluate(Net(base: base, split: split), split.trainable, rows)
end
```

## Run it

From `transformers/`, with `data` from the fetch step and `out` as a new folder for the saved model:

```sh
mkdir -p out
agentica --max-steps 1000000000000 tools/finetune_stsb.agc data/all-minilm-l6-v2 data/stsb out
```

The optional fourth and fifth arguments are the backend and the number of training pairs. The backend is the only thing that changes:

```sh
agentica --max-steps 1000000000000 tools/finetune_stsb.agc data/all-minilm-l6-v2 data/stsb out cpu 1000
agentica --max-steps 1000000000000 tools/finetune_stsb.agc data/all-minilm-l6-v2 data/stsb out vulkan 1000
agentica --max-steps 1000000000000 tools/finetune_stsb.agc data/all-minilm-l6-v2 data/stsb out xla 1000
agentica --max-steps 1000000000000 tools/finetune_stsb.agc data/all-minilm-l6-v2 data/stsb out torch 1000
```

`cpu` and `vulkan` are ggml on the CPU and on the first Vulkan device, `xla` is XLA on the CPU (`tensor/tools/install_xla.agc` installs its plugin) and `torch` is libtorch on the CPU (`tensor/tools/build-torch.sh`). A backend that is not on the machine is an `error` that names it.

## What it prints

A run on the first 1,000 training pairs (62 steps of 16) with the ggml CPU backend, scored on the first 500 validation pairs:

```text
1000 training pairs, 500 validation pairs, on Backend.Ggml(Device.Cpu)
before: Spearman 0.925684, cosine loss 0.020719
training loss 0.017299
after: Spearman 0.906469, cosine loss 0.019092
loaded: Spearman 0.906469, cosine loss 0.019092
```

The loss on the validation pairs falls from 0.0207 to 0.0191: the cosines of the trained model are nearer the scores. Spearman's correlation, which looks only at the order of the cosines, goes from 0.9257 to 0.9065, a little down. The model was trained on over a billion sentence pairs, so its ordering was good to begin with, and a thousand pairs for one epoch adjust the scale of its cosines more than their order. `loaded` equals `after` to 1e-6 because the program fails when they differ, and `out` then holds `model.safetensors`.

## One argument for the backend

The same program on the first 160 training pairs (10 steps), once for each backend. The quality numbers agree to the digits that float arithmetic leaves, which is the point of the page: the loop does not change with the backend.

| Backend | Spearman before | Spearman after | Cosine loss before | Cosine loss after |
|---|---|---|---|---|
| ggml CPU | 0.925684 | 0.917515 | 0.020720 | 0.017936 |
| ggml Vulkan | 0.925684 | 0.917515 | 0.020720 | 0.017936 |
| XLA CPU | 0.925684 | 0.917515 | 0.020720 | 0.017936 |
| libtorch CPU | not finished in 2.5 GB | | | |

On libtorch the word table stays frozen, and the run did not finish in a 2.5 GB cap: its measurement of the model before the training was killed by the cap, so there is no line for it. The libtorch memory of a step and of `Trace.eval` is tracked in [ml#394](https://github.com/agentica-lang/ml/issues/394); until it is fixed, `torch` needs a larger cap than the other backends.

## Next steps

- More rows or epochs. `Train.fit_batches` takes the number of epochs, and the loss it returns has one entry per epoch; a shuffle, a warmup and dropout are what the usual recipe adds. With more rows the loss has more steps to follow the scores.
- Another dataset. Any table with two text columns and a score from 0 to 1 fits the loop; `Table.read` reads a CSV file too.
- Fewer trained weights. [LoRA](_index.md#lora) trains an adapter on the attention projections instead of every weight.
