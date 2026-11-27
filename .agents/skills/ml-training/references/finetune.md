# Freezing, LoRA, fine-tuning a Hub model

## Train part of a model

`Params` splits a model's weights (a `Map[String, Tensor]` by name, such as `Qwen3.weights(m)` or
`DistilBert.weights(m)` gives) into the trainable ones and the frozen ones. `fit` trains the list `split.trainable`;
inside the loss `Params.join(split, ps)` puts the traced tensors back by name, and the frozen weights enter the trace as
constants, so they get no gradient and the optimizer keeps no state for them.

```agentica
import Data from "pkg/train/data"
import Loss from "pkg/train/loss"
import Optim from "pkg/train/optim"
import Params from "pkg/train/params"
import Tensor from "pkg/tensor/tensor"
import Train from "pkg/train/train"

def loss(split: Params.Split, ps: List[Tensor.Tensor], batch: Data.Batch) -> Result[Tensor.Tensor]
  w = try Params.join(split, ps)
  hidden = try Tensor.matmul(batch.inputs, w["first.weight"])
  Loss.mse(try Tensor.matmul(hidden, w["second.weight"]), batch.targets)
end

def main() -> Void
  a = Tensor.floats([1.0, 0.0, 0.0, 1.0], [2, 2]).unwrap!
  b = Tensor.floats([0.0, 0.0, 0.0, 0.0], [2, 2]).unwrap!
  weights = {"first.weight": a, "second.weight": b}
  split = Params.split(weights, ->(name: String) { name.start_with?("second.") }).unwrap!
  puts(split.names)

  inputs = Tensor.floats([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], [3, 2]).unwrap!
  targets = Tensor.floats([2.0, 0.0, 0.0, 2.0, 2.0, 2.0], [3, 2]).unwrap!
  loader = Data.Loader(dataset: Data.dataset(inputs, targets).unwrap!, batch_size: 3, shuffle: false)
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1))
  fit = Train.fit(
    ->(ps: List[Tensor.Tensor], batch: Data.Batch) { loss(split, ps, batch) },
    split.trainable,
    loader,
    sgd,
    50,
  ).unwrap!

  puts(fit.history[49] < fit.history[0])
  trained = Params.named(split, fit.params).unwrap!
  puts(trained.keys)
end
```

```output
["second.weight"]
true
["second.weight"]
```

- `Params.split(weights, trainable)` answers `names` (sorted), `trainable` (their tensors, the list `fit` takes) and
  `frozen`. A selection that matches no name is an error.
- `Params.named(split, ps)` gives the trained weights alone by name: what to save with `SafetensorsWrite.write`.
- `Params.update(weights, changed)` puts tensors in place of the ones of the same name: load a saved part with
  `Params.update(Qwen3.weights(m), try Hub.tensors(try Hub.open(path), backend))`, then `Qwen3.with_weights`.
- With `AdamW`, `decay` follows `split.names`. Hugging Face's rule leaves out biases and layer norms:
  `decay = split.names.map { |n: String| !(n.end_with?(".bias") or n.include?("LayerNorm") or n.include?("layer_norm")) }`.

## LoRA

A frozen weight `W` of a linear layer (`[out, in]`) gets two small matrices, `A` `[r, in]` and `B` `[out, r]`, and the
layer computes with `W + (alpha / r) B A`. `B` starts at zero, so a new adapter changes nothing. Only `A` and `B` train.

| Call | Does |
|---|---|
| `Lora.Config(targets:, r: 8, alpha: 16.0, dropout: 0.0)` | `targets` name layers as PEFT's `target_modules` do: `q_proj` takes every `...self_attn.q_proj` |
| `Lora.init(weights, config, seed, backend)` | a new adapter: `A` uniform, `B` zero |
| `Lora.params(adapter)` | the list `fit` trains (`A` then `B` of each adapted weight) |
| `Lora.with_params(adapter, ps)` | the adapter with those tensors: inside the loss, and after `fit` |
| `Lora.applied(weights, adapter)` | every weight by name, the adapted ones as `W + (alpha / r) B A`: the training forward |
| `Lora.merged(weights, adapter)` | the same, computed once, for inference (any backend, ggml included) |
| `Lora.save(dir, adapter, embeddings)` and `Lora.open(dir, backend)` | PEFT's files: `adapter_config.json` and `adapter_model.safetensors` |

This program trains an adapter on a tiny weight until it matches a target, merges it, then saves and reopens it:

```agentica
import Data from "pkg/train/data"
import Loss from "pkg/train/loss"
import Lora from "pkg/train/lora"
import Optim from "pkg/train/optim"
import Tensor from "pkg/tensor/tensor"
import Train from "pkg/train/train"

NAME = "layer.q_proj.weight"

def apply(w: Tensor.Tensor, x: Tensor.Tensor) -> Result[Tensor.Tensor] = Tensor.matmul(x, try Tensor.transpose(w))

def loss(
  frozen: Map[String, Tensor.Tensor],
  adapter: Lora.Adapter,
  ps: List[Tensor.Tensor],
  b: Data.Batch,
) -> Result[Tensor.Tensor]
  weights = try Lora.applied(frozen, try Lora.with_params(adapter, ps))
  Loss.mse(try apply(weights[NAME], b.inputs), b.targets)
end

def floats(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("floats only")
    end
  end

  ok(out)
end

def main() -> Void
  backend = Tensor.Backend.Reference
  frozen = {NAME: Tensor.floats([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 3]).unwrap!}
  adapter = Lora.init(frozen, Lora.Config(targets: ["q_proj"], r: 2, alpha: 2.0), 1, backend).unwrap!
  x = Tensor.floats([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [4, 3]).unwrap!
  y = Tensor.floats([1.0, 0.0, 0.0, 1.0, 0.5, -0.5, 1.5, 0.5], [4, 2]).unwrap!
  loader = Data.Loader(dataset: Data.dataset(x, y).unwrap!, batch_size: 4, shuffle: false)
  adam = Optim.Optimizer.Adam(Optim.Adam(lr: 0.05))
  fit = Train.fit(
    ->(ps: List[Tensor.Tensor], b: Data.Batch) { loss(frozen, adapter, ps, b) },
    Lora.params(adapter).unwrap!,
    loader,
    adam,
    200,
  ).unwrap!

  puts(fit.history[199] < 0.01)

  trained = Lora.with_params(adapter, fit.params).unwrap!
  merged = Lora.merged(frozen, trained).unwrap!
  before = floats(merged[NAME]).unwrap!

  dir = "/tmp/ml-skill-lora-#{Random.int(0, 999999999)}"
  Dir.mkdir_private(dir, 5000).unwrap!
  Lora.save(dir, trained).unwrap!
  reopened = Lora.open(dir, backend).unwrap!
  after = floats(Lora.merged(frozen, reopened).unwrap![NAME]).unwrap!
  gap = 0.0
  for i in 0...before.size
    gap = [gap, (before[i] - after[i]).abs].max.unwrap!
  end

  puts(gap < 0.00001)
  Path.remove(dir, 5000).unwrap!
end
```

```output
true
true
```

- A fresh `Lora.init` with the same `seed` gives the same `A`. The reference and ggml draw the same values.
- `Lora.params` refuses an adapter with `dropout` above 0: LoRA trains without dropout here.
- `Lora.merged` shares the tensors of weights it does not adapt, and makes new ones for the adapted weights. Both must
  be on one backend. The merged model runs at the cost of the model alone.
- `Lora.save` writes every adapted weight as a linear layer's. PEFT keeps an embedding's adapter in another layout, so
  pass the model's embedding tables as the third argument (`Qwen3.embeddings(m)`, `DistilBert.embeddings(m)`): an adapter
  on one of them is then an error instead of a file PEFT would misread.
- `Lora.open` reads PEFT's files as `F32`. An adapter of another kind (DoRA, rsLoRA, `fan_in_fan_out`, a trained bias,
  `modules_to_save`, a regular-expression `target_modules`, an adapter of an embedding) is an error that names it.
- On Qwen3 the output head is tied to the embedding table, so an adapter on the table also changes the head here.

## Fine-tune DistilBERT for a classification task

A whole program that fine-tunes `distilbert-base-uncased` on SST-2 (a linear head on the mean-pooled output, AdamW, saving and loading the weights) is at `https://www.agentica-lang.com/ml/train/finetune-distilbert/`, with the pinned downloads and the run's numbers.

## Fine-tune DistilBERT with LoRA

The recipe for a Hub model: load it as `F32`, take its weights by name, make an adapter on its attention projections,
write the loss on the model rebuilt from `Lora.applied`, train with `fit_batches`, merge, and save for PEFT. This
trains a sentence-embedding model on scored pairs with the cosine loss of sentence-transformers. Get
`sentence-transformers/msmarco-distilbert-cos-v5` into a folder first (`ml-inference`, `references/models.md`).

```agentica
import Data from "pkg/train/data"
import DistilBert from "pkg/transformers/distilbert"
import Lora from "pkg/train/lora"
import Optim from "pkg/train/optim"
import Sentence from "pkg/transformers/sentence"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "pkg/train/train"
import WordPiece from "pkg/transformers/tokenizer/wordpiece"

struct Pair
  left: List[Int]
  right: List[Int]
  score: Float
end

def pairs(tok: WordPiece.WordPiece) -> Result[List[Pair]]
  rows = [
    ["A man is eating food.", "A man is eating a meal.", "0.95"],
    ["A plane is taking off.", "An air plane is taking off.", "1.0"],
    ["A man plays a flute.", "A woman is slicing an onion.", "0.0"],
    ["Two dogs run in a field.", "Dogs are running outside.", "0.8"],
    ["A girl is playing a guitar.", "A woman plays the guitar.", "0.85"],
    ["A man is riding a horse.", "A cat is sleeping on a sofa.", "0.0"],
  ]

  out: List[Pair] = []
  for r in rows
    left = (try WordPiece.encode(tok, r[0])).ids
    right = (try WordPiece.encode(tok, r[1])).ids
    out = out.push(Pair(left: left, right: right, score: try r[2].to_f))
  end

  ok(out)
end

def token(ids: List[Int], k: Int, pad: Int) -> Int
  return ids[k] if k < ids.size

  pad
end

def batch(rows: List[Pair], pad: Int, backend: Tensor.Backend) -> Result[Data.Batch]
  length = rows.reduce(1) do |acc: Int, p: Pair| [acc, p.left.size, p.right.size].max.unwrap! end
  ids: List[Tensor.Values] = []
  for p in rows
    for side in [p.left, p.right]
      for k in 0...length
        ids = ids.push(Tensor.Values.Int(token(side, k, pad)))
      end
    end
  end

  inputs = try Tensor.from_list(ids, [rows.size, 2, length], Tensor.Dtype.I32, backend)
  scores = rows.map { |p: Pair| Tensor.Values.Float(p.score) }
  ok(Data.Batch(inputs: inputs, targets: try Tensor.from_list(scores, [rows.size], Tensor.Dtype.F32, backend)))
end

def loss(m: DistilBert.DistilBert, weights: Map[String, Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  model = try DistilBert.with_weights(m, weights)
  n = Tensor.shape(b.inputs)[0]
  length = Tensor.shape(b.inputs)[2]
  width = m.config.dim
  ids = try Tensor.reshape(b.inputs, [2 * n, length])
  backend = Tensor.backend(ids)
  pads = try Tensor.full([2 * n, length], Tensor.Values.Int(m.config.pad_token_id), Tensor.Dtype.I32, backend)
  ones = try Tensor.full([2 * n, length], Tensor.Values.Float(1.0), Tensor.Dtype.F32, backend)
  mask = try Tensor.sub(ones, try Tensor.to_dtype(try Tensor.eq(ids, pads), Tensor.Dtype.F32))
  sides = try Tensor.reshape(try DistilBert.pooled(model, ids, mask), [n, 2, width])
  left = try Tensor.reshape(try Tensor.slice(sides, [0, 0, 0], [n, 1, width]), [n, width])
  right = try Tensor.reshape(try Tensor.slice(sides, [0, 1, 0], [n, 2, width]), [n, width])
  Sentence.cosine_similarity_loss(left, right, b.targets)
end

def validated(
  m: DistilBert.DistilBert,
  weights: Map[String, Tensor.Tensor],
  adapter: Lora.Adapter,
  ps: List[Tensor.Tensor],
  b: Data.Batch,
) -> Result[Float]
  n = ps.size
  out = try Trace.eval(ps + [b.inputs, b.targets]) do |xs|
    loss(
      m,
      try Lora.applied(weights, try Lora.with_params(adapter, xs.take(n))),
      Data.Batch(inputs: xs[n], targets: xs[n + 1]),
    )
  end

  case (try Tensor.to_list(out))[0]
    Tensor.Values.Float(f) -> ok(f)
    _ -> error("a loss is a float")
  end
end

def main() -> Void
  args = System.args()
  if args.size != 1
    puts("usage: app.agc MODEL_DIR")
    System.exit(2)
  end

  backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  dir = args[0]
  tok = WordPiece.open("#{dir}/tokenizer.json").unwrap!
  m = DistilBert.open(dir, backend).unwrap!
  pad = m.config.pad_token_id
  all = pairs(tok).unwrap!
  rows = all.take(4)
  held = batch(all.drop(4), pad, backend).unwrap!

  weights = DistilBert.weights(m)
  config = Lora.Config(targets: ["q_lin", "v_lin"], r: 8, alpha: 16.0)
  adapter = Lora.init(weights, config, 1, backend).unwrap!
  puts("validation loss before: #{validated(m, weights, adapter, Lora.params(adapter).unwrap!, held).unwrap!}")
  fit = Train.fit_batches(
    ->(ps: List[Tensor.Tensor], b: Data.Batch) {
      loss(m, try Lora.applied(weights, try Lora.with_params(adapter, ps)), b)
    },
    Lora.params(adapter).unwrap!,
    ->(_e: Int, _i: Int) { batch(rows, pad, backend) },
    1,
    Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.0002, weight_decay: 0.01)),
    10,
    some(->(e: Train.Epoch) { puts("step #{e.epoch}: #{e.loss}") }),
  ).unwrap!

  puts("validation loss after: #{validated(m, weights, adapter, fit.params, held).unwrap!}")
  trained = Lora.with_params(adapter, fit.params).unwrap!
  Lora.save("adapter", trained, DistilBert.embeddings(m)).unwrap!
  puts("saved adapter/adapter_config.json and adapter/adapter_model.safetensors")
  DistilBert.close_model(m)
end
```

- Each "epoch" here is one batch (`count` 1, `epochs` 10): `on_epoch` then prints one loss per step. With real data,
  make `count` batches of an epoch and `batch(epoch, i)` answers batch `i`.
- Pad each batch to the longest sentence, rounded up to a multiple of 8 when lengths vary: on ggml every new shape is a
  compile of about half a second.
- Run it as `agentica app.agc msmarco-distilbert-cos-v5`. About 1.0 s per step at batch 16 pairs on a 4-core CPU.
- `validated` is the check after the fit: the loss of the trained adapter on pairs it did not train on, as
  `references/fit.md` does for a plain `fit`. Build the batch outside the trace: `Tensor.reshape` does not run eagerly
  on ggml.
- After training, the model with the adapter merged in is
  `DistilBert.with_weights(m, Lora.merged(weights, trained).unwrap!)`; `DistilBert.embed` then runs it at the cost of
  the model alone.
- Load the saved adapter in PyTorch: `PeftModel.from_pretrained(base, "adapter")` (`verify.md`). The files do not say
  which model the adapter is for (`base_model_name_or_path` in `adapter_config.json` is `null`), so give PEFT the base
  model yourself.

## Use an adapter for Qwen3

A PEFT adapter of a `Qwen3ForCausalLM` (or one trained here with `targets: ["q_proj", "v_proj"]`) loads as it is:

```agentica
import Bpe from "pkg/transformers/tokenizer/bpe"
import Generate from "pkg/transformers/generate"
import Lora from "pkg/train/lora"
import Qwen3 from "pkg/transformers/qwen3"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  tok = Bpe.open("Qwen3-0.6B/tokenizer.json").unwrap!
  base = Qwen3.open("Qwen3-0.6B", backend).unwrap!
  adapter = Lora.open("adapter", backend).unwrap!
  merged = Qwen3.with_weights(base, Lora.merged(Qwen3.weights(base), adapter).unwrap!).unwrap!
  g = Generate.open(merged).unwrap!
  prompt = Bpe.encode(tok, "The capital of France is").unwrap!
  puts(Bpe.decode(tok, Generate.generate(g, prompt, 16).unwrap!).unwrap!)
  Generate.close(g)
end
```

To train one, take the Qwen3 weights (`Qwen3.weights(m)`), build the loss from `Qwen3.logits(Qwen3.with_weights(m,
Lora.applied(weights, adapter)), ids)` and a cross entropy on the next token, and use `fit_batches` as above. Qwen3 is
heavy to train on a CPU: the weights are `F32` (2.5 GB for 0.6B) and each step runs the whole model. Pages:
`https://www.agentica-lang.com/ml/train/` and `https://www.agentica-lang.com/ml/transformers/`.
