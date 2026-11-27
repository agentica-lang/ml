## A short fine-tune of sentence-transformers/all-MiniLM-L6-v2 (6 layers, width 384) on scored sentence pairs, on ggml,
## XLA or libtorch, as a check that a BERT-family model loads and trains: its embeddings put a paraphrase closer than an
## unrelated sentence, the cosine loss falls over a few `Train.fit_batches` steps, and a sample of its weights moved. CI
## checks and formats this file but does not run it: it runs by hand, as it needs the network (the checkpoint).
## From `transformers/`:
##
## ```sh
## sh tools/fetch_minilm.sh                       # the pinned checkpoint, into ~/.cache/agentica-ml/hf/all-minilm-l6-v2
## agentica --max-steps 1000000000000 tools/minilm_finetune.agc MODEL_DIR [cpu|vulkan|xla|torch [STEPS [LR [PAIRS]]]]
## ```
##
## It trains every weight (AdamW, no weight decay, no dropout, `F32`) on PAIRS of its eight pairs (default all), all of
## them in each step, so the loss must fall. Only on libtorch the word table (12M of the 22M weights) stays frozen: a
## step there held about 2.5 GB at its peak (2,505,192 kB, 4 pairs, table frozen) and more with the table trained, which
## went over a 2.5 GB cap (https://github.com/agentica-lang/ml/issues/394).

import Bert from "../bert"
import Data from "pkg/train/data"
import Lift from "../lift"
import Optim from "pkg/train/optim"
import Params from "pkg/train/params"
import Sentence from "../sentence"
import Tensor from "pkg/tensor/tensor"
import Train from "pkg/train/train"
import WordPiece from "../tokenizer/wordpiece"

MAX_LENGTH = 16
DEFAULT_STEPS = 6
DEFAULT_LR = 0.0001

## The weights whose values the check reads before and after (the largest tables are too big to copy into lists).
WATCHED = [
  "embeddings.token_type_embeddings.weight",
  "embeddings.LayerNorm.weight",
  "encoder.layer.0.attention.self.query.weight",
  "encoder.layer.2.intermediate.dense.weight",
  "encoder.layer.5.output.dense.weight",
  "encoder.layer.5.output.LayerNorm.bias",
]

struct Pair
  left: String
  right: String
  score: Float
end

def main() -> Void
  args = System.args
  if args.empty?
    puts("usage: minilm_finetune.agc MODEL_DIR [cpu|vulkan|xla|torch [STEPS [LR [PAIRS]]]]")
    System.exit(2)
  end

  case run(args)
    ok(_) -> nil
    error(e) ->
      puts("minilm_finetune: #{e.message}")
      System.exit(1)
  end
end

private

## The first `count` pairs, a paraphrase and an unrelated pair in turn.
def pairs(count: Int) -> List[Pair]
  all = [
    Pair(left: "A man is eating food.", right: "A man is eating a piece of bread.", score: 0.9),
    Pair(left: "A man is eating food.", right: "The stock market fell on Monday.", score: 0.0),
    Pair(left: "A girl is playing a guitar.", right: "A woman plays the guitar.", score: 0.85),
    Pair(left: "A girl is playing a guitar.", right: "Two trains stand in a station.", score: 0.05),
    Pair(left: "A man is riding a horse.", right: "A man rides a horse across a field.", score: 0.9),
    Pair(left: "A man is riding a horse.", right: "A chef is slicing an onion.", score: 0.1),
    Pair(left: "The cat sits on the sofa.", right: "A cat is lying on the couch.", score: 0.8),
    Pair(left: "The cat sits on the sofa.", right: "Scientists found water on Mars.", score: 0.0),
  ]

  all.take(count)
end

def run(args: List[String]) -> Result[Void]
  device = Tensor.Device.Cpu
  if args.size > 1 and args[1] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  backend = Tensor.Backend.Ggml(device)
  if args.size > 1 and args[1] == "xla"
    backend = Tensor.Backend.Xla(Tensor.Device.Cpu)
  end

  if args.size > 1 and args[1] == "torch"
    backend = Tensor.Backend.Torch(Tensor.Device.Cpu)
  end

  steps = DEFAULT_STEPS
  if args.size > 2
    steps = try args[2].to_i
  end

  lr = DEFAULT_LR
  if args.size > 3
    lr = try args[3].to_f
  end

  count = 8
  if args.size > 4
    count = try args[4].to_i
  end

  return error("PAIRS is 2 to 8, got #{count}") if count < 2 or count > 8

  m = try Bert.open(args[0], backend)
  tok = try WordPiece.open("#{args[0]}/tokenizer.json")
  puts("loaded #{m.config.num_hidden_layers} layers of width #{m.config.hidden_size} on #{backend}")
  try expect_paraphrase_closer(m, tok)
  train_table = true
  case backend
    Tensor.Backend.Torch(_) -> train_table = false
    _ -> nil
  end

  split = try Params.split(
    Bert.weights(m),
    ->(name: String) { train_table or !name.end_with?("word_embeddings.weight") },
  )

  watched = WATCHED.map { |name: String| split.names.index(name).unwrap! }
  before = try snapshot(watched.map { |i: Int| split.trainable[i] })
  ids = try encoded(m, tok, count)
  fit = try Train.fit_batches(
    ->(ps: List[Tensor.Tensor], b: Data.Batch) { loss(m, split, ps, b) },
    split.trainable,
    ->(_e: Int, _i: Int) { batch(m, ids, count) },
    1,
    Optim.Optimizer.AdamW(Optim.AdamW(lr: lr, weight_decay: 0.0)),
    steps,
  )

  puts("loss #{fit.history}")
  return error("the loss did not fall: #{fit.history}") if !(fit.history[steps - 1] < fit.history[0])

  after = try snapshot(watched.map { |i: Int| fit.params[i] })
  for k in 0...watched.size
    return error("#{WATCHED[k]} did not move") if after[k] == before[k]
  end

  puts("the loss fell from #{fit.history[0]} to #{fit.history[steps - 1]}, and #{watched.size} watched weights moved")
  ok()
end

## The two sentences of each pair as ids, `[pair, side, MAX_LENGTH]` padded with the model's pad id, in one shape.
def encoded(m: Bert.Bert, tok: WordPiece.WordPiece, count: Int) -> Result[List[Int]]
  out: List[Int] = []
  for p in pairs(count)
    for text in [p.left, p.right]
      e = try WordPiece.encode(tok, text, none, true, some(MAX_LENGTH))
      out = out + e.ids + List.repeat(m.config.pad_token_id, MAX_LENGTH - e.ids.size)
    end
  end

  ok(out)
end

def batch(m: Bert.Bert, ids: List[Int], count: Int) -> Result[Data.Batch]
  backend = Tensor.backend(m.words.table)
  n = count
  inputs = try Tensor.from_list(
    ids.map { |i: Int| Tensor.Values.Int(i) },
    [n, 2, MAX_LENGTH],
    Tensor.Dtype.I32,
    backend,
  )

  case Tensor.from_list(pairs(count).map { |p: Pair| Tensor.Values.Float(p.score) }, [n], Tensor.Dtype.F32, backend)
    ok(targets) -> ok(Data.Batch(inputs: inputs, targets: targets))
    error(e) ->
      Data.free(inputs)
      error(e)
  end
end

## The cosine loss of a batch: the mask is `ids != pad`, both sides go through the model as one batch.
def loss(m: Bert.Bert, split: Params.Split, ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  model = try Bert.with_weights(m, try Params.join(split, ps))
  n = Tensor.shape(b.inputs)[0]
  seq = Tensor.shape(b.inputs)[2]
  width = m.config.hidden_size
  flat = try Tensor.reshape(b.inputs, [2 * n, seq])
  backend = Tensor.backend(flat)
  pads = try Tensor.full([2 * n, seq], Tensor.Values.Int(m.config.pad_token_id), Tensor.Dtype.I32, backend)
  ones = try Tensor.full([2 * n, seq], Tensor.Values.Float(1.0), Tensor.Dtype.F32, backend)
  mask = try Tensor.sub(ones, try Tensor.to_dtype(try Tensor.eq(flat, pads), Tensor.Dtype.F32))
  types = try Tensor.full([2 * n, seq], Tensor.Values.Int(0), Tensor.Dtype.I32, backend)
  sides = try Tensor.reshape(try Bert.pooled(model, flat, types, mask), [n, 2, width])
  left = try Tensor.reshape(try Tensor.slice(sides, [0, 0, 0], [n, 1, width]), [n, width])
  right = try Tensor.reshape(try Tensor.slice(sides, [0, 1, 0], [n, 2, width]), [n, width])
  Sentence.cosine_similarity_loss(left, right, b.targets)
end

## The model as it is: a paraphrase is closer than an unrelated sentence.
def expect_paraphrase_closer(m: Bert.Bert, tok: WordPiece.WordPiece) -> Result[Void]
  texts = ["A man is eating food.", "A man is eating a piece of bread.", "The stock market fell on Monday."]
  encodings: List[WordPiece.Encoding] = []
  for t in texts
    encodings = encodings.push(try WordPiece.encode(tok, t))
  end

  unit = try Bert.embed(m, encodings)
  v = try values(unit)
  Lift.close([unit])
  width = m.config.hidden_size
  near = cosine(v.take(width), v.drop(width).take(width))
  far = cosine(v.take(width), v.drop(2 * width))
  puts("cosine of a paraphrase #{near}, of an unrelated sentence #{far}")
  return error("a paraphrase is not closer than an unrelated sentence") if !(near > far)
  ok()
end

def cosine(a: List[Float], b: List[Float]) -> Float
  dot = 0.0
  na = 0.0
  nb = 0.0
  for i in 0...a.size
    dot = dot + a[i] * b[i]
    na = na + a[i] * a[i]
    nb = nb + b[i] * b[i]
  end

  dot / (Math.sqrt(na) * Math.sqrt(nb))
end

def values(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end

def snapshot(ts: List[Tensor.Tensor]) -> Result[List[List[Float]]]
  out: List[List[Float]] = []
  for t in ts
    out = out.push(try values(t))
  end

  ok(out)
end
