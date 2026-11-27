## Fine-tunes a sentence-transformers DistilBERT on scored sentence pairs (STS-B) on ggml CPU, the Agentica side of
## `benchmarks/stsb/` (ml#182); `benchmarks/stsb/distilbert_stsb.py` is the same run in PyTorch. Manual, never in CI;
## from `transformers/`:
##
## ```sh
## agentica tools/distilbert_stsb.agc \
##   MODEL_DIR TRAIN_JSONL VALIDATION_JSONL [PAIRS [BATCH [PAD_TO [WEIGHT_DECAY [ADAPTER_DIR]]]]]
## ```
##
## The rules, the same on both sides: the first PAIRS pairs of TRAIN_JSONL (1000) in order, BATCH pairs a step (16),
## one epoch; the checkpoint's WordPiece with a limit of 128 tokens; both sentences of a batch go through the model as
## one batch of `2 BATCH` rows, padded to the longest of them rounded up to a multiple of PAD_TO (1: the longest);
## mean pooling, and the cosine loss of the pair against its score; AdamW (lr 2e-5, betas 0.9 and 0.999, eps 1e-8,
## weight decay WEIGHT_DECAY (0.01), none on a bias or a layer-norm weight), no warmup, no clipping, no dropout; `F32`.
##
## With ADAPTER_DIR the run is a LoRA fine-tune (ml#137): the model stays frozen and an adapter on `q_lin` and
## `v_lin` of every layer is trained (r 8, alpha 16, no dropout, every tensor decayed, lr 2e-4). The new adapter is
## saved into ADAPTER_DIR before the first step, in PEFT's files, and the peer starts from that folder, so both sides
## train the same tensors. Spearman after the training is of the model with the adapter merged in.
##
## It prints JSON lines: `checks` (how many weights decay, a SHA-256 of their names and one of every token id, which
## `summary.py` compares with the peer's), `spearman` (of the cosine of the normalized embeddings against the scores
## of VALIDATION_JSONL, before and after the training), and one `step` per step with its loss, its wall time in ms
## (`batch_ms` of it is building the batch), the padded length and `new_shape`, true for a step that compiled the
## graph of its shape, which a median leaves out.

import Data from "pkg/train/data"
import DistilBert from "../distilbert"
import Lift from "../lift"
import Lora from "pkg/train/lora"
import Optim from "pkg/train/optim"
import Params from "pkg/train/params"
import Sentence from "../sentence"
import Tensor from "pkg/tensor/tensor"
import Train from "pkg/train/train"
import WordPiece from "../tokenizer/wordpiece"

MAX_LENGTH = 128
LR = 0.00002

## The LoRA run: the learning rate, and the adapter of PEFT's usual start.
LORA_LR = 0.0002
LORA = Lora.Config(targets: ["q_lin", "v_lin"], r: 8, alpha: 16.0)
LORA_SEED = 1

## How many sentences an inference batch of the validation has.
EMBED_BATCH = 32

struct Row
  sentence1: String
  sentence2: String
  score: Float
end

## A tokenized pair with its score.
struct Pair
  left: List[Int]
  right: List[Int]
  score: Float
end

## The lines of the output.
struct Checks
  kind: String
  decayed: Int
  decayed_names: String
  token_ids: String
end

struct Correlation
  kind: String
  when: String
  value: Float
end

struct Step
  kind: String
  step: Int
  loss: Float
  ms: Float
  batch_ms: Float
  length: Int
  new_shape: Bool
end

## What a run trains: the tensors `fit` gets, which of them decay, their names (as the peer's, for the checks), the
## learning rate, every weight of the model inside a loss, and the model after the training.
struct Trained
  params: List[Tensor.Tensor]
  decay: List[Bool]
  names: List[String]
  lr: Float
  weights: Fn[List[Tensor.Tensor], Result[Map[String, Tensor.Tensor]]]
  model: Fn[List[Tensor.Tensor], Result[DistilBert.DistilBert]]
end

struct Setup
  model: DistilBert.DistilBert
  trained: Trained
  train: List[Pair]
  validation: List[Pair]
  batch: Int
  pad_to: Int
  weight_decay: Float
end

def main() -> Void
  case run(System.args())
    ok(_) -> nil
    error(e) ->
      puts("distilbert_stsb: #{e.message}")
      System.exit(1)
  end
end

private

def run(args: List[String]) -> Result[Void]
  if args.size < 3
    return error("usage: MODEL_DIR TRAIN_JSONL VALIDATION_JSONL [PAIRS [BATCH [PAD_TO [WEIGHT_DECAY [ADAPTER_DIR]]]]]")
  end

  pairs = try number(args, 3, 1000)
  batch = try number(args, 4, 16)
  pad_to = try number(args, 5, 1)
  weight_decay = 0.01
  if args.size > 6
    weight_decay = try args[6].to_f
  end

  tok = try WordPiece.open("#{args[0]}/tokenizer.json")
  m = try DistilBert.open(args[0], Tensor.Backend.Ggml(Tensor.Device.Cpu))
  pad = tok.pad_id.unwrap_or(m.config.pad_token_id)
  if pad != m.config.pad_token_id
    return error("the tokenizer pads with #{pad}, the model with #{m.config.pad_token_id}")
  end

  train = (try tokenized(tok, args[1])).take(pairs)
  return error("#{args[1]} has #{train.size} pairs, fewer than #{pairs}") if train.size < pairs
  chosen = case args.size > 7
    true -> adapted(m, args[7])
    _ -> whole(m)
  end

  trained = try chosen

  s = Setup(
    model: m,
    trained: trained,
    train: train,
    validation: try tokenized(tok, args[2]),
    batch: batch,
    pad_to: pad_to,
    weight_decay: weight_decay,
  )

  puts(JSON.encode(checks(s)))
  puts(JSON.encode(Correlation(kind: "spearman", when: "before", value: try spearman(m, s.validation))))
  after = try fit(s)
  puts(JSON.encode(Correlation(kind: "spearman", when: "after", value: try spearman(after, s.validation))))
  ok()
end

## The whole model trained: every weight, the biases and the layer-norm weights without decay.
def whole(m: DistilBert.DistilBert) -> Result[Trained]
  split = try Params.split(DistilBert.weights(m), ->(_name: String) { true })
  decayed = split.names.select { |name: String| decays?(name) }
  ok(
    Trained(
      params: split.trainable,
      decay: split.names.map { |name: String| decays?(name) },
      names: decayed.map { |name: String| name.slice(m.prefix.length, name.length - m.prefix.length) },
      lr: LR,
      weights: ->(ps: List[Tensor.Tensor]) { Params.join(split, ps) },
      model: ->(ps: List[Tensor.Tensor]) { DistilBert.with_weights(m, try Params.join(split, ps)) },
    ),
  )
end

## A new LoRA adapter trained, the model frozen; the adapter is saved into `dir` as it starts.
def adapted(m: DistilBert.DistilBert, dir: String) -> Result[Trained]
  weights = DistilBert.weights(m)
  adapter = try Lora.init(weights, LORA, LORA_SEED, Tensor.backend(m.words.table))
  try Lora.save(dir, adapter, DistilBert.embeddings(m))
  params = try Lora.params(adapter)
  names = adapter.names.flat_map do |name: String|
    layer = name.slice(m.prefix.length, name.length - m.prefix.length - ".weight".length)
    ["#{layer}.lora_A.weight", "#{layer}.lora_B.weight"]
  end

  ok(
    Trained(
      params: params,
      decay: params.map { |_p: Tensor.Tensor| true },
      names: names,
      lr: LORA_LR,
      weights: ->(ps: List[Tensor.Tensor]) { Lora.applied(weights, try Lora.with_params(adapter, ps)) },
      model: ->(ps: List[Tensor.Tensor]) {
        DistilBert.with_weights(m, try Lora.merged(weights, try Lora.with_params(adapter, ps)))
      },
    ),
  )
end

def number(args: List[String], i: Int, default: Int) -> Result[Int]
  return ok(default) if args.size <= i
  n = try args[i].to_i
  return error("argument #{i + 1} must be at least 1, got #{n}") if n < 1
  ok(n)
end

## The pairs of a JSONL file (`sentence1`, `sentence2`, `score`), tokenized with the limit of the run.
def tokenized(tok: WordPiece.WordPiece, path: String) -> Result[List[Pair]]
  out: List[Pair] = []
  for line in (try File.read_text(path)).split("\n")
    if line.strip != ""
      row = try JSON.decode(line, Row)
      left = try WordPiece.encode(tok, row.sentence1, none, true, some(MAX_LENGTH))
      right = try WordPiece.encode(tok, row.sentence2, none, true, some(MAX_LENGTH))
      out = out.push(Pair(left: left.ids, right: right.ids, score: row.score))
    end
  end

  ok(out)
end

## What both sides must agree on before a loss means anything: the tensors that decay (their names without the
## prefix of the file, sorted; for an adapter, PEFT's names of `A` and `B`) and every token id, the training pairs
## first.
def checks(s: Setup) -> Checks
  bare = s.trained.names.sort
  ids = (s.train + s.validation).map { |p: Pair| "#{text(p.left)}\n#{text(p.right)}\n" }
  Checks(
    kind: "checks",
    decayed: bare.size,
    decayed_names: OpenSSL.Digest.sha256_hexdigest(bare.join("\n").bytes),
    token_ids: OpenSSL.Digest.sha256_hexdigest(ids.join("").bytes),
  )
end

def text(ids: List[Int]) -> String = ids.map { |id: Int| id.to_s }.join(",")

## Whether AdamW decays the weight `name`: not a bias, not a layer-norm weight (Hugging Face's rule).
def decays?(name: String) -> Bool
  !(name.end_with?(".bias") or name.include?("LayerNorm") or name.include?("layer_norm"))
end

## The padded length of batch `i`: its longest sentence, rounded up to a multiple of `pad_to`.
def length(s: Setup, i: Int) -> Int
  longest = 1
  for p in s.train.drop(i * s.batch).take(s.batch)
    longest = [longest, p.left.size, p.right.size].max.unwrap!
  end

  ((longest + s.pad_to - 1) / s.pad_to) * s.pad_to
end

## Batch `i`: the ids `[pairs, 2, length]` (`I32`, padded with the model's pad id) and the scores `[pairs]`.
def batch(s: Setup, i: Int) -> Result[Data.Batch]
  picked = s.train.drop(i * s.batch).take(s.batch)
  seq = length(s, i)
  pad = s.model.config.pad_token_id
  ids: List[Tensor.Values] = []
  for p in picked
    for side in [p.left, p.right]
      for k in 0...seq
        if k < side.size
          ids = ids.push(Tensor.Values.Int(side[k]))
        else
          ids = ids.push(Tensor.Values.Int(pad))
        end
      end
    end
  end

  backend = Tensor.backend(s.model.words.table)
  inputs = try Tensor.from_list(ids, [picked.size, 2, seq], Tensor.Dtype.I32, backend)
  scores = picked.map { |p: Pair| Tensor.Values.Float(p.score) }
  case Tensor.from_list(scores, [picked.size], Tensor.Dtype.F32, backend)
    ok(targets) -> ok(Data.Batch(inputs: inputs, targets: targets))
    error(e) ->
      Data.free(inputs)
      error(e)
  end
end

## `batch(s, i)`, with the ms it took sent to `builds`.
def timed_batch(s: Setup, i: Int, builds: Channel[Float]) -> Result[Data.Batch]
  started = Time.monotonic_ns()
  made = try batch(s, i)
  _ = builds.send((Time.monotonic_ns() - started).to_f / 1000000.0, 0)
  ok(made)
end

## The cosine loss of a batch: the mask is `ids != pad`, both sides go through the model as one batch, and each
## pair's two mean embeddings are compared with its score.
def loss(s: Setup, ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  m = s.model
  model = try DistilBert.with_weights(m, try s.trained.weights.call(ps))
  n = Tensor.shape(b.inputs)[0]
  seq = Tensor.shape(b.inputs)[2]
  width = m.config.dim
  ids = try Tensor.reshape(b.inputs, [2 * n, seq])
  backend = Tensor.backend(ids)
  pads = try Tensor.full([2 * n, seq], Tensor.Values.Int(m.config.pad_token_id), Tensor.Dtype.I32, backend)
  ones = try Tensor.full([2 * n, seq], Tensor.Values.Float(1.0), Tensor.Dtype.F32, backend)
  mask = try Tensor.sub(ones, try Tensor.to_dtype(try Tensor.eq(ids, pads), Tensor.Dtype.F32))
  sides = try Tensor.reshape(try DistilBert.pooled(model, ids, mask), [n, 2, width])
  left = try Tensor.reshape(try Tensor.slice(sides, [0, 0, 0], [n, 1, width]), [n, width])
  right = try Tensor.reshape(try Tensor.slice(sides, [0, 1, 0], [n, 2, width]), [n, width])
  Sentence.cosine_similarity_loss(left, right, b.targets)
end

## One epoch over the training pairs, a step per batch, and the model with the trained weights. Each step is an
## "epoch" of one batch for `fit_batches`, so that its history is the loss of every step and `on_epoch` marks the
## end of each one.
def fit(s: Setup) -> Result[DistilBert.DistilBert]
  steps = (s.train.size + s.batch - 1) / s.batch
  ends: Channel[Int] = Channel.new(steps)
  builds: Channel[Float] = Channel.new(steps)
  started = Time.monotonic_ns()
  fit = try Train.fit_batches(
    ->(ps: List[Tensor.Tensor], b: Data.Batch) { loss(s, ps, b) },
    s.trained.params,
    ->(e: Int, _i: Int) { timed_batch(s, e - 1, builds) },
    1,
    Optim.Optimizer.AdamW(Optim.AdamW(lr: s.trained.lr, weight_decay: s.weight_decay, decay: s.trained.decay)),
    steps,
    some(
      ->(_) {
        _ = ends.send(Time.monotonic_ns(), 0)
        ok(Train.Control.Continue)
      },
    ),
  )

  seen: List[Int] = []
  last = started
  for i in 0...steps
    done = try ends.recv(0)
    seq = length(s, i)
    rows = [s.batch, s.train.size - i * s.batch].min.unwrap!
    shape = rows * 1000 + seq
    line = Step(
      kind: "step",
      step: i + 1,
      loss: fit.history[i],
      ms: (done - last).to_f / 1000000.0,
      batch_ms: try builds.recv(0),
      length: seq,
      new_shape: !seen.include?(shape),
    )

    puts(JSON.encode(line))
    seen = seen.push(shape)
    last = done
  end

  s.trained.model.call(fit.params)
end

## Spearman's correlation of the cosine of the two sentences of each pair with its score.
def spearman(m: DistilBert.DistilBert, pairs: List[Pair]) -> Result[Float]
  sentences: List[List[Int]] = []
  for p in pairs
    sentences = sentences.push(p.left).push(p.right)
  end

  width = m.config.dim
  vectors: List[Float] = []
  at = 0
  while at < sentences.size
    part = sentences.drop(at).take(EMBED_BATCH)
    encodings = part.map { |ids: List[Int]| WordPiece.Encoding(ids: ids, type_ids: List.repeat(0, ids.size)) }
    unit = try DistilBert.embed(m, encodings, true)
    for v in try Tensor.to_list(unit)
      case v
        Tensor.Values.Float(f) -> vectors = vectors.push(f)
        _ -> return error("an embedding holds Float values")
      end
    end

    Lift.close([unit])
    at = at + EMBED_BATCH
  end

  cosines: List[Float] = []
  for i in 0...pairs.size
    dot = 0.0
    for k in 0...width
      dot = dot + vectors[2 * i * width + k] * vectors[(2 * i + 1) * width + k]
    end

    cosines = cosines.push(dot)
  end

  ok(pearson(ranks(cosines), ranks(pairs.map { |p: Pair| p.score })))
end

## The rank of each value from 1, equal values sharing the mean of their ranks.
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
