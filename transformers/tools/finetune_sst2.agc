## Fine-tunes DistilBERT (`distilbert-base-uncased`) for the sentiment of SST-2 (`stanfordnlp/sst2`): a linear head
## on the mean-pooled output, AdamW, one epoch over the first ROWS rows of the training split, with no dropout,
## shuffle, warmup or clipping. It prints the loss and the accuracy on the whole validation split before and after,
## the time of each phase and of the steps, saves the weights into OUT_DIR (a folder that exists) and loads them back
## for a prediction. The guide is `site/content/train/finetune-distilbert.md`. From `transformers/`:
##
## ```sh
## sh tools/fetch_sst2.sh DIR
## mkdir -p out
## agentica tools/finetune_sst2.agc DIR/distilbert-base-uncased DIR/sst2 out [cpu|vulkan|xla|torch [ROWS]]
## ```
##
## The default is the ggml CPU. `vulkan` is ggml on the first Vulkan device, multiplying in `F32` as the checks do.
## `xla` and `torch` are refused until the ops this model needs exist on them (ml#243, ml#203).

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
