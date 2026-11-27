## A whole fine-tune of sentence-transformers/all-MiniLM-L6-v2 on STS-B, the scored sentence pairs of
## sentence-transformers/stsb: it reads the pairs, tokenizes them, cuts them into batches of one shape, trains with
## AdamW and the cosine loss through `Loop`, measures Spearman's correlation of the cosines with the scores on a sample
## of the validation split before, after every epoch and after, saves the weights, loads them back and measures again.
## The backend, the number of training pairs, the epochs, the size of the validation sample and an optional time budget
## are arguments. The budget counts only the training steps (and the batches they take): with it the model is measured
## before and after, not after each epoch, and a line `trained ... steps in ... ms` says how long the steps took. It is the program of the guide `site/content/train/finetune-minilm.md`. From `transformers/`:
##
## ```sh
## sh tools/fetch_stsb.sh data
## mkdir -p out
## agentica --max-steps 1000000000000 tools/finetune_stsb.agc data/all-minilm-l6-v2 data/stsb out [cpu|vulkan|xla|torch [ROWS [EPOCHS [VALIDATION [SECONDS]]]]]
## ```

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
VALIDATION_ROWS = 500
LOG_EVERY = 10
SEED = 1
BATCH = 16
EVAL_BATCH = 32
LENGTH = 64
LR = 0.00002
WEIGHT_DECAY = 0.01
TOLERANCE = 0.000001

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
  return ok(options) if args.size < 8

  seconds = try args[7].to_i
  return error("SECONDS is at least 1, got #{seconds}") if seconds < 1

  ok(options.with(seconds: some(seconds)))
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
def trainable(base: Bert.Bert) -> Result[Params.Split] = Params.split(Bert.weights(base), ->(_) { true })

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
# batch is left out). The schedule warms up over the first tenth of all the steps and decays to 0 at the last, which a
# run stopped by its budget does not reach. Every `LOG_EVERY` steps the step and its loss are printed, and a last line
# says how many steps ran, in how long, and how many pairs per second.
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

  ends: Channel[Int] = Channel.new(total)
  started = Time.monotonic_ms()
  fit = (
    Loop.trainer(->(ps, b) { loss(net, ps, b) }, net.split.trainable, optimizer)
    |> Loop.handle(
      Loop.Event.StepCompleted,
      ->(_) {
        _ = ends.send(Time.monotonic_ms(), 0)
        ok()
      },
    )
    |> Loop.handle(
      Loop.Event.StepCompleted,
      ->(info) {
        puts("step #{info.step} loss #{info.loss}")
        ok()
      },
      every: LOG_EVERY,
    )
    |> measured_each_epoch(net, validation, options.seconds)
    |> within(options.seconds, started)
    |> try Loop.run_batches(
      ->(epoch, i) { batch(net.base, picked(rows, orders[epoch - 1], i)) },
      steps,
      epochs: options.epochs,
    )
  )

  puts(throughput(ends, started))
  ok(fit)
end

# Without a budget, the two measures of the validation sample at the end of every epoch, printed. With one,
# nothing: the
# clock of the budget counts the training steps and not the measuring.
def measured_each_epoch(t: Loop.Trainer, net: Net, validation: List[Pair], seconds: Optional[Int]) -> Loop.Trainer
  case seconds
    some(_) -> t
    none ->
      (
        t
        |> Loop.metric(gauge("spearman", &Loop.spearman, net, validation))
        |> Loop.metric(gauge("cosine loss", &squared_gap, net, validation))
        |> Loop.handle(
          Loop.Event.EpochCompleted,
          ->(info) {
            puts(
              "epoch #{info.epoch}: Spearman #{info.metrics["spearman"]}, cosine loss #{info.metrics["cosine loss"]}",
            )

            ok()
          },
        )
      )
  end
end

# `trained STEPS steps in MS ms, N pairs/s, N pairs/s after the first step`, from the time at which each step ended and
# the time the training started. The first step holds the compile and the warmup of the backend.
def throughput(ends: Channel[Int], started: Int) -> String
  times: List[Int] = []
  got = ends.recv(0)
  while got.ok?
    times = times.push(got.unwrap!)
    got = ends.recv(0)
  end

  return "trained 0 steps" if times.empty?

  ms = times[times.size - 1] - started
  rest = times[times.size - 1] - times[0]
  after_first = 0.0
  if times.size > 1 and rest > 0
    after_first = ((times.size - 1) * BATCH * 1000).to_f / rest.to_f
  end

  rate = (times.size * BATCH * 1000).to_f / ms.to_f
  "trained #{times.size} steps in #{ms} ms, #{rate} pairs/s, #{after_first} pairs/s after the first step"
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

# The validation measures of the weights `ps`: the cosines of the whole sample once, reduced twice.
def evaluate(net: Net, ps: List[Tensor.Tensor], rows: List[Pair]) -> Result[Score]
  all: List[List[Float]] = []
  for i in 0...((rows.size + EVAL_BATCH - 1) / EVAL_BATCH)
    b = try batch(net.base, rows.drop(i * EVAL_BATCH).take(EVAL_BATCH))
    got = cosines(net, ps, b)
    Data.release(b)
    for row in try got
      all = all.push(row)
    end
  end

  ok(Score(spearman: Loop.spearman(all), loss: squared_gap(all)))
end

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
