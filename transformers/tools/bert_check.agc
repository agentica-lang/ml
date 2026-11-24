## Checks BERT on a real checkpoint against Hugging Face's `BertModel`
## (`test/bert_golden.py`): each sentence alone and the padded batch, the last
## hidden state, the pooler output and the mean embedding, in `float32` on
## ggml. Manual (it needs the checkpoint); run from `transformers/`:
##
##   python -I test/bert_golden.py MODEL_DIR > GOLDEN.json
##   agentica tools/bert_check.agc MODEL_DIR GOLDEN.json [cpu|vulkan]

import Bert from "../bert"
import Load from "pkg/hub/load"
import Source from "../tokenizer/source"
import Tensor from "pkg/tensor/tensor"
import WordPiece from "../tokenizer/wordpiece"

## The gap that passes: `float32` through 12 layers.
ABS = 0.0001
REL = 0.0001

struct Case
  ids: List[Int]
  type_ids: List[Int]
  hidden: List[List[Float]]
  pooled: List[Float]
  mean: List[Float]
  batch_hidden: List[List[Float]]
  batch_pooled: List[Float]
end

struct Golden
  cases: List[Case]
end

def main() -> Void
  args = System.args
  if args.size < 2
    puts("usage: bert_check.agc MODEL_DIR GOLDEN_JSON [cpu|vulkan]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 2 and args[2] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  case run(args[0], args[1], Tensor.Backend.Ggml(device))
    ok(gap) -> puts("bert: matches Hugging Face, largest gap #{gap}")
    error(e) ->
      puts("bert: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, golden: String, backend: Tensor.Backend) -> Result[Float]
  want = try JSON.decode(try Source.read_text(golden), Golden)
  try Load.vulkan_fp16_from_env()
  m = try Bert.open(dir, backend)
  worst = 0.0
  batch: List[WordPiece.Encoding] = []
  for c in want.cases
    batch = batch.push(WordPiece.Encoding(ids: c.ids, type_ids: c.type_ids))
  end

  for i in 0...want.cases.size
    c = want.cases[i]
    out = try Bert.run(m, [batch[i]])
    worst = [worst, try compare("sentence #{i} hidden", try floats(out.hidden), flat(c.hidden))].max.unwrap!
    worst = [worst, try compare("sentence #{i} pooled", try floats(out.pooled.unwrap!), c.pooled)].max.unwrap!
    mean = try Bert.embed(m, [batch[i]])
    worst = [worst, try compare("sentence #{i} mean", try floats(mean), c.mean)].max.unwrap!
  end

  out = try Bert.run(m, batch)
  seq = Tensor.shape(out.hidden)[1]
  width = Tensor.shape(out.hidden)[2]
  all = try floats(out.hidden)
  pooled = try floats(out.pooled.unwrap!)
  for i in 0...want.cases.size
    c = want.cases[i]
    real = all.drop(i * seq * width).take(c.ids.size * width)
    worst = [worst, try compare("batch sentence #{i} hidden", real, flat(c.batch_hidden))].max.unwrap!
    worst = [
      worst,
      try compare("batch sentence #{i} pooled", pooled.drop(i * width).take(width), c.batch_pooled),
    ].max.unwrap!
  end

  ok(worst)
end

def flat(rows: List[List[Float]]) -> List[Float]
  out: List[Float] = []
  for r in rows
    out = out + r
  end

  out
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

## The largest gap of `got` from `want`, or an error past the tolerance.
def compare(what: String, got: List[Float], want: List[Float]) -> Result[Float]
  return error("#{what}: #{got.size} values, expected #{want.size}") if got.size != want.size
  worst = 0.0
  for i in 0...want.size
    gap = (got[i] - want[i]).abs
    return error("#{what}: value #{i} is #{got[i]}, expected #{want[i]}") if gap > ABS + REL * want[i].abs
    worst = [worst, gap].max.unwrap!
  end

  ok(worst)
end
