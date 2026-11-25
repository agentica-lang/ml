## Checks DistilBERT on a real sentence-transformers checkpoint against Hugging Face (`test/distilbert_golden.py`):
## the ids of each text from `WordPiece` (the file's truncation included), the last hidden state of each sentence
## alone, and the mean and the normalized embedding of the padded batch, in `float32` on ggml. It fails when the
## tokenizer pads with another id than the model's `pad_token_id`. Manual (it needs the checkpoint); run from
## `transformers/`:
##
##   python -I test/distilbert_golden.py MODEL_DIR > GOLDEN.json
##   agentica tools/distilbert_check.agc MODEL_DIR GOLDEN.json [cpu|vulkan]

import DistilBert from "../distilbert"
import Load from "pkg/hub/load"
import Source from "../tokenizer/source"
import Tensor from "pkg/tensor/tensor"
import WordPiece from "../tokenizer/wordpiece"

## The gap that passes: `float32` through 6 layers.
ABS = 0.0001
REL = 0.0001

struct Case
  text: String
  ids: List[Int]
  hidden: List[List[Float]]
  mean: List[Float]
  unit: List[Float]
end

struct Golden
  cases: List[Case]
end

def main() -> Void
  args = System.args
  if args.size < 2
    puts("usage: distilbert_check.agc MODEL_DIR GOLDEN_JSON [cpu|vulkan]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 2 and args[2] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  case run(args[0], args[1], Tensor.Backend.Ggml(device))
    ok(gap) -> puts("distilbert: matches Hugging Face, largest gap #{gap}")
    error(e) ->
      puts("distilbert: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, golden: String, backend: Tensor.Backend) -> Result[Float]
  want = try JSON.decode(try Source.read_text(golden), Golden)
  try Load.vulkan_fp16_from_env()
  tok = try WordPiece.open("#{dir}/tokenizer.json")
  m = try DistilBert.open(dir, backend)
  pad = tok.pad_id.unwrap_or(m.config.pad_token_id)
  if pad != m.config.pad_token_id
    return error("the tokenizer pads with #{pad}, the model with #{m.config.pad_token_id}")
  end

  worst = 0.0
  batch: List[WordPiece.Encoding] = []
  for i in 0...want.cases.size
    c = want.cases[i]
    e = try WordPiece.encode(tok, c.text)
    return error("sentence #{i}: the ids are #{e.ids}, expected #{c.ids}") if e.ids != c.ids
    hidden = try DistilBert.run(m, [e])
    worst = [worst, try compare("sentence #{i} hidden", try floats(hidden), flat(c.hidden))].max.unwrap!
    batch = batch.push(e)
  end

  width = m.config.dim
  mean = try floats(try DistilBert.embed(m, batch))
  unit = try floats(try DistilBert.embed(m, batch, true))
  for i in 0...want.cases.size
    c = want.cases[i]
    worst = [worst, try compare("batch sentence #{i} mean", mean.drop(i * width).take(width), c.mean)].max.unwrap!
    worst = [worst, try compare("batch sentence #{i} unit", unit.drop(i * width).take(width), c.unit)].max.unwrap!
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
