## Checks LFM3 on a real checkpoint (`LiquidAI/LFM3-1.7B`) against the official code (`test/lfm3_check_golden.py`, which
## says how to run it): for each prompt the logits of the last position over the whole vocabulary, and the ids of the five
## largest logits of every position (a prompt of more than 100 tokens, the last one, is over the sliding window of 256 and is
## checked at its last position only), on ggml. Manual (it needs the checkpoint); run from `transformers/`:
##
##   python -I test/lfm3_check_golden.py MODEL_DIR > GOLDEN.json
##   WEIGHTS=f16 agentica tools/lfm3_check.agc MODEL_DIR GOLDEN.json [cpu|vulkan]
##
## `WEIGHTS=f16` is what a 12 GB machine runs: the `F32` model is 6.9 GB, the `F16` one 3.4 GB, and a `bfloat16` weight is
## exactly an `F16` one, so the same gap passes. `VULKAN_FP16=1` runs Vulkan's `mul_mat` in fp16, which this gap does not
## hold. On a shared machine run it as `flock LOCK systemd-run --user --scope -q -p MemoryMax=5G agentica ...`.

import Lfm3 from "../lfm3"
import Load from "pkg/hub/load"
import Lift from "../lift"
import Source from "../tokenizer/source"
import Tensor from "pkg/tensor/tensor"

## The gap that passes: `float32` through 24 layers.
ABS = 0.002
REL = 0.0005

## A prompt longer than this is checked at its last position only.
LONG = 100

struct Case
  ids: List[Int]
  last: List[Float]
  top5: List[List[Int]]
end

struct Golden
  cases: List[Case]
end

def main() -> Void
  args = System.args
  if args.size < 2
    puts("usage: lfm3_check.agc MODEL_DIR GOLDEN_JSON [cpu|vulkan]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 2 and args[2] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  case run(args[0], args[1], Tensor.Backend.Ggml(device))
    ok(gap) -> puts("matches Hugging Face, largest gap of a logit #{gap}")
    error(e) ->
      puts("lfm3_check: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, golden: String, backend: Tensor.Backend) -> Result[Float]
  want = try JSON.decode(try Source.read_text(golden), Golden)
  try Load.vulkan_fp16_from_env()
  m = try Lfm3.open(dir, backend, try Load.weights_from_env())
  gap = checked(m, want)
  Lfm3.close_model(m)
  gap
end

## The largest gap of a logit over the prompts of `want`, or the first error.
def checked(m: Lfm3.Lfm3, want: Golden) -> Result[Float]
  worst = 0.0
  for i in 0...want.cases.size
    c = want.cases[i]
    started = Time.monotonic_ms
    # Every position's logits as host floats is 250,000 values a token: past LONG tokens only the last one is read.
    long = c.ids.size > LONG
    all = try Lfm3.run(m, [c.ids], long)
    took = Time.monotonic_ms - started
    vocab = Tensor.shape(all)[2]
    read = floats(all)
    Lift.close([all])
    values = try read
    last = values.drop(values.size - vocab)
    worst = [worst, try compare("prompt #{i} last position", last, c.last)].max.unwrap!
    for at in 0...c.ids.size
      next if long
      mine = top5(values.drop(at * vocab).take(vocab))
      if mine.sort != c.top5[at].sort
        return error("prompt #{i} position #{at}: top five #{mine}, expected #{c.top5[at]}")
      end
    end

    puts("prompt #{i}: #{c.ids.size} tokens in #{took} ms")
  end

  ok(worst)
end

## The ids of the five largest of `values`.
def top5(values: List[Float]) -> List[Int]
  best: List[Int] = []
  for _ in 0...5
    top = -1
    for i in 0...values.size
      if not best.include?(i) and (top < 0 or values[i] > values[top])
        top = i
      end
    end

    best = best.push(top)
  end

  best
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
    return error("#{what}: logit #{i} is #{got[i]}, expected #{want[i]}") if gap > ABS + REL * want[i].abs
    worst = [worst, gap].max.unwrap!
  end

  ok(worst)
end
