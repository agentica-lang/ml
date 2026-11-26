## Checks greedy generation with the hybrid cache (`generate_lfm3.agc`) against the official `generate`
## (`test/generate_lfm3_check_golden.py`, which says how to run it) on `LiquidAI/LFM3-1.7B` on ggml: the 24 ids it adds
## to each prompt must be the same (the last prompt is over the sliding window of 256), the largest gap of a logit at the
## five ids the official code found largest is printed, and the speed: the first call (which traces and plans the steps),
## then a second one with the steps compiled. Manual (it needs the checkpoint); run from `transformers/`:
##
##   python -I test/generate_lfm3_check_golden.py MODEL_DIR > GOLDEN.json
##   WEIGHTS=f16 agentica tools/generate_lfm3_check.agc MODEL_DIR GOLDEN.json [cpu|vulkan]
##
## `WEIGHTS=f16` is what a 12 GB machine runs (3.4 GB; see `tools/lfm3_check.agc`). `VULKAN_FP16=0` is the default of the
## check. On a shared machine run it as `flock LOCK systemd-run --user --scope -q -p MemoryMax=5G agentica ...`.

import Generate from "../generate_lfm3"
import Lfm3 from "../lfm3"
import Load from "pkg/hub/load"
import Source from "../tokenizer/source"
import Tensor from "pkg/tensor/tensor"

struct Case
  ids: List[Int]
  generated: List[Int]
  gaps: List[Float]
  step_top_ids: List[List[Int]]
  step_top_logits: List[List[Float]]
end

struct Golden
  cases: List[Case]
end

def main() -> Void
  args = System.args
  if args.size < 2
    puts("usage: generate_lfm3_check.agc MODEL_DIR GOLDEN_JSON [cpu|vulkan]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 2 and args[2] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  case run(args[0], args[1], Tensor.Backend.Ggml(device))
    ok(_) -> puts("generates what the official code does")
    error(e) ->
      puts("generate_lfm3_check: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, golden: String, backend: Tensor.Backend) -> Result[Void]
  want = try JSON.decode(try Source.read_text(golden), Golden)
  try Load.vulkan_fp16_from_env()
  m = try Lfm3.open(dir, backend, try Load.weights_from_env())
  puts("model_type #{m.config.model_type}, #{m.layers.size} layers")
  g = try Generate.open(m, System.env("USABLE").unwrap_or("0").to_i.unwrap_or(0))
  done = checked(g, want)
  Generate.close(g)
  Lfm3.close_model(m)
  puts(memory())
  done
end

def checked(g: Generate.Generator, want: Golden) -> Result[Void]
  worst = 0.0
  for i in 0...want.cases.size
    c = want.cases[i]
    started = Time.monotonic_ms
    first = try Generate.generate(g, c.ids, c.generated.size)
    cold = Time.monotonic_ms - started
    started = Time.monotonic_ms
    _ = try Generate.generate(g, c.ids, 1)
    prompt = Time.monotonic_ms - started
    started = Time.monotonic_ms
    again = try Generate.generate(g, c.ids, c.generated.size)
    warm = Time.monotonic_ms - started
    return error("prompt #{i}: ids #{again}, expected #{c.generated}") if again != c.generated
    return error("prompt #{i}: the first call gave #{first}, the second #{again}") if first != again
    steps = again.size - 1
    per = 0.0
    if steps > 0 and warm > prompt
      per = (steps * 1000).to_f / (warm - prompt).to_f
    end

    worst = [worst, try logits_gap(g, c)].max.unwrap!
    puts(
      "prompt #{i}: #{c.ids.size} prompt tokens, #{again.size} new, smallest gap of the official code " +
      "#{c.gaps.min.unwrap!}; first call #{cold} ms, then prompt and one token #{prompt} ms, #{again.size} tokens " +
      "#{warm} ms, #{per} tokens/s after the prompt",
    )
  end

  puts("largest gap of a step's logit from the official code's, at its five largest: #{worst}")
  ok()
end

## The largest gap, over the steps of `c`, between our logits at the five ids the official code found largest and its
## own: the prompt and then the ids it generated are fed to `step_logits`.
def logits_gap(g: Generate.Generator, c: Case) -> Result[Float]
  cache = Generate.start()
  feed = c.ids
  worst = 0.0
  for i in 0...c.generated.size
    s = try Generate.step_logits(g, cache, feed)
    cache = s.cache
    for k in 0...c.step_top_ids[i].size
      gap = (s.logits[c.step_top_ids[i][k]] - c.step_top_logits[i][k]).abs
      worst = [worst, gap].max.unwrap!
    end

    feed = [c.generated[i]]
  end

  Generate.finish(cache)
  ok(worst)
end

## The peak and the current resident memory of the process.
def memory() -> String
  text = File.read_text("/proc/self/status").unwrap_or("")
  out: List[String] = []
  for line in text.split("\n")
    if line.start_with?("VmHWM") or line.start_with?("VmRSS")
      out = out.push(line.replace("\t", " ").replace("  ", " "))
    end
  end

  out.join(", ")
end
