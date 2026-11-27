## Times Qwen3.5 generation on ggml (`benchmarks/README.md`, "Qwen3.5-0.8B (ml#150), ours only"): the load, then one warm-up pass (which compiles
## the steps), then one timed pass: a prompt of 128 token ids in one step, and 64 one-token steps after it. Prints
## `RESULT load_ms prompt_tokens_per_s decode_tokens_per_s`. Run from `transformers/`:
##
##   agentica tools/bench_qwen35.agc MODEL_DIR [cpu|vulkan]
##
## `VULKAN_FP16=0` makes Vulkan multiply in exact `float32` (fp16 is the default). The weights are `F32`.

import Generate from "../generate_qwen35"
import Qwen3_5 from "../qwen3_5"
import Load from "pkg/hub/load"
import Tensor from "pkg/tensor/tensor"

PROMPT = 128
NEW = 64

def main() -> Void
  args = System.args
  if args.size < 1
    puts("usage: bench_qwen35.agc MODEL_DIR [cpu|vulkan]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 1 and args[1] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  case run(args[0], Tensor.Backend.Ggml(device))
    ok(line) -> puts(line)
    error(e) ->
      puts("bench_qwen35: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, backend: Tensor.Backend) -> Result[String]
  started = Time.monotonic_ms
  try Load.vulkan_fp16_from_env(true)
  m = try Qwen3_5.open(dir, backend)
  load = Time.monotonic_ms - started
  opened = Generate.open(m)
  if not opened.ok?
    Qwen3_5.close_model(m)
    return error(opened.message)
  end

  g = opened.unwrap!
  prompt = (0...PROMPT).map { |i| 1000 + (i * 7) % 5000 }
  ran = timed_pass(g, prompt)
  Generate.close(g)
  Qwen3_5.close_model(m)
  timed = try ran
  prompt_rate = (PROMPT * 1000).to_f / timed[0].to_f
  decode_rate = (NEW * 1000).to_f / timed[1].to_f
  ok("RESULT #{load} #{prompt_rate} #{decode_rate}")
end

## The warm-up pass (which compiles the steps), then the timed one.
def timed_pass(g: Generate.Generator, prompt: List[Int]) -> Result[List[Int]]
  try pass(g, prompt)
  pass(g, prompt)
end

## The prompt in one step, then `NEW` steps of one token: the milliseconds of each part.
def pass(g: Generate.Generator, prompt: List[Int]) -> Result[List[Int]]
  started = Time.monotonic_ms
  s = try Generate.step(g, Generate.start(), prompt)
  fed = Time.monotonic_ms - started
  cache = s.cache
  id = s.id
  started = Time.monotonic_ms
  for _ in 0...NEW
    s = try Generate.step(g, cache, [id])
    cache = s.cache
    id = s.id
  end

  decoded = Time.monotonic_ms - started
  Generate.finish(cache)
  ok([fed, decoded])
end
