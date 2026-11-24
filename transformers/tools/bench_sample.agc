## Times a decode step of Qwen3 on ggml with the greedy pick in the graph against a sampled one (the logits read back
## and `Sample.pick` on the host), and splits the sampled step into the readback and the pick. Prints
## `greedy MS`, `sampled MS`, `readback MS` and `pick MS` per step (the means of 32 one-token steps after a prompt of
## 128 ids). Manual. Run from `transformers/`:
##
##   agentica tools/bench_sample.agc MODEL_DIR [cpu|vulkan]

import Generate from "../generate"
import Qwen3 from "../qwen3"
import Sample from "../sample"
import Tensor from "pkg/tensor/tensor"

PROMPT = 128
STEPS = 32

def main() -> Void
  args = System.args
  if args.size < 1
    puts("usage: bench_sample.agc MODEL_DIR [cpu|vulkan]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 1 and args[1] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  case run(args[0], Tensor.Backend.Ggml(device))
    ok(_) -> nil
    error(e) ->
      puts("bench_sample: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, backend: Tensor.Backend) -> Result[Void]
  m = try Qwen3.open(dir, backend)
  g = try Generate.open(m)
  prompt = (0...PROMPT).map { |i| 1000 + (i * 7) % 5000 }
  try greedy(g, prompt)
  try logits(g, prompt)
  greedy_ms = try greedy(g, prompt)
  both = try logits(g, prompt)
  puts("greedy #{greedy_ms.to_f / STEPS.to_f}")
  puts("sampled #{both[0].to_f / STEPS.to_f}")
  puts("readback #{(both[0] - both[1]).to_f / STEPS.to_f}")
  puts("pick #{both[1].to_f / STEPS.to_f}")
  Generate.close(g)
  Qwen3.close_model(m)
  ok()
end

## The milliseconds of `STEPS` greedy steps.
def greedy(g: Generate.Generator, prompt: List[Int]) -> Result[Int]
  s = try Generate.step(g, Generate.start(), prompt)
  cache = s.cache
  id = s.id
  started = Time.monotonic_ms
  for _ in 0...STEPS
    s = try Generate.step(g, cache, [id])
    cache = s.cache
    id = s.id
  end

  took = Time.monotonic_ms - started
  Generate.finish(cache)
  ok(took)
end

## The milliseconds of `STEPS` steps that read the logits, and of the picks among them.
def logits(g: Generate.Generator, prompt: List[Int]) -> Result[List[Int]]
  sampling = Sample.Sampling(temperature: 0.8, top_k: 40, top_p: 0.95, seed: 7)
  state = Sample.start(sampling)
  s = try Generate.step_logits(g, Generate.start(), prompt)
  cache = s.cache
  last = try Sample.pick(s.logits, sampling, state)
  picking = 0
  started = Time.monotonic_ms
  for _ in 0...STEPS
    s = try Generate.step_logits(g, cache, [last.id])
    cache = s.cache
    began = Time.monotonic_ms
    last = try Sample.pick(s.logits, sampling, last.state)
    picking += Time.monotonic_ms - began
  end

  took = Time.monotonic_ms - started
  Generate.finish(cache)
  ok([took, picking])
end
