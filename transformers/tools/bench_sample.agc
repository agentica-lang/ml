## Times a decode step of Qwen3 on ggml with the greedy pick in the graph against a sampled one (the `top_k` cut in the graph,
## then `Sample.pick_top` on the host), and splits the sampled step into the graph run with its readback and the pick. Prints
## `greedy MS`, `sampled MS`, `graph MS` and `pick MS` per step with the top_k of the draws (40 unless given; 0 is the window of 256) (the means of 32 one-token steps after a prompt of
## 128 ids). Manual. Run from `transformers/`:
##
##   agentica tools/bench_sample.agc MODEL_DIR [cpu|vulkan] [TOP_K]

import Generate from "../generate"
import Qwen3 from "../qwen3"
import Sample from "../sample"
import Tensor from "pkg/tensor/tensor"

PROMPT = 128
STEPS = 32

def main() -> Void
  args = System.args
  if args.size < 1
    puts("usage: bench_sample.agc MODEL_DIR [cpu|vulkan] [TOP_K]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 1 and args[1] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  top_k = 40
  if args.size > 2
    top_k = args[2].to_i.unwrap_or(40)
  end

  case run(args[0], Tensor.Backend.Ggml(device), top_k)
    ok(_) -> nil
    error(e) ->
      puts("bench_sample: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, backend: Tensor.Backend, top_k: Int) -> Result[Void]
  m = try Qwen3.open(dir, backend)
  g = try Generate.open(m)
  prompt = (0...PROMPT).map { |i| 1000 + (i * 7) % 5000 }
  try greedy(g, prompt)
  try logits(g, prompt, top_k)
  greedy_ms = try greedy(g, prompt)
  both = try logits(g, prompt, top_k)
  puts("greedy #{greedy_ms.to_f / STEPS.to_f}")
  puts("sampled #{both[0].to_f / STEPS.to_f}")
  puts("graph #{(both[0] - both[1]).to_f / STEPS.to_f}")
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

## The milliseconds of `STEPS` sampled steps (the in-graph cut and `Sample.pick_top`), and of the picks among them.
def logits(g: Generate.Generator, prompt: List[Int], top_k: Int) -> Result[List[Int]]
  sampling = Sample.Sampling(temperature: 0.8, top_k: top_k, top_p: 0.95, seed: 7)
  k = Sample.window(sampling, g.model.config.vocab_size)
  state = Sample.start(sampling)
  s = try Generate.step_top(g, Generate.start(), prompt, k)
  cache = s.cache
  last = try Sample.pick_top(s.ids, s.logits, sampling, state)
  picking = 0
  started = Time.monotonic_ms
  for _ in 0...STEPS
    s = try Generate.step_top(g, cache, [last.id], k)
    cache = s.cache
    began = Time.monotonic_ms
    last = try Sample.pick_top(s.ids, s.logits, sampling, last.state)
    picking += Time.monotonic_ms - began
  end

  took = Time.monotonic_ms - started
  Generate.finish(cache)
  ok([took, picking])
end
