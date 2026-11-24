## Checks greedy generation with the cache (`generate.agc`) against Hugging Face's
## `generate` (`test/qwen_golden.py`) on a Qwen3 or Qwen2.5 checkpoint in `float32` on ggml: the ids
## it adds to each prompt must be the same, and the speed is printed: the first
## call (which traces and plans the steps), then a second one with the steps
## compiled. Manual (it needs the checkpoint); run from `transformers/`:
##
##   python -I test/qwen_golden.py MODEL_DIR > GOLDEN.json
##   agentica tools/generate_check.agc MODEL_DIR GOLDEN.json [cpu|vulkan]

import Generate from "../generate"
import Qwen3 from "../qwen3"
import Load from "pkg/hub/load"
import Source from "../tokenizer/source"
import Tensor from "pkg/tensor/tensor"

## The end ids of `generation_config.json`: `<|im_end|>` and `<|endoftext|>`.
STOP = [151645, 151643]

struct Case
  ids: List[Int]
  generated: List[Int]
  gaps: List[Float]
end

struct Golden
  cases: List[Case]
end

def main() -> Void
  args = System.args
  if args.size < 2
    puts("usage: generate_check.agc MODEL_DIR GOLDEN_JSON [cpu|vulkan]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 2 and args[2] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  case run(args[0], args[1], Tensor.Backend.Ggml(device))
    ok(_) -> puts("generates what Hugging Face does")
    error(e) ->
      puts("generate_check: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, golden: String, backend: Tensor.Backend) -> Result[Void]
  want = try JSON.decode(try Source.read_text(golden), Golden)
  try Load.vulkan_fp16_from_env()
  m = try Qwen3.open(dir, backend, try Load.weights_from_env())
  puts("model_type #{m.config.model_type}")
  g = try Generate.open(m)
  done = checked(g, want)
  Generate.close(g)
  Qwen3.close_model(m)
  puts(memory())
  done
end

def checked(g: Generate.Generator, want: Golden) -> Result[Void]
  for i in 0...want.cases.size
    c = want.cases[i]
    started = Time.monotonic_ms
    first = try Generate.generate(g, c.ids, c.generated.size, STOP)
    cold = Time.monotonic_ms - started
    started = Time.monotonic_ms
    _ = try Generate.generate(g, c.ids, 1, STOP)
    prompt = Time.monotonic_ms - started
    started = Time.monotonic_ms
    again = try Generate.generate(g, c.ids, c.generated.size, STOP)
    warm = Time.monotonic_ms - started
    # Hugging Face keeps the end id it stopped at; ours stops before it.
    expect = c.generated.filter { |id| not STOP.include?(id) }
    return error("prompt #{i}: ids #{again}, expected #{expect}") if again != expect
    return error("prompt #{i}: the first call gave #{first}, the second #{again}") if first != again
    steps = again.size - 1
    per = 0.0
    if steps > 0 and warm > prompt
      per = (steps * 1000).to_f / (warm - prompt).to_f
    end

    puts(
      "prompt #{i}: #{c.ids.size} prompt tokens, #{again.size} new, smallest gap of HF #{c.gaps.min.unwrap!}; " +
      "first call #{cold} ms, then prompt and one token #{prompt} ms, #{again.size} tokens #{warm} ms, " +
      "#{per} tokens/s after the prompt",
    )
  end

  ok()
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
