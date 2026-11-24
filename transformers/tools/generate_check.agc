## Checks greedy generation with the cache (`generate.agc`) against Hugging Face's
## `generate` (`test/qwen_golden.py`) on `Qwen3-0.6B` in `float32` on ggml: the ids
## it adds to each prompt must be the same, and the speed is printed: the first
## call (which traces and plans the steps), then a second one with the steps
## compiled. Manual (it needs the checkpoint); run from `transformers/`:
##
##   python -I test/qwen_golden.py MODEL_DIR > GOLDEN.json
##   agentica tools/generate_check.agc MODEL_DIR GOLDEN.json [cpu|vulkan]

import Generate from "../generate"
import Qwen3 from "../qwen3"
import Source from "../tokenizer/source"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

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
    ok(_) -> puts("qwen3: generates what Hugging Face does")
    error(e) ->
      puts("qwen3: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, golden: String, backend: Tensor.Backend) -> Result[Void]
  want = try JSON.decode(try Source.read_text(golden), Golden)
  m = try Qwen3.open(dir, backend)
  done = checked(m, want)
  Qwen3.close_model(m)
  puts(memory())
  done
end

def checked(m: Qwen3.Qwen3, want: Golden) -> Result[Void]
  graphs: Map[String, Trace.Compiled] = {}
  for i in 0...want.cases.size
    c = want.cases[i]
    started = Time.monotonic_ms
    first = try Generate.generate(m, c.ids, c.generated.size, STOP, graphs)
    cold = Time.monotonic_ms - started
    started = Time.monotonic_ms
    one = try Generate.generate(m, c.ids, 1, STOP, first.graphs)
    prompt = Time.monotonic_ms - started
    started = Time.monotonic_ms
    again = try Generate.generate(m, c.ids, c.generated.size, STOP, one.graphs)
    warm = Time.monotonic_ms - started
    graphs = again.graphs
    # Hugging Face keeps the end id it stopped at; ours stops before it.
    expect = c.generated.filter { |id| not STOP.include?(id) }
    return error("prompt #{i}: ids #{again.ids}, expected #{expect}") if again.ids != expect
    return error("prompt #{i}: the first call gave #{first.ids}, the second #{again.ids}") if first.ids != again.ids
    steps = again.ids.size - 1
    per = 0.0
    if steps > 0 and warm > prompt
      per = (steps * 1000).to_f / (warm - prompt).to_f
    end

    puts(
      "prompt #{i}: #{c.ids.size} prompt tokens, #{again.ids.size} new, smallest gap of HF #{c.gaps.min.unwrap!}; " +
      "first call #{cold} ms, then prompt and one token #{prompt} ms, #{again.ids.size} tokens #{warm} ms, " +
      "#{per} tokens/s after the prompt",
    )
  end

  Generate.close_graphs(graphs)
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
