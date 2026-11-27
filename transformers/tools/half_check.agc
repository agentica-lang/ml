## Prints the greedy ids that `generate.agc` makes for the prompts of a Hugging Face golden (`test/qwen_golden.py`) on a
## Qwen3 or Qwen2.5 checkpoint, with the weights as `F32` or `F16`, on ggml CPU or XLA, and the peak resident memory. Run it
## once per backend and compare the `ids` lines: the ids of `F16` weights on XLA must be those of `F16` weights on ggml.
## Manual (it needs the checkpoint); run from `transformers/`:
##
##   agentica tools/half_check.agc MODEL_DIR GOLDEN_JSON cpu|xla f32|f16 [PROMPTS]
##
## PROMPTS is how many prompts of the golden to run, from the first (default all).

import Generate from "../generate"
import Qwen3 from "../qwen3"
import Source from "../tokenizer/source"
import Tensor from "pkg/tensor/tensor"

## The end ids of `generation_config.json`: `<|im_end|>` and `<|endoftext|>`.
STOP = [151645, 151643]

struct Case
  ids: List[Int]
  generated: List[Int]
end

struct Golden
  cases: List[Case]
end

def main() -> Void
  args = System.args
  if args.size < 4 or args.size > 5 or !["cpu", "xla"].include?(args[2]) or !["f32", "f16"].include?(args[3])
    puts("usage: half_check.agc MODEL_DIR GOLDEN_JSON cpu|xla f32|f16 [PROMPTS]")
    System.exit(2)
  end

  backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  if args[2] == "xla"
    backend = Tensor.Backend.Xla(Tensor.Device.Cpu)
  end

  weights = Tensor.Dtype.F32
  if args[3] == "f16"
    weights = Tensor.Dtype.F16
  end

  prompts = 1000
  if args.size == 5
    prompts = args[4].to_i.unwrap_or(1000)
  end

  case run(args[0], args[1], backend, weights, prompts)
    ok(_) -> puts("done")
    error(e) ->
      puts("half_check: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, golden: String, backend: Tensor.Backend, weights: Tensor.Dtype, prompts: Int) -> Result[Void]
  want = try JSON.decode(try Source.read_text(golden), Golden)
  m = try Qwen3.open(dir, backend, weights)
  puts("loaded: #{memory()}")
  g = try Generate.open(m)
  for i in 0...[want.cases.size, prompts].min.unwrap!
    c = want.cases[i]
    made = try Generate.generate(g, c.ids, c.generated.size, STOP)
    puts("ids #{i}: #{made}")
  end

  Generate.close(g)
  Qwen3.close_model(m)
  puts(memory())
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
