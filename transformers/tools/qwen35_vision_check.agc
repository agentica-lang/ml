## Checks the vision tower of Qwen3.5 on a real checkpoint against Hugging Face's `Qwen3_5VisionModel`
## (`test/qwen35_vision_check_golden.py`): the merged features of a grid of 6 by 8 patches whose values come from an
## integer formula, in `float32` on ggml. Manual (it needs the checkpoint); run from `transformers/`:
##
##   python -I test/qwen35_vision_check_golden.py MODEL_DIR > GOLDEN.json
##   agentica tools/qwen35_vision_check.agc MODEL_DIR GOLDEN.json [cpu|vulkan]

import Lift from "../lift"
import Load from "pkg/hub/load"
import Qwen3_5Vision from "../qwen3_5_vision"
import Source from "../tokenizer/source"
import Tensor from "pkg/tensor/tensor"

## The gap that passes: about twice the largest measured (2.4e-5 on CPU), `float32` through 12 blocks.
ABS = 0.0001
REL = 0.0001

struct Golden
  grid: List[Int]
  features: List[Float]
end

def main() -> Void
  args = System.args
  if args.size < 2
    puts("usage: qwen35_vision_check.agc MODEL_DIR GOLDEN_JSON [cpu|vulkan]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 2 and args[2] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  case run(args[0], args[1], Tensor.Backend.Ggml(device))
    ok(gap) -> puts("matches Hugging Face, largest gap of a feature #{gap}")
    error(e) ->
      puts("qwen35_vision_check: #{e.message}")
      System.exit(1)
  end
end

private

## The value of the `i`th sample, the formula of the golden script.
def pixel(i: Int) -> Float = ((i * 7919 + 13) % 255).to_f / 127.5 - 1.0

def run(dir: String, golden: String, backend: Tensor.Backend) -> Result[Float]
  want = try JSON.decode(try Source.read_text(golden), Golden)
  try Load.vulkan_fp16_from_env()
  m = try Qwen3_5Vision.open(dir, backend)
  n = want.grid[0] * want.grid[1]
  width = m.config.in_channels * m.config.temporal_patch_size * m.config.patch_size * m.config.patch_size
  patches = try Tensor.floats((0...(n * width)).to_a.map { |i: Int| pixel(i) }, [n, width], backend)
  out = try Qwen3_5Vision.encode(m, patches, want.grid[0], want.grid[1])
  got = try values(out)
  Lift.close([patches, out])
  Qwen3_5Vision.close_model(m)
  return error("#{got.size} features, Hugging Face's are #{want.features.size}") if got.size != want.features.size
  gap = 0.0
  for i in 0...got.size
    d = (got[i] - want.features[i]).abs
    if d > ABS + REL * want.features[i].abs
      return error("feature #{i} is #{got[i]}, Hugging Face's is #{want.features[i]}")
    end

    if d > gap
      gap = d
    end
  end

  ok(gap)
end

def values(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end
