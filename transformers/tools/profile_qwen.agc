## Splits a Qwen3 step on ggml with the profile of the VM and the per-kernel timer of ggml-vulkan. The model is
## loaded once and one warm-up pass (a prompt of 128 ids, then 8 one-token steps) compiles the steps; then PREFILLS
## more prompts and DECODES more one-token steps run, which a report with both at 0 does not hold, so
## `benchmarks/step_profile.py` can subtract the two reports (`AGENTICA_VM_TRACE=profile`, the `ml.*` sections). With
## `GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1` ggml-vulkan prints the GPU time of each kernel of each
## graph to stderr; the `MARK` lines on stderr tell where the timed steps start. Manual. From `transformers/`:
##
##   agentica tools/profile_qwen.agc MODEL_DIR PREFILLS DECODES [cpu|vulkan|vulkan32]
##
## `vulkan` multiplies in fp16 (the default), `vulkan32` in exact `float32`.

import Generate from "../generate"
import Qwen3 from "../qwen3"
import Tensor from "pkg/tensor/tensor"

PROMPT = 128

def main() -> Void
  args = System.args
  if args.size < 3
    puts("usage: profile_qwen.agc MODEL_DIR PREFILLS DECODES [cpu|vulkan|vulkan32]")
    System.exit(2)
  end

  device = Tensor.Device.Vulkan(0)
  if args.size > 3 and args[3] == "cpu"
    device = Tensor.Device.Cpu
  end

  if args.size > 3 and args[3] == "vulkan32"
    _ = Tensor.vulkan_fp16(false)
  end

  case run(args[0], args[1].to_i.unwrap!, args[2].to_i.unwrap!, Tensor.Backend.Ggml(device))
    ok(_) -> nil
    error(e) ->
      puts("profile_qwen: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, prefills: Int, decodes: Int, backend: Tensor.Backend) -> Result[Void]
  m = try Qwen3.open(dir, backend)
  g = try Generate.open(m)
  prompt = (0...PROMPT).map { |i| 1000 + (i * 7) % 5000 }
  s = try Generate.step(g, Generate.start(), prompt)
  cache = s.cache
  id = s.id
  for _ in 0...8
    s = try Generate.step(g, cache, [id])
    cache = s.cache
    id = s.id
  end

  IO.warn("MARK prefill #{prefills}")
  started = Time.monotonic_ms
  for _ in 0...prefills
    p = try Generate.step(g, Generate.start(), prompt)
    Generate.finish(p.cache)
  end

  puts("prefills #{prefills} #{Time.monotonic_ms - started} ms")
  IO.warn("MARK decode #{decodes}")
  started = Time.monotonic_ms
  for _ in 0...decodes
    s = try Generate.step(g, cache, [id])
    cache = s.cache
    id = s.id
  end

  puts("decodes #{decodes} #{Time.monotonic_ms - started} ms")
  IO.warn("MARK end")
  Generate.finish(cache)
  Generate.close(g)
  Qwen3.close_model(m)
  ok()
end
