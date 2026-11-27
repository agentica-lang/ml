## Times Qwen3 generation on ggml (or XLA) for `benchmarks/bench_qwen.sh`: the load, then one warm-up pass
## (which compiles the steps), then one timed pass: a prompt of 128 token ids in one step, and 64 one-token
## steps after it. Prints `RESULT load_ms prompt_tokens_per_s decode_tokens_per_s`, and on the CPU `THREADS n` first (the
## threads of the ggml CPU backend: `TENSOR_THREADS`, else the physical cores). Run from `transformers/`:
##
##   agentica tools/bench_qwen.agc MODEL_DIR [cpu|vulkan|xla]
##
## `xla` runs on `Backend.Xla` with the plugin of the cache (the CPU one, or a CUDA one put there). `WEIGHTS=f16`
## (or `bf16`) loads the big matrices as half weights (off by default) and `VULKAN_FP16=0` makes Vulkan multiply in
## exact `float32` (fp16 is the default). `AGENTICA_ML_KV=host` keeps the KV cache in host memory (`Placement.Kv.Host`).
## `AGENTICA_ML_MAX_DEVICE_BYTES=n` puts on a Vulkan device the most blocks that fit `n` bytes of it (`Qwen3.layers_that_fit`,
## printed as `LAYERS`), and it excludes `AGENTICA_ML_LAYERS_ON_DEVICE`.

import Generate from "../generate"
import Placement from "../placement"
import Qwen3 from "../qwen3"
import Load from "pkg/hub/load"
import Lower from "pkg/tensor/src/lower"
import Tensor from "pkg/tensor/tensor"

PROMPT = 128
NEW = 64

def main() -> Void
  args = System.args
  if args.size < 1
    puts("usage: bench_qwen.agc MODEL_DIR [cpu|vulkan|xla]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 1 and args[1] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  backend = Tensor.Backend.Ggml(device)
  if args.size > 1 and args[1] == "xla"
    backend = Tensor.Backend.Xla(Tensor.Device.Cpu)
  end

  case run(args[0], backend, backend == Tensor.Backend.Ggml(Tensor.Device.Cpu))
    ok(line) -> puts(line)
    error(e) ->
      puts("bench_qwen: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, backend: Tensor.Backend, cpu: Bool) -> Result[String]
  started = Time.monotonic_ms
  try Load.vulkan_fp16_from_env(true)
  weights = try Load.weights_from_env()
  layers = try Load.layers_on_device_from_env()
  cap = try Load.max_device_bytes_from_env()
  if cap.some?
    if layers.some?
      return error("AGENTICA_ML_MAX_DEVICE_BYTES and AGENTICA_ML_LAYERS_ON_DEVICE are two ways to say one thing")
    end

    layers = try Qwen3.layers_that_fit(dir, backend, weights, cap, PROMPT + NEW)
    puts("LAYERS #{layers}")
  end

  m = try Qwen3.open(dir, backend, weights, layers)
  load = Time.monotonic_ms - started
  g = try Generate.open(m, kv: try Placement.kv_from_env())
  prompt = (0...PROMPT).map { |i| 1000 + (i * 7) % 5000 }
  try pass(g, prompt)
  timed = try pass(g, prompt)
  Generate.close(g)
  Qwen3.close_model(m)
  prompt_rate = (PROMPT * 1000).to_f / timed[0].to_f
  decode_rate = (NEW * 1000).to_f / timed[1].to_f
  if cpu
    puts("THREADS #{try Lower.cpu_threads()}")
  end

  ok("RESULT #{load} #{prompt_rate} #{decode_rate}")
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
