## Runs STEPS one-token steps of `Generate` after a prompt of 16 ids on `Qwen3-0.6B` and prints the resident memory of
## the process every 100 steps: it must stay flat once the cache has its capacity (the cache grows with the length, so
## expect a step up at each power of two of rows, then no change). Manual (it needs the checkpoint); from `transformers/`:
##
##   agentica tools/leak_check.agc MODEL_DIR STEPS [cpu|vulkan] [device|host]
##
## `host` opens the generator with `kv: Placement.Kv.Host`: the cache then grows in host memory.

import Generate from "../generate"
import Placement from "../placement"
import Qwen3 from "../qwen3"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  args = System.args
  if args.size < 2
    puts("usage: leak_check.agc MODEL_DIR STEPS [cpu|vulkan] [device|host]")
    System.exit(2)
  end

  device = Tensor.Device.Vulkan(0)
  if args.size > 2 and args[2] == "cpu"
    device = Tensor.Device.Cpu
  end

  kv = Placement.Kv.Device
  if args.size > 3 and args[3] == "host"
    kv = Placement.Kv.Host
  end

  case run(args[0], args[1].to_i.unwrap!, Tensor.Backend.Ggml(device), kv)
    ok(_) -> nil
    error(e) ->
      puts("leak_check: #{e.message}")
      System.exit(1)
  end
end

private

## The resident set of this process, from /proc/self/status.
def resident() -> String
  text = File.read_text("/proc/self/status").unwrap_or("")
  for line in text.split("\n")
    return line.replace("\t", " ").replace("  ", " ") if line.start_with?("VmRSS")
  end

  "no VmRSS"
end

def run(dir: String, steps: Int, backend: Tensor.Backend, kv: Placement.Kv) -> Result[Void]
  m = try Qwen3.open(dir, backend)
  g = try Generate.open(m, kv: kv)
  prompt = (0...16).map { |i| 1000 + (i * 7) % 5000 }
  s = try Generate.step(g, Generate.start(), prompt)
  for i in 1...(steps + 1)
    s = try Generate.step(g, s.cache, [s.id])
    if i % 100 == 0
      puts("step #{i}: rows #{Tensor.shape(s.cache.keys[0])[2]}, #{resident()}")
    end
  end

  Generate.finish(s.cache)
  Generate.close(g)
  Qwen3.close_model(m)
  ok()
end
