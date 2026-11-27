## Times ViT on ggml: the forward of a batch of `BATCH` images of 224 by 224 (one warm-up run first, then the median of
## `REPS` runs). Vulkan multiplies in `F32`, as the checks do (`Tensor.vulkan_fp16(false)`). Manual; run from
## `transformers/`:
##
##   agentica tools/bench_vit.agc MODEL_DIR [cpu|vulkan] [BATCH] [REPS]

import Tensor from "pkg/tensor/tensor"
import ViT from "../vit"

SIDE = 224

def main() -> Void
  args = System.args
  if args.empty?
    puts("usage: bench_vit.agc MODEL_DIR [cpu|vulkan] [BATCH] [REPS]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 1 and args[1] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  batch = 1
  reps = 5
  if args.size > 2
    batch = args[2].to_i!
  end

  if args.size > 3
    reps = args[3].to_i!
  end

  case measure(args[0], Tensor.Backend.Ggml(device), batch, reps)
    ok(line) -> puts(line)
    error(e) ->
      puts("bench_vit: #{e.message}")
      System.exit(1)
  end
end

private

def measure(dir: String, backend: Tensor.Backend, batch: Int, reps: Int) -> Result[String]
  try Tensor.vulkan_fp16(false)
  started = Time.monotonic_ms
  m = try ViT.open(dir, backend)
  load = Time.monotonic_ms - started
  images = try Tensor.full([batch, 3, SIDE, SIDE], Tensor.Values.Float(0.5), Tensor.Dtype.F32, backend)
  try forward(m, images)
  times: List[Int] = []
  for _ in 0...reps
    started = Time.monotonic_ms
    try forward(m, images)
    times = times.push(Time.monotonic_ms - started)
  end

  median = times.sort[times.size / 2]
  ok(
    "load #{load} ms; forward, batch #{batch}: median #{median} ms of #{reps} runs, #{tenths(batch * 10000 / [median, 1].max.unwrap!)} images/s",
  )
end

## `n` tenths as a decimal number.
def tenths(n: Int) -> String = "#{n / 10}.#{n % 10}"

def forward(m: ViT.ViT, images: Tensor.Tensor) -> Result[Void]
  ViT.close(try ViT.run(m, images))
  ok()
end
