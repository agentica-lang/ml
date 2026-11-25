## Times CLIP on ggml: the image embeddings of a batch of `BATCH` images of 224 by 224, or the text embeddings of `BATCH`
## rows of 77 ids (the longest a prompt can be), as `Clip.image_embeds` and `Clip.text_embeds` give them (one warm-up
## run first, then the median of `REPS` runs). Vulkan multiplies in `F32`, as the checks do
## (`Tensor.vulkan_fp16(false)`). Manual; run from `transformers/`:
##
##   agentica --max-steps 1000000000000 tools/bench_clip.agc MODEL_DIR [cpu|vulkan] [BATCH] [REPS] [image|text]

import Clip from "../clip"
import Tensor from "pkg/tensor/tensor"

SIDE = 224
LENGTH = 77

def main() -> Void
  args = System.args
  if args.empty?
    puts("usage: bench_clip.agc MODEL_DIR [cpu|vulkan] [BATCH] [REPS] [image|text]")
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

  text = args.size > 4 and args[4] == "text"
  case measure(args[0], Tensor.Backend.Ggml(device), batch, reps, text)
    ok(line) -> puts(line)
    error(e) ->
      puts("bench_clip: #{e.message}")
      System.exit(1)
  end
end

private

def measure(dir: String, backend: Tensor.Backend, batch: Int, reps: Int, text: Bool) -> Result[String]
  try Tensor.vulkan_fp16(false)
  started = Time.monotonic_ms
  m = try Clip.open(dir, backend)
  load = Time.monotonic_ms - started
  images = try Tensor.full([batch, 3, SIDE, SIDE], Tensor.Values.Float(0.5), Tensor.Dtype.F32, backend)
  rows = (0...batch).to_a.map { |_| List.repeat(320, LENGTH) }
  step = -> { image_step(m, images) }
  what = "image embeddings"
  unit = "images"
  if text
    step = -> { text_step(m, rows) }
    what = "text embeddings of #{LENGTH} ids"
    unit = "texts"
  end

  try step.call()
  times: List[Int] = []
  for _ in 0...reps
    started = Time.monotonic_ms
    try step.call()
    times = times.push(Time.monotonic_ms - started)
  end

  median = times.sort[times.size / 2]
  ok(
    "load #{load} ms; #{what}, batch #{batch}: median #{median} ms of #{reps} runs, #{tenths(batch * 10000 / [median, 1].max.unwrap!)} #{unit}/s",
  )
end

## `n` tenths as a decimal number.
def tenths(n: Int) -> String = "#{n / 10}.#{n % 10}"

def image_step(m: Clip.Clip, images: Tensor.Tensor) -> Result[Void]
  Clip.close(try Clip.image_embeds(m, images))
  ok()
end

def text_step(m: Clip.Clip, rows: List[List[Int]]) -> Result[Void]
  Clip.close(try Clip.text_embeds(m, rows))
  ok()
end
