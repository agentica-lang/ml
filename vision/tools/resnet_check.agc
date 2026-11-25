## Checks ResNet and the image pipeline against Hugging Face (`test/resnet_golden.py`): each image of
## `test/fixtures/resnet` goes through `Resize(256)`, `CenterCrop(224)`, `to_tensor` and `normalize`, as the image
## processor of `microsoft/resnet-*` does, and then through the model in `float32` on ggml. It compares the pixels
## with the golden (the mean of each channel and a grid of 8 by 8), the best 10 logits and the top-5 classes. With
## the file of `pixel_values` that the golden script writes, it runs the model on Hugging Face's own pixels too, which
## tests the model alone. Manual (it needs the checkpoints); run from `vision/`:
##
##   python -I test/resnet_golden.py test/fixtures/resnet PIXELS_FILE MODEL_DIR... > GOLDEN.json
##   agentica tools/resnet_check.agc MODEL_DIR test/fixtures/resnet GOLDEN.json [PIXELS_FILE] [cpu|vulkan]
##
## `MODEL_DIR` holds `model.safetensors` and `config.json`, and its folder name is its name in the golden.

import ResNet from "pkg/transformers/resnet"
import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

SIZE = 224
SHORTER = 256
GRID = 8
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

## What passes. The model alone, on Hugging Face's pixels: `float32` through up to 50 layers.
MODEL_ABS = 0.001
MODEL_REL = 0.001

## The pipeline: libvips and Pillow resize with different kernels, so the pixels differ a little (`PIXEL_ABS`, in
## normalized units) and the logits follow.
PIXEL_ABS = 0.1
PIPELINE_ABS = 0.25
PIPELINE_REL = 0.01

struct Pixel
  name: String
  mean: List[Float]
  grid: List[List[List[Float]]]
end

struct Entry
  id: Int
  label: String
  logit: Float
end

struct Golden
  images: List[Pixel]
  models: Map[String, List[List[Entry]]]
end

def main() -> Void
  args = System.args
  if args.size < 3
    puts("usage: resnet_check.agc MODEL_DIR IMAGES_DIR GOLDEN_JSON [PIXELS_FILE] [cpu|vulkan]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  pixels = ""
  for arg in args.drop(3)
    if arg == "vulkan"
      device = Tensor.Device.Vulkan(0)
    elsif arg != "cpu"
      pixels = arg
    end
  end

  case run(args[0], args[1], args[2], pixels, Tensor.Backend.Ggml(device))
    ok(report) -> puts(report)
    error(e) ->
      puts("resnet_check: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, images: String, golden: String, pixels: String, backend: Tensor.Backend) -> Result[String]
  want = try JSON.decode(try File.read_text(golden), Golden)
  name = dir.split("/").select { |part| part != "" }.last
  cases = try want.models.get(name).to_result("the golden has no model #{name}")
  m = try ResNet.open(dir, backend)
  batch = try preprocessed(images, want.images, backend)
  pixel = try pixel_gap(batch, want.images)
  return error("the pixels are #{pixel} from Hugging Face's, more than #{PIXEL_ABS}") if pixel > PIXEL_ABS
  logits = try ResNet.run(m, batch)
  pipeline = try logit_gap("pipeline", m, logits, cases, PIPELINE_ABS, PIPELINE_REL)
  report = "#{name}: largest gap of the pixels #{pixel}, of the logits after the pipeline #{pipeline}"
  if pixels != ""
    own = try Tensor.from_file(pixels, 0, [want.images.size, 3, SIZE, SIZE], backend: backend)
    alone = try logit_gap("model", m, try ResNet.run(m, own), cases, MODEL_ABS, MODEL_REL)
    report = report + ", of the logits of the model alone #{alone}"
  end

  ok(report)
end

## The images of `golden` as one normalized batch `[n, 3, 224, 224]`.
def preprocessed(dir: String, golden: List[Pixel], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  set = Vision.Folder(classes: ["images"], paths: golden.map { |g| "#{dir}/#{g.name}" }, labels: golden.map { |_| 0 })

  steps = [Vision.Step.Resize(SHORTER), Vision.Step.CenterCrop(SIZE, SIZE)]
  batch = try Vision.samples(set, (0...golden.size).to_a, steps, 0, backend)
  Vision.normalize(batch.inputs, MEAN, STD)
end

## The largest gap of the mean of each channel and of the grid of `batch` from the golden.
def pixel_gap(batch: Tensor.Tensor, golden: List[Pixel]) -> Result[Float]
  values = try floats(batch)
  plane = SIZE * SIZE
  worst = 0.0
  for i in 0...golden.size
    for c in 0...3
      base = (i * 3 + c) * plane
      sum = 0.0
      for v in values.drop(base).take(plane)
        sum = sum + v
      end

      worst = [worst, (sum / plane.to_f - golden[i].mean[c]).abs].max.unwrap!
      for y in 0...GRID
        for x in 0...GRID
          at = values[base + step(y) * SIZE + step(x)]
          worst = [worst, (at - golden[i].grid[c][y][x]).abs].max.unwrap!
        end
      end
    end
  end

  ok(worst)
end

## The pixel of the grid that Python's `round(i * (SIZE - 1) / (GRID - 1))` picks.
def step(i: Int) -> Int = (2 * i * (SIZE - 1) + GRID - 1) / (2 * (GRID - 1))

## The largest gap of the best 10 logits of each image from the golden; an error when a logit is past the tolerance or
## the best 5 classes are not the same, in the same order.
def logit_gap(
  what: String,
  m: ResNet.ResNet,
  logits: Tensor.Tensor,
  cases: List[List[Entry]],
  abs: Float,
  rel: Float,
) -> Result[Float]
  values = try floats(logits)
  classes = Tensor.shape(logits)[1]
  best = try ResNet.top(m, logits, 5)
  worst = 0.0
  for i in 0...cases.size
    ours = best[i].map { |p| p.id }
    theirs = cases[i].take(5).map { |e| e.id }
    return error("#{what}, image #{i}: the top 5 is #{ours}, expected #{theirs}") if ours != theirs
    for e in cases[i]
      got = values[i * classes + e.id]
      gap = (got - e.logit).abs
      return error("#{what}, image #{i}, class #{e.id}: #{got}, expected #{e.logit}") if gap > abs + rel * e.logit.abs

      worst = [worst, gap].max.unwrap!
    end
  end

  ok(worst)
end

def floats(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end
