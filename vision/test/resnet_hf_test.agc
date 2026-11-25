## ResNet and the image pipeline against Hugging Face (`resnet_golden.py`): each image of `fixtures/resnet` goes through
## `Resize(256)`, `CenterCrop(224)` and `normalize`, as the image processor of `microsoft/resnet-*` does, and then
## through the model in `float32` on ggml. The pixels are compared with the golden (the mean of each channel and a grid
## of 8 by 8), the best 10 logits and the top-5 classes too. The 8-bit crops that Hugging Face's processor made
## (`fixtures/resnet/crops`) go through the model alone, which is the strict check.
##
## It needs the checkpoints, which are not in the repository, so it runs only when `RESNET_MODELS` names them: a list of
## folders (the revisions of `transformers/DESIGN.md`, each with `model.safetensors` and `config.json`) separated by
## `:`, and the folder name is the model's name in the golden. `RESNET_DEVICE=vulkan` runs on Vulkan, the CPU otherwise.
## Run from `vision/`:
## `RESNET_MODELS=resnet-50:resnet-18 agentica test --max-steps 1000000000000 test/resnet_hf_test.agc`.

import ResNet from "pkg/transformers/resnet"
import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

SIZE = 224
SHORTER = 256
GRID = 8
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

## What passes. The pixels and the logits after the pipeline have about twice the gap measured (0.0175 and 0.158): the
## pipeline differs from Hugging Face's in the resize (libvips and Pillow use different kernels), and the logits follow
## the pixels. The model alone is `float32` through up to 50 layers and measured 1.6e-5 on CPU and 2.1e-5 on Vulkan; its
## 1e-3 is looser on purpose, for the other machines and ggml builds (the sums of a convolution depend on the kernel).
MODEL_ABS = 0.001
MODEL_REL = 0.001
PIXEL_ABS = 0.035
PIPELINE_ABS = 0.32

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

private

def test_the_model_and_the_pipeline_match_hugging_face() -> Result[Void]
  listed = System.env("RESNET_MODELS")
  if !listed.ok? or try listed == ""
    puts("resnet_hf_test: skipped, RESNET_MODELS names no checkpoint folder")
    return ok()
  end

  device = Tensor.Device.Cpu
  chosen = System.env("RESNET_DEVICE")
  if chosen.ok? and try chosen == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  try Tensor.vulkan_fp16(false)
  backend = Tensor.Backend.Ggml(device)
  return error("RESNET_MODELS is set and ggml is not built") if !Tensor.available(backend)
  want = try JSON.decode(try File.read_text("test/fixtures/resnet/golden.json"), Golden)
  for dir in (try listed).split(":")
    report = try check(dir, want, backend)
    puts(report)
  end

  ok()
end

def check(dir: String, want: Golden, backend: Tensor.Backend) -> Result[String]
  name = dir.split("/").select { |part| part != "" }.last
  cases = try want.models.get(name).to_result("the golden has no model #{name}")
  m = try ResNet.open(dir, backend)
  names = want.images.map { |g| g.name }
  crops = try loaded("test/fixtures/resnet/crops", names, [], backend)
  alone = try logit_gap(
    "model alone",
    m,
    try ResNet.run(m, try Vision.normalize(crops, MEAN, STD)),
    cases,
    MODEL_ABS,
    MODEL_REL,
  )

  pixels = try preprocessed("test/fixtures/resnet", names, backend)
  pixel = try pixel_gap(pixels, want.images)
  return error("the pixels are #{pixel} from Hugging Face's, more than #{PIXEL_ABS}") if pixel > PIXEL_ABS
  pipeline = try logit_gap("pipeline", m, try ResNet.run(m, pixels), cases, PIPELINE_ABS, 0.0)
  ok(
    "#{name}: largest gap of the logits of the model alone #{alone}, of the pixels #{pixel}, of the logits after the pipeline #{pipeline}",
  )
end

## The images `names` of `dir` after `steps`, as one `[n, 3, h, w]` batch in `0..1`.
def loaded(
  dir: String,
  names: List[String],
  steps: List[Vision.Step],
  backend: Tensor.Backend,
) -> Result[Tensor.Tensor]
  set = Vision.Folder(classes: ["images"], paths: names.map { |n| "#{dir}/#{n}" }, labels: names.map { |_| 0 })
  batch = try Vision.samples(set, (0...names.size).to_a, steps, 0, backend)
  ok(batch.inputs)
end

## The images as the image processor of the checkpoints makes them: normalized `[n, 3, 224, 224]`.
def preprocessed(dir: String, names: List[String], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  steps = [Vision.Step.Resize(SHORTER), Vision.Step.CenterCrop(SIZE, SIZE)]
  Vision.normalize(try loaded(dir, names, steps, backend), MEAN, STD)
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
