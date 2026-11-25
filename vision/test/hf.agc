## What the tests against Hugging Face share (`resnet_hf_test.agc`, `vit_hf_test.agc`): the golden of their scripts
## (the pixels that the image processor made, and the best 10 logits), the images through `vision`, and the gaps.

import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

SIZE = 224
GRID = 8

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

## `RESNET_DEVICE`-style choice: Vulkan when `name` is set to `vulkan`, the CPU otherwise.
def device_named(name: String) -> Result[Tensor.Backend]
  device = Tensor.Device.Cpu
  chosen = System.env(name)
  if chosen.ok? and try chosen == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  try Tensor.vulkan_fp16(false)
  ok(Tensor.Backend.Ggml(device))
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
## the best 5 class ids `best` are not Hugging Face's, in the same order. Without `ordered` they may be reordered by the
## near-ties of the logits: each of ours is in Hugging Face's best 10, and each class of Hugging Face's best 10 that is
## more than `abs` above its sixth is in ours.
def logit_gap(
  what: String,
  best: List[List[Int]],
  logits: Tensor.Tensor,
  cases: List[List[Entry]],
  abs: Float,
  rel: Float,
  ordered: Bool = true,
) -> Result[Float]
  values = try floats(logits)
  classes = Tensor.shape(logits)[1]
  worst = 0.0
  for i in 0...cases.size
    ours = best[i]
    theirs = cases[i].take(5).map { |e| e.id }
    if ordered
      return error("#{what}, image #{i}: the top 5 is #{ours}, expected #{theirs}") if ours != theirs
    else
      try tied_top(what, i, ours, cases[i], abs)
    end

    for e in cases[i]
      got = values[i * classes + e.id]
      gap = (got - e.logit).abs
      return error("#{what}, image #{i}, class #{e.id}: #{got}, expected #{e.logit}") if gap > abs + rel * e.logit.abs

      worst = [worst, gap].max.unwrap!
    end
  end

  ok(worst)
end

## `ours` (the best 5 ids) against the best 10 of Hugging Face, `want`, allowing for ties within `abs` of its sixth.
def tied_top(what: String, image: Int, ours: List[Int], want: List[Entry], abs: Float) -> Result[Void]
  sixth = want[5].logit
  for id in ours
    if !want.any? { |e| e.id == id }
      return error("#{what}, image #{image}: class #{id} of the top 5 is not in Hugging Face's best 10")
    end
  end

  for e in want
    if e.logit > sixth + abs and !ours.include?(e.id)
      return error("#{what}, image #{image}: class #{e.id} (#{e.logit}) is clearly in the top 5 and is not in #{ours}")
    end
  end

  ok()
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
