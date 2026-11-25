## What the tests against Hugging Face share (`resnet_hf_test.agc`, `vit_hf_test.agc`): the golden of their scripts
## (the pixels that the image processor made, and the best 10 logits), the images through `vision`, and the gaps.

import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

SIZE = 224
GRID = 8

## A value that is off by more than this is not Hugging Face's: the smallest level of any model here is 0.0078, and our
## samples are the same bytes up to float rounding.
LEVEL_FRACTION = 0.001

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

## What `pixel_gap` measures, in normalized units: the largest gap, the mean gap and the share of the values (the
## means and the grid points) that are not Hugging Face's at all. The pixels are 8-bit, so a gap is a whole number of
## levels, and a level is `1 / 255 / std` (0.0175 for ResNet's red channel).
struct PixelGap
  largest: Float
  mean: Float
  differing: Float
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

def describe(g: PixelGap) -> String = "#{g.largest} (mean #{g.mean}, #{g.differing * 100.0}% of the values differ)"

## The gap of the mean of each channel and of the grid of `batch` from the golden.
def pixel_gap(batch: Tensor.Tensor, golden: List[Pixel]) -> Result[PixelGap]
  values = try floats(batch)
  plane = SIZE * SIZE
  worst = 0.0
  total = 0.0
  count = 0
  differing = 0
  for i in 0...golden.size
    for c in 0...3
      base = (i * 3 + c) * plane
      sum = 0.0
      for v in values.drop(base).take(plane)
        sum = sum + v
      end

      gaps = [(sum / plane.to_f - golden[i].mean[c]).abs]
      for y in 0...GRID
        for x in 0...GRID
          at = values[base + step(y) * SIZE + step(x)]
          gaps = gaps.push((at - golden[i].grid[c][y][x]).abs)
        end
      end

      for gap in gaps
        worst = [worst, gap].max.unwrap!
        total = total + gap
        count = count + 1
        if gap > LEVEL_FRACTION
          differing = differing + 1
        end
      end
    end
  end

  ok(PixelGap(largest: worst, mean: total / count.to_f, differing: differing.to_f / count.to_f))
end

## The pixel of the grid that Python's `round(i * (SIZE - 1) / (GRID - 1))` picks.
def step(i: Int) -> Int = (2 * i * (SIZE - 1) + GRID - 1) / (2 * (GRID - 1))

## The largest gap of the best 10 logits of each image from the golden; an error when a logit is past the tolerance or
## the best 5 class ids `best` are not Hugging Face's, in the same order.
def logit_gap(
  what: String,
  best: List[List[Int]],
  logits: Tensor.Tensor,
  cases: List[List[Entry]],
  abs: Float,
  rel: Float,
) -> Result[Float]
  values = try floats(logits)
  classes = Tensor.shape(logits)[1]
  worst = 0.0
  for i in 0...cases.size
    ours = best[i]
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

## The largest gap of `t` (`[rows, columns]`) from `want`; an error when one is past the tolerance `abs` plus `rel` of
## the value wanted.
def matrix_gap(what: String, t: Tensor.Tensor, want: List[List[Float]], abs: Float, rel: Float) -> Result[Float]
  values = try floats(t)
  columns = Tensor.shape(t)[1]
  return error("#{what}: #{Tensor.shape(t)} is not #{want.size} rows") if Tensor.shape(t)[0] != want.size
  worst = 0.0
  for r in 0...want.size
    return error("#{what}: a row of #{want[r].size} values, not #{columns}") if want[r].size != columns
    for c in 0...columns
      gap = (values[r * columns + c] - want[r][c]).abs
      if gap > abs + rel * want[r][c].abs
        return error("#{what}, row #{r}, column #{c}: #{values[r * columns + c]}, expected #{want[r][c]}")
      end

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
