## The image input of Qwen3.5 against Hugging Face's image processor (`qwen_pixels_golden.py`,
## `fixtures/qwen/golden.json`): `smart_resize` on a table of sizes, `Vision.patches` on the 8-bit images that the
## processor made (`fixtures/qwen/crops`), which must give its `pixel_values` to float rounding, and the whole input
## (`Step.SmartResize` with the bicubic kernel, `to_tensor`, `normalize` with the mean and std 0.5, `patches`) on the
## drawn images of `fixtures/resnet`, which is a resize of libvips against Pillow's. The values compared are about 500
## of each image (every `stride`th of its `pixel_values`) and their mean. No checkpoint is needed. Run from `vision/`.

import Image from "pkg/image/image"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Vision from "../vision"

PATCH = 16
MERGE = 2
FACTOR = 32
## The reference computes slowly: it takes the images of up to this many patches.
REF_PATCHES = 8
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

## The patches of the processor's own 8-bit images are its `pixel_values` up to the rounding of `float32`.
EXACT = 0.00001

## What passes for our own resize: two 8-bit levels (1 / 255 / 0.5 = 0.0078 each) from Pillow's at the most. The 64 by 32
## case measured two and the others one, the grow (the image is under `min_pixels`: area 2.5 times, 1.5 and 1.67 per axis)
## one too, since `image-0.3.0` aligns the centres of an enlarged pixel like Pillow (agentica-lang/packages#266).
PIXEL_ABS = 0.016
PIXEL_MEAN = 0.002

struct Case
  name: String
  image: String
  min_pixels: Int
  max_pixels: Int
  grid: List[Int]
  crop: Bool
  stride: Int
  mean: Float
  samples: List[Float]
end

struct Golden
  cases: List[Case]
  sizes: List[List[Int]]
end

private

## The largest gap between `got` at every `stride`th place and `want`, and the mean over all of `got`.
struct Gap
  largest: Float
  ## The share of the sampled values more than `PIXEL_ABS` apart.
  far: Float
  mean: Float
end

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/qwen/golden.json"), Golden)

## The pixel values of the image at `path` after `steps`, as `[patches, 1536]` on `backend`.
def pixel_values(path: String, steps: List[Vision.Step], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Native.arena do
    image = try Image.colourspace(try Image.load(path), Image.Space.Srgb)
    x = try Vision.to_tensor(try Vision.apply(image, steps), backend)
    Vision.patches(try Vision.normalize(x, MEAN, STD), PATCH, MERGE)
  end
end

def measure(got: Tensor.Tensor, c: Case) -> Result[Gap]
  shape = Tensor.shape(got)
  total = shape[0] * shape[1]
  at = (0...c.samples.size).to_a.map { |i| Tensor.Values.Int(i * c.stride) }
  ids = try Tensor.from_list(at, [at.size], Tensor.Dtype.I32, Tensor.backend(got))
  # Only the sampled values and the mean come back to the host: a grid of 22 by 30 patches has a million values.
  outs = try Trace.eval_many([got, ids]) do |xs|
    column = try Tensor.reshape(xs[0], [total, 1])
    ok([try Tensor.embedding(xs[1], column), try Tensor.reshape(try Tensor.mean(xs[0]), [1])])
  end

  values = try S.floats_of(outs[0])
  largest = 0.0
  far = 0
  for i in 0...c.samples.size
    gap = (values[i] - c.samples[i]).abs
    if gap > PIXEL_ABS
      far = far + 1
    end

    if gap > largest
      largest = gap
    end
  end

  mean = (try S.floats_of(outs[1]))[0]
  ok(Gap(largest: largest, far: far.to_f / c.samples.size.to_f, mean: (mean - c.mean).abs))
end

def test_smart_resize_gives_hugging_faces_sizes() -> Result[Void]
  for row in (try golden()).sizes
    size = try Vision.smart_resize(row[0], row[1], FACTOR, row[2], row[3])
    try Test.eq([size.height, size.width], [row[4], row[5]])
  end

  ok()
end

## The aspect ratio above 200 is Hugging Face's error (200 itself is allowed: the table has 1 by 200); the sizes under 1
## and `min_pixels` above `max_pixels` are this function's own checks.
def test_smart_resize_refuses_bad_arguments() -> Result[Void]
  try Test.error?(Vision.smart_resize(1, 201, FACTOR, 1, 100000))
  try Test.error?(Vision.smart_resize(201, 1, FACTOR, 1, 100000))
  try Test.error?(Vision.smart_resize(0, 5, FACTOR, 1, 100000))
  Test.error?(Vision.smart_resize(40, 40, FACTOR, 200, 100))
end

def test_the_patches_of_the_processors_images_are_its_pixel_values() -> Result[Void]
  g = try golden()
  for backend in try S.backends()
    for c in g.cases
      if c.crop and (backend != S.REF or c.grid[0] * c.grid[1] <= REF_PATCHES)
        got = try pixel_values("test/fixtures/qwen/crops/#{c.name}.png", [], backend)
        try Test.eq(Tensor.shape(got), [c.grid[0] * c.grid[1], 3 * 2 * PATCH * PATCH])
        gap = try measure(got, c)
        try Test.assert(gap.largest <= EXACT, "#{c.name}: the largest gap is #{gap.largest}")
        try Test.assert(gap.mean <= EXACT, "#{c.name}: the mean is #{gap.mean} from Hugging Face's")
      end
    end
  end

  ok()
end

def test_the_whole_input_is_hugging_faces_within_the_resize_of_libvips() -> Result[Void]
  g = try golden()
  backends = try S.backends()
  backend = backends[backends.size - 1]
  for c in g.cases
    if backend != S.REF or c.grid[0] * c.grid[1] <= REF_PATCHES
      steps = [Vision.Step.SmartResize(FACTOR, c.min_pixels, c.max_pixels, Image.Kernel.Cubic)]
      got = try pixel_values("test/fixtures/#{c.image}", steps, backend)
      try Test.eq(Tensor.shape(got), [c.grid[0] * c.grid[1], 3 * 2 * PATCH * PATCH])
      gap = try measure(got, c)
      puts("qwen_pixels #{c.name}: largest gap #{gap.largest}, #{gap.far * 100.0}% far, mean #{gap.mean}")
      try Test.assert(gap.largest <= PIXEL_ABS, "#{c.name}: the largest gap is #{gap.largest}")
      try Test.assert(gap.mean <= PIXEL_MEAN, "#{c.name}: the mean is #{gap.mean} from Hugging Face's")
    end
  end

  ok()
end

def test_patches_refuses_what_it_cannot_lay_out() -> Result[Void]
  x = try Tensor.floats((0...(3 * 32 * 32)).to_a.map { |i| i.to_f }, [3, 32, 32])
  try Test.eq(Tensor.shape(try Vision.patches(x, PATCH, MERGE)), [4, 1536])
  try Test.error?(Vision.patches(x, PATCH, 3))
  try Test.error?(Vision.patches(try Tensor.floats([1.0], [1, 1, 1, 1]), PATCH, MERGE))
  try Test.error?(Vision.patches(x, 0, MERGE))
  try Test.error?(Vision.patches(x, PATCH, MERGE, 0))
  try Test.error?(
    Vision.patches(try Tensor.floats((0...(32 * 32)).to_a.map { |i| i.to_f }, [1, 32, 32]), PATCH, MERGE),
  )

  ints = (0...(3 * 32 * 32)).to_a.map { |i| Tensor.Values.Int(i) }
  Test.error?(Vision.patches(try Tensor.from_list(ints, [3, 32, 32], Tensor.Dtype.I32), PATCH, MERGE))
end
