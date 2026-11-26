## Images for tensors. libvips (the `image` package) decodes and transforms them, and their pixels reach a tensor as
## one `Bytes` buffer that the backend reads itself, never as a list of floats. The geometry runs in libvips; scaling,
## the layout and `normalize` run as tensor ops on the backend.
##
## A function that makes an `Image.Image` leaves it open until its task ends or a `Native.arena` closes it; `samples`
## closes every image it decodes.

import Image from "pkg/image/image"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

export Image.Kernel

PI = 3.141592653589793

## The longest edge that SmolVLM's image processor scales an image to, and the side of its tiles.
IDEFICS3_LONGEST_EDGE = 2048
IDEFICS3_TILE = 512

## How a tensor lays out an image: `Hwc` is `[height, width, channels]`, as libvips keeps the pixels, and `Chw` is
## `[channels, height, width]`. A batch has a first axis `[n, ...]`.
enum Layout
  Hwc
  Chw
end

## One transform of `apply`. `Resize(n, kernel)` scales the shorter side to `n` and keeps the aspect ratio; the longer
## side is cut to a whole number, as torchvision does. `ResizeTo(width, height, kernel)` scales to exactly that size,
## whatever the aspect ratio. The `kernel` (`Vision.Kernel`: `Nearest`, `Linear`, `Cubic`, `Lanczos2`, `Lanczos3`) is the
## filter that libvips resizes with: the one of the model's image processor (bicubic is `Cubic`, bilinear `Linear`).
## `CenterCrop(width, height)` and `RandomCrop(width, height)` cut an area of that size.
## `HorizontalFlip(p)` mirrors the image with probability `p` (`1.0` always). `SmartResize(factor, min_pixels,
## max_pixels, kernel)` scales to the size that `smart_resize` gives (Qwen2-VL's image processor).
enum Step
  Resize(Int, Image.Kernel)
  ResizeTo(Int, Int, Image.Kernel)
  SmartResize(Int, Int, Int, Image.Kernel)
  CenterCrop(Int, Int)
  RandomCrop(Int, Int)
  HorizontalFlip(Float)
end

## The images of a folder per class, in order: `labels[i]` is the index in `classes` of the folder of `paths[i]`.
struct Folder
  classes: List[String]
  paths: List[String]
  labels: List[Int]
end

## A batch: `inputs` is `[n, 3, height, width]` of `F32` in `0..1`, `targets` is `[n]` of `I32` class indexes.
struct Samples
  inputs: Tensor.Tensor
  targets: Tensor.Tensor
end

## A size in pixels.
struct Size
  height: Int
  width: Int
end

## How SmolVLM's image processor (Idefics3's) cuts an image of a given size (`idefics3_tiling`): the size `resized`
## that its longest edge makes, the `grid` (a multiple of the tile on both sides, which the image is resized to again)
## and the `rows` and `cols` of tiles that the grid holds.
struct Tiling
  resized: Size
  grid: Size
  rows: Int
  cols: Int
end

## The tiles of an image (`idefics3_tiles`): `rows * cols` tensors of them, row by row, then the whole image resized to
## a tile (the global image), each `[3, 512, 512]` in `F32` in `0.0..1.0`, as `to_tensor` makes them.
struct Tiles
  tiles: List[Tensor.Tensor]
  rows: Int
  cols: Int
end

## The size that Hugging Face's `smart_resize` (the image processor of Qwen2-VL and its successors) gives an image of
## `height` by `width`: each side is the nearest multiple of `factor` (a half goes to the even multiple, as Python's
## `round` does, and may be 0), and when that has more than `max_pixels` pixels the image shrinks, or when it has fewer
## than `min_pixels` it grows, by the same ratio on both sides, to multiples of `factor`, keeping the aspect ratio as
## closely as it can. An aspect ratio above 200 is an `error`.
def smart_resize(height: Int, width: Int, factor: Int, min_pixels: Int, max_pixels: Int) -> Result[Size]
  if height < 1 or width < 1 or factor < 1 or min_pixels < 1 or max_pixels < min_pixels
    return error("smart_resize needs sizes of 1 or more and min_pixels <= max_pixels, got #{height}x#{width}")
  end

  if [height, width].max.unwrap! > 200 * [height, width].min.unwrap!
    return error("smart_resize needs an aspect ratio of at most 200, got #{height}x#{width}")
  end

  h_bar = nearest_multiple(height, factor)
  w_bar = nearest_multiple(width, factor)
  if h_bar * w_bar > max_pixels
    beta = Math.sqrt((height * width).to_f / max_pixels.to_f)
    h_bar = [factor, (height.to_f / beta / factor.to_f).floor * factor].max.unwrap!
    w_bar = [factor, (width.to_f / beta / factor.to_f).floor * factor].max.unwrap!
  elsif h_bar * w_bar < min_pixels
    beta = Math.sqrt(min_pixels.to_f / (height * width).to_f)
    h_bar = ((height.to_f * beta) / factor.to_f).ceil * factor
    w_bar = ((width.to_f * beta) / factor.to_f).ceil * factor
  end

  ok(Size(height: h_bar, width: w_bar))
end

## The sizes that Hugging Face's `Idefics3ImageProcessor` works out for an image of `height` by `width` with the
## settings of SmolVLM (a longest edge of 2048 and tiles of 512 by 512): the longest edge is scaled to 2048 (up as well
## as down) and the other side keeps the aspect ratio, cut to a whole number and made even (`resized`); then each side is
## made a multiple of 512 (the second side from the aspect ratio of the first: `grid`), and the grid is cut into `rows`
## by `cols` tiles. Python's floats do the arithmetic. The processor's `do_image_splitting: false` (one 512 by 512 image,
## no tiles) is not supported.
def idefics3_tiling(height: Int, width: Int) -> Result[Tiling]
  return error("idefics3_tiling needs sizes of 1 or more, got #{height}x#{width}") if height < 1 or width < 1

  aspect = width.to_f / height.to_f
  h = height
  w = width
  if width >= height
    w = IDEFICS3_LONGEST_EDGE
    h = (w.to_f / aspect).floor
    h = even(h)
  else
    h = IDEFICS3_LONGEST_EDGE
    w = (h.to_f * aspect).floor
    w = even(w)
  end

  resized = Size(height: [h, 1].max.unwrap!, width: [w, 1].max.unwrap!)
  ratio = resized.width.to_f / resized.height.to_f
  gh = resized.height
  gw = resized.width
  if resized.width >= resized.height
    gw = ceiling(resized.width, IDEFICS3_TILE)
    gh = ceiling((gw.to_f / ratio).floor, IDEFICS3_TILE)
  else
    gh = ceiling(resized.height, IDEFICS3_TILE)
    gw = ceiling((gh.to_f * ratio).floor, IDEFICS3_TILE)
  end

  ok(Tiling(resized: resized, grid: Size(height: gh, width: gw), rows: gh / IDEFICS3_TILE, cols: gw / IDEFICS3_TILE))
end

## The tiles of `image` as Hugging Face's `Idefics3ImageProcessor` cuts them (`idefics3_tiling`): the image resized
## to `resized`, then to the `grid`, cut into tiles of 512 by 512 row by row, then the whole grid resized to one
## tile, as tensors on `backend` (`Tiles`). The image is an sRGB `U8` one of 3 bands (`Image.colourspace` of the
## `image` package; the processor converts to RGB itself, this function does not), else an error. Each resize is
## Pillow's Lanczos (`ImagingResample`: the filter with its support scaled when shrinking, the centre of output
## pixel `i` at `(i + 0.5) * scale`, weights normalized for each output pixel, a pass across and then one down,
## each held to 0..255), as products by matrices of weights in one graph. libvips' resize is not used: it enlarges
## with bicubic interpolation whatever the kernel. The passes are not rounded to 8 bits as Pillow's are (ggml has
## no rounding), so a sample is within a level or two of the processor's.
def idefics3_tiles(image: Image.Image, backend: Tensor.Backend = Tensor.Backend.Reference) -> Result[Tiles]
  return error("idefics3_tiles needs an sRGB image of 3 bands, got #{Image.bands(image)}") if Image.bands(image) != 3
  kind = Image.pixel_format(image)
  return error("idefics3_tiles needs a U8 image, got #{kind}") if kind != Image.Format.U8

  Native.arena do
    t = try idefics3_tiling(Image.height(image), Image.width(image))
    pixels = try pixels_of(image)
    raw = try Tensor.from_bytes(pixels.data, [pixels.height, pixels.width, 3], Tensor.FloatFormat.U8, backend)
    steps = [
      Resize(from_h: pixels.height, from_w: pixels.width, to_h: t.resized.height, to_w: t.resized.width),
      Resize(from_h: t.resized.height, from_w: t.resized.width, to_h: t.grid.height, to_w: t.grid.width),
      Resize(from_h: t.grid.height, from_w: t.grid.width, to_h: IDEFICS3_TILE, to_w: IDEFICS3_TILE),
    ]

    weights: List[Tensor.Tensor] = []
    for r in steps
      if r.to_w != r.from_w
        weights = weights + try lanczos_taps(r.from_w, r.to_w, backend)
      end

      if r.to_h != r.from_h
        weights = weights + try lanczos_taps(r.from_h, r.to_h, backend)
      end
    end

    tiles = try Trace.eval_many([raw] + weights) do |xs| grid_and_whole(xs, steps, t) end

    ok(Tiles(tiles: tiles, rows: t.rows, cols: t.cols))
  end
end

## The tensor of one image: the samples of a `U8` image scaled to `0..1` (a `U16` image by 65535, a `F32` one kept)
## with `dtype: F32`, or as they are with `dtype: U8` (a `U8` image only). `Chw` is the layout that convolutions
## take. The pixels cross as one buffer, and the image keeps all its bands, alpha included.
def to_tensor(
  image: Image.Image,
  backend: Tensor.Backend = Tensor.Backend.Reference,
  dtype: Tensor.Dtype = Tensor.Dtype.F32,
  layout: Layout = Layout.Chw,
) -> Result[Tensor.Tensor]
  Native.arena do one_tensor(image, backend, dtype, layout) end
end

## The image after `steps`, in order. The random steps draw from `seed`: one seed gives one result.
def apply(image: Image.Image, steps: List[Step], seed: Int = 0) -> Result[Image.Image]
  Native.arena do transformed(image, steps, seed) end
end

## `x` minus `mean` and divided by `std`, for each channel: `x` is `F32`, of rank 3 or more, with `Chw` or `Hwc`
## channels (batch axes first), and `mean` and `std` have one value for each channel.
def normalize(
  x: Tensor.Tensor,
  mean: List[Float],
  std: List[Float],
  layout: Layout = Layout.Chw,
) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  return error("normalize needs an F32 tensor, got #{Tensor.dtype(x)}") if Tensor.dtype(x) != Tensor.Dtype.F32
  return error("normalize needs a tensor of rank 3 or more, got #{shape}") if shape.size < 3
  axis = shape.size - 1
  if layout == Layout.Chw
    axis = shape.size - 3
  end

  channels = shape[axis]
  if mean.size != channels or std.size != channels
    return error("normalize needs #{channels} means and #{channels} stds, got #{mean.size} and #{std.size}")
  end

  return error("normalize needs stds above 0.0, got #{std}") if std.any? { |s| s <= 0.0 }
  Native.arena do scaled(x, mean, std, axis) end
end

## The patches of one normalized image `x` (`[channels, height, width]`, `F32`) as Qwen2-VL's image processor lays
## them out for its vision tower (`pixel_values`): `[(height / patch_size) * (width / patch_size), channels *
## temporal_patch_size * patch_size * patch_size]`. A row is one patch, as `channels`, `temporal_patch_size` frames (the
## image repeated: a still image is its own frames) and the pixels of the patch. The patches come in the order of the
## merge squares of `merge_size` by `merge_size` patches: the patches of one square together (by row, then column), the
## squares row by row. `height` and `width` must be multiples of `patch_size * merge_size`, which `SmartResize` makes. `x`
## has 3 channels: an image with an alpha band is an `error` (Hugging Face's processor flattens it first).
def patches(x: Tensor.Tensor, patch_size: Int, merge_size: Int, temporal_patch_size: Int = 2) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  return error("patches needs an F32 tensor, got #{Tensor.dtype(x)}") if Tensor.dtype(x) != Tensor.Dtype.F32
  return error("patches needs [channels, height, width], got #{shape}") if shape.size != 3
  return error("patches needs 3 channels (sRGB, no alpha), got #{shape[0]}") if shape[0] != 3
  if patch_size < 1 or merge_size < 1 or temporal_patch_size < 1
    return error("patches needs a patch_size, merge_size and temporal_patch_size of 1 or more")
  end

  square = patch_size * merge_size
  if shape[1] % square != 0 or shape[2] % square != 0
    return error("patches needs a height and width that are multiples of #{square}, got #{shape[1]}x#{shape[2]}")
  end

  Native.arena do patched(x, patch_size, merge_size, temporal_patch_size) end
end

## The folders of `root` as classes, in name order, and the images in each (`png`, `jpg`, `jpeg`, `webp`, `tif`,
## `tiff`, `bmp`, `gif`), in name order. A name that starts with `.` is skipped.
def folder(root: String) -> Result[Folder]
  classes: List[String] = []
  paths: List[String] = []
  labels: List[Int] = []
  for name in try Dir.list(root)
    next if name.start_with?(".")
    dir = try Path.join([root, name])
    next if (try File.stat(dir)).kind != "directory"
    label = classes.size
    classes = classes.push(name)
    for file in try Dir.list(dir)
      path = try Path.join([dir, file])
      next if !image_name?(file) or (try File.stat(path)).kind != "file"
      paths = paths.push(path)
      labels = labels.push(label)
    end
  end

  return error("#{root} has no folder with images") if paths.empty?
  ok(Folder(classes: classes, paths: paths, labels: labels))
end

## The images `rows` of `f` as one batch on `backend`: each is loaded, turned to sRGB, put through `steps` and read as
## pixels, and the whole batch becomes one tensor. The images must all end with the same size (end `steps` with a
## crop). Row `i` draws its random steps from `seed` and `i`, so a new `seed` for each epoch gives new crops and flips.
## A batch of every row is a dataset for the loader of `train/`: `Data.dataset(s.inputs, s.targets)`.
def samples(
  f: Folder,
  rows: List[Int],
  steps: List[Step],
  seed: Int,
  backend: Tensor.Backend = Tensor.Backend.Reference,
) -> Result[Samples]
  return error("samples needs at least one row") if rows.empty?
  parts: List[Pixels] = []
  for row in rows
    return error("row #{row} is outside the #{f.paths.size} images of the folder") if row < 0 or row >= f.paths.size
    decoded = Native.arena do decode(f.paths[row], steps, row_seed(seed, row)) end
    parts = parts.push(try labelled(decoded, parts, f.paths[row]))
  end

  Native.arena do batch(parts, rows.map { |row| f.labels[row] }, backend) end
end

private

## The seed of the random steps starts the generator of Park-Miller, as in `train/data.agc`.
MODULUS = 2147483647
MULTIPLIER = 48271
SUFFIXES = [".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"]

struct Resample
  image: Tensor.Tensor
  at: Int
end

struct Resize
  from_h: Int
  from_w: Int
  to_h: Int
  to_w: Int
end

## An image's samples, interleaved: one byte each for a `U8` image, little-endian `F32` for the others.
struct Pixels
  data: Bytes
  width: Int
  height: Int
  bands: Int
  pixel_format: Image.Format
end

## The weights of Pillow's Lanczos resize of a line of `from` samples to `to` that are not zero, as two graph inputs:
## the place `i * from + x` in the matrix `[to, from]` that gives the weight of sample `x` in output sample `i`, and
## the weights (`[n, 1]`). `matrix` scatters them.
def lanczos_taps(from: Int, to: Int, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  scale = from.to_f / to.to_f
  spread = [scale, 1.0].max.unwrap!
  support = 3.0 * spread
  places: List[Tensor.Values] = []
  values: List[Float] = []
  for i in 0...to
    centre = (i.to_f + 0.5) * scale
    low = [(centre - support + 0.5).floor, 0].max.unwrap!
    high = [(centre + support + 0.5).floor, from].min.unwrap!
    weights = (low...high).to_a.map { |x| lanczos((x.to_f - centre + 0.5) / spread) }
    total = weights.sum
    for k in 0...weights.size
      places = places.push(Tensor.Values.Int(i * from + low + k))
      values = values.push(weights[k] / total)
    end
  end

  ok([
    try Tensor.from_list(places, [places.size], Tensor.Dtype.I32, backend),
    try Tensor.floats(values, [values.size, 1], backend),
  ])
end

## The `[to, from]` matrix of `lanczos_taps`.
def matrix(places: Tensor.Tensor, values: Tensor.Tensor, to: Int, from: Int) -> Result[Tensor.Tensor]
  Tensor.reshape(try Tensor.scatter_add(places, values, to * from), [to, from])
end

## The samples of a pass held to `0.0..255.0`, as Pillow's clip to 8 bits does (`relu` twice: ggml has no `clamp`).
## They are not rounded to a level: ggml has no rounding either.
def held(x: Tensor.Tensor) -> Result[Tensor.Tensor]
  top = try Tensor.full(Tensor.shape(x), Tensor.Values.Float(255.0), Tensor.Dtype.F32, Tensor.backend(x))
  Tensor.sub(try Tensor.relu(x), try Tensor.relu(try Tensor.sub(x, top)))
end

## `x` of `[height, width, 3]` resized across to `w` of `[to, width]`.
def across(x: Tensor.Tensor, w: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  columns = try Tensor.reshape(try Tensor.permute(x, [1, 0, 2]), [shape[1], shape[0] * 3])
  moved = try Tensor.reshape(try Tensor.matmul(w, columns), [Tensor.shape(w)[0], shape[0], 3])
  held(try Tensor.permute(moved, [1, 0, 2]))
end

## `x` of `[height, width, 3]` resized down by `w` of `[to, height]`.
def down(x: Tensor.Tensor, w: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  rows = try Tensor.reshape(x, [shape[0], shape[1] * 3])
  held(try Tensor.reshape(try Tensor.matmul(w, rows), [Tensor.shape(w)[0], shape[1], 3]))
end

## `x` resized as `r` says by the weights from `xs[at]` on (across first, then down); the weights left at `at`.
def resized_by(x: Tensor.Tensor, r: Resize, xs: List[Tensor.Tensor], at: Int) -> Result[Resample]
  out = x
  next_at = at
  if r.to_w != r.from_w
    out = try across(out, try matrix(xs[next_at], xs[next_at + 1], r.to_w, r.from_w))
    next_at += 2
  end

  if r.to_h != r.from_h
    out = try down(out, try matrix(xs[next_at], xs[next_at + 1], r.to_h, r.from_h))
    next_at += 2
  end

  ok(Resample(image: out, at: next_at))
end

## The tiles of `idefics3_tiles` from its graph inputs: the pixels and the weights of each step.
def grid_and_whole(xs: List[Tensor.Tensor], steps: List[Resize], t: Tiling) -> Result[List[Tensor.Tensor]]
  first = try resized_by(xs[0], steps[0], xs, 1)
  grid = try resized_by(first.image, steps[1], xs, first.at)
  whole = try resized_by(grid.image, steps[2], xs, grid.at)
  outs: List[Tensor.Tensor] = []
  for r in 0...t.rows
    for c in 0...t.cols
      top = r * IDEFICS3_TILE
      left = c * IDEFICS3_TILE
      outs = outs.push(try Tensor.slice(grid.image, [top, left, 0], [top + IDEFICS3_TILE, left + IDEFICS3_TILE, 3]))
    end
  end

  outs = outs.push(whole.image)
  done: List[Tensor.Tensor] = []
  for o in outs
    done = done.push(try unit(o))
  end

  ok(done)
end

## A tile of `[512, 512, 3]` levels as `to_tensor` gives it: `[3, 512, 512]` in `0.0..1.0`.
def unit(x: Tensor.Tensor) -> Result[Tensor.Tensor]
  scale = try Tensor.full(Tensor.shape(x), Tensor.Values.Float(1.0 / 255.0), Tensor.Dtype.F32, Tensor.backend(x))
  Tensor.permute(try Tensor.mul(x, scale), [2, 0, 1])
end

def lanczos(x: Float) -> Float
  return 0.0 if x <= -3.0 or x >= 3.0
  return 1.0 if x == 0.0
  a = PI * x
  b = a / 3.0
  (Math.sin(a) / a) * (Math.sin(b) / b)
end

def image_name?(file: String) -> Bool
  lower = file.downcase
  SUFFIXES.any? { |suffix| lower.end_with?(suffix) }
end

## The first draws of a generator follow its seed linearly, so it is warmed up.
def start(seed: Int) -> Int
  state = seed.abs % (MODULUS - 1) + 1
  for _ in 0...4
    state = advance(state)
  end

  state
end

def advance(state: Int) -> Int = (state * MULTIPLIER) % MODULUS

## One seed for each (seed, row): the rows of a seed, and the same row under other seeds, draw unrelated numbers. `mix`
## breaks the line that a generator of Park-Miller draws along its seed.
def row_seed(seed: Int, row: Int) -> Int = mix((mix(seed.abs % MODULUS) + row) % MODULUS)

def mix(x: Int) -> Int
  a = (x.bit_xor(x.shift_right(15)) * MULTIPLIER) % MODULUS
  (a.bit_xor(a.shift_right(13)) * 69621) % MODULUS
end

## `x` less `mean` over `std` along the channel `axis`, in one evaluation; the arena of `normalize` closes the tensors
## of `mean` and `std`.
def scaled(x: Tensor.Tensor, mean: List[Float], std: List[Float], axis: Int) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  view = shape.map { |_| 1 }.set(axis, shape[axis])
  means = try Tensor.floats(mean, view, Tensor.backend(x))
  stds = try Tensor.floats(std, view, Tensor.backend(x))
  Trace.eval([x, means, stds]) do |xs|
    centered = try Tensor.sub(xs[0], try Tensor.broadcast_to(xs[1], shape))
    Tensor.div(centered, try Tensor.broadcast_to(xs[2], shape))
  end
end

## The rows of `patches`, one strip of patch rows at a time (a tensor holds four axes at most on ggml): the patches of
## a row are `[columns, channels * patch * patch]`, and the rows of one merge square stack into `[squares, merge, merge,
## ...]`.
def patched(x: Tensor.Tensor, patch: Int, merge: Int, frames: Int) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  channels = shape[0]
  columns = shape[2] / patch
  depth = channels * patch * patch
  Trace.eval([x]) do |xs|
    strips: List[Tensor.Tensor] = []
    for square_row in 0...(shape[1] / patch / merge)
      rows: List[Tensor.Tensor] = []
      for r in 0...merge
        top = (square_row * merge + r) * patch
        strip = try Tensor.slice(xs[0], [0, top, 0], [channels, top + patch, shape[2]])
        cells = try Tensor.permute(try Tensor.reshape(strip, [channels, patch, columns, patch]), [2, 0, 1, 3])
        rows = rows.push(try Tensor.reshape(cells, [columns / merge, 1, merge, depth]))
      end

      strips = strips.push(try Tensor.reshape(try Tensor.concat(rows, 1), [columns * merge, depth]))
    end

    flat = try Tensor.concat(strips, 0)
    n = Tensor.shape(flat)[0]
    framed = try Tensor.reshape(flat, [n, channels, 1, patch * patch])
    repeated = try Tensor.broadcast_to(framed, [n, channels, frames, patch * patch])
    Tensor.reshape(repeated, [n, channels * frames * patch * patch])
  end
end

## The batch of the images `parts` with their class `labels`.
def batch(parts: List[Pixels], labels: List[Int], backend: Tensor.Backend) -> Result[Samples]
  ids = labels.map { |label| Tensor.Values.Int(label) }
  inputs = try tensorize(parts, backend, Tensor.Dtype.F32, Layout.Chw, true)
  ok(Samples(inputs: inputs, targets: try Tensor.from_list(ids, [parts.size], Tensor.Dtype.I32, backend)))
end

def one_tensor(
  image: Image.Image,
  backend: Tensor.Backend,
  dtype: Tensor.Dtype,
  layout: Layout,
) -> Result[Tensor.Tensor]
  pixels = try pixels_of(image)
  tensorize([pixels], backend, dtype, layout, false)
end

## The pixels of the image at `path` as sRGB after `steps`: 3 bands.
def decode(path: String, steps: List[Step], seed: Int) -> Result[Pixels]
  image = try Image.colourspace(try Image.load(path), Image.Space.Srgb)
  pixels = try pixels_of(try transformed(image, steps, seed))
  return error("an image of #{pixels.bands} bands: samples take 3, so no alpha band") if pixels.bands != 3
  ok(pixels)
end

## `decoded` when it is a good image of the size, bands and format of the first one of `parts`; the error names `path`.
def labelled(decoded: Result[Pixels], parts: List[Pixels], path: String) -> Result[Pixels]
  case decoded
    error(e) -> error(e.with(message: "#{path}: #{e.message}"))
    ok(p) ->
      return ok(p) if parts.empty?
      first = parts[0]
      if (
        p.width != first.width or
        p.height != first.height or
        p.bands != first.bands or
        p.pixel_format != first.pixel_format
      )
        size = "#{p.width}x#{p.height} with #{p.bands} bands of #{p.pixel_format}"
        wanted = "#{first.width}x#{first.height} with #{first.bands} of #{first.pixel_format}"
        return error("#{path} is #{size}, not #{wanted}: end the steps with a crop")
      end

      ok(p)
  end
end

## The steps of `apply`; the arena of `apply` closes the images that the steps leave behind.
def transformed(image: Image.Image, steps: List[Step], seed: Int) -> Result[Image.Image]
  current = image
  state = start(seed)
  for step in steps
    case step
      Step.Resize(shorter, kernel) -> current = try resize_shorter(current, shorter, kernel)
      Step.ResizeTo(width, height, kernel) -> current = try resize_to(current, width, height, kernel)
      Step.SmartResize(factor, min_pixels, max_pixels, kernel) ->
        size = try smart_resize(Image.height(current), Image.width(current), factor, min_pixels, max_pixels)
        current = try resize_to(current, size.width, size.height, kernel)
      Step.CenterCrop(width, height) ->
        left = centered(Image.width(current), width)
        current = try crop_at(current, width, height, left, centered(Image.height(current), height))
      Step.RandomCrop(width, height) ->
        iw = Image.width(current)
        ih = Image.height(current)
        return error("RandomCrop #{width}x#{height} does not fit the #{iw}x#{ih} image") if width > iw or height > ih
        state = advance(state)
        left = state % (iw - width + 1)
        state = advance(state)
        current = try crop_at(current, width, height, left, state % (ih - height + 1))
      Step.HorizontalFlip(p) ->
        return error("HorizontalFlip needs a probability in 0.0..1.0, got #{p}") if p < 0.0 or p > 1.0
        state = advance(state)
        if (state % 1000000).to_f / 1000000.0 < p
          current = try Image.flip(current, Image.Direction.Horizontal)
        end
    end
  end

  ok(current)
end

## The size of the shorter side is `shorter`; the other one is cut to a whole number, and is at least 1.
def resize_shorter(image: Image.Image, shorter: Int, kernel: Image.Kernel) -> Result[Image.Image]
  return error("Resize needs a side of 1 or more, got #{shorter}") if shorter < 1
  w = Image.width(image)
  h = Image.height(image)
  width = shorter
  height = [(h * shorter) / w, 1].max.unwrap!
  if w > h
    width = [(w * shorter) / h, 1].max.unwrap!
    height = shorter
  end

  return ok(image) if width == w and height == h
  Image.resize(image, width, height, kernel)
end

def resize_to(image: Image.Image, width: Int, height: Int, kernel: Image.Kernel) -> Result[Image.Image]
  return error("ResizeTo needs sides of 1 or more, got #{width}x#{height}") if width < 1 or height < 1
  return ok(image) if width == Image.width(image) and height == Image.height(image)
  Image.resize(image, width, height, kernel)
end

## `length` made even by adding 1 when it is odd.
def even(length: Int) -> Int = length + length % 2

## `length` rounded up to a multiple of `factor`.
def ceiling(length: Int, factor: Int) -> Int = ((length + factor - 1) / factor) * factor

## The multiple of `factor` nearest to `length`, a half to the even multiple, as Python's `round` goes.
def nearest_multiple(length: Int, factor: Int) -> Int
  below = length / factor
  twice = 2 * (length % factor)
  return (below + 1) * factor if twice > factor
  return (below + below % 2) * factor if twice == factor
  below * factor
end

## The offset that centers `size` in `length`, rounded to the even one when it falls between two, as Python rounds
## and torchvision crops.
def centered(length: Int, size: Int) -> Int
  half = (length - size) / 2
  return half + 1 if (length - size) % 2 == 1 and half % 2 == 1
  half
end

def crop_at(image: Image.Image, width: Int, height: Int, left: Int, top: Int) -> Result[Image.Image]
  return error("a crop needs sizes of 1 or more, got #{width}x#{height}") if width < 1 or height < 1
  iw = Image.width(image)
  ih = Image.height(image)
  return error("a crop of #{width}x#{height} does not fit the #{iw}x#{ih} image") if width > iw or height > ih
  Image.crop(image, left, top, width, height)
end

## The samples of `image`, with the check that a tensor can be made of them.
def pixels_of(image: Image.Image) -> Result[Pixels]
  kind = Image.pixel_format(image)
  case kind
    Image.Format.U8 -> nil
    Image.Format.U16 -> nil
    Image.Format.F32 -> nil
    _ -> return error("an image of #{kind} samples has no tensor: use U8, U16 or F32")
  end

  samples = image
  if kind != Image.Format.U8
    samples = try Image.cast(image, Image.Format.F32)
  end

  data = try Image.to_pixels(samples)
  ok(
    Pixels(
      data: data,
      width: Image.width(image),
      height: Image.height(image),
      bands: Image.bands(image),
      pixel_format: kind,
    ),
  )
end

## What turns the samples of a format into `0..1`.
def scale_of(kind: Image.Format) -> Float
  case kind
    Image.Format.U8 -> 1.0 / 255.0
    Image.Format.U16 -> 1.0 / 65535.0
    _ -> 1.0
  end
end

## The images in `parts` as one tensor: `[n, ...]` for a batch, one image without
## the `n`. The images have one size, band count and format.
def tensorize(
  parts: List[Pixels],
  backend: Tensor.Backend,
  dtype: Tensor.Dtype,
  layout: Layout,
  batch: Bool,
) -> Result[Tensor.Tensor]
  first = parts[0]
  if dtype != Tensor.Dtype.F32 and dtype != Tensor.Dtype.U8
    return error("a tensor of #{dtype} is not made of images: F32 or U8")
  end

  if dtype == Tensor.Dtype.U8 and first.pixel_format != Image.Format.U8
    return error("U8 tensors take U8 images, got #{first.pixel_format}")
  end

  stored = Tensor.FloatFormat.F32
  if first.pixel_format == Image.Format.U8
    stored = Tensor.FloatFormat.U8
  end

  raw = try Tensor.from_bytes(
    Bytes.concat(parts.map { |p| p.data }),
    [parts.size, first.height, first.width, first.bands],
    stored,
    backend,
  )

  scale = scale_of(first.pixel_format)
  Trace.eval([raw]) do |xs|
    x = xs[0]
    if dtype == Tensor.Dtype.F32 and scale != 1.0
      x = try Tensor.mul(
        x,
        try Tensor.full(Tensor.shape(x), Tensor.Values.Float(scale), Tensor.Dtype.F32, Tensor.backend(x)),
      )
    end

    if layout == Layout.Chw
      x = try Tensor.permute(x, [0, 3, 1, 2])
    end

    if !batch
      x = try Tensor.squeeze(x, some(0))
    end

    if dtype == Tensor.Dtype.U8
      x = try Tensor.to_dtype(x, Tensor.Dtype.U8)
    end

    ok(x)
  end
end
