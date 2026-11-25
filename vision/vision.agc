## Images for tensors. libvips (the `image` package) decodes and transforms them, and their pixels reach a tensor as
## one `Bytes` buffer that the backend reads itself, never as a list of floats. The geometry runs in libvips; scaling,
## the layout and `normalize` run as tensor ops on the backend.
##
## A function that makes an `Image.Image` leaves it open until its task ends or a `Native.arena` closes it; `samples`
## closes every image it decodes.

import Image from "pkg/image/image"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## How a tensor lays out an image: `Hwc` is `[height, width, channels]`, as libvips keeps the pixels, and `Chw` is
## `[channels, height, width]`. A batch has a first axis `[n, ...]`.
enum Layout
  Hwc
  Chw
end

## One transform of `apply`. `Resize(n)` scales the shorter side to `n` and keeps the aspect ratio; the longer side is
## cut to a whole number, as torchvision does. `CenterCrop(width, height)` and `RandomCrop(width, height)` cut an
## area of that size. `HorizontalFlip(p)` mirrors the image with probability `p` (`1.0` always).
enum Step
  Resize(Int)
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

## An image as `F32` pixels, interleaved, with what makes them a tensor.
struct Pixels
  data: Bytes
  width: Int
  height: Int
  bands: Int
  pixel_format: Image.Format
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
        return error(
          "#{path} is #{p.width}x#{p.height} with #{p.bands} bands of #{p.pixel_format}, not #{first.width}x#{first.height} with #{first.bands} of #{first.pixel_format}: end the steps with a crop",
        )
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
      Step.Resize(shorter) -> current = try resize_shorter(current, shorter)
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
def resize_shorter(image: Image.Image, shorter: Int) -> Result[Image.Image]
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
  Image.resize(image, width, height)
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

## The pixels of `image` as `F32`, with the check that a tensor can be made of them.
def pixels_of(image: Image.Image) -> Result[Pixels]
  kind = Image.pixel_format(image)
  case kind
    Image.Format.U8 -> nil
    Image.Format.U16 -> nil
    Image.Format.F32 -> nil
    _ -> return error("an image of #{kind} samples has no tensor: use U8, U16 or F32")
  end

  as_float = try Image.cast(image, Image.Format.F32)
  data = try Image.to_pixels(as_float)
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

  raw = try Tensor.from_bytes(
    Bytes.concat(parts.map { |p| p.data }),
    [parts.size, first.height, first.width, first.bands],
    Tensor.FloatFormat.F32,
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
