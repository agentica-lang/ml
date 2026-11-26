---
title: vision
description: Load images with libvips, cut them with seeded transforms, and hand their pixels to a tensor as one buffer, with a folder-per-class dataset for the data loader.
weight: 8
---

`vision` turns image files into tensors. [libvips](https://www.libvips.org/) (through the [image](https://www.agentica-lang.com/packages/image/) package) decodes and resizes them, and the pixels cross to the backend as one `Bytes` buffer, with no `List` of floats. The geometry runs in libvips. Scaling to `0..1`, the layout and `normalize` run as tensor ops on the backend.

## Install

`vision` needs `tensor`, the `image` package, and libvips with its development files (`libvips-dev` on Debian and Ubuntu, `brew install vips` on macOS). The `image` page has the details.

```toml
[dependencies]
vision = { git = "https://github.com/agentica-lang/ml", tag = "vision-0.6.0", subdir = "vision" }
tensor = { git = "https://github.com/agentica-lang/ml", tag = "tensor-0.6.0", subdir = "tensor" }
image = { git = "https://github.com/agentica-lang/packages", tag = "image-0.3.0", subdir = "image" }
```

## Example

Make an image of 3 by 2 pixels, cut one column off it, and read it as an `F32` tensor in `[channels, height, width]`:

```agentica
import Image from "pkg/image/image"
import Tensor from "pkg/tensor/tensor"
import Vision from "pkg/vision/vision"

def main() -> Void
  pixels = Bytes.from_list([0, 51, 102, 153, 204, 255]).unwrap!
  image = Image.from_pixels(pixels, 3, 2, 1, Image.Format.U8).unwrap!
  cut = Vision.apply(image, [Vision.Step.CenterCrop(2, 2)]).unwrap!
  t = Vision.to_tensor(cut).unwrap!
  puts(Tensor.shape(t))
  puts(Tensor.to_list(t).unwrap!)
end
```

```output
[1, 2, 2]
[Values.Float(0.0), Values.Float(0.2), Values.Float(0.6), Values.Float(0.8)]
```

## To a tensor

- `to_tensor(image, backend, dtype, layout)` makes the tensor of one image. With `dtype: F32` (the default) the samples of a `U8` image are scaled to `0..1` (a `U16` one by 65535, an `F32` one is kept). With `dtype: U8` they are the samples as they are, which needs a `U8` image and a backend that converts to `U8` (not ggml). `layout` is `Chw` (`[channels, height, width]`, the default) or `Hwc` (`[height, width, channels]`). The image keeps all its bands, alpha included. A pixel format that is not `U8`, `U16` or `F32` is an `error`.
- The pixels become a tensor with `Tensor.from_bytes`: the backend reads the interleaved samples itself. A `U8` image crosses into the backend as bytes (`FloatFormat.U8`), a quarter of the size of `F32` ones, and the backend widens them to `F32` in C and uploads that; the scale to `0..1` is then an op on the device. An `F32` image, or a `U16` one cast to `F32`, crosses as `F32`.
- `normalize(x, mean, std, layout)` gives `(x - mean) / std` for each channel. `x` is `F32` of rank 3 or more with the channels in the last three axes of `Chw` or the last axis of `Hwc`, so one call normalizes a whole batch. `mean` and `std` have one value for each channel, and a `std` of 0 is an `error`.

## Transforms

`apply(image, steps, seed)` runs a list of `Step` values in order and gives the new image:

| Step | Does |
|---|---|
| `Resize(n, kernel)` | scales the shorter side to `n` and keeps the aspect ratio; the longer side is cut to a whole number, as torchvision does |
| `ResizeTo(width, height, kernel)` | scales to exactly that size, whatever the aspect ratio (what the processor of ViT does) |
| `CenterCrop(width, height)` | cuts the area at the center; when the gap is odd and falls between two pixels, the offset is the even one, as Python's `round` and torchvision do |
| `RandomCrop(width, height)` | cuts the area at a random place |
| `HorizontalFlip(p)` | mirrors the image with probability `p` |
| `SmartResize(factor, min_pixels, max_pixels, kernel)` | scales to the size that `smart_resize` gives, the one of Qwen2-VL's image processor (see below) |

`kernel` is a `Vision.Kernel` (`Image.Kernel` of the `image` package, which `vision` re-exports; `Nearest`, `Linear`, `Cubic`, `Lanczos2`, `Lanczos3`): the filter that libvips resizes with. Use the kernel of the model's image processor: Hugging Face's ResNet and CLIP processors resize with Pillow's bicubic (`Cubic`) and ViT's with its bilinear (`Linear`). With the kernel of the processor the pixels are at most one 8-bit level from Pillow's (0.0175 for ResNet, 0.0078 for ViT and 0.015 for CLIP, after normalizing): on the four drawn images of the tests 92% to 96% of the samples are Pillow's exactly, 4% to 8% are one level off and under 0.1% two levels off (see [the transformers page](../transformers/_index.md)).

The random steps draw from `seed`, so one seed gives one result. A crop larger than the image, a `Resize` below 1 or a `p` outside `0.0..1.0` is an `error`. The images that `apply` makes along the way are closed when it returns, but the one it gives stays open (see [image](https://www.agentica-lang.com/packages/image/) for the lifetime of an image).

## Patches for a vision-language model

`smart_resize(height, width, factor, min_pixels, max_pixels)` is Hugging Face's function of the same name (the image processor of Qwen2-VL and its successors, Qwen3.5 among them) and gives a `Size(height, width)`: each side is the nearest multiple of `factor` (a half goes to the even multiple, as Python's `round` does), and an image with more than `max_pixels` pixels shrinks, or one with fewer than `min_pixels` grows, by the same ratio on both sides. An aspect ratio above 200 is an `error`. `Step.SmartResize` resizes to that size.

`patches(x, patch_size, merge_size, temporal_patch_size)` lays one normalized image `x` (`[3, height, width]`, `F32`: sRGB with 3 bands) out as that processor's `pixel_values`: one row of `channels * temporal_patch_size * patch_size * patch_size` values for each patch, in the order of the merge squares of `merge_size` by `merge_size` patches, with the image repeated for the frames of the temporal patch. `height` and `width` must be multiples of `patch_size * merge_size`, which `SmartResize` makes. An image with an alpha band is an `error`: Hugging Face's processor flattens it first (`convert_to_rgb`) and `vision` does not yet, so flatten it before. For the processor of `Qwen/Qwen3.5-0.8B` (`patch_size` 16, `merge_size` 2, `shortest_edge` 65,536 and `longest_edge` 16,777,216 pixels in its `preprocessor_config.json`, mean and std 0.5):

```agentica {mode="fragment"}
steps = [Vision.Step.SmartResize(32, 65536, 16777216, Vision.Kernel.Cubic)]
srgb = Image.colourspace(Image.load("photo.jpg").unwrap!, Image.Space.Srgb).unwrap!
image = Vision.apply(srgb, steps).unwrap!
x = Vision.normalize(Vision.to_tensor(image, backend).unwrap!, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).unwrap!
pixel_values = Vision.patches(x, 16, 2).unwrap!
```

The tower of Qwen3.5 reads these patches, and the text model the features it gives: the chat with an image is in [Images with Qwen3.5](../transformers/models.md#images-with-qwen35). The patches of an image that Hugging Face's processor resized are its `pixel_values` to float rounding. A resized image, shrunk or grown (under `min_pixels`), is at most one 8-bit level (0.0078 normalized) from the processor's, two for a shrink of several times, as for the other models; the grow needs `image-0.3.0`, which aligns the centres of an enlarged pixel like Pillow's bicubic.

## Tiles for SmolVLM

`idefics3_tiling(height, width)` gives the sizes that Hugging Face's `Idefics3ImageProcessor` works out with the settings of SmolVLM: the longest edge goes to 2048 (up as well as down) and the other side keeps the aspect ratio, cut to a whole number and made even (`resized`); then each side is rounded up to a multiple of 512 (`grid`), and the grid holds `rows` by `cols` tiles of 512 by 512. `idefics3_tiles(image, backend)` makes the tiles as tensors on `backend`: the image resized to `resized` and then to `grid`, cut row by row, and the whole grid resized to one tile, last (the global image); each is `[3, 512, 512]` in `F32`, `0.0..1.0`, as `to_tensor` gives it. The image must be an sRGB one of 3 bands (`Image.colourspace`; the processor converts itself, this function gives an error for another band count). The processor's `do_image_splitting: false` (one image, no tiles) is not supported, and the sizes are those of the settings above only.

Each tile goes through `normalize` with mean and std 0.5, as the `pixel_values` of the processor do. The resize is Pillow's Lanczos, made here as products by matrices of its weights (libvips enlarges with bicubic interpolation whatever the kernel), with each pass held to 0..255 but not rounded to a level: the frames are within 1.5 levels of 8 bits (0.0112 after the normalization) of Hugging Face's, with hard edges too. A 3000 by 2000 image takes under 2 seconds on the CPU. The ids of the prompt for the tiles are in [`SmolVlmPrompt`](../transformers/models.md#smolvlm).

```agentica {mode="fragment"}
image = Image.colourspace(Image.load("photo.jpg").unwrap!, Image.Space.Srgb).unwrap!
tiles = Vision.idefics3_tiles(image, backend).unwrap!
# tiles.tiles: rows * cols tiles of [3, 512, 512], then the global image
frames = tiles.tiles.map { |t| Vision.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).unwrap! }
```

## A folder per class

`folder(root)` reads a folder per class, as ImageNet does (`root/cats/1.jpg`, `root/dogs/2.png`). The classes are the folder names in name order, and the images in each (`png`, `jpg`, `jpeg`, `webp`, `tif`, `tiff`, `bmp`, `gif`), in name order. A name that starts with `.` is skipped. It gives a `Folder` with `classes`, `paths` and `labels`.

`samples(folder, rows, steps, seed, backend)` loads the images `rows` and gives a `Samples`: `inputs` is `[n, 3, height, width]` of `F32` in `0..1` and `targets` is `[n]` of `I32` class indexes. Each image is turned to sRGB (a grey one gets 3 bands), put through `steps`, and read as pixels; the batch is one tensor, made in one call. Every image must end with the same size, so end `steps` with a crop. An image with an alpha band, or one of another size, is an `error` that names its path. Image `i` draws its random steps from `seed` and `i`, so a new `seed` for each epoch gives new crops and flips.

A dataset of the whole folder feeds the loader of `train`, as in [the data loader](../train/data.md):

```agentica {mode="fragment"}
folder = Vision.folder("photos").unwrap!
steps = [Vision.Step.Resize(36, Vision.Kernel.Lanczos3), Vision.Step.CenterCrop(32, 32)]
s = Vision.samples(folder, (0...folder.paths.size).to_a, steps, 0, backend).unwrap!
dataset = Data.dataset(s.inputs, s.targets).unwrap!
```

`samples` of every row holds all the images as tensors, so it fits a dataset that fits in memory. For a larger one, or for new random crops each epoch, take `samples` of the rows of one batch inside a batch function of `Train.fit_batches`, and give it `prefetch: 2`: a task then decodes the next batches while the device runs the step.

```agentica {mode="fragment"}
batch = ->(epoch: Int, i: Int) {
  s = try Vision.samples(folder, rows_of(order, i), steps, epoch, backend)
  ok(Data.Batch(inputs: s.inputs, targets: s.targets))
}

fit = try Train.fit_batches(&loss, params, batch, batches, optimizer, epochs, prefetch: 2)
```

## Limits

- The `[n, 3, height, width]` layout is the one `Nn.conv2d` takes. [ResNet](../transformers/models.md#resnet) classifies a batch of them after `normalize`, as [ViT](../transformers/models.md#vit) does, and [CLIP](../transformers/generation.md#zero-shot-with-clip) embeds it.
- `dtype: U8` is not for ggml, which cannot convert to `U8`.
- The samples cross as little-endian `F32`. A big-endian machine is untested, and its pixels would be wrong.
- The `U8` cast of `dtype: U8` is tested on the reference only, not on libtorch or XLA.
- `samples` runs in the task that calls it. In `Train.fit_batches` that is the task of `prefetch` when you give one, and the step's own task when you do not.
