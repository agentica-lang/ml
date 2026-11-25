## `Vision`: the pixels of an image reach a tensor exactly, the transforms cut where torchvision cuts, and a folder per
## class gives batches. Images are made in the tests (`Image.from_pixels`, `Image.save`), so no fixture is committed.

import Image from "pkg/image/image"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

private

## 3 by 2 pixels of 3 bands: red 10 20 30, 40 50 60, 70 80 90 on the first row and the same plus 100 on the second.
RGB = [10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150, 160, 170, 180, 190]

def test_u8_pixels_reach_a_tensor_and_come_back_exact() -> Result[Void]
  image = try S.u8(RGB, 3, 2, 3)
  hwc = try Vision.to_tensor(image, S.REF, Tensor.Dtype.U8, Vision.Layout.Hwc)
  try Test.eq(Tensor.shape(hwc), [2, 3, 3])
  try Test.eq(Tensor.dtype(hwc), Tensor.Dtype.U8)
  got = try S.ints_of(hwc)
  try Test.eq(got, RGB)
  back = try Image.from_pixels(try Bytes.from_list(got), 3, 2, 3, Image.Format.U8)
  try Test.eq(try S.samples_of(back), RGB)
  chw = try Vision.to_tensor(image, S.REF, Tensor.Dtype.U8)
  try Test.eq(Tensor.shape(chw), [3, 2, 3])
  try Test.eq(try S.ints_of(chw), [10, 40, 70, 110, 140, 170, 20, 50, 80, 120, 150, 180, 30, 60, 90, 130, 160, 190])
  ok()
end

## Distinct values in each band, so a wrong band order or permute on a backend shows.
def test_f32_keeps_the_band_order_and_the_layout_on_each_backend() -> Result[Void]
  image = try S.u8(RGB, 3, 2, 3)
  unit = RGB.map { |v| v.to_f / 255.0 }
  planes = [0, 1, 2].flat_map { |c| [0, 1, 2, 3, 4, 5].map { |i| unit[i * 3 + c] } }
  for backend in try S.backends()
    hwc = try Vision.to_tensor(image, backend, Tensor.Dtype.F32, Vision.Layout.Hwc)
    try Test.eq(Tensor.shape(hwc), [2, 3, 3])
    try S.close(try S.floats_of(hwc), unit, 0.000001)
    chw = try Vision.to_tensor(image, backend)
    try Test.eq(Tensor.shape(chw), [3, 2, 3])
    try S.close(try S.floats_of(chw), planes, 0.000001)
  end

  ok()
end

def test_f32_scales_to_the_unit_range_and_keeps_the_layout() -> Result[Void]
  image = try S.u8([0, 51, 255, 102], 2, 2, 1)
  for backend in try S.backends()
    t = try Vision.to_tensor(image, backend)
    try Test.eq(Tensor.shape(t), [1, 2, 2])
    try Test.eq(Tensor.dtype(t), Tensor.Dtype.F32)
    try S.close(try S.floats_of(t), [0.0, 0.2, 1.0, 0.4], 0.000001)
  end

  grey16 = try Image.from_pixels(try Bytes.from_list([0, 0, 255, 255]), 2, 1, 1, Image.Format.U16)
  half = try Vision.to_tensor(grey16)
  try S.close(try S.floats_of(half), [0.0, 1.0], 0.000001)
  try Test.error?(Vision.to_tensor(grey16, S.REF, Tensor.Dtype.U8))
  Test.error?(Vision.to_tensor(image, S.REF, Tensor.Dtype.I32))
end

def test_resize_keeps_the_aspect_ratio_and_crops_center_like_torchvision() -> Result[Void]
  wide = try S.u8(List.repeat(1, 8 * 4), 8, 4, 1)
  small = try Vision.apply(wide, [Vision.Step.Resize(2)])
  try Test.eq([Image.width(small), Image.height(small)], [4, 2])
  tall = try S.u8(List.repeat(1, 3 * 5), 3, 5, 1)
  cut = try Vision.apply(tall, [Vision.Step.Resize(2)])
  try Test.eq([Image.width(cut), Image.height(cut)], [2, 3])
  try Test.error?(Vision.apply(wide, [Vision.Step.Resize(0)]))
  line = try S.u8([0, 1, 2, 3, 4, 5, 6, 7, 8], 9, 1, 1)
  # The gaps 1, 3 and 5 fall on a half: Python rounds them to the even offset, 0, 2 and 2.
  five = try Vision.apply(try S.u8([0, 1, 2, 3, 4], 5, 1, 1), [Vision.Step.CenterCrop(4, 1)])
  try Test.eq(try S.samples_of(five), [0, 1, 2, 3])
  seven = try Vision.apply(try S.u8([0, 1, 2, 3, 4, 5, 6], 7, 1, 1), [Vision.Step.CenterCrop(4, 1)])
  try Test.eq(try S.samples_of(seven), [2, 3, 4, 5])
  nine = try Vision.apply(line, [Vision.Step.CenterCrop(4, 1)])
  try Test.eq(try S.samples_of(nine), [2, 3, 4, 5])
  Test.error?(Vision.apply(line, [Vision.Step.CenterCrop(10, 1)]))
end

def test_random_steps_follow_the_seed() -> Result[Void]
  pixels = (0...64).to_a
  image = try S.u8(pixels, 8, 8, 1)
  steps = [Vision.Step.RandomCrop(4, 4), Vision.Step.HorizontalFlip(0.5)]
  a = try S.samples_of(try Vision.apply(image, steps, 7))
  try Test.eq(try S.samples_of(try Vision.apply(image, steps, 7)), a)
  seen: Set[String] = Set.new([])
  for seed in 0...12
    seen = seen.add((try S.samples_of(try Vision.apply(image, steps, seed))).map { |v| v.to_s }.join(","))
  end

  try Test.assert(seen.size > 4, "12 seeds gave #{seen.size} different crops")
  always = try Vision.apply(try S.u8([1, 2, 3, 4], 4, 1, 1), [Vision.Step.HorizontalFlip(1.0)], 3)
  try Test.eq(try S.samples_of(always), [4, 3, 2, 1])
  never = try Vision.apply(try S.u8([1, 2, 3, 4], 4, 1, 1), [Vision.Step.HorizontalFlip(0.0)], 3)
  try Test.eq(try S.samples_of(never), [1, 2, 3, 4])
  try Test.error?(Vision.apply(image, [Vision.Step.HorizontalFlip(1.5)]))
  Test.error?(Vision.apply(image, [Vision.Step.RandomCrop(9, 4)]))
end

def test_normalize_subtracts_the_mean_and_divides_by_the_std_of_each_channel() -> Result[Void]
  for backend in try S.backends()
    x = try Tensor.floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 1, 3], backend)
    chw = try Vision.normalize(x, [1.0, 4.0], [1.0, 2.0])
    try S.close(try S.floats_of(chw), [0.0, 1.0, 2.0, 0.0, 0.5, 1.0], 0.000001)
    hwc = try Vision.normalize(
      try Tensor.floats([1.0, 2.0, 3.0, 4.0], [1, 2, 2], backend),
      [1.0, 2.0],
      [1.0, 0.5],
      Vision.Layout.Hwc,
    )

    try S.close(try S.floats_of(hwc), [0.0, 0.0, 2.0, 4.0], 0.000001)
    try Test.error?(Vision.normalize(x, [1.0], [1.0]))
    try Test.error?(Vision.normalize(x, [1.0, 4.0], [1.0, 0.0]))
  end

  Test.error?(Vision.normalize(try Tensor.floats([1.0], [1], S.REF), [1.0], [1.0]))
end

## `name` ("a/1.png") under `root`, an image of `pixels` of `bands` bands, 2 by 2.
def save(root: String, name: String, pixels: List[Int], bands: Int) -> Result[Void]
  dir = "#{root}/#{name.split("/")[0]}"
  if !File.stat(dir, 5000).ok?
    try Dir.mkdir(dir, 5000)
  end

  Image.save(try S.u8(pixels, 2, 2, bands), "#{root}/#{name}")
end

def make_folder(root: String) -> Result[Void]
  try save(root, "b/2.png", [20, 20, 20, 20], 1)
  try save(root, "b/1.PNG", [10, 10, 10, 10], 1)
  try save(root, "a/x.png", [30, 30, 30, 30], 1)
  try save(root, "c/empty.png", [40, 40, 40, 40], 1)
  try File.write_text("#{root}/c/notes.txt", "x", 5000)
  try Dir.mkdir("#{root}/.hidden", 5000)
  try save(root, ".hidden/1.png", [1, 1, 1, 1], 1)
  try File.write_text("#{root}/readme.txt", "x", 5000)
  ok()
end

def test_a_folder_lists_its_classes_and_images_in_order() -> Result[Void]
  S.with_dir do |root|
    try make_folder(root)
    f = try Vision.folder(root)
    try Test.eq(f.classes, ["a", "b", "c"])
    try Test.eq(f.paths, ["#{root}/a/x.png", "#{root}/b/1.PNG", "#{root}/b/2.png", "#{root}/c/empty.png"])
    try Test.eq(f.labels, [0, 1, 1, 2])
    Test.error?(Vision.folder("#{root}/.hidden"))
  end
end

def test_samples_make_one_batch_of_srgb_images_on_each_backend() -> Result[Void]
  S.with_dir do |root|
    try make_folder(root)
    f = try Vision.folder(root)
    for backend in try S.backends()
      s = try Vision.samples(f, [2, 0], [], 0, backend)
      try Test.eq(Tensor.shape(s.inputs), [2, 3, 2, 2])
      try Test.eq(try S.ints_of(s.targets), [1, 0])
      try Test.eq(Tensor.dtype(s.targets), Tensor.Dtype.I32)
      values = try S.floats_of(s.inputs)
      try S.close(values.take(12), List.repeat(20.0 / 255.0, 12), 0.000001)
      try S.close(values.drop(12), List.repeat(30.0 / 255.0, 12), 0.000001)
    end

    try Test.error?(Vision.samples(f, [4], [], 0))
    Test.error?(Vision.samples(f, [], [], 0))
  end
end

def test_images_that_differ_or_have_alpha_are_an_error_that_names_the_path() -> Result[Void]
  S.with_dir do |root|
    try Dir.mkdir("#{root}/a", 5000)
    try Image.save(try S.u8(List.repeat(5, 4 * 4), 4, 4, 1), "#{root}/a/big.png")
    try Image.save(try S.u8(List.repeat(5, 2 * 2), 2, 2, 1), "#{root}/a/small.png")
    f = try Vision.folder(root)
    mixed = Vision.samples(f, [0, 1], [], 0)
    try Test.error?(mixed)
    try Test.assert(mixed.message.include?("a/small.png"), "the error names no path: #{mixed.message}")
    alike = try Vision.samples(f, [0, 1], [Vision.Step.Resize(2), Vision.Step.CenterCrop(2, 2)], 0)
    try Test.eq(Tensor.shape(alike.inputs), [2, 3, 2, 2])
    try Image.save(try S.u8(List.repeat(9, 16), 2, 2, 4), "#{root}/a/alpha.png")
    withalpha = Vision.samples(try Vision.folder(root), [0], [], 0)
    try Test.error?(withalpha)
    Test.assert(withalpha.message.include?("a/alpha.png"), "the error names no path: #{withalpha.message}")
  end
end

def test_samples_keep_the_band_order_of_a_colour_image_on_each_backend() -> Result[Void]
  S.with_dir do |root|
    try Dir.mkdir("#{root}/a", 5000)
    rgb = [10, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130]
    try Image.save(try S.u8(rgb, 2, 2, 3), "#{root}/a/1.png")
    planes = [0, 1, 2].flat_map { |c| [0, 1, 2, 3].map { |i| rgb[i * 3 + c].to_f / 255.0 } }
    f = try Vision.folder(root)
    for backend in try S.backends()
      s = try Vision.samples(f, [0], [], 0, backend)
      try Test.eq(Tensor.shape(s.inputs), [1, 3, 2, 2])
      try S.close(try S.floats_of(s.inputs), planes, 0.000001)
    end

    ok()
  end
end
