## The image pipeline, timed, in front of a model on ggml: real JPEG files (made here, 640 by 480) go through
## `Resize`, `CenterCrop`, `Vision.samples` and `Vision.normalize` to a batch of `BATCH` images, and the model's image
## forward runs on it, for `STEPS` batches. Three runs with the same batches:
##
##   device    the batch is decoded once and only the model runs: what the device takes, with no decoding
##   sync      each batch is decoded in the step, then the model runs (what `fit_batches` does with no prefetch)
##   prefetch  a task decodes the next batches while the model runs (`Data.ahead`, 2 ahead; `fit_batches` with `prefetch`)
##
## For each, the mean time of a step and the mean time that the step waited for its batch: the device idles that long
## (for `sync` it is the whole decode); `PREFETCH_OUT` names the file that gets the lines, since a passing test shows no
## output. Vulkan multiplies in `F32`, as the checks do (`Tensor.vulkan_fp16(false)`).
## A test only because `train` and `transformers` are test dependencies of `vision/`: it runs only when
## `PREFETCH_MODEL` is `resnet` or `clip`, so CI skips it. `PREFETCH_DIR` is the checkpoint folder, `PREFETCH_DEVICE`
## `cpu` (the default) or `vulkan`, `PREFETCH_BATCH` (32) and `PREFETCH_STEPS` (6) the sizes, and `PREFETCH_MODE` the run
## (`device`, `sync` by default, or `prefetch`): one a time, since a test has 30 seconds. Run from `vision/`:
##
##   PREFETCH_MODEL=resnet PREFETCH_DIR=resnet-50 agentica test --max-steps 1000000000000 test/prefetch_bench_test.agc

import Clip from "pkg/transformers/clip"
import Data from "pkg/train/data"
import Image from "pkg/image/image"
import ResNet from "pkg/transformers/resnet"
import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

SIDE = 224
FILES = 32
WIDTH = 640
HEIGHT = 480

private

## What the model's pipeline takes: the shorter side after the resize, and the channel mean and std.
struct Recipe
  shorter: Int
  mean: List[Float]
  std: List[Float]
end

def env_or(name: String, fallback: String) -> String
  case System.env(name)
    ok(value) ->
      return value if value != ""
      fallback
    error(_) -> fallback
  end
end

def test_the_pipeline_in_front_of_a_model_is_timed() -> Result[Void]
  kind = env_or("PREFETCH_MODEL", "")
  if kind == ""
    puts("prefetch_bench_test: skipped, PREFETCH_MODEL names no model")
    return ok()
  end

  device = Tensor.Device.Cpu
  if env_or("PREFETCH_DEVICE", "cpu") == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  lines = try measure(
    kind,
    env_or("PREFETCH_DIR", ""),
    Tensor.Backend.Ggml(device),
    env_or("PREFETCH_BATCH", "32").to_i!,
    env_or("PREFETCH_STEPS", "6").to_i!,
  )

  out = env_or("PREFETCH_OUT", "")
  if out != ""
    try File.write_text(out, lines.join("\n") + "\n")
  end

  ok()
end

def measure(kind: String, dir: String, backend: Tensor.Backend, batch: Int, steps: Int) -> Result[List[String]]
  try Tensor.vulkan_fp16(false)
  recipe = recipe_of(kind)
  forward = try forward_of(kind, dir, backend)
  root = "/tmp/agentica-bench-prefetch-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(root, 5000)
  result = run_all(root, kind, recipe, forward, backend, batch, steps)
  _ = Path.remove(root, 5000)
  result
end

def recipe_of(kind: String) -> Recipe
  if kind == "clip"
    return Recipe(shorter: 224, mean: [0.48145466, 0.4578275, 0.40821073], std: [0.26862954, 0.26130258, 0.27577711])
  end

  Recipe(shorter: 256, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])
end

def forward_of(kind: String, dir: String, backend: Tensor.Backend) -> Result[Fn[Tensor.Tensor, Result[Void]]]
  return clip_forward(dir, backend) if kind == "clip"
  resnet_forward(dir, backend)
end

def resnet_forward(dir: String, backend: Tensor.Backend) -> Result[Fn[Tensor.Tensor, Result[Void]]]
  m = try ResNet.open(dir, backend)
  ok(
    ->(images: Tensor.Tensor) {
      ResNet.close(try ResNet.run(m, images))
      ok()
    },
  )
end

def clip_forward(dir: String, backend: Tensor.Backend) -> Result[Fn[Tensor.Tensor, Result[Void]]]
  m = try Clip.open(dir, backend)
  ok(
    ->(images: Tensor.Tensor) {
      Clip.close(try Clip.image_embeds(m, images))
      ok()
    },
  )
end

## `FILES` JPEG files in one class folder, all the same picture: a gradient with a little noise, so that the decoder has
## work to do. Answers the size of a file.
def make_images(root: String) -> Result[Int]
  try Dir.mkdir("#{root}/c", 5000)
  state = 12345
  values: List[Int] = []
  for y in 0...HEIGHT
    for x in 0...WIDTH
      for c in 0...3
        state = (state * 48271) % 2147483647
        values = values.push((x * 3 + y * 2 + c * 60 + state % 24) % 256)
      end
    end
  end

  image = try Image.from_pixels(try Bytes.from_list(values), WIDTH, HEIGHT, 3, Image.Format.U8)
  for k in 0...FILES
    try Image.save(image, "#{root}/c/#{k}.jpg")
  end

  ok((try File.stat("#{root}/c/0.jpg")).size_bytes)
end

## Batch `step`: the images of its rows, through the steps of `recipe`, normalized, with the class ids.
def decoded(f: Vision.Folder, recipe: Recipe, backend: Tensor.Backend, batch: Int, step: Int) -> Result[Data.Batch]
  rows = (0...batch).map { |k| (step * batch + k) % FILES }
  steps = [Vision.Step.Resize(recipe.shorter), Vision.Step.CenterCrop(SIDE, SIDE)]
  s = try Vision.samples(f, rows, steps, 0, backend)
  normal = try Vision.normalize(s.inputs, recipe.mean, recipe.std)
  Data.free(s.inputs)
  ok(Data.Batch(inputs: normal, targets: s.targets))
end

## The mean of `total` over `steps`, in tenths of a millisecond as a decimal number.
def ms(total_ns: Int, steps: Int) -> String
  tenths = total_ns / 100000 / steps
  "#{tenths / 10}.#{tenths % 10}"
end

def report(label: String, step_ns: Int, wait_ns: Int, steps: Int) -> String
  "#{label}: step #{ms(step_ns, steps)} ms, waited for the batch #{ms(wait_ns, steps)} ms"
end

## One batch through the model, which compiles its graph, so that no timed step pays for it.
def warm(
  f: Vision.Folder,
  recipe: Recipe,
  forward: Fn[Tensor.Tensor, Result[Void]],
  backend: Tensor.Backend,
  batch: Int,
) -> Result[Void]
  b = try decoded(f, recipe, backend, batch, 0)
  try forward.call(b.inputs)
  Data.release(b)
  ok()
end

## Only the model: one batch decoded once, run `steps` times.
def device_run(
  f: Vision.Folder,
  recipe: Recipe,
  forward: Fn[Tensor.Tensor, Result[Void]],
  backend: Tensor.Backend,
  batch: Int,
  steps: Int,
) -> Result[String]
  once = try decoded(f, recipe, backend, batch, 0)
  started = Time.monotonic_ns()
  for _ in 0...steps
    try forward.call(once.inputs)
  end

  line = report("device  ", Time.monotonic_ns() - started, 0, steps)
  Data.release(once)
  ok(line)
end

## Each batch decoded in the step, then the model.
def sync_run(
  f: Vision.Folder,
  recipe: Recipe,
  forward: Fn[Tensor.Tensor, Result[Void]],
  backend: Tensor.Backend,
  batch: Int,
  steps: Int,
) -> Result[String]
  waited = 0
  started = Time.monotonic_ns()
  for i in 0...steps
    asked = Time.monotonic_ns()
    b = try decoded(f, recipe, backend, batch, i)
    waited = waited + Time.monotonic_ns() - asked
    try forward.call(b.inputs)
    Data.release(b)
  end

  ok(report("sync    ", Time.monotonic_ns() - started, waited, steps))
end

## The batches decoded by a task two ahead, the model beside it.
def prefetch_run(
  f: Vision.Folder,
  recipe: Recipe,
  forward: Fn[Tensor.Tensor, Result[Void]],
  backend: Tensor.Backend,
  batch: Int,
  steps: Int,
) -> Result[String]
  waited = 0
  stream = try Data.ahead(->(_e: Int, i: Int) { decoded(f, recipe, backend, batch, i) }, steps, 1, 2)
  started = Time.monotonic_ns()
  for _ in 0...steps
    asked = Time.monotonic_ns()
    got = Data.next(stream)
    waited = waited + Time.monotonic_ns() - asked
    if !got.ok?
      Data.finish(stream)
      return error(got.message)
    end

    b = got.unwrap!
    try forward.call(b.inputs)
    Data.release(b)
  end

  line = report("prefetch", Time.monotonic_ns() - started, waited, steps)
  Data.finish(stream)
  ok(line)
end

def run_all(
  root: String,
  kind: String,
  recipe: Recipe,
  forward: Fn[Tensor.Tensor, Result[Void]],
  backend: Tensor.Backend,
  batch: Int,
  steps: Int,
) -> Result[List[String]]
  bytes = try make_images(root)
  f = try Vision.folder(root)
  lines = [
    "#{kind}, #{backend}, batch #{batch}, #{steps} steps, #{FILES} JPEG files of #{WIDTH}x#{HEIGHT}, #{bytes} bytes each",
  ]

  try warm(f, recipe, forward, backend, batch)
  mode = env_or("PREFETCH_MODE", "sync")
  case mode
    "device" -> lines = lines.push(try device_run(f, recipe, forward, backend, batch, steps))
    "sync" -> lines = lines.push(try sync_run(f, recipe, forward, backend, batch, steps))
    "prefetch" -> lines = lines.push(try prefetch_run(f, recipe, forward, backend, batch, steps))
    _ -> return error("PREFETCH_MODE is device, sync or prefetch, got #{mode}")
  end

  ok(lines)
end
