## The vision tower and the connector of SmolVLM against Hugging Face (`smolvlm_hf_golden.py`,
## `fixtures/smolvlm/golden.json`), on the real `HuggingFaceTB/SmolVLM-256M-Instruct` in `float32`: each 512 by 512 tile
## of `fixtures/smolvlm/tiles` (8-bit, sha256 in the golden) goes through `to_tensor`, `normalize` (mean and std 0.5)
## and `SmolVlmVision.encode`, and its 64 rows of 576 are compared with `get_image_features`: the mean, the mean of the
## absolute values and one value of every `stride`th place.
##
## It needs the checkpoint, which is not in the repository, so it runs only when `SMOLVLM_MODEL` names its folder
## (`config.json` and `model.safetensors`). `SMOLVLM_DEVICE=vulkan` runs on Vulkan, the CPU otherwise. Run from `vision/`:
## `SMOLVLM_MODEL=SmolVLM-256M-Instruct agentica test --max-steps 1000000000000 test/smolvlm_hf_test.agc`.

import Hf from "hf"
import Image from "pkg/image/image"
import Lift from "pkg/transformers/lift"
import SmolVlmVision from "pkg/transformers/smolvlm_vision"
import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

SIZE = 512

## What passes for a sampled value or a mean: about 2.7 times the largest gap measured (3.6e-5 and 5.3e-5 on the CPU,
## 1.1e-4 and 9.2e-5 on Vulkan, for values of about 4 in absolute mean).
SAMPLE_ABS = 0.0003

struct Case
  name: String
  tile: String
  tile_sha256: String
  shape: List[Int]
  stride: Int
  mean: Float
  abs_mean: Float
  samples: List[Float]
end

struct Golden
  cases: List[Case]
end

private

def test_the_tiles_are_the_files_of_the_golden() -> Result[Void]
  want = try JSON.decode(try File.read_text("test/fixtures/smolvlm/golden.json"), Golden)
  try Test.eq(want.cases.size, 2)
  for c in want.cases
    try Test.eq(OpenSSL.Digest.hexencode(try File.sha256("test/fixtures/#{c.tile}")), c.tile_sha256)
  end

  ok()
end

def test_the_wide_tile_gives_hugging_faces_rows() -> Result[Void] = checked("wide")

def test_the_tall_tile_gives_hugging_faces_rows() -> Result[Void] = checked("tall")

## The case `name` of the golden on the checkpoint of `SMOLVLM_MODEL`, or nothing when it names none.
def checked(name: String) -> Result[Void]
  listed = System.env("SMOLVLM_MODEL")
  if !listed.ok? or try listed == ""
    puts("smolvlm_hf_test: skipped, SMOLVLM_MODEL names no checkpoint folder")
    return ok()
  end

  backend = try Hf.device_named("SMOLVLM_DEVICE")
  return error("SMOLVLM_MODEL is set and ggml is not built") if !Tensor.available(backend)
  want = try JSON.decode(try File.read_text("test/fixtures/smolvlm/golden.json"), Golden)
  c = try want.cases.find { |k| k.name == name }
  puts(try check(try listed, c, backend))
  ok()
end

## The tile of `c` as `[3, 512, 512]`, normalized as the checkpoint's image processor does.
def tile_tensor(c: Case, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Native.arena do
    image = try Image.colourspace(try Image.load("test/fixtures/#{c.tile}"), Image.Space.Srgb)
    Vision.normalize(try Vision.to_tensor(image, backend), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
  end
end

def check(dir: String, c: Case, backend: Tensor.Backend) -> Result[String]
  m = try SmolVlmVision.open(dir, backend)
  tile = try tile_tensor(c, backend)
  rows = try SmolVlmVision.encode(m, tile)
  try Test.eq(Tensor.shape(rows), [1, c.shape[0], c.shape[1]])
  values = try floats_of(rows)
  SmolVlmVision.close_model(m)
  Lift.close([tile, rows])
  gap = 0.0
  for i in 0...c.samples.size
    gap = [gap, (values[i * c.stride] - c.samples[i]).abs].max.unwrap!
  end

  mean = values.sum / values.size.to_f
  abs_mean = values.map { |v| v.abs }.sum / values.size.to_f
  gap = [gap, (mean - c.mean).abs, (abs_mean - c.abs_mean).abs].max.unwrap!
  try Test.assert(gap <= SAMPLE_ABS, "a value is #{gap} from Hugging Face's, more than #{SAMPLE_ABS}")
  ok("#{c.name}: #{c.samples.size} values and the means within #{gap}")
end

def floats_of(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end
