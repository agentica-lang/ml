## A prompt with one image through SmolVLM against Hugging Face (`smolvlm_image_hf_golden.py`,
## `fixtures/smolvlm/image_golden.json`), on the real `HuggingFaceTB/SmolVLM-256M-Instruct` in `float32`: each 512 by 512
## tile of `fixtures/smolvlm/tiles` (the ones of `smolvlm_hf_test.agc`) goes through `to_tensor`, `normalize` and
## `SmolVlmVision.encode`; its 64 rows are spliced into the prompt of the golden at the 64 `<image>` tokens by a
## `KvDecoder.Sight` over `Qwen3.open_idefics3` (the text model of the same folder), and the logits of the last
## position are compared with Hugging Face's: the mean, the mean of the absolute values, one value of every `stride`th
## place and the ten largest with their ids.
##
## It needs the checkpoint, which is not in the repository, so it runs only when `SMOLVLM_MODEL` names its folder
## (`config.json` and `model.safetensors`). `SMOLVLM_DEVICE=vulkan` runs on Vulkan, the CPU otherwise. Run from `vision/`:
## `SMOLVLM_MODEL=SmolVLM-256M-Instruct agentica test --max-steps 1000000000000 test/smolvlm_image_hf_test.agc`.

import Hf from "hf"
import Image from "pkg/image/image"
import Generate from "pkg/transformers/generate"
import KvDecoder from "pkg/transformers/kv_decoder"
import Lift from "pkg/transformers/lift"
import Mrope from "pkg/transformers/mrope"
import Qwen3 from "pkg/transformers/qwen3"
import SmolVlmVision from "pkg/transformers/smolvlm_vision"
import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

## What passes for a logit or a mean: about 3 times the largest gap measured (2.4e-5 and 2.7e-5 on the CPU, 2.3e-5 and
## 3.2e-5 on Vulkan, for logits of about 20 at the top).
LOGIT_ABS = 0.0001

struct Case
  name: String
  tile: String
  tile_sha256: String
  stride: Int
  mean: Float
  abs_mean: Float
  samples: List[Float]
  top_ids: List[Int]
  top_values: List[Float]
end

struct Golden
  ids: List[Int]
  image_token: Int
  cases: List[Case]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/smolvlm/image_golden.json"), Golden)

def test_the_tiles_are_the_files_of_the_golden() -> Result[Void]
  want = try golden()
  try Test.eq(want.cases.size, 2)
  try Test.eq(want.ids.select { |id| id == want.image_token }.size, 64)
  for c in want.cases
    try Test.eq(OpenSSL.Digest.hexencode(try File.sha256("test/fixtures/#{c.tile}")), c.tile_sha256)
  end

  ok()
end

def test_the_wide_tile_in_a_prompt_gives_hugging_faces_logits() -> Result[Void] = checked("wide")

def test_the_tall_tile_in_a_prompt_gives_hugging_faces_logits() -> Result[Void] = checked("tall")

## The case `name` of the golden on the checkpoint of `SMOLVLM_MODEL`, or nothing when it names none.
def checked(name: String) -> Result[Void]
  listed = System.env("SMOLVLM_MODEL")
  if !listed.ok? or try listed == ""
    puts("smolvlm_image_hf_test: skipped, SMOLVLM_MODEL names no checkpoint folder")
    return ok()
  end

  backend = try Hf.device_named("SMOLVLM_DEVICE")
  return error("SMOLVLM_MODEL is set and ggml is not built") if !Tensor.available(backend)
  want = try golden()
  c = try want.cases.find { |k| k.name == name }
  puts(try check(try listed, want, c, backend))
  ok()
end

## The 64 rows of the tile of `c` as `[64, 576]`, normalized as the checkpoint's image processor does.
def features(dir: String, c: Case, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  m = try SmolVlmVision.open(dir, backend)
  tile = try tile_tensor(c, backend)
  rows = try SmolVlmVision.encode(m, tile)
  shape = Tensor.shape(rows)
  values = try Tensor.to_list(rows)
  SmolVlmVision.close_model(m)
  Lift.close([tile, rows])
  Tensor.from_list(values, [shape[1], shape[2]], Tensor.Dtype.F32, backend)
end

def tile_tensor(c: Case, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Native.arena do
    image = try Image.colourspace(try Image.load("test/fixtures/#{c.tile}"), Image.Space.Srgb)
    Vision.normalize(try Vision.to_tensor(image, backend), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
  end
end

def check(dir: String, want: Golden, c: Case, backend: Tensor.Backend) -> Result[String]
  rows = try features(dir, c, backend)
  sight = KvDecoder.Sight(
    features: rows,
    image_token: want.image_token,
    layout: Mrope.plain(want.ids, want.image_token),
  )

  m = try Qwen3.open_idefics3(dir, backend)
  gen = try Generate.open(m)
  s = try Generate.step_logits(gen, Generate.start(some(sight)), want.ids)
  Generate.finish(s.cache)
  Generate.close(gen)
  Qwen3.close_model(m)
  Lift.close([rows])
  values = s.logits
  gap = 0.0
  for i in 0...c.samples.size
    gap = [gap, (values[i * c.stride] - c.samples[i]).abs].max.unwrap!
  end

  mean = values.sum / values.size.to_f
  abs_mean = values.map { |v| v.abs }.sum / values.size.to_f
  gap = [gap, (mean - c.mean).abs, (abs_mean - c.abs_mean).abs].max.unwrap!
  for k in 0...c.top_ids.size
    gap = [gap, (values[c.top_ids[k]] - c.top_values[k]).abs].max.unwrap!
  end

  best = values.max.unwrap!
  try Test.eq(values.index(best).unwrap!, c.top_ids[0])
  try Test.assert(gap <= LOGIT_ABS, "a logit is #{gap} from Hugging Face's, more than #{LOGIT_ABS}")
  ok("#{c.name}: #{c.samples.size} logits, the means and the top 10 within #{gap}")
end
