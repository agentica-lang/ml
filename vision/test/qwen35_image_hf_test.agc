## Qwen3.5 with an image against Hugging Face (`qwen35_image_hf_golden.py`, `fixtures/qwen/chat_golden.json`), on the
## real `Qwen/Qwen3.5-0.8B` in `float32`: for each case the resized picture that the image processor made (the 8-bit
## crop `chat-NAME.png`, which `qwen_pixels_test` does not need to resize again) goes through `to_tensor`, `normalize`
## and `patches` to the vision tower, and the chat template and the tokenizer make the prompt with the tokens of the
## image (`Chat.IMAGE`, `Mrope.expand`); the prompt must be the ids of Hugging Face's processor, and the greedy ids that
## `generate_qwen35` adds from the images' features, spliced in with their M-RoPE positions, must be the ones of Hugging
## Face's `generate`. The five largest logits of every step are compared too, within `LOGIT_ABS`: with the crop as the
## input, the gap is the model's, so the bound is tight enough to fail when the positions or the splice are off.
##
## It needs the checkpoint, which is not in the repository, so it runs only when `QWEN35_MODEL` names its folder (the
## revision of `transformers/DESIGN.md`, with `model.safetensors` or its index, `config.json` and `tokenizer.json`).
## `QWEN35_DEVICE=vulkan` runs on Vulkan, the CPU otherwise. The vision tower is closed before the text model opens, so
## the two never share the memory. Run from `vision/`, in an environment where `QWEN35_MODEL` is the folder of the
## checkpoint:
## `QWEN35_MODEL=Qwen3.5-0.8B agentica test --max-steps 1000000000000 test/qwen35_image_hf_test.agc`.

import Bpe from "pkg/transformers/tokenizer/bpe"
import Chat from "pkg/transformers/chat"
import Generate from "pkg/transformers/generate_qwen35"
import Hf from "hf"
import Image from "pkg/image/image"
import KvDecoder from "pkg/transformers/kv_decoder"
import Lift from "pkg/transformers/lift"
import Mrope from "pkg/transformers/mrope"
import Qwen3_5 from "pkg/transformers/qwen3_5"
import Qwen3_5Vision from "pkg/transformers/qwen3_5_vision"
import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

PATCH = 16
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]
## The end ids of the chat model: `<|im_end|>` and `<|endoftext|>`.
STOP = [248046, 248044]

## What passes for the five largest logits of a step (they are up to about 20): twice the largest gap measured, 0.0023 on
## the CPU and 0.0021 on Vulkan for the three cases. It is the model's own (the tower alone is within 2.4e-5 of Hugging
## Face's and the text model within 2e-3, `transformers/DESIGN.md`), since the tower reads the crop of the processor.
## A change of the positions or of the splice moves a logit by far more (`DESIGN.md` has the mutants).
LOGIT_ABS = 0.005

struct Case
  name: String
  crop: String
  question: String
  grid: List[Int]
  ids: List[Int]
  generated: List[Int]
  gaps: List[Float]
  step_top_ids: List[List[Int]]
  step_top_logits: List[List[Float]]
end

struct Golden
  cases: List[Case]
end

## The features of a crop and the merge of the tower that made them.
struct Encoded
  features: Tensor.Tensor
  merge: Int
end

private

def test_a_shrunk_wide_picture_gives_hugging_faces_ids() -> Result[Void] = checked("wide")

def test_a_shrunk_tall_picture_gives_hugging_faces_ids() -> Result[Void] = checked("tall")

def test_an_enlarged_picture_gives_hugging_faces_ids() -> Result[Void] = checked("grow")

## The case `name` of the golden on the checkpoint of `QWEN35_MODEL`, or nothing when it names none. A test has 30
## seconds, so each case loads its own models.
def checked(name: String) -> Result[Void]
  listed = System.env("QWEN35_MODEL")
  if !listed.ok? or try listed == ""
    puts("qwen35_image_hf_test: skipped, QWEN35_MODEL names no checkpoint folder")
    return ok()
  end

  backend = try Hf.device_named("QWEN35_DEVICE")
  return error("QWEN35_MODEL is set and ggml is not built") if !Tensor.available(backend)
  want = try JSON.decode(try File.read_text("test/fixtures/qwen/chat_golden.json"), Golden)
  c = try want.cases.find { |k| k.name == name }
  puts(try check(try listed, c, backend))
  ok()
end

## The patches of the crop of `c` (the resized picture of Hugging Face's processor), `[grid, 1536]` on `backend`.
def crop_patches(c: Case, merge: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Native.arena do
    image = try Image.colourspace(try Image.load("test/fixtures/#{c.crop}"), Image.Space.Srgb)
    x = try Vision.to_tensor(image, backend)
    Vision.patches(try Vision.normalize(x, MEAN, STD), PATCH, merge)
  end
end

## The features of the crop of `c` through the tower of the checkpoint in `dir`, and the merge of the tower.
def encoded(dir: String, c: Case, backend: Tensor.Backend) -> Result[Encoded]
  vision = try Qwen3_5Vision.open(dir, backend)
  merge = Qwen3_5Vision.merge(vision)
  patches = try crop_patches(c, merge, backend)
  try Test.eq(Tensor.shape(patches)[0], c.grid[0] * c.grid[1])
  features = try Qwen3_5Vision.encode(vision, patches, c.grid[0], c.grid[1])
  Lift.close([patches])
  Qwen3_5Vision.close_model(vision)
  ok(Encoded(features: features, merge: merge))
end

## The prompt of `c` as ids: the template with `Chat.IMAGE` in the user's message, the tokenizer, and the tokens of the
## image in place of its pad.
def prompt_of(tok: Bpe.ByteLevelBpe, image_token: Int, merge: Int, c: Case) -> Result[List[Int]]
  text = try Chat.apply("qwen3_5", [Chat.Message(role: "user", content: Chat.IMAGE + c.question)])
  Mrope.expand(try Bpe.encode(tok, text), image_token, [Mrope.Grid(height: c.grid[0], width: c.grid[1])], merge)
end

def check(dir: String, c: Case, backend: Tensor.Backend) -> Result[String]
  tok = try Bpe.open("#{dir}/tokenizer.json")
  image_token = try Qwen3_5.image_token(dir)
  seen = try encoded(dir, c, backend)
  m = try Qwen3_5.open(dir, backend)
  g = try Generate.open(m, tok.pieces.size)
  ids = try prompt_of(tok, image_token, seen.merge, c)
  try Test.eq(ids, c.ids)
  grid = Mrope.Grid(height: c.grid[0], width: c.grid[1])
  layout = try Mrope.layout(ids, image_token, [grid], seen.merge)
  sight = KvDecoder.Sight(features: seen.features, image_token: image_token, layout: layout)
  made = try Generate.generate(g, ids, c.generated.size, STOP, none, some(sight))
  # Hugging Face keeps the end id it stopped at; ours stops before it.
  try Test.eq(made, c.generated.filter { |id| not STOP.include?(id) })
  gap = try logits_gap(g, c, sight)
  Generate.close(g)
  Qwen3_5.close_model(m)
  Lift.close([seen.features])
  try Test.assert(gap <= LOGIT_ABS, "a logit is #{gap} from Hugging Face's, more than #{LOGIT_ABS}")
  ok(
    "#{c.name}: #{ids.size} tokens, #{made.size} ids equal, logits within #{gap}, HF smallest gap #{c.gaps.min.unwrap!}",
  )
end

## The largest gap, over the steps of `c`, between our logits at the five ids Hugging Face found largest and its own:
## the prompt (with its images) and then the ids it generated are fed to `step_logits`.
def logits_gap(g: Generate.Generator, c: Case, sight: KvDecoder.Sight) -> Result[Float]
  cache = Generate.start(some(sight))
  feed = c.ids
  worst = 0.0
  for i in 0...c.generated.size
    s = try Generate.step_logits(g, cache, feed)
    cache = s.cache
    for k in 0...c.step_top_ids[i].size
      gap = (s.logits[c.step_top_ids[i][k]] - c.step_top_logits[i][k]).abs
      worst = [worst, gap].max.unwrap!
    end

    feed = [c.generated[i]]
  end

  Generate.finish(cache)
  ok(worst)
end
