## A chat with one image on SmolVLM against Hugging Face (`smolvlm_chat_hf_golden.py`,
## `fixtures/smolvlm/chat_golden.json`), the whole way on the real `HuggingFaceTB/SmolVLM-256M-Instruct` in `float32`:
## `Chat.apply` writes the text of the prompt, `Bpe` makes its ids, `Vision.idefics3_tiles` cuts the image, each tile
## goes through `normalize` and `SmolVlmVision.encode`, `SmolVlmPrompt.expand` writes the ids of the tiles into the
## prompt, and a greedy `Generate` with the features spliced in at the `<image>` ids gives the new ids. The text and
## the ids of the prompt must be Hugging Face's, and the new ids too (24, or fewer when the model ends its turn, which
## `generate` of Hugging Face marks with `<|im_end|>` after the `<end_of_utterance>`).
##
## It needs the checkpoint, which is not in the repository, so it runs only when `SMOLVLM_MODEL` names its folder
## (`config.json`, `model.safetensors`, `tokenizer.json` and `processor_config.json`). It runs
## on Vulkan (`SMOLVLM_DEVICE=vulkan`), where each case takes some 15 seconds; on the CPU the 13 tiles and the 876 ids
## of the prompt do not fit the 30 seconds that a test has (`RUNTIME-TEST-TIMEOUT`), so with another device it is
## skipped. Run from `vision/`: `SMOLVLM_MODEL=SmolVLM-256M-Instruct SMOLVLM_DEVICE=vulkan agentica test --max-steps
## 1000000000000 test/smolvlm_chat_hf_test.agc`.

import Bpe from "pkg/transformers/tokenizer/bpe"
import Chat from "pkg/transformers/chat"
import Hf from "hf"
import Image from "pkg/image/image"
import Generate from "pkg/transformers/generate"
import KvDecoder from "pkg/transformers/kv_decoder"
import Lift from "pkg/transformers/lift"
import Qwen3 from "pkg/transformers/qwen3"
import SmolVlmPrompt from "pkg/transformers/smolvlm_prompt"
import SmolVlmVision from "pkg/transformers/smolvlm_vision"
import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

## The id that ends a turn of Hugging Face's `generate` (`<|im_end|>`, `eos_token_id` of the checkpoint).
STOP = 2

struct Case
  name: String
  image: String
  image_sha256: String
  question: String
  text: String
  prompt_ids: List[Int]
  new_ids: List[Int]
  reply: String
end

struct Golden
  cases: List[Case]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/smolvlm/chat_golden.json"), Golden)

def test_the_images_are_the_files_of_the_golden() -> Result[Void]
  want = try golden()
  try Test.eq(want.cases.size, 2)
  for c in want.cases
    try Test.eq(OpenSSL.Digest.hexencode(try File.sha256("test/fixtures/#{c.image}")), c.image_sha256)
  end

  ok()
end

def test_a_wide_image_is_described_with_hugging_faces_ids() -> Result[Void] = checked("wide")

def test_a_tall_image_is_described_with_hugging_faces_ids() -> Result[Void] = checked("tall")

## The case `name` of the golden on the checkpoint of `SMOLVLM_MODEL`, or nothing when it names none.
def checked(name: String) -> Result[Void]
  listed = System.env("SMOLVLM_MODEL")
  if !listed.ok? or try listed == ""
    puts("smolvlm_chat_hf_test: skipped, SMOLVLM_MODEL names no checkpoint folder")
    return ok()
  end

  device = System.env("SMOLVLM_DEVICE")
  if !device.ok? or try device != "vulkan"
    puts("smolvlm_chat_hf_test: skipped, SMOLVLM_DEVICE is not vulkan")
    return ok()
  end

  backend = try Hf.device_named("SMOLVLM_DEVICE")
  return error("SMOLVLM_MODEL is set and ggml is not built") if !Tensor.available(backend)
  want = try golden()
  c = try want.cases.find { |k| k.name == name }
  puts(try check(try listed, c, backend))
  ok()
end

## The rows of the tiles of `tiles` as `[tiles * 64, 576]`: each tile through the tower, one at a time, in order.
def features(dir: String, tiles: Vision.Tiles, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  m = try SmolVlmVision.open(dir, backend)
  values: List[Tensor.Values] = []
  width = 0
  for t in tiles.tiles
    x = try Vision.normalize(t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    rows = try SmolVlmVision.encode(m, x)
    width = Tensor.shape(rows)[2]
    values = values + try Tensor.to_list(rows)
    Lift.close([x, rows])
  end

  SmolVlmVision.close_model(m)
  Tensor.from_list(values, [values.size / width, width], Tensor.Dtype.F32, backend)
end

def tiles_of(c: Case, backend: Tensor.Backend) -> Result[Vision.Tiles]
  Native.arena do
    image = try Image.colourspace(try Image.load("test/fixtures/#{c.image}"), Image.Space.Srgb)
    Vision.idefics3_tiles(image, backend)
  end
end

def check(dir: String, c: Case, backend: Tensor.Backend) -> Result[String]
  text = try Chat.apply("idefics3", [Chat.Message(role: "user", content: Chat.SMOLVLM_IMAGE + c.question)])
  try Test.eq(text, c.text)

  tok = try Bpe.open("#{dir}/tokenizer.json")
  prompt = try SmolVlmPrompt.open(dir)
  image_token = try Qwen3.image_token(dir)
  tiles = try tiles_of(c, backend)

  ids = try SmolVlmPrompt.expand(
    try Bpe.encode(tok, text),
    prompt,
    [SmolVlmPrompt.Grid(rows: tiles.rows, cols: tiles.cols)],
  )

  try Test.eq(ids, c.prompt_ids)

  rows = try features(dir, tiles, backend)
  sight = KvDecoder.plain_sight(rows, image_token, ids)
  m = try Qwen3.open_idefics3(dir, backend)
  gen = try Generate.open(m)
  made = try Generate.generate(gen, ids, c.new_ids.size, [STOP], none, some(sight))
  Generate.close(gen)
  Qwen3.close_model(m)
  Lift.close([rows])

  want = c.new_ids.select { |id| id != STOP }
  try Test.eq(made, want)
  ok("#{c.name}: #{ids.size} ids of the prompt, #{made.size} new ids: #{try Bpe.decode(tok, made)}")
end
