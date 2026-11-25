## CLIP and the image pipeline against Hugging Face (`clip_golden.py`): each image of `fixtures/resnet` goes through
## `Resize(224)`, `CenterCrop(224)` and `normalize` (the CLIP mean and std), as the image processor of
## `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` does, and then through the vision tower in `float32` on ggml; the prompts
## go through the tokenizer and the text tower. The embeddings, the scaled cosines (`logits_per_image`) and their
## softmax over the prompts are compared with the golden, and the ids of each prompt are compared exactly. The pixels
## are compared too (the mean of each channel and a grid of 8 by 8). The 8-bit crops that Hugging Face's processor made
## (`fixtures/clip/crops`) go through the model alone, which is the strict check.
##
## It needs the checkpoint, which is not in the repository, so it runs only when `CLIP_MODEL` names its folder (the
## revision of `transformers/DESIGN.md`, with `model.safetensors`, `config.json`, `vocab.json` and `merges.txt`).
## `CLIP_DEVICE=vulkan` runs on Vulkan, the CPU otherwise. Run from `vision/`:
## `CLIP_MODEL=clip-laion-b32 agentica test --max-steps 1000000000000 test/clip_hf_test.agc`.

import Clip from "pkg/transformers/clip"
import ClipTokenizer from "pkg/transformers/tokenizer/clip"
import Hf from "hf"
import Image from "pkg/image/image"
import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
SHORTER = 224

## What passes. The model alone is `float32` through 12 blocks and measured, on CPU and Vulkan, 5e-6 (text) and 2e-5
## (image) for an embedding, 4e-5 for a logit (the scale of 100 multiplies the gap of the cosine) and 2e-6 for a
## probability; the tolerances leave room for other ggml builds. Through our own pipeline, with the `Cubic` kernel of
## the processor's `resample` 3, the pixels measured 0.015 and the logits 0.21 (0.015 and 0.43 with libvips' Lanczos 3),
## from the pixels, on CPU and Vulkan; those tolerances are about twice that.
MODEL_ABS = 0.0001
MODEL_REL = 0.0
LOGIT_ABS = 0.001
PROBABILITY_ABS = 0.0001
PIXEL_ABS = 0.03
PIPELINE_ABS = 0.43

struct Text
  text: String
  ids: List[Int]
end

struct Golden
  images: List[Hf.Pixel]
  texts: List[Text]
  image_embeds: List[List[Float]]
  text_embeds: List[List[Float]]
  logits: List[List[Float]]
  probabilities: List[List[Float]]
  scale: Float
end

private

def test_the_model_and_the_pipeline_match_hugging_face() -> Result[Void]
  dir = System.env("CLIP_MODEL")
  if !dir.ok? or try dir == ""
    puts("clip_hf_test: skipped, CLIP_MODEL names no checkpoint folder")
    return ok()
  end

  backend = try Hf.device_named("CLIP_DEVICE")
  return error("CLIP_MODEL is set and ggml is not built") if !Tensor.available(backend)
  want = try JSON.decode(try File.read_text("test/fixtures/clip/golden.json"), Golden)
  m = try Clip.open(try dir, backend)
  tok = try ClipTokenizer.open(try dir)
  rows = try ids_of(tok, want.texts)
  texts = try Clip.text_embeds(m, rows)
  text_gap = try Hf.matrix_gap("text embeddings", texts, want.text_embeds, MODEL_ABS, MODEL_REL)
  names = want.images.map { |g| g.name }
  crops = try Hf.loaded("test/fixtures/clip/crops", names, [], backend)
  alone = try Clip.image_embeds(m, try Vision.normalize(crops, MEAN, STD))
  alone_gap = try Hf.matrix_gap("image embeddings of the crops", alone, want.image_embeds, MODEL_ABS, MODEL_REL)
  logits = try Clip.logits(m, alone, texts)
  probabilities = try Clip.probabilities(m, alone, texts)
  logit_gap = try Hf.matrix_gap("logits of the crops", logits, want.logits, LOGIT_ABS, 0.0)
  probability_gap = try Hf.matrix_gap(
    "probabilities of the crops",
    probabilities,
    want.probabilities,
    PROBABILITY_ABS,
    0.0,
  )

  try same_best("the crops", logits, want.logits)
  return error("the scale is #{m.scale}, Hugging Face's is #{want.scale}") if (m.scale - want.scale).abs > 0.001

  pixels = try preprocessed("test/fixtures/resnet", names, backend)
  pixel = try Hf.pixel_gap(pixels, want.images)
  return error("the pixels are #{pixel} from Hugging Face's, more than #{PIXEL_ABS}") if pixel > PIXEL_ABS
  pipeline_embeds = try Clip.image_embeds(m, pixels)
  pipeline_logits = try Clip.logits(m, pipeline_embeds, texts)
  pipeline_gap = try Hf.matrix_gap("logits of the pipeline", pipeline_logits, want.logits, PIPELINE_ABS, 0.0)
  try same_best("the pipeline", pipeline_logits, want.logits)
  puts(
    "clip: largest gap of the text embeddings #{text_gap}, of the image embeddings of the crops #{alone_gap}, of their logits #{logit_gap} and probabilities #{probability_gap}; of the pixels #{pixel} and of the logits after the pipeline #{pipeline_gap}",
  )

  ok()
end

## The ids of the prompts from our tokenizer, which must be Hugging Face's.
def ids_of(tok: ClipTokenizer.Clip, texts: List[Text]) -> Result[List[List[Int]]]
  rows: List[List[Int]] = []
  for t in texts
    ids = try ClipTokenizer.encode(tok, t.text)
    return error("#{t.text.inspect}: the ids are #{ids}, Hugging Face's are #{t.ids}") if ids != t.ids
    rows = rows.push(ids)
  end

  ok(rows)
end

## The images as the image processor of the checkpoint makes them: normalized `[n, 3, 224, 224]`.
def preprocessed(dir: String, names: List[String], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  steps = [Vision.Step.Resize(SHORTER, Image.Kernel.Cubic), Vision.Step.CenterCrop(Hf.SIZE, Hf.SIZE)]
  Vision.normalize(try Hf.loaded(dir, names, steps, backend), MEAN, STD)
end

## An error when the best prompt of an image in `got` (`[images, prompts]`) is not the one of `want`.
def same_best(what: String, got: Tensor.Tensor, want: List[List[Float]]) -> Result[Void]
  values = try Hf.floats(got)
  columns = Tensor.shape(got)[1]
  for r in 0...want.size
    row = values.drop(r * columns).take(columns)
    ours = (0...columns).to_a.sort_by { |i| -row[i] }[0]
    theirs = (0...columns).to_a.sort_by { |i| -want[r][i] }[0]
    return error("#{what}, image #{r}: the best prompt is #{ours}, Hugging Face's is #{theirs}") if ours != theirs
  end

  ok()
end
