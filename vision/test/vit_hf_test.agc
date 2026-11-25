## ViT and the image pipeline against Hugging Face (`vit_golden.py`): each image of `fixtures/resnet` goes through
## `ResizeTo(224, 224)` and `normalize` (the mean and std of 0.5), as the image processor of `google/vit-base-patch16-224`
## does, and then through the model in `float32` on ggml. The pixels are compared with the golden (the mean of each
## channel and a grid of 8 by 8), the best 10 logits and the top-5 classes too. The 8-bit images that Hugging Face's
## processor made (`fixtures/vit/crops`) go through the model alone, which is the strict check.
##
## It needs the checkpoint, which is not in the repository, so it runs only when `VIT_MODEL` names its folder (the
## revision of `transformers/DESIGN.md`, with `model.safetensors` and `config.json`). `VIT_DEVICE=vulkan` runs on
## Vulkan, the CPU otherwise. Run from `vision/`:
## `VIT_MODEL=vit-base-patch16-224 agentica test --max-steps 1000000000000 test/vit_hf_test.agc`.

import Hf from "hf"
import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"
import ViT from "pkg/transformers/vit"

MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

## What passes: about twice the gap measured for the pixels (0.0235) and the logits after the pipeline (0.156), which
## differ from Hugging Face's in the resize (Lanczos in libvips, Pillow's bilinear). The model alone is `float32`
## through 12 blocks and measured 1.2e-5 on CPU and 9.6e-6 on Vulkan; its 1e-4 leaves room for other ggml builds.
## The logits of these drawn images lie within 0.03 to 0.3 of each other, so the resize reorders the top 5: the
## pipeline keeps the same five classes, and the model alone keeps their order too.
MODEL_ABS = 0.0001
MODEL_REL = 0.0
PIXEL_ABS = 0.05
PIPELINE_ABS = 0.32

struct Golden
  images: List[Hf.Pixel]
  cases: List[List[Hf.Entry]]
end

private

def test_the_model_and_the_pipeline_match_hugging_face() -> Result[Void]
  dir = System.env("VIT_MODEL")
  if !dir.ok? or try dir == ""
    puts("vit_hf_test: skipped, VIT_MODEL names no checkpoint folder")
    return ok()
  end

  backend = try Hf.device_named("VIT_DEVICE")
  return error("VIT_MODEL is set and ggml is not built") if !Tensor.available(backend)
  want = try JSON.decode(try File.read_text("test/fixtures/vit/golden.json"), Golden)
  m = try ViT.open(try dir, backend)
  names = want.images.map { |g| g.name }
  crops = try Hf.loaded("test/fixtures/vit/crops", names, [], backend)
  alone_logits = try ViT.run(m, try Vision.normalize(crops, MEAN, STD))
  alone = try Hf.logit_gap(
    "model alone",
    try ViT.top(m, alone_logits, 5),
    alone_logits,
    want.cases,
    MODEL_ABS,
    MODEL_REL,
  )

  steps = [Vision.Step.ResizeTo(Hf.SIZE, Hf.SIZE)]
  pixels = try Vision.normalize(try Hf.loaded("test/fixtures/resnet", names, steps, backend), MEAN, STD)
  pixel = try Hf.pixel_gap(pixels, want.images)
  return error("the pixels are #{pixel} from Hugging Face's, more than #{PIXEL_ABS}") if pixel > PIXEL_ABS
  logits = try ViT.run(m, pixels)
  pipeline = try Hf.logit_gap("pipeline", try ViT.top(m, logits, 5), logits, want.cases, PIPELINE_ABS, 0.0, false)
  puts(
    "vit: largest gap of the logits of the model alone #{alone}, of the pixels #{pixel}, of the logits after the pipeline #{pipeline}",
  )

  ok()
end
