## ResNet and the image pipeline against Hugging Face (`resnet_golden.py`): each image of `fixtures/resnet` goes through
## `Resize(256)`, `CenterCrop(224)` and `normalize`, as the image processor of `microsoft/resnet-*` does, and then
## through the model in `float32` on ggml. The pixels are compared with the golden (the mean of each channel and a grid
## of 8 by 8), the best 10 logits and the top-5 classes too. The 8-bit crops that Hugging Face's processor made
## (`fixtures/resnet/crops`) go through the model alone, which is the strict check.
##
## It needs the checkpoints, which are not in the repository, so it runs only when `RESNET_MODELS` names them: a list of
## folders (the revisions of `transformers/DESIGN.md`, each with `model.safetensors` and `config.json`) separated by
## `:`, and the folder name is the model's name in the golden. `RESNET_DEVICE=vulkan` runs on Vulkan, the CPU otherwise.
## Run from `vision/`:
## `RESNET_MODELS=resnet-50:resnet-18 agentica test --max-steps 1000000000000 test/resnet_hf_test.agc`.

import Hf from "hf"
import ResNet from "pkg/transformers/resnet"
import Tensor from "pkg/tensor/tensor"
import Vision from "../vision"

SHORTER = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

## What passes. The pixels and the logits after the pipeline have about twice the gap measured (0.0175 and 0.158): the
## pipeline differs from Hugging Face's in the resize (libvips and Pillow use different kernels), and the logits follow
## the pixels. The model alone is `float32` through up to 50 layers and measured 1.6e-5 on CPU and 2.1e-5 on Vulkan; its
## 1e-3 is looser on purpose, for the other machines and ggml builds (the sums of a convolution depend on the kernel).
MODEL_ABS = 0.001
MODEL_REL = 0.001
PIXEL_ABS = 0.035
PIPELINE_ABS = 0.32

struct Golden
  images: List[Hf.Pixel]
  models: Map[String, List[List[Hf.Entry]]]
end

private

def test_the_model_and_the_pipeline_match_hugging_face() -> Result[Void]
  listed = System.env("RESNET_MODELS")
  if !listed.ok? or try listed == ""
    puts("resnet_hf_test: skipped, RESNET_MODELS names no checkpoint folder")
    return ok()
  end

  backend = try Hf.device_named("RESNET_DEVICE")
  return error("RESNET_MODELS is set and ggml is not built") if !Tensor.available(backend)
  want = try JSON.decode(try File.read_text("test/fixtures/resnet/golden.json"), Golden)
  for dir in (try listed).split(":")
    report = try check(dir, want, backend)
    puts(report)
  end

  ok()
end

def check(dir: String, want: Golden, backend: Tensor.Backend) -> Result[String]
  name = dir.split("/").select { |part| part != "" }.last
  cases = try want.models.get(name).to_result("the golden has no model #{name}")
  m = try ResNet.open(dir, backend)
  names = want.images.map { |g| g.name }
  crops = try Hf.loaded("test/fixtures/resnet/crops", names, [], backend)
  alone_logits = try ResNet.run(m, try Vision.normalize(crops, MEAN, STD))
  alone = try Hf.logit_gap(
    "model alone",
    try ResNet.top(m, alone_logits, 5),
    alone_logits,
    cases,
    MODEL_ABS,
    MODEL_REL,
  )

  pixels = try preprocessed("test/fixtures/resnet", names, backend)
  pixel = try Hf.pixel_gap(pixels, want.images)
  return error("the pixels are #{pixel} from Hugging Face's, more than #{PIXEL_ABS}") if pixel > PIXEL_ABS
  logits = try ResNet.run(m, pixels)
  pipeline = try Hf.logit_gap("pipeline", try ResNet.top(m, logits, 5), logits, cases, PIPELINE_ABS, 0.0)
  ok(
    "#{name}: largest gap of the logits of the model alone #{alone}, of the pixels #{pixel}, of the logits after the pipeline #{pipeline}",
  )
end

## The images as the image processor of the checkpoints makes them: normalized `[n, 3, 224, 224]`.
def preprocessed(dir: String, names: List[String], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  steps = [Vision.Step.Resize(SHORTER), Vision.Step.CenterCrop(Hf.SIZE, Hf.SIZE)]
  Vision.normalize(try Hf.loaded(dir, names, steps, backend), MEAN, STD)
end
