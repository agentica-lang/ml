## A tiny random Qwen3.5 vision tower (`qwen35_vision_golden.py`, `fixtures/qwen35_vision_golden.json`): its weights,
## written here as a safetensors file under the names of an official checkpoint, with its `config.json`, are opened by
## `Qwen3_5Vision.open`, and the merged features of two images (a grid of 4 by 6 patches and one of 6 by 2, over a table
## of 4 by 4 positions) match what Hugging Face's `Qwen3_5VisionModel` gave, on the reference and on ggml (CPU). Run from
## `transformers/`.

import Qwen3_5Vision from "../qwen3_5_vision"
import Write from "pkg/hub/safetensors_write"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The golden is in `float32`, the reference in doubles.
ABS = 0.0001
REL = 0.0001

## ggml computes in `F32` and sums in another order.
F32_ABS = 0.001
F32_REL = 0.001

struct Image
  grid: List[Int]
  pixels: List[Float]
  features: List[Float]
end

struct Golden
  config: String
  weights: Map[String, List[Float]]
  shapes: Map[String, List[Int]]
  images: List[Image]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/qwen35_vision_golden.json"), Golden)

## The model folder of the golden with `config` in place of its `config.json`, and its weights as `model.safetensors`
## but for the ones that `skip` names.
def with_model(g: Golden, config: String, skip: String, body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-qwen35-vision-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = populated(g, config, skip, dir, body)
  _ = Path.remove(dir)
  result
end

def populated(g: Golden, config: String, skip: String, dir: String, body: Fn[String, Result[Void]]) -> Result[Void]
  tensors: Map[String, Tensor.Tensor] = {}
  for name in g.weights.keys
    if name != skip
      tensors = tensors.put(name, try S.f32(g.weights[name], g.shapes[name]))
    end
  end

  try Write.write("#{dir}/model.safetensors", tensors)
  try File.write_text("#{dir}/config.json", config)
  body.call(dir)
end

def features(m: Qwen3_5Vision.Vision, backend: Tensor.Backend, image: Image) -> Result[List[Float]]
  patches = try S.f32(
    image.pixels,
    [image.grid[0] * image.grid[1], image.pixels.size / (image.grid[0] * image.grid[1])],
    backend,
  )

  out = try Qwen3_5Vision.encode(m, patches, image.grid[0], image.grid[1])
  ok(try S.floats_of(out))
end

def matches(g: Golden, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  with_model(g, g.config, "") do |dir|
    m = try Qwen3_5Vision.open(dir, backend)
    for image in g.images
      try S.expect_close(try features(m, backend, image), image.features, abs, rel, "features of #{image.grid}")
    end

    Qwen3_5Vision.close_model(m)
    ok()
  end
end

def test_the_features_match_hugging_face_on_the_reference() -> Result[Void] = matches(try golden(), S.REF, ABS, REL)

def test_the_features_match_hugging_face_on_ggml() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends()
    try matches(g, backend, F32_ABS, F32_REL)
  end

  ok()
end

def test_a_grid_the_patches_do_not_fit_is_an_error() -> Result[Void]
  g = try golden()
  image = g.images[0]
  with_model(g, g.config, "") do |dir|
    m = try Qwen3_5Vision.open(dir, S.REF)
    patches = try S.f32(image.pixels, [24, 96])
    try S.expect_error(Qwen3_5Vision.encode(m, patches, 3, 8), "spatial_merge_size")
    try S.expect_error(Qwen3_5Vision.encode(m, patches, 4, 4), "the patches need")
    S.expect_error(Qwen3_5Vision.encode(m, try S.f32([1.0, 2.0], [1, 2]), 4, 6), "the patches need")
  end
end

def test_a_checkpoint_the_tower_cannot_read_is_an_error_that_names_why() -> Result[Void]
  g = try golden()
  try with_model(g, g.config, "model.visual.blocks.1.attn.qkv.bias") do |dir|
    S.expect_error(Qwen3_5Vision.open(dir, S.REF), "blocks.1.attn.qkv.bias")
  end

  for change in [
    ["\"model_type\": \"qwen3_5\", \"vision_config\"", "\"model_type\": \"qwen3_5_moe\", \"vision_config\""],
    ["\"gelu_pytorch_tanh\"", "\"gelu\""],
    ["\"num_position_embeddings\": 16", "\"num_position_embeddings\": 15"],
    ["\"hidden_size\": 16", "\"hidden_size\": 18"],
  ]
    wrong = try S.replaced(g.config, change[0], change[1])
    try with_model(g, wrong, "") do |dir| Test.error?(Qwen3_5Vision.open(dir, S.REF)) end
  end

  ok()
end
