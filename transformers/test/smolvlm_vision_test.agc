## A tiny random SmolVLM (`smolvlm_vision_golden.py`, `fixtures/smolvlm_vision_golden.json`): the weights of its SigLIP
## tower and its connector, written here as a safetensors file under the names of an official checkpoint with its
## `config.json`, are opened by `SmolVlmVision.open`, and the rows of three tiles (16 by 16 pixels, patches of 4, a pixel
## shuffle by 2: 4 rows of 12 for each) match what Hugging Face's `Idefics3Model.get_image_features` gave, on the
## reference and on ggml (CPU). Run from `transformers/`.

import SmolVlmVision from "../smolvlm_vision"
import Write from "pkg/hub/safetensors_write"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The golden is in `float32`, the reference in doubles.
ABS = 0.00002
REL = 0.00002

## ggml computes in `F32` and sums in another order.
F32_ABS = 0.0001
F32_REL = 0.0001

struct Golden
  config: String
  weights: Map[String, List[Float]]
  shapes: Map[String, List[Int]]
  pixels: List[Float]
  tiles: Int
  features: List[Float]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/smolvlm_vision_golden.json"), Golden)

## The model folder of the golden with `config` in place of its `config.json`, and its weights as `model.safetensors`
## but for the ones that `skip` names.
def with_model(g: Golden, config: String, skip: String, body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-smolvlm-vision-#{Random.int(0, 999999999)}"
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

def matches(g: Golden, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  with_model(g, g.config, "") do |dir|
    m = try SmolVlmVision.open(dir, backend)
    try Test.eq(SmolVlmVision.tokens(m), 4)
    tiles = try S.f32(g.pixels, [g.tiles, 3, 16, 16], backend)
    out = try SmolVlmVision.encode(m, tiles)
    try Test.eq(Tensor.shape(out), [g.tiles, 4, 12])
    one = try SmolVlmVision.encode(m, try S.f32(g.pixels.take(3 * 16 * 16), [3, 16, 16], backend))
    try Test.eq(Tensor.shape(one), [1, 4, 12])
    try S.expect_close(try S.floats_of(one), g.features.take(4 * 12), abs, rel, "one tile")
    try S.expect_close(try S.floats_of(out), g.features, abs, rel, "the rows of the tiles")
    SmolVlmVision.close_model(m)
    ok()
  end
end

def test_the_rows_match_hugging_face_on_the_reference() -> Result[Void] = matches(try golden(), S.REF, ABS, REL)

def test_the_rows_match_hugging_face_on_ggml() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends()
    try matches(g, backend, F32_ABS, F32_REL)
  end

  ok()
end

def test_a_tile_of_another_size_is_an_error() -> Result[Void]
  g = try golden()
  with_model(g, g.config, "") do |dir|
    m = try SmolVlmVision.open(dir, S.REF)
    try S.expect_error(SmolVlmVision.encode(m, try S.f32(g.pixels.take(3 * 8 * 8), [1, 3, 8, 8])), "the tiles need")
    try S.expect_error(SmolVlmVision.encode(m, try S.f32(g.pixels.take(3 * 16), [3, 4, 4])), "the tiles need")
    S.expect_error(SmolVlmVision.encode(m, try S.f32([], [0, 3, 16, 16])), "the tiles need")
  end
end

def test_a_checkpoint_the_tower_cannot_read_is_an_error_that_names_why() -> Result[Void]
  g = try golden()
  try with_model(g, g.config, "model.vision_model.encoder.layers.1.self_attn.q_proj.bias") do |dir|
    S.expect_error(SmolVlmVision.open(dir, S.REF), "layers.1.self_attn.q_proj.bias")
  end

  try with_model(g, g.config, "model.connector.modality_projection.proj.weight") do |dir|
    S.expect_error(SmolVlmVision.open(dir, S.REF), "modality_projection")
  end

  for change in [
    ["\"idefics3\", \"scale_factor\"", "\"smolvlm\", \"scale_factor\"", "model_type"],
    ["\"gelu_pytorch_tanh\"", "\"gelu\"", "hidden_act"],
    ["\"scale_factor\": 2", "\"scale_factor\": 3", "scale_factor"],
    ["\"scale_factor\": 2", "\"scale_factor\": 0", "scale_factor"],
    ["\"num_attention_heads\": 2", "\"num_attention_heads\": 5", "heads"],
    ["\"patch_size\": 4", "\"patch_size\": 5", "patch_size"],
    ["\"hidden_size\": 12", "\"hidden_size\": 13", "modality_projection"],
  ]
    wrong = try S.replaced(g.config, change[0], change[1])
    try with_model(g, wrong, "") do |dir| S.expect_error(SmolVlmVision.open(dir, S.REF), change[2]) end
  end

  ok()
end
