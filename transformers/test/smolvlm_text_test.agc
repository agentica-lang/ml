## A tiny random SmolVLM (`smolvlm_text_golden.py`, `fixtures/smolvlm_text_golden.json`): the weights of its text model
## and its head, written here as a safetensors file under the names of an official checkpoint
## (`model.text_model.*`, `lm_head.weight`) with its `config.json` (the Llama in `text_config`), are opened by
## `Qwen3.open_idefics3`, and the logits of every position of two rows of token ids match what Hugging Face's
## `Idefics3ForConditionalGeneration` gave, on the reference and on ggml (CPU). Run from `transformers/`.

import Qwen3 from "../qwen3"
import Write from "pkg/hub/safetensors_write"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The golden is in `float32`, the reference in doubles.
ABS = 0.00001
REL = 0.00001

## ggml computes in `F32` and sums in another order.
F32_ABS = 0.0001
F32_REL = 0.0001

struct Golden
  config: String
  weights: Map[String, List[Float]]
  shapes: Map[String, List[Int]]
  rows: List[List[Int]]
  logits: List[Float]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/smolvlm_text_golden.json"), Golden)

## The model folder of the golden with `config` in place of its `config.json`, and its weights as `model.safetensors`
## but for the ones that `skip` names.
def with_model(g: Golden, config: String, skip: String, body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-smolvlm-text-#{Random.int(0, 999999999)}"
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
    m = try Qwen3.open_idefics3(dir, backend)
    out = try Qwen3.run(m, g.rows)
    try Test.eq(Tensor.shape(out), [2, 6, 31])
    try S.expect_close(try S.floats_of(out), g.logits, abs, rel, "the logits of every position")
    Qwen3.close_model(m)
    ok()
  end
end

def test_the_logits_match_hugging_face_on_the_reference() -> Result[Void] = matches(try golden(), S.REF, ABS, REL)

def test_the_logits_match_hugging_face_on_ggml() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends()
    try matches(g, backend, F32_ABS, F32_REL)
  end

  ok()
end

def test_a_folder_the_text_model_cannot_read_is_an_error_that_names_why() -> Result[Void]
  g = try golden()
  try with_model(g, g.config, "model.text_model.layers.1.mlp.up_proj.weight") do |dir|
    S.expect_error(Qwen3.open_idefics3(dir, S.REF), "layers.1.mlp.up_proj")
  end

  try with_model(g, g.config, "lm_head.weight") do |dir| S.expect_error(Qwen3.open_idefics3(dir, S.REF), "lm_head") end

  for change in [
    ["\"model_type\": \"idefics3\"", "\"model_type\": \"smolvlm\"", "idefics3"],
    ["\"mlp_bias\": false", "\"mlp_bias\": true", "mlp_bias"],
    ["\"hidden_size\": 24", "\"hidden_size\": 28", "embed_tokens"],
  ]
    wrong = try S.replaced(g.config, change[0], change[1])
    try with_model(g, wrong, "") do |dir| S.expect_error(Qwen3.open_idefics3(dir, S.REF), change[2]) end
  end

  ok()
end

## A plain Llama folder is not a SmolVLM one: its tensors are not under `model.text_model.`.
def test_the_text_model_of_a_plain_llama_is_not_found() -> Result[Void]
  g = try golden()
  plain = try S.replaced(g.config, "\"model_type\": \"idefics3\"", "\"model_type\": \"llama\"")
  with_model(g, plain, "") do |dir| S.expect_error(Qwen3.open_idefics3(dir, S.REF), "idefics3") end
end
