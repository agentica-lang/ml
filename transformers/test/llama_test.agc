## A tiny random Llama (`llama_golden.py`, `fixtures/llama_golden.json`, the architecture of SmolLM2): its weights,
## written here as a safetensors file under the names of an official checkpoint with its `config.json`, are opened by
## `Qwen3.open`, and the logits of every position of two rows match what Hugging Face's `LlamaForCausalLM` gave, on the
## reference and on ggml (CPU). Run from `transformers/`.

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

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/llama_golden.json"), Golden)

## The model folder of the golden with `config` in place of its `config.json`.
def with_model(g: Golden, config: String, body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-llama-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = populated(g, config, dir, body)
  _ = Path.remove(dir)
  result
end

def populated(g: Golden, config: String, dir: String, body: Fn[String, Result[Void]]) -> Result[Void]
  tensors: Map[String, Tensor.Tensor] = {}
  for name in g.weights.keys
    tensors = tensors.put(name, try S.f32(g.weights[name], g.shapes[name]))
  end

  try Write.write("#{dir}/model.safetensors", tensors)
  try File.write_text("#{dir}/config.json", config)
  body.call(dir)
end

def matches(g: Golden, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  with_model(g, g.config) do |dir|
    m = try Qwen3.open(dir, backend)
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

def test_a_llama_the_model_cannot_run_is_an_error_that_names_why() -> Result[Void]
  g = try golden()
  for change in [
    ["\"mlp_bias\": false", "\"mlp_bias\": true", "mlp_bias"],
    ["\"attention_bias\": false", "\"attention_bias\": true", "attention_bias"],
    ["\"rope_scaling\": null", "\"rope_scaling\": {\"rope_type\": \"linear\"}", "rope_scaling"],
    ["\"model_type\": \"llama\"", "\"model_type\": \"mistral\"", "model_type"],
  ]
    wrong = try S.replaced(g.config, change[0], change[1])
    try with_model(g, wrong) do |dir| S.expect_error(Qwen3.open(dir, S.REF), change[2]) end
  end

  ok()
end
