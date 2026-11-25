## A tiny random Qwen3.5 (`qwen35_golden.py`, `fixtures/qwen35_golden.json`): its weights, written here as a safetensors
## file under the names of an official checkpoint, with its `config.json`, are opened by `Qwen3_5.open`, and the logits of
## every position match what Hugging Face's `Qwen3_5ForCausalLM` gave, on the reference and on ggml (CPU). Run from
## `transformers/`.

import Write from "pkg/hub/safetensors_write"
import Qwen3_5 from "../qwen3_5"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The golden is in `float32`, the reference in doubles.
ABS = 0.0001
REL = 0.0001

## ggml computes in `F32` and sums in another order.
F32_ABS = 0.001
F32_REL = 0.001

struct Golden
  config: String
  weights: Map[String, List[Float]]
  shapes: Map[String, List[Int]]
  rows: List[List[Int]]
  logits: List[Float]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/qwen35_golden.json"), Golden)

## The model folder of the golden with `config` in place of its `config.json`, and its weights as `model.safetensors`.
def with_model(g: Golden, config: String, body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-qwen35-#{Random.int(0, 999999999)}"
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
    m = try Qwen3_5.open(dir, backend)
    out = try Qwen3_5.run(m, g.rows)
    try Test.eq(Tensor.shape(out), [2, 7, 16])
    try S.expect_close(try S.floats_of(out), g.logits, abs, rel, "logits")
    Qwen3_5.close_model(m)
    ok()
  end
end

def test_every_position_matches_hugging_face_on_the_reference() -> Result[Void]
  matches(try golden(), S.REF, ABS, REL)
end

def test_every_position_matches_hugging_face_on_ggml() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends()
    try matches(g, backend, F32_ABS, F32_REL)
  end

  ok()
end

def test_the_last_position_alone_is_the_last_row_of_the_logits() -> Result[Void]
  g = try golden()
  with_model(g, g.config) do |dir|
    m = try Qwen3_5.open(dir, S.REF)
    last = try Qwen3_5.run(m, g.rows, true)
    try Test.eq(Tensor.shape(last), [2, 1, 16])
    want = g.logits.drop(6 * 16).take(16) + g.logits.drop(13 * 16).take(16)
    S.expect_close(try S.floats_of(last), want, ABS, REL, "last row")
  end
end

def test_a_config_the_model_does_not_run_is_an_error() -> Result[Void]
  g = try golden()
  for change in [
    ["\"model_type\": \"qwen3_5\"", "\"model_type\": \"qwen3\""],
    ["\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\""],
    ["\"attn_output_gate\": true", "\"attn_output_gate\": false"],
    [
      "\"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\"",
      "\"linear_attention\", \"full_attention\"",
    ],
    ["\"tie_word_embeddings\": true}", "\"tie_word_embeddings\": false}"],
  ]
    wrong = try S.replaced(g.config, change[0], change[1])
    try with_model(g, wrong) do |dir|
      try Test.error?(Qwen3_5.open(dir, S.REF))
      ok()
    end
  end

  ok()
end

def test_a_token_outside_the_vocabulary_is_an_error() -> Result[Void]
  g = try golden()
  with_model(g, g.config) do |dir|
    m = try Qwen3_5.open(dir, S.REF)
    try Test.error?(Qwen3_5.run(m, [[16]]))
    try Test.error?(Qwen3_5.run(m, [[1, 2], [3]]))
    Test.error?(Qwen3_5.run(m, []))
  end
end
