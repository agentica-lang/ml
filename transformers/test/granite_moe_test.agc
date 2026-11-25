## A tiny random Granite MoE (`granite_moe_golden.py`, `fixtures/granite_moe_golden.json`): its weights, written here as a
## safetensors file under the names of an official checkpoint, with its `config.json`, are opened by `GraniteMoe.open`,
## and the logits of every position match what Hugging Face's `GraniteMoeForCausalLM` gave, on the reference and on ggml
## (CPU). Run from `transformers/`.

import Write from "pkg/hub/safetensors_write"
import GraniteMoe from "../granite_moe"
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

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/granite_moe_golden.json"), Golden)

## The model folder of the golden with `config` in place of its `config.json`, and its weights as `model.safetensors`.
def with_model(g: Golden, config: String, body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-granite-moe-#{Random.int(0, 999999999)}"
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
    m = try GraniteMoe.open(dir, backend)
    out = try GraniteMoe.run(m, g.rows)
    try Test.eq(Tensor.shape(out), [2, 7, 16])
    try S.expect_close(try S.floats_of(out), g.logits, abs, rel, "logits")
    GraniteMoe.close_model(m)
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
    m = try GraniteMoe.open(dir, S.REF)
    last = try GraniteMoe.run(m, g.rows, true)
    try Test.eq(Tensor.shape(last), [2, 1, 16])
    want = g.logits.drop(6 * 16).take(16) + g.logits.drop(13 * 16).take(16)
    S.expect_close(try S.floats_of(last), want, ABS, REL, "last row")
  end
end

## What the loader would read and ignore, or run another way: each is an error and not a quiet difference.
def test_a_config_that_the_model_does_not_run_is_an_error() -> Result[Void]
  g = try golden()
  for change in [
    ["\"model_type\": \"granitemoe\"", "\"model_type\": \"granite\""],
    ["\"hidden_act\": \"silu\"", "\"hidden_act\": \"gelu\""],
    ["\"num_experts_per_tok\": 2", "\"num_experts_per_tok\": 7"],
    ["\"attention_bias\": false", "\"attention_bias\": true"],
    ["\"rope_scaling\": null", "\"rope_scaling\": {\"rope_type\": \"linear\", \"factor\": 2.0}"],
  ]
    wrong = try S.replaced(g.config, change[0], change[1])
    try with_model(g, wrong) do |dir| Test.error?(GraniteMoe.open(dir, S.REF)) end
  end

  ok()
end

def test_only_f32_weights_are_loaded() -> Result[Void]
  g = try golden()
  with_model(g, g.config) do |dir| S.expect_error(GraniteMoe.open(dir, S.REF, Tensor.Dtype.F16), "F32") end
end

## An `lm_head.weight` of twice the embedding table gives twice the logits: with `tie_word_embeddings` false the head is
## the file's.
def test_an_untied_head_is_read_from_the_file() -> Result[Void]
  g = try golden()
  table = "model.embed_tokens.weight"
  untied = g.with(
    weights: g.weights.put("lm_head.weight", g.weights[table].map { |x| 2.0 * x }),
    shapes: g.shapes.put("lm_head.weight", g.shapes[table]),
  )

  config = try S.replaced(g.config, "\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false")
  with_model(untied, config) do |dir|
    m = try GraniteMoe.open(dir, S.REF)
    out = try GraniteMoe.run(m, g.rows)
    S.expect_close(try S.floats_of(out), g.logits.map { |x| 2.0 * x }, ABS, REL, "untied logits")
  end
end

def test_a_token_outside_the_vocabulary_is_an_error() -> Result[Void]
  g = try golden()
  with_model(g, g.config) do |dir|
    m = try GraniteMoe.open(dir, S.REF)
    try Test.error?(GraniteMoe.run(m, [[16]]))
    try Test.error?(GraniteMoe.run(m, [[1, 2], [3]]))
    Test.error?(GraniteMoe.run(m, []))
  end
end
