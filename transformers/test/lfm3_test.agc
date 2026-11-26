## A tiny random LFM3 (`lfm3_golden.py`, `fixtures/lfm3_golden.json`): its weights, written here as a safetensors file under
## the names of an official checkpoint, with its `config.json`, are opened by `Lfm3.open`, and the logits of every position
## match what the official code gave (a short convolution of 7 taps, a sliding window of 4 over rows of 9 tokens, the
## zero-tail rotary of the global layer, the sandwich norms), on the reference and on ggml (CPU). Run from
## `transformers/`.

import Write from "pkg/hub/safetensors_write"
import Lfm3 from "../lfm3"
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

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/lfm3_golden.json"), Golden)

## The model folder of the golden with `config` in place of its `config.json`, and its weights as `model.safetensors`.
def with_model(g: Golden, config: String, body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-lfm3-#{Random.int(0, 999999999)}"
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

def matches(g: Golden, backend: Tensor.Backend, weights: Tensor.Dtype, abs: Float, rel: Float) -> Result[Void]
  with_model(g, g.config) do |dir|
    m = try Lfm3.open(dir, backend, weights)
    out = try Lfm3.run(m, g.rows)
    try Test.eq(Tensor.shape(out), [2, 9, 32])
    try S.expect_close(try S.floats_of(out), g.logits, abs, rel, "logits")
    Lfm3.close_model(m)
    ok()
  end
end

def test_every_position_matches_the_official_code_on_the_reference() -> Result[Void]
  matches(try golden(), S.REF, Tensor.Dtype.F32, ABS, REL)
end

## The real model's `F16` weights are checked by `tools/lfm3_check.agc`: a matrix under 100,000 values stays `F32`, so
## the tiny model would not show them.
def test_every_position_matches_the_official_code_on_ggml() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends()
    try matches(g, backend, Tensor.Dtype.F32, F32_ABS, F32_REL)
  end

  ok()
end

def test_the_last_position_alone_is_the_last_row_of_the_logits() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends()
    try with_model(g, g.config) do |dir|
      m = try Lfm3.open(dir, backend)
      last = try Lfm3.run(m, g.rows, true)
      try Test.eq(Tensor.shape(last), [2, 1, 32])
      want = g.logits.drop(8 * 32).take(32) + g.logits.drop(17 * 32).take(32)
      try S.expect_close(try S.floats_of(last), want, F32_ABS, F32_REL, "last row")
      Lfm3.close_model(m)
      ok()
    end
  end

  ok()
end

## What the loader does not run is an error that names it, never a quiet difference.
def test_a_model_this_file_does_not_run_is_an_error() -> Result[Void]
  g = try golden()
  for change in [
    ["\"model_type\": \"lfm3\"", "\"model_type\": \"lfm2\"", "not lfm3"],
    ["\"conv_bias\": false", "\"conv_bias\": true", "conv_bias"],
    ["\"use_sandwich_norm\": true", "\"use_sandwich_norm\": false", "use_sandwich_norm"],
    ["\"sliding_attention\"]", "\"mamba\"]", "layer type mamba"],
    ["\"sliding_window\": 4", "\"sliding_window\": null", "sliding_window"],
    ["\"rope_theta\": 10000.0", "\"rope_theta\": 10000.0, \"rope_type\": \"yarn\"", "rope_type"],
    ["\"partial_rotary_factor\": 0.5", "\"partial_rotary_factor\": 0.1", "rotary width"],
  ]
    wrong = try S.replaced(g.config, change[0], change[1])
    try with_model(g, wrong) do |dir| S.expect_error(Lfm3.open(dir, S.REF), change[2]) end
  end

  ok()
end

## The `weights` the loader does not keep are an error.
def test_the_weights_option_is_checked() -> Result[Void]
  g = try golden()
  try with_model(g, g.config) do |dir| S.expect_error(Lfm3.open(dir, S.REF, Tensor.Dtype.Q8_0), "F32 or F16") end
  with_model(g, g.config) do |dir| S.expect_error(Lfm3.open(dir, S.REF, Tensor.Dtype.F16), "ggml") end
end

## An `lm_head.weight` of twice the embedding table gives twice the logits when the config says untied.
def test_an_untied_head_is_read_from_the_file() -> Result[Void]
  g = try golden()
  table = "model.embed_tokens.weight"
  untied = g.with(
    weights: g.weights.put("lm_head.weight", g.weights[table].map { |x| 2.0 * x }),
    shapes: g.shapes.put("lm_head.weight", g.shapes[table]),
  )

  config = try S.replaced(g.config, "\"tie_word_embeddings\": true", "\"tie_word_embeddings\": false")
  for backend in try S.ggml_backends()
    try with_model(untied, config) do |dir|
      m = try Lfm3.open(dir, backend)
      out = try Lfm3.run(m, g.rows)
      try S.expect_close(try S.floats_of(out), g.logits.map { |x| 2.0 * x }, F32_ABS, F32_REL, "untied logits")
      Lfm3.close_model(m)
      ok()
    end
  end

  ok()
end

def test_a_token_outside_the_vocabulary_is_an_error() -> Result[Void]
  g = try golden()
  with_model(g, g.config) do |dir|
    m = try Lfm3.open(dir, S.REF)
    try Test.error?(Lfm3.run(m, [[32]]))
    try Test.error?(Lfm3.run(m, [[1, 2], [3]]))
    try Test.error?(Lfm3.run(m, [(0...65).map { |_| 1 }]))
    Test.error?(Lfm3.run(m, []))
  end
end
