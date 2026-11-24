## A tiny random Qwen3 that the Python peer `qwen_peer.py` writes with Hugging
## Face's names, in `bfloat16` with tied embeddings and in `float32` with its
## own `lm_head`: the logits of every position match the peer's forward in
## doubles, on the reference and on ggml. Run from `transformers/`.

import Qwen3 from "../qwen3"
import Lift from "../lift"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The reference computes in doubles, as the peer does.
ABS = 0.000000001
REL = 0.000000001

## ggml computes in `F32`.
F32_ABS = 0.0001
F32_REL = 0.0001

## The logits (about 5 at most) of the half model are within a few 1e-3 of the F32 model's: every weight rounds once, by 2^-11.
HALF_ABS = 0.01
HALF_REL = 0.002

def test_every_position_matches_the_peer_on_the_reference() -> Result[Void]
  with_peer do |dir|
    try matches(dir, "tiny", S.REF, ABS, REL)
    matches(dir, "untied", S.REF, ABS, REL)
  end
end

def test_every_position_matches_the_peer_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      try matches(dir, "tiny", backend, F32_ABS, F32_REL)
      try matches(dir, "untied", backend, F32_ABS, F32_REL)
    end

    ok()
  end
end

def test_the_last_position_alone_is_the_last_row_of_the_logits() -> Result[Void]
  with_peer do |dir|
    m = try Qwen3.open("#{dir}/tiny", S.REF)
    rows = try ids(dir)
    all = try S.floats_of(try Qwen3.run(m, rows))
    last = try Qwen3.run(m, rows, true)
    try Test.eq(Tensor.shape(last), [2, 1, 13])
    got = try S.floats_of(last)
    want = all.drop(5 * 13).take(13) + all.drop(11 * 13).take(13)
    S.expect_close(got, want, ABS, REL, "last row")
  end
end

## `weights: F16` is an error off ggml, and on ggml a model whose matrices are all under `Load.HALF_MIN_ELEMENTS` is the
## same model (a tiny one: the half path itself is `hub/test/load_test.agc` and the checkpoints of `tools/`).
def test_f16_weights_need_ggml_and_leave_a_tiny_model_alone() -> Result[Void]
  with_peer do |dir|
    refused = Qwen3.open("#{dir}/tiny", S.REF, Tensor.Dtype.F16)
    try Test.error?(refused)
    for backend in try S.ggml_backends()
      half = try Qwen3.open("#{dir}/tiny", backend, Tensor.Dtype.F16)
      whole = try Qwen3.open("#{dir}/tiny", backend)
      try Test.eq(
        try S.floats_of(try Qwen3.run(half, try ids(dir))),
        try S.floats_of(try Qwen3.run(whole, try ids(dir))),
      )
    end

    ok()
  end
end

## The half path through a model: with `weights: F16` the table (which is also the tied head) and the big MLP
## matrices are half, the small ones stay `F32`, and the logits stay within the half-weight bound of the `F32` model's.
def test_f16_weights_keep_the_big_matrices_half_and_the_logits_close() -> Result[Void]
  with_peer do |dir|
    rows = [[1, 5, 2, 7, 3, 12], [399, 200, 9, 0, 6, 11]]
    for backend in try S.ggml_backends()
      half = try Qwen3.open("#{dir}/half", backend, Tensor.Dtype.F16)
      whole = try Qwen3.open("#{dir}/half", backend)
      block = half.stack.blocks[0]
      try Test.eq(Tensor.dtype(half.tokens.table), Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(half.head.w), Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(block.mlp.up.w), Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(block.attention.query.w), Tensor.Dtype.F32)
      try Test.eq(Tensor.dtype(whole.tokens.table), Tensor.Dtype.F32)
      out = try Qwen3.run(half, rows)
      try Test.eq(Tensor.shape(out), [2, 6, 400])
      got = try S.floats_of(out)
      want = try S.floats_of(try Qwen3.run(whole, rows))
      try Test.assert(got != want, "the half model computed the F32 model's logits exactly")
      try S.expect_close(got, want, HALF_ABS, HALF_REL, "half against F32")
    end

    ok()
  end
end

## `close_model` frees what ggml holds; a model that was closed is an error.
def test_a_closed_model_is_an_error_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      m = try Qwen3.open("#{dir}/tiny", backend)
      out = try Qwen3.run(m, try ids(dir))
      try Test.eq(Tensor.shape(out), [2, 6, 13])
      Lift.close([out])
      Qwen3.close_model(m)
      try Test.error?(Qwen3.run(m, try ids(dir)))
    end

    ok()
  end
end

def test_a_model_that_is_not_qwen3_is_refused() -> Result[Void]
  with_peer do |dir|
    config = try File.read_text("#{dir}/tiny/config.json")
    for wrong in [
      try S.replaced(config, "\"qwen3\"", "\"qwen2\""),
      try S.replaced(config, "\"silu\"", "\"gelu\""),
      try S.replaced(config, "\"attention_bias\": false", "\"attention_bias\": true"),
      try S.replaced(config, "\"use_sliding_window\": false", "\"use_sliding_window\": true"),
      try S.replaced(config, "\"rope_scaling\": null", "\"rope_scaling\": {\"rope_type\": \"yarn\"}"),
      try S.replaced(config, "\"num_key_value_heads\": 2", "\"num_key_value_heads\": 3"),
      try S.replaced(config, "\"head_dim\": 4", "\"head_dim\": 5"),
      try S.replaced(config, "\"head_dim\": 4", "\"head_dim\": 6"),
      try S.replaced(config, "\"hidden_size\": 8", "\"hidden_size\": 9"),
      try S.replaced(config, "\"vocab_size\": 13", "\"vocab_size\": 14"),
    ]
      try File.write_text("#{dir}/tiny/config.json", wrong)
      try Test.error?(Qwen3.open("#{dir}/tiny", S.REF))
    end

    try Test.error?(Qwen3.open("#{dir}/missing", S.REF))
    Test.error?(Qwen3.open("#{dir}/absent", S.REF))
  end
end

## Hugging Face's defaults: the embeddings are tied when the config does not say, and a
## `sliding_window` size does nothing while `use_sliding_window` is false.
def test_the_defaults_and_an_unused_sliding_window_are_read_as_hugging_face_does() -> Result[Void]
  with_peer do |dir|
    config = try File.read_text("#{dir}/tiny/config.json")
    try File.write_text("#{dir}/tiny/config.json", try S.replaced(config, "\"tie_word_embeddings\": true, ", ""))
    try matches(dir, "tiny", S.REF, ABS, REL)
    try File.write_text(
      "#{dir}/tiny/config.json",
      try S.replaced(config, "\"sliding_window\": null", "\"sliding_window\": 4096"),
    )

    matches(dir, "tiny", S.REF, ABS, REL)
  end
end

def test_bad_inputs_are_errors() -> Result[Void]
  with_peer do |dir|
    m = try Qwen3.open("#{dir}/tiny", S.REF)
    try Test.error?(Qwen3.run(m, []))
    try Test.error?(Qwen3.run(m, [[]]))
    try Test.error?(Qwen3.run(m, [[1, 2], [3]]))
    try Test.error?(Qwen3.run(m, [List.repeat(1, 33)]))
    Test.error?(Qwen3.run(m, [[1, 13]]))
  end
end

private

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/qwen_peer.py", "agentica-qwen-", body)
end

def expected(dir: String) -> Result[JSON.Value] = JSON.parse(try File.read_text("#{dir}/expected.json"))

def ids(dir: String) -> Result[List[List[Int]]]
  out: List[List[Int]] = []
  for row in try JSON.list(try JSON.get(try expected(dir), "ids"))
    ints: List[Int] = []
    for v in try JSON.list(row)
      ints = ints.push(try JSON.int(v))
    end

    out = out.push(ints)
  end

  ok(out)
end

## The model `folder` on `backend` against the peer's logits of the batch.
def matches(dir: String, folder: String, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  m = try Qwen3.open("#{dir}/#{folder}", backend)
  got = try Qwen3.run(m, try ids(dir))
  try Test.eq(Tensor.shape(got), [2, 6, 13])
  S.expect_close(try S.floats_of(got), try S.flat(try JSON.get(try expected(dir), folder)), abs, rel, folder)
end
