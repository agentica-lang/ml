## A tiny random Qwen3 that the Python peer `qwen_peer.py` writes with Hugging
## Face's names, in `bfloat16` with tied embeddings and in `float32` with its
## own `lm_head`: the logits of every position match the peer's forward in
## doubles, on the reference and on ggml. Run from `transformers/`.

import Qwen3 from "../qwen3"
import Qwen3Gguf from "../qwen3_gguf"
import Bpe from "../tokenizer/bpe"
import Lift from "../lift"
import Gguf from "pkg/hub/gguf"
import Hub from "pkg/hub/hub"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The reference computes in doubles, as the peer does.
ABS = 0.000000001
REL = 0.000000001

## ggml computes in `F32`.
F32_ABS = 0.0001
F32_REL = 0.0001

## The Q8_0 weights keep the blocks, and ggml rounds every activation to Q8_0 blocks (to F16 too on Vulkan): the logits
## (up to 5) are within this of the dequantized model's. It is a measured margin, not a derived bound: the largest gap
## on this model is 0.08 on the CPU, and a weight read in the wrong layout is off by whole units.
Q8_ABS = 0.1
Q8_REL = 0.0

## The logits (about 5 at most) of the half model are within a few 1e-3 of the F32 model's: every weight rounds once, by 2^-11.
HALF_ABS = 0.01
HALF_REL = 0.002

def test_every_position_matches_the_peer_on_the_reference() -> Result[Void]
  with_peer do |dir|
    try matches(dir, "tiny", S.REF, ABS, REL)
    matches(dir, "untied", S.REF, ABS, REL)
  end
end

## The Qwen2 layout: no norms on the query and the key, a bias on the query, key and value, and no `head_dim`.
def test_a_qwen2_model_matches_the_peer() -> Result[Void]
  with_peer do |dir|
    try matches(dir, "qwen2", S.REF, ABS, REL)
    for backend in try S.ggml_backends()
      try matches(dir, "qwen2", backend, F32_ABS, F32_REL)
    end

    ok()
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

## The same models as GGUF files (llama.cpp's names; the tied one with `output.weight` as llama.cpp writes it, and
## without) give the peer's logits, from the key-values alone: no `config.json`.
def test_a_gguf_gives_the_same_logits_as_the_safetensors() -> Result[Void]
  with_peer do |dir|
    try matches_path(dir, "tiny.gguf", "tiny", S.REF, ABS, REL)
    try matches_path(dir, "tied.gguf", "tiny", S.REF, ABS, REL)
    try matches_path(dir, "untied.gguf", "untied", S.REF, ABS, REL)
    for backend in try S.ggml_backends()
      try matches_path(dir, "tiny.gguf", "tiny", backend, F32_ABS, F32_REL)
      try matches_path(dir, "tied.gguf", "tiny", backend, F32_ABS, F32_REL)
    end

    ok()
  end
end

## An F16 GGUF read with `weights: F16` keeps its big matrices half, which an F32 read widens; both are the one file.
def test_an_f16_gguf_keeps_half_weights() -> Result[Void]
  with_peer do |dir|
    rows = [[1, 5, 2, 7, 3, 12], [399, 200, 9, 0, 6, 11]]
    for backend in try S.ggml_backends()
      half = try Qwen3.open("#{dir}/half.gguf", backend, Tensor.Dtype.F16)
      whole = try Qwen3.open("#{dir}/half.gguf", backend)
      try Test.eq(Tensor.dtype(half.tokens.table), Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(half.stack.blocks[0].mlp.up.w), Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(whole.tokens.table), Tensor.Dtype.F32)
      try S.expect_close(
        try S.floats_of(try Qwen3.run(half, rows)),
        try S.floats_of(try Qwen3.run(whole, rows)),
        F32_ABS,
        F32_REL,
        "half against widened",
      )
    end

    ok()
  end
end

## A GGUF whose matrices are Q8_0 blocks (the peer's `q8.gguf`, width 32) dequantizes at load into the peer's model: the
## logits match its forward in doubles, on the reference and on ggml.
def test_a_q8_0_gguf_gives_the_peers_logits() -> Result[Void]
  with_peer do |dir|
    try matches_q8(dir, S.REF, Tensor.Dtype.F32, ABS, REL)
    for backend in try S.ggml_backends()
      try matches_q8(dir, backend, Tensor.Dtype.F32, F32_ABS, F32_REL)
    end

    ok()
  end
end

## With `weights: Q8_0` the matrices (and the table, the tied head) stay Q8_0 blocks on ggml, the norms stay F32, and the
## logits stay within the bound of ggml's rounding of the activations to Q8_0 blocks in every product.
def test_q8_0_weights_keep_the_blocks_and_the_logits_close() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      m = try Qwen3.open("#{dir}/q8.gguf", backend, Tensor.Dtype.Q8_0)
      block = m.stack.blocks[0]
      try Test.eq(Tensor.dtype(m.tokens.table), Tensor.Dtype.Q8_0)
      try Test.eq(Tensor.dtype(m.head.w), Tensor.Dtype.Q8_0)
      try Test.eq(Tensor.dtype(block.attention.query.w), Tensor.Dtype.Q8_0)
      try Test.eq(Tensor.dtype(block.mlp.down.w), Tensor.Dtype.Q8_0)
      try matches_q8(dir, backend, Tensor.Dtype.Q8_0, Q8_ABS, Q8_REL)
    end

    try S.expect_error(Qwen3.open("#{dir}/q8.gguf", S.REF, Tensor.Dtype.Q8_0), "needs the ggml backend")
    ok()
  end
end

## What a GGUF cannot say is an error naming it.
def test_a_gguf_of_another_architecture_or_with_rope_scaling_is_refused() -> Result[Void]
  with_peer do |dir|
    try S.expect_error(Qwen3.open("#{dir}/llama.gguf", S.REF), "llama")
    S.expect_error(Qwen3.open("#{dir}/scaled.gguf", S.REF), "rope_scaling")
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
      try S.replaced(config, "\"qwen3\"", "\"llama\""),
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

## The model folders of the peer written as GGUF by `Qwen3Gguf.convert` read back as the same model: `F32` and `BF16`
## give the peer's logits (the weights are `bfloat16` or `float32` already), `F16` the logits of the weights rounded to
## binary16 that the peer computes with `struct`. A tied model writes no `output.weight`, an untied one does.
def test_a_converted_gguf_gives_the_peers_logits() -> Result[Void]
  with_peer do |dir|
    # the untied model holds `float32` values, which `BF16` would round
    for format in ["F32", "BF16", "F16"]
      for folder in ["tiny", "untied"]
        out = "#{dir}/#{folder}_#{format}.gguf"
        if not (folder == "untied" and format == "BF16")
          try Qwen3Gguf.convert("#{dir}/#{folder}", out, try format_of(format))
          key = folder
          if format == "F16"
            key = "#{folder}_f16"
          end

          try matches_path(dir, "#{folder}_#{format}.gguf", key, S.REF, ABS, REL)
          for backend in try S.ggml_backends()
            try matches_path(dir, "#{folder}_#{format}.gguf", key, backend, F32_ABS, F32_REL)
          end
        end
      end
    end

    tied = try Hub.open("#{dir}/tiny_F16.gguf")
    try Test.eq(Hub.has?(tied, "lm_head.weight"), false)
    try Test.eq(Hub.has?(try Hub.open("#{dir}/untied_F16.gguf"), "lm_head.weight"), true)
    # a matrix takes the format, a norm stays F32, as in a file of llama.cpp
    shard = tied.shards["tiny_F16.gguf"]
    try Test.eq(shard.entries["model.layers.0.self_attn.q_proj.weight"].dtype, "F16")
    try Test.eq(shard.entries["model.layers.0.input_layernorm.weight"].dtype, "F32")
    try Test.eq(shard.entries["model.embed_tokens.weight"].dtype, "F16")
    ok()
  end
end

## The tokenizer of the written file is the one of `tokenizer.json` (the tables are equal, and so are the ids of a text
## with added tokens), the ids of the config are written, an id with no token is padding (type 5), and the chat
## template is kept. An added token written `<|...|>` is a control token though `tokenizer.json` does not say special.
def test_a_converted_gguf_carries_the_tokenizer_and_the_ids() -> Result[Void]
  with_peer do |dir|
    out = "#{dir}/tokens.gguf"
    try Qwen3Gguf.convert("#{dir}/tiny", out, Tensor.FloatFormat.F16)
    json = try Bpe.open("#{dir}/tiny/tokenizer.json")
    back = try Bpe.open_gguf(out)
    try Test.eq(back.vocab, json.vocab)
    try Test.eq(back.pieces, json.pieces)
    try Test.eq(back.ranks, json.ranks)
    try Test.eq(back.nfc, json.nfc)
    try Test.eq(back.added.map { |a| a.id }, [9, 10, 11])
    try Test.eq(back.added.map { |a| a.content }, json.added.map { |a| a.content })
    try Test.eq(back.added.map { |a| a.special }, [true, true, false])
    try Test.eq(json.added.map { |a| a.special }, [true, false, false])
    text = "abc ab<think>de <|endoftext|>a"
    try Test.eq(try Bpe.encode(back, text), try Bpe.encode(json, text))
    h = (try Hub.open(out)).gguf.unwrap!
    try Test.eq(try Gguf.ints(h, "tokenizer.ggml.token_type"), [1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 4, 5])
    try Test.eq((try Gguf.strings(h, "tokenizer.ggml.tokens"))[12], "[PAD12]")
    try Test.eq(try Gguf.strings(h, "tokenizer.ggml.merges"), ["a b", "Ġ a", "ab c"])
    try Test.eq(try Gguf.int(h, "tokenizer.ggml.bos_token_id"), 9)
    try Test.eq(try Gguf.int(h, "tokenizer.ggml.eos_token_id"), 10)
    try Test.eq(try Gguf.int(h, "tokenizer.ggml.padding_token_id"), 9)
    try Test.eq(try Gguf.string(h, "tokenizer.ggml.model"), "gpt2")
    try Test.eq(try Gguf.string(h, "tokenizer.ggml.pre"), "qwen2")
    try Test.eq(
      try Gguf.string(h, "tokenizer.chat_template"),
      "{% for m in messages %}<|im_start|>{{ m.content }}{% endfor %}",
    )

    try Test.eq(try Gguf.int(h, "general.file_type"), 1)
    try Test.eq(try Gguf.string(h, "general.architecture"), "qwen3")
    try Test.eq(try Gguf.int(h, "qwen3.block_count"), 2)
    ok()
  end
end

def test_a_model_that_cannot_be_written_is_refused() -> Result[Void]
  with_peer do |dir|
    out = "#{dir}/refused.gguf"
    try S.expect_error(
      Qwen3Gguf.convert("#{dir}/missing", out, Tensor.FloatFormat.F32),
      "model.layers.1.mlp.down_proj.weight",
    )

    try Test.eq(Path.real(out).ok?, false)
    try Test.eq(Path.real(out + ".part").ok?, false)
    S.expect_error(Qwen3Gguf.convert("#{dir}/half", out, Tensor.FloatFormat.F32), "tokenizer.json")
  end
end

## The chat template is the one of `tokenizer_config.json`, else `chat_template.jinja`, and `add_bos_token` and
## `add_eos_token` are the file's booleans, else false.
def test_the_template_and_the_bos_and_eos_flags_come_from_the_tokenizer_config_or_its_file() -> Result[Void]
  with_peer do |dir|
    try File.write_text("#{dir}/tiny/chat_template.jinja", "FROM THE FILE")
    out = "#{dir}/template.gguf"
    try Qwen3Gguf.convert("#{dir}/tiny", out, Tensor.FloatFormat.F32)
    h = (try Hub.open(out)).gguf.unwrap!
    try Test.eq(
      try Gguf.string(h, "tokenizer.chat_template"),
      "{% for m in messages %}<|im_start|>{{ m.content }}{% endfor %}",
    )

    try Test.eq(try Gguf.bool(h, "tokenizer.ggml.add_bos_token"), false)
    try File.write_text(
      "#{dir}/tiny/tokenizer_config.json",
      "{\"pad_token\": \"<|endoftext|>\", \"add_bos_token\": true, \"add_eos_token\": false}",
    )

    try Qwen3Gguf.convert("#{dir}/tiny", out, Tensor.FloatFormat.F32)
    h = (try Hub.open(out)).gguf.unwrap!
    try Test.eq(try Gguf.string(h, "tokenizer.chat_template"), "FROM THE FILE")
    try Test.eq(try Gguf.bool(h, "tokenizer.ggml.add_bos_token"), true)
    Test.eq(try Gguf.bool(h, "tokenizer.ggml.add_eos_token"), false)
  end
end

## An added token whose id is also in the base vocabulary is written with the type of an added token, not NORMAL: here
## `<think>` takes id 7, where the base vocabulary has `d`.
def test_an_added_token_over_a_base_token_keeps_its_added_type() -> Result[Void]
  with_peer do |dir|
    tokens = "#{dir}/tiny/tokenizer.json"
    try File.write_text(tokens, (try File.read_text(tokens)).replace("\"id\": 11", "\"id\": 7"))
    out = "#{dir}/over.gguf"
    try Qwen3Gguf.convert("#{dir}/tiny", out, Tensor.FloatFormat.F32)
    h = (try Hub.open(out)).gguf.unwrap!
    try Test.eq(try Gguf.ints(h, "tokenizer.ggml.token_type"), [1, 1, 1, 1, 1, 1, 1, 4, 1, 3, 3, 5, 5])
    Test.eq((try Gguf.strings(h, "tokenizer.ggml.tokens"))[7], "<think>")
  end
end

## A config that does not fit the weights (a `vocab_size` too large, or no `head_dim`, which is then the hidden size over
## the heads), and merges whose ranks are not 0 to n-1, are errors.
def test_a_config_or_tokenizer_that_does_not_fit_the_weights_is_refused() -> Result[Void]
  with_peer do |dir|
    out = "#{dir}/fit.gguf"
    config = "#{dir}/tiny/config.json"
    text = try File.read_text(config)
    try File.write_text(config, text.replace("\"vocab_size\": 13", "\"vocab_size\": 14"))
    try S.expect_error(Qwen3Gguf.convert("#{dir}/tiny", out, Tensor.FloatFormat.F32), "needs [14,")
    try Test.eq(Path.real(out).ok?, false)
    try File.write_text(config, text.replace("\"head_dim\": 4,", ""))
    try S.expect_error(Qwen3Gguf.convert("#{dir}/tiny", out, Tensor.FloatFormat.F32), "needs [8, 8]")
    try File.write_text(config, text)
    tokens = "#{dir}/tiny/tokenizer.json"
    merges = try File.read_text(tokens)
    try File.write_text(tokens, merges.replace("[\"ab\", \"c\"]", "[\"a\", \"b\"]"))
    S.expect_error(Qwen3Gguf.convert("#{dir}/tiny", "#{dir}/ranks.gguf", Tensor.FloatFormat.F32), "ranks")
  end
end

private

def format_of(name: String) -> Result[Tensor.FloatFormat]
  case name
    "F32" -> ok(Tensor.FloatFormat.F32)
    "F16" -> ok(Tensor.FloatFormat.F16)
    "BF16" -> ok(Tensor.FloatFormat.BF16)
    _ -> error("no format #{name}")
  end
end

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/qwen_peer.py", "agentica-qwen-", body)
end

def expected(dir: String) -> Result[JSON.Value] = JSON.parse(try File.read_text("#{dir}/expected.json"))

def ids(dir: String) -> Result[List[List[Int]]] = ids_of(dir, "ids")

## The batch of the peer's `key`.
def ids_of(dir: String, key: String) -> Result[List[List[Int]]]
  out: List[List[Int]] = []
  for row in try JSON.list(try JSON.get(try expected(dir), key))
    ints: List[Int] = []
    for v in try JSON.list(row)
      ints = ints.push(try JSON.int(v))
    end

    out = out.push(ints)
  end

  ok(out)
end

## The Q8_0 model of the peer on `backend`, loaded with `weights`, against the logits of its dequantized weights.
def matches_q8(dir: String, backend: Tensor.Backend, weights: Tensor.Dtype, abs: Float, rel: Float) -> Result[Void]
  m = try Qwen3.open("#{dir}/q8.gguf", backend, weights)
  rows = try ids_of(dir, "ids_q8")
  got = try Qwen3.run(m, rows)
  try Test.eq(Tensor.shape(got), [2, 6, 20])
  S.expect_close(try S.floats_of(got), try S.flat(try JSON.get(try expected(dir), "q8")), abs, rel, "q8.gguf")
end

## The model file `file` on `backend` against the peer's logits of `key`.
def matches_path(
  dir: String,
  file: String,
  key: String,
  backend: Tensor.Backend,
  abs: Float,
  rel: Float,
) -> Result[Void]
  m = try Qwen3.open("#{dir}/#{file}", backend)
  got = try Qwen3.run(m, try ids(dir))
  try Test.eq(Tensor.shape(got), [2, 6, 13])
  S.expect_close(try S.floats_of(got), try S.flat(try JSON.get(try expected(dir), key)), abs, rel, file)
end

## The model `folder` on `backend` against the peer's logits of the batch.
def matches(dir: String, folder: String, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  m = try Qwen3.open("#{dir}/#{folder}", backend)
  got = try Qwen3.run(m, try ids(dir))
  try Test.eq(Tensor.shape(got), [2, 6, 13])
  S.expect_close(try S.floats_of(got), try S.flat(try JSON.get(try expected(dir), folder)), abs, rel, folder)
end
