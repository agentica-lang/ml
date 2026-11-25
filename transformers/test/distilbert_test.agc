## A tiny random DistilBERT that the Python peer `distilbert_peer.py` writes with Hugging Face's names (and under the
## `distilbert.` prefix of a `DistilBertFor...` checkpoint): the last hidden state, the mean embedding and the
## normalized one of a padded batch match the peer's forward in doubles, on the reference and on ggml. Run from
## `transformers/`.

import DistilBert from "../distilbert"
import Lift from "../lift"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import WordPiece from "../tokenizer/wordpiece"

## The reference computes in doubles, as the peer does.
ABS = 0.000000001
REL = 0.000000001

## ggml computes in `F32`.
F32_ABS = 0.0001
F32_REL = 0.0001

def test_a_padded_batch_matches_the_peer_on_the_reference() -> Result[Void]
  with_peer do |dir|
    try matches(dir, "tiny", S.REF, ABS, REL)
    matches(dir, "prefixed", S.REF, ABS, REL)
  end
end

def test_a_padded_batch_matches_the_peer_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      try matches(dir, "tiny", backend, F32_ABS, F32_REL)
      try matches(dir, "prefixed", backend, F32_ABS, F32_REL)
    end

    ok()
  end
end

## `forward` takes the tensors `run` builds: the same hidden state.
def test_forward_takes_ids_and_a_mask() -> Result[Void]
  with_peer do |dir|
    m = try DistilBert.open("#{dir}/tiny", S.REF)
    ids = try Tensor.from_list(
      [1, 5, 2, 7, 3, 1, 4, 6, 3, 0].map { |i| Tensor.Values.Int(i) },
      [2, 5],
      Tensor.Dtype.I32,
      S.REF,
    )

    mask = try S.f32([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [2, 5])
    hidden = try DistilBert.forward(m, ids, mask)
    want = try S.flat(try JSON.get(try expected(dir), "hidden"))
    try S.expect_close(try S.floats_of(hidden), want, ABS, REL, "hidden")
    Test.error?(DistilBert.forward(m, ids, try S.f32([1.0, 1.0], [1, 2])))
  end
end

def test_a_model_that_is_not_distilbert_is_refused() -> Result[Void]
  with_peer do |dir|
    config = try File.read_text("#{dir}/tiny/config.json")
    for wrong in [
      config.replace("\"distilbert\"", "\"bert\""),
      config.replace("\"gelu\"", "\"relu\""),
      config.replace("\"sinusoidal_pos_embds\": false", "\"sinusoidal_pos_embds\": true"),
      config.replace("\"dim\": 8", "\"dim\": 9"),
      config.replace("\"vocab_size\": 11", "\"vocab_size\": 12"),
      config.replace("\"hidden_dim\": 16", "\"hidden_dim\": 12"),
    ]
      try Test.assert(wrong != config, "an edit changed nothing")
      try File.write_text("#{dir}/tiny/config.json", wrong)
      try Test.error?(DistilBert.open("#{dir}/tiny", S.REF))
    end

    try Test.error?(DistilBert.open("#{dir}/missing", S.REF))
    Test.error?(DistilBert.open("#{dir}/absent", S.REF))
  end
end

def test_bad_inputs_are_errors() -> Result[Void]
  with_peer do |dir|
    m = try DistilBert.open("#{dir}/tiny", S.REF)
    try Test.error?(DistilBert.run(m, []))
    try Test.error?(DistilBert.run(m, [WordPiece.Encoding(ids: List.repeat(1, 9), type_ids: List.repeat(0, 9))]))
    full = WordPiece.Encoding(ids: [1, 2, 3], type_ids: [0, 0, 0])
    try Test.error?(DistilBert.run(m, [full, WordPiece.Encoding(ids: [], type_ids: [])]))
    Test.error?(DistilBert.embed(m, [full, WordPiece.Encoding(ids: [], type_ids: [])]))
  end
end

## `close_model` frees what ggml holds; a model that was closed is an error, not a read of freed memory.
def test_a_closed_model_is_an_error_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      m = try DistilBert.open("#{dir}/tiny", backend)
      batch = try encodings(dir)
      hidden = try DistilBert.run(m, batch)
      try Test.eq(Tensor.shape(hidden), [2, 5, 8])
      Lift.close([hidden])
      DistilBert.close_model(m)
      try Test.error?(DistilBert.run(m, batch))
    end

    ok()
  end
end

private

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/distilbert_peer.py", "agentica-distilbert-", body)
end

def expected(dir: String) -> Result[JSON.Value] = JSON.parse(try File.read_text("#{dir}/expected.json"))

## The batch of `expected.json` as encodings.
def encodings(dir: String) -> Result[List[WordPiece.Encoding]]
  out: List[WordPiece.Encoding] = []
  for row in try JSON.list(try JSON.get(try expected(dir), "ids"))
    ids: List[Int] = []
    for v in try JSON.list(row)
      ids = ids.push(try JSON.int(v))
    end

    out = out.push(WordPiece.Encoding(ids: ids, type_ids: List.repeat(0, ids.size)))
  end

  ok(out)
end

## The model `folder` on `backend` against the peer's hidden state, mean embedding and normalized embedding.
def matches(dir: String, folder: String, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  want = try expected(dir)
  m = try DistilBert.open("#{dir}/#{folder}", backend)
  batch = try encodings(dir)
  hidden = try DistilBert.run(m, batch)
  try Test.eq(Tensor.shape(hidden), [2, 5, 8])
  try S.expect_close(try S.floats_of(hidden), try S.flat(try JSON.get(want, "hidden")), abs, rel, "#{folder} hidden")
  mean = try DistilBert.embed(m, batch)
  try S.expect_close(try S.floats_of(mean), try S.flat(try JSON.get(want, "mean")), abs, rel, "#{folder} mean")
  unit = try DistilBert.embed(m, batch, true)
  S.expect_close(try S.floats_of(unit), try S.flat(try JSON.get(want, "unit")), abs, rel, "#{folder} unit")
end
