## A tiny random BERT that the Python peer `bert_peer.py` writes with Hugging
## Face's names (and with the `bert.` prefix and `gamma`/`beta` norms of
## `bert-base-uncased`): the last hidden state, the pooled output and the mean
## embedding of a padded batch match the peer's forward in doubles, on the
## reference and on ggml. Run from `transformers/`.

import Bert from "../bert"
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
    matches(dir, "legacy", S.REF, ABS, REL)
  end
end

def test_a_padded_batch_matches_the_peer_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      try matches(dir, "tiny", backend, F32_ABS, F32_REL)
      try matches(dir, "legacy", backend, F32_ABS, F32_REL)
    end

    ok()
  end
end

def test_a_model_without_a_pooler_gives_the_hidden_state_alone() -> Result[Void]
  with_peer do |dir|
    m = try Bert.open("#{dir}/nopooler", S.REF)
    try Test.assert(m.pooler.none?, "the pooler is there")
    out = try Bert.run(m, try encodings(dir))
    try Test.assert(out.pooled.none?, "a pooled output without a pooler")
    want = try S.flat(try JSON.get(try expected(dir), "hidden"))
    S.expect_close(try S.floats_of(out.hidden), want, ABS, REL, "hidden")
  end
end

def test_a_model_that_is_not_bert_is_refused() -> Result[Void]
  with_peer do |dir|
    config = try File.read_text("#{dir}/tiny/config.json")
    for wrong in [
      config.replace("\"bert\"", "\"roberta\""),
      config.replace("\"gelu\"", "\"swish\""),
      config.replace("\"absolute\"", "\"relative_key\""),
      config.replace("\"hidden_size\": 8", "\"hidden_size\": 9"),
      config.replace("\"vocab_size\": 11", "\"vocab_size\": 12"),
    ]
      try File.write_text("#{dir}/tiny/config.json", wrong)
      try Test.error?(Bert.open("#{dir}/tiny", S.REF))
    end

    try Test.error?(Bert.open("#{dir}/missing", S.REF))
    Test.error?(Bert.open("#{dir}/absent", S.REF))
  end
end

def test_bad_inputs_are_errors() -> Result[Void]
  with_peer do |dir|
    m = try Bert.open("#{dir}/tiny", S.REF)
    try Test.error?(Bert.run(m, []))
    long = WordPiece.Encoding(ids: List.repeat(1, 9), type_ids: List.repeat(0, 9))
    try Test.error?(Bert.run(m, [long]))
    try Test.error?(Bert.run(m, [WordPiece.Encoding(ids: [1, 2], type_ids: [0])]))
    Test.error?(Bert.run(m, [WordPiece.Encoding(ids: [], type_ids: [])]))
  end
end

private

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/bert_peer.py", "agentica-bert-", body)
end

def expected(dir: String) -> Result[JSON.Value] = JSON.parse(try File.read_text("#{dir}/expected.json"))

## The batch of `expected.json` as encodings.
def encodings(dir: String) -> Result[List[WordPiece.Encoding]]
  want = try expected(dir)
  ids = try JSON.list(try JSON.get(want, "ids"))
  types = try JSON.list(try JSON.get(want, "type_ids"))
  out: List[WordPiece.Encoding] = []
  for i in 0...ids.size
    out = out.push(WordPiece.Encoding(ids: try ints(ids[i]), type_ids: try ints(types[i])))
  end

  ok(out)
end

def ints(json: JSON.Value) -> Result[List[Int]]
  out: List[Int] = []
  for v in try JSON.list(json)
    out = out.push(try JSON.int(v))
  end

  ok(out)
end

## The model `folder` on `backend` against the peer's hidden state, pooled
## output and mean embedding.
def matches(dir: String, folder: String, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  want = try expected(dir)
  m = try Bert.open("#{dir}/#{folder}", backend)
  batch = try encodings(dir)
  out = try Bert.run(m, batch)
  try Test.eq(Tensor.shape(out.hidden), [2, 5, 8])
  try S.expect_close(
    try S.floats_of(out.hidden),
    try S.flat(try JSON.get(want, "hidden")),
    abs,
    rel,
    "#{folder} hidden",
  )

  try S.expect_close(
    try S.floats_of(out.pooled.unwrap!),
    try S.flat(try JSON.get(want, "pooled")),
    abs,
    rel,
    "#{folder} pooled",
  )

  mean = try Bert.embed(m, batch)
  S.expect_close(try S.floats_of(mean), try S.flat(try JSON.get(want, "mean")), abs, rel, "#{folder} mean")
end
