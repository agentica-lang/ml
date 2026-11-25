## A tiny random CLIP that the Python peer `clip_peer.py` writes with Hugging Face's names: the embeddings of two images
## and of three rows of text ids, their scaled cosines and the probabilities of each text match the peer's forward in
## doubles, on the reference and on ggml. Run from `transformers/`.

import Clip from "../clip"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The reference computes in doubles, as the peer does; ggml in `F32`.
ABS = 0.000000001
REL = 0.000000001
F32_ABS = 0.0001
F32_REL = 0.0001

private

def test_the_embeddings_logits_and_probabilities_match_the_peer_on_the_reference() -> Result[Void]
  with_peer do |dir| matches(dir, S.REF, ABS, REL) end
end

def test_the_embeddings_logits_and_probabilities_match_the_peer_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      try matches(dir, backend, F32_ABS, F32_REL)
    end

    ok()
  end
end

def test_a_checkpoint_the_model_cannot_read_is_an_error_that_names_why() -> Result[Void]
  with_peer do |dir|
    try S.expect_error(Clip.open("#{dir}/missing", S.REF), "layers.1.self_attn.k_proj.weight")
    config = try File.read_text("#{dir}/clip/config.json")
    for wrong in [
      try S.replaced(config, "\"clip\"", "\"vit\""),
      try S.replaced(config, "\"gelu\"", "\"quick_gelu\""),
      try S.replaced(config, "\"patch_size\": 4", "\"patch_size\": 3"),
      try S.replaced(config, "\"projection_dim\": 6", "\"projection_dim\": 7"),
    ]
      try File.write_text("#{dir}/clip/config.json", wrong)
      try Test.error?(Clip.open("#{dir}/clip", S.REF))
    end

    try File.write_text("#{dir}/clip/config.json", config)
    m = try Clip.open("#{dir}/clip", S.REF)
    try Test.error?(Clip.image_embeds(m, try S.f32([1.0, 2.0], [1, 2, 1, 1])))
    try Test.error?(Clip.text_embeds(m, []))
    try Test.error?(Clip.text_embeds(m, [[]]))
    try Test.error?(Clip.text_embeds(m, [[1, 2, 3, 4, 5, 6, 7]]))
    Test.error?(Clip.logits(m, try S.f32([1.0, 2.0], [1, 2]), try S.f32([1.0, 2.0], [1, 2])))
  end
end

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/clip_peer.py", "agentica-clip-", body)
end

def expected(dir: String) -> Result[JSON.Value] = JSON.parse(try File.read_text("#{dir}/expected.json"))

def floats_at(dir: String, key: String) -> Result[List[Float]] = S.flat(try JSON.get(try expected(dir), key))

def rows(dir: String) -> Result[List[List[Int]]]
  out: List[List[Int]] = []
  for row in try JSON.list(try JSON.get(try expected(dir), "rows"))
    ids: List[Int] = []
    for id in try JSON.list(row)
      ids = ids.push(try JSON.int(id))
    end

    out = out.push(ids)
  end

  ok(out)
end

## The model on `backend` against the peer: image and text embeddings, logits and probabilities.
def matches(dir: String, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  m = try Clip.open("#{dir}/clip", backend)
  images = try S.f32(try floats_at(dir, "images"), [2, 3, 8, 8], backend)
  i = try Clip.image_embeds(m, images)
  t = try Clip.text_embeds(m, try rows(dir))
  try Test.eq(Tensor.shape(i), [2, 6])
  try Test.eq(Tensor.shape(t), [3, 6])
  try S.expect_close(
    try S.floats_of(i),
    try floats_at(dir, "image_embeds"),
    abs,
    rel,
    "image embeddings on #{backend}",
  )

  try S.expect_close(try S.floats_of(t), try floats_at(dir, "text_embeds"), abs, rel, "text embeddings on #{backend}")
  try S.expect_close([m.scale], try floats_at(dir, "scale"), abs, rel, "the scale")
  logits = try Clip.logits(m, i, t)
  probabilities = try Clip.probabilities(m, i, t)
  try S.expect_close(try S.floats_of(logits), try floats_at(dir, "logits"), abs, rel, "logits on #{backend}")
  try S.expect_close(
    try S.floats_of(probabilities),
    try floats_at(dir, "probabilities"),
    abs,
    rel,
    "probabilities on #{backend}",
  )

  for x in [i, t, logits, probabilities]
    Clip.close(x)
  end

  Clip.close_model(m)
  ok()
end
