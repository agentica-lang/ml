## A tiny random ViT that the Python peer `vit_peer.py` writes with Hugging Face's names (with and without the biases
## of the query, key and value projections): the logits of two images match the peer's forward in doubles, on the
## reference and on ggml. Run from `transformers/`.

import Lift from "../lift"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import ViT from "../vit"

## The reference computes in doubles, as the peer does; ggml in `F32`.
ABS = 0.000000001
REL = 0.000000001
F32_ABS = 0.0001
F32_REL = 0.0001

private

def test_the_logits_match_the_peer_on_the_reference() -> Result[Void]
  with_peer do |dir|
    for name in ["biased", "unbiased"]
      try matches(dir, name, S.REF, ABS, REL)
    end

    ok()
  end
end

def test_the_logits_match_the_peer_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      for name in ["biased", "unbiased"]
        try matches(dir, name, backend, F32_ABS, F32_REL)
      end
    end

    ok()
  end
end

def test_top_lists_the_best_classes_first_with_their_names() -> Result[Void]
  with_peer do |dir|
    m = try ViT.open("#{dir}/biased", S.REF)
    logits = try ViT.run(m, try images(dir))
    best = try ViT.top(m, logits, 2)
    want = try S.flat(try JSON.get(try expected(dir), "biased"))
    for row in 0...2
      scores = want.drop(row * 5).take(5)
      first = (0...5).to_a.sort_by { |i| -scores[i] }
      try Test.eq(best[row].map { |p| p.id }, first.take(2))
      try Test.eq(best[row].map { |p| p.label }, first.take(2).map { |i| "class #{i}" })
      try S.expect_close([best[row][0].logit], [scores[first[0]]], ABS, REL, "the best logit")
    end

    try Test.error?(ViT.top(m, logits, 0))
    Test.error?(ViT.top(m, logits, 6))
  end
end

def test_a_checkpoint_the_model_cannot_read_is_an_error_that_names_why() -> Result[Void]
  with_peer do |dir|
    try S.expect_error(ViT.open("#{dir}/missing", S.REF), "layer.1.attention.attention.key.weight")
    config = try File.read_text("#{dir}/biased/config.json")
    for wrong in [
      try S.replaced(config, "\"vit\"", "\"bert\""),
      try S.replaced(config, "\"gelu\"", "\"relu\""),
      try S.replaced(config, "\"patch_size\": 4", "\"patch_size\": 3"),
      try S.replaced(config, "\"intermediate_size\": 16", "\"intermediate_size\": 12"),
      try S.replaced(config, "\"4\": \"class 4\"", "\"4\": \"class 4\", \"5\": \"class 5\""),
    ]
      try File.write_text("#{dir}/biased/config.json", wrong)
      try Test.error?(ViT.open("#{dir}/biased", S.REF))
    end

    try File.write_text("#{dir}/biased/config.json", config)
    m = try ViT.open("#{dir}/biased", S.REF)
    try Test.error?(ViT.run(m, try S.f32([1.0, 2.0], [1, 2, 1, 1])))
    Test.error?(ViT.run(m, try S.f32((0...(3 * 12 * 12)).to_a.map { |_| 0.5 }, [1, 3, 12, 12])))
  end
end

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/vit_peer.py", "agentica-vit-", body)
end

def expected(dir: String) -> Result[JSON.Value] = JSON.parse(try File.read_text("#{dir}/expected.json"))

def images(dir: String, backend: Tensor.Backend = S.REF) -> Result[Tensor.Tensor]
  S.f32(try S.flat(try JSON.get(try expected(dir), "images")), [2, 3, 8, 8], backend)
end

## The model `folder` on `backend` against the peer's logits.
def matches(dir: String, folder: String, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  m = try ViT.open("#{dir}/#{folder}", backend)
  batch = try images(dir, backend)
  logits = try ViT.run(m, batch)
  try Test.eq(Tensor.shape(logits), [2, 5])
  want = try S.flat(try JSON.get(try expected(dir), folder))
  try S.expect_close(try S.floats_of(logits), want, abs, rel, "#{folder} on #{backend}")
  Lift.close([batch])
  ViT.close(logits)
  ViT.close_model(m)
  ok()
end
