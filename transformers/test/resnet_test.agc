## A tiny random ResNet that the Python peer `resnet_peer.py` writes with Hugging Face's names (a bottleneck one, a basic
## one, and a bottleneck one with the strides of `downsample_in_first_stage` and `downsample_in_bottleneck`), the
## batch norms with statistics of their own: the logits of two images match the peer's forward in doubles, with the
## batch norms not folded, on the reference and on ggml. Run from `transformers/`.

import Lift from "../lift"
import ResNet from "../resnet"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The reference computes in doubles, as the peer does; ggml in `F32`.
ABS = 0.000000001
REL = 0.000000001
F32_ABS = 0.0001
F32_REL = 0.0001

private

def test_the_logits_of_every_layout_match_the_peer_on_the_reference() -> Result[Void]
  with_peer do |dir|
    for name in ["bottleneck", "basic", "strided"]
      try matches(dir, name, S.REF, ABS, REL)
    end

    ok()
  end
end

def test_the_logits_of_every_layout_match_the_peer_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      for name in ["bottleneck", "basic", "strided"]
        try matches(dir, name, backend, F32_ABS, F32_REL)
      end
    end

    ok()
  end
end

def test_top_lists_the_best_classes_first_with_their_names() -> Result[Void]
  with_peer do |dir|
    m = try ResNet.open("#{dir}/bottleneck", S.REF)
    logits = try ResNet.run(m, try images(dir))
    best = try ResNet.top(m, logits, 2)
    want = try S.flat(try JSON.get(try expected(dir), "bottleneck"))
    for row in 0...2
      scores = want.drop(row * 5).take(5)
      first = (0...5).to_a.sort_by { |i| -scores[i] }
      try Test.eq(best[row].map { |p| p.id }, first.take(2))
      try Test.eq(best[row].map { |p| p.label }, first.take(2).map { |i| "class #{i}" })
      try S.expect_close([best[row][0].logit], [scores[first[0]]], ABS, REL, "the best logit")
    end

    try Test.error?(ResNet.top(m, logits, 0))
    Test.error?(ResNet.top(m, logits, 6))
  end
end

def test_a_checkpoint_the_model_cannot_read_is_an_error_that_names_why() -> Result[Void]
  with_peer do |dir|
    try S.expect_error(ResNet.open("#{dir}/missing", S.REF), "stages.1.layers.0.shortcut")
    config = try File.read_text("#{dir}/basic/config.json")
    for wrong in [
      try S.replaced(config, "\"resnet\"", "\"vit\""),
      try S.replaced(config, "\"relu\"", "\"gelu\""),
      try S.replaced(config, "\"basic\"", "\"mixed\""),
    ]
      try File.write_text("#{dir}/basic/config.json", wrong)
      try Test.error?(ResNet.open("#{dir}/basic", S.REF))
    end

    try File.write_text("#{dir}/basic/config.json", config)
    m = try ResNet.open("#{dir}/basic", S.REF)
    try Test.error?(ResNet.run(m, try S.f32([1.0, 2.0], [1, 2, 1, 1])))
    ok()
  end
end

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/resnet_peer.py", "agentica-resnet-", body)
end

def expected(dir: String) -> Result[JSON.Value] = JSON.parse(try File.read_text("#{dir}/expected.json"))

def images(dir: String) -> Result[Tensor.Tensor]
  S.f32(try S.flat(try JSON.get(try expected(dir), "images")), [2, 3, 21, 19])
end

## The model `folder` on `backend` against the peer's logits.
def matches(dir: String, folder: String, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  m = try ResNet.open("#{dir}/#{folder}", backend)
  batch = try S.f32(try S.flat(try JSON.get(try expected(dir), "images")), [2, 3, 21, 19], backend)
  logits = try ResNet.run(m, batch)
  try Test.eq(Tensor.shape(logits), [2, 5])
  want = try S.flat(try JSON.get(try expected(dir), folder))
  try S.expect_close(try S.floats_of(logits), want, abs, rel, "#{folder} on #{backend}")
  Lift.close([batch])
  ResNet.close(logits)
  ResNet.close_model(m)
  ok()
end
