## The weights of a DistilBERT by name (`DistilBert.weights`, `DistilBert.with_weights`) and its forward inside a trace
## (`DistilBert.pooled`), on the tiny models of `distilbert_peer.py`: the names are the checkpoint's, a model rebuilt
## from its own weights gives the same embedding, the cosine loss of two scored pairs and its gradient for every
## weight agree on the reference (in doubles) and ggml, the loss and the gradient of 14 values match the peer's
## central difference, and `Train.fit` with AdamW lowers the loss on ggml. libtorch has no `batch_matmul`, which the
## gradient of an attention needs, so it is not run (ml#203). Run from `transformers/`.

import Data from "pkg/train/data"
import DistilBert from "../distilbert"
import Hub from "pkg/hub/hub"
import Optim from "pkg/train/optim"
import Params from "pkg/train/params"
import S from "support"
import Sentence from "../sentence"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "pkg/train/train"

## Two pairs of sentences, `[pair, side, token]`, padded with 0, and the score of each pair.
IDS = [1, 5, 2, 7, 3, 1, 6, 3, 0, 0, 1, 4, 6, 3, 0, 1, 2, 2, 5, 3]
MASK = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
SCORES = [0.8, 0.1]

## `F32` on ggml against the doubles of the reference.
ABS = 0.00001
REL = 0.001

## The loss in doubles against the peer's, and a gradient against a central difference with a step of 1e-5.
PEER_ABS = 0.000000000001
PEER_REL = 0.000000001
SLOPE_ABS = 0.00000001
SLOPE_REL = 0.00001

private

## One value of the weight `name` and the slope of the loss for it, a central difference of the peer.
struct Slope
  name: String
  at: Int
  slope: Float
end

## What `expected.json` says of the loss of the pairs.
struct Expected
  loss: Float
  slopes: List[Slope]
end

## The loss and its gradient for every weight of `tiny` on `backend`, by name.
struct Graded
  split: Params.Split
  value: Float
  grads: List[List[Float]]
end

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/distilbert_peer.py", "agentica-distilbert-", body)
end

def ids(backend: Tensor.Backend, shape: List[Int]) -> Result[Tensor.Tensor]
  Tensor.from_list(IDS.map { |i| Tensor.Values.Int(i) }, shape, Tensor.Dtype.I32, backend)
end

def test_the_names_are_the_checkpoints_and_the_weights_go_back() -> Result[Void]
  with_peer do |dir|
    for folder in ["tiny", "prefixed"]
      m = try DistilBert.open("#{dir}/#{folder}", S.REF)
      w = DistilBert.weights(m)
      names = Hub.names(try Hub.open("#{dir}/#{folder}/model.safetensors"))
      if folder == "prefixed"
        names = names.select { |name: String| name.start_with?("distilbert.") }
      end

      try Test.eq(w.keys.sort, names)
      try Test.eq(w.size, 4 + 16 * m.config.n_layers)
      flat = try ids(S.REF, [4, 5])
      mask = try S.f32(MASK, [4, 5])
      own = try S.floats_of(try DistilBert.pooled(m, flat, mask, true))
      scaled: Map[String, Tensor.Tensor] = {}
      for name in w.keys
        values = (try S.floats_of(w[name])).map { |x: Float| x * 1.5 }
        scaled = scaled.put(name, try S.f32(values, Tensor.shape(w[name])))
      end

      other = try DistilBert.with_weights(m, scaled)
      try Test.assert(own != try S.floats_of(try DistilBert.pooled(other, flat, mask, true)), "the weights were ignored")
      back = try DistilBert.with_weights(other, w)
      try Test.eq(try S.floats_of(try DistilBert.pooled(back, flat, mask, true)), own)

      norm = "#{m.prefix}embeddings.LayerNorm.weight"
      try S.expect_error(DistilBert.with_weights(m, w.delete(norm)), norm)
      try S.expect_error(DistilBert.with_weights(m, w.put("extra.weight", w[norm])), "extra.weight")
      table = w["#{m.prefix}embeddings.word_embeddings.weight"]
      try S.expect_error(DistilBert.with_weights(m, w.put(norm, table)), "#{norm}: the model holds")
    end

    ok()
  end
end

## The cosine loss of the pairs `pairs` (`[pair, side, token]` ids) with `mask` and `scores`: the two sides go through
## the model as one batch, then each pair's two embeddings are compared.
def loss(
  m: DistilBert.DistilBert,
  split: Params.Split,
  ps: List[Tensor.Tensor],
  pairs: Tensor.Tensor,
  mask: Tensor.Tensor,
  scores: Tensor.Tensor,
) -> Result[Tensor.Tensor]
  model = try DistilBert.with_weights(m, try Params.join(split, ps))
  n = Tensor.shape(pairs)[0]
  seq = Tensor.shape(pairs)[2]
  width = m.config.dim
  pooled = try DistilBert.pooled(model, try Tensor.reshape(pairs, [2 * n, seq]), mask)
  sides = try Tensor.reshape(pooled, [n, 2, width])
  a = try Tensor.reshape(try Tensor.slice(sides, [0, 0, 0], [n, 1, width]), [n, width])
  b = try Tensor.reshape(try Tensor.slice(sides, [0, 1, 0], [n, 2, width]), [n, width])
  Sentence.cosine_similarity_loss(a, b, scores)
end

def graded(dir: String, backend: Tensor.Backend) -> Result[Graded]
  m = try DistilBert.open("#{dir}/tiny", backend)
  split = try Params.split(DistilBert.weights(m), ->(_name: String) { true })
  pairs = try ids(backend, [2, 2, 5])
  mask = try S.f32(MASK, [4, 5], backend)
  scores = try S.f32(SCORES, [2], backend)
  both = try Trace.value_and_grad(
    split.trainable,
    ->(ps: List[Tensor.Tensor]) {
      loss(m, split, ps, Trace.constant(pairs), Trace.constant(mask), Trace.constant(scores))
    },
  )

  grads: List[List[Float]] = []
  for g in both.grads
    grads = grads.push(try S.floats_of(g))
  end

  ok(Graded(split: split, value: (try S.floats_of(both.value))[0], grads: grads))
end

## The peer computes the loss in doubles with its own forward and takes its slope for one value of 14 weights
## (tables, attention, norms, MLP) as a central difference: no code of `Trace` is in it. The key bias of an attention
## has a gradient of zero, since a softmax does not change when every score of a row moves by the same amount; every
## other weight has a gradient that is not zero.
def test_the_loss_and_its_gradients_match_the_peers_central_difference() -> Result[Void]
  with_peer do |dir|
    peer = try JSON.decode(try File.read_text("#{dir}/expected.json"), Expected)
    ours = try graded(dir, S.REF)
    try S.expect_close([ours.value], [peer.loss], PEER_ABS, PEER_REL, "loss")
    try Test.eq(peer.slopes.size, 14)
    for s in peer.slopes
      index = try ours.split.names.index(s.name).to_result("no weight #{s.name}")
      try S.expect_close([ours.grads[index][s.at]], [s.slope], SLOPE_ABS, SLOPE_REL, "#{s.name}[#{s.at}]")
    end

    for i in 0...ours.grads.size
      largest = ours.grads[i].map { |g: Float| g.abs }.max.unwrap!
      if ours.split.names[i].end_with?(".k_lin.bias")
        try Test.assert(largest < SLOPE_ABS, "#{ours.split.names[i]} has a gradient of #{largest}")
      else
        try Test.assert(largest > SLOPE_ABS, "#{ours.split.names[i]} has no gradient")
      end
    end

    ok()
  end
end

def test_the_loss_and_every_gradient_agree_on_the_backends() -> Result[Void]
  with_peer do |dir|
    want = try graded(dir, S.REF)
    try Test.eq(want.grads.size, 36)
    for backend in try S.ggml_backends()
      got = try graded(dir, backend)
      try S.expect_close([got.value], [want.value], ABS, REL, "loss on #{backend}")
      for i in 0...want.grads.size
        try S.expect_close(got.grads[i], want.grads[i], ABS, REL, "#{want.split.names[i]} on #{backend}")
      end
    end

    ok()
  end
end

def trains(dir: String, backend: Tensor.Backend) -> Result[Void]
  m = try DistilBert.open("#{dir}/tiny", backend)
  split = try Params.split(DistilBert.weights(m), ->(_name: String) { true })
  norm = ->(name: String) { name.include?("LayerNorm") or name.include?("layer_norm") }
  decay = split.names.map { |name: String| !(name.end_with?(".bias") or norm.call(name)) }
  try Test.eq(decay.select { |d| d }.size, 14)
  mask = try S.f32(MASK, [4, 5], backend)
  fit = try Train.fit(
    ->(ps: List[Tensor.Tensor], b: Data.Batch) { loss(m, split, ps, b.inputs, Trace.constant(mask), b.targets) },
    split.trainable,
    Data.Loader(
      dataset: try Data.dataset(try ids(backend, [2, 2, 5]), try S.f32(SCORES, [2], backend)),
      batch_size: 2,
      shuffle: false,
    ),
    Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.01, weight_decay: 0.01, decay: decay)),
    3,
  )

  try Test.assert(fit.history[2] < fit.history[0], "the loss went from #{fit.history[0]} to #{fit.history[2]}")
  trained = try DistilBert.with_weights(m, try Params.join(split, fit.params))
  Test.eq(Tensor.shape(try DistilBert.forward(trained, try ids(backend, [4, 5]), mask)), [4, 5, 8])
end

## On ggml CPU alone: tracing three steps of the model takes most of a test's step budget.
def test_fit_with_adamw_lowers_the_loss() -> Result[Void]
  with_peer do |dir|
    return ok() if (try S.ggml_backends()).empty?
    trains(dir, S.GGML)
  end
end
