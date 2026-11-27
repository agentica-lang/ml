## The weights of a BERT by name (`Bert.weights`, `Bert.with_weights`) and its forward inside a trace (`Bert.pooled`),
## on the tiny models of `bert_peer.py`: the names are the checkpoint's, a model rebuilt from its own weights gives the
## same embedding, the gradient of the cosine loss of a scored pair agrees on the reference (in doubles) with a central
## difference of the loss itself for three values and is not zero for the encoder, and `Train.fit_batches` with AdamW
## lowers the loss on ggml and moves every weight that has a gradient. Run from `transformers/`.

import Bert from "../bert"
import Data from "pkg/train/data"
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
SCORES = [0.8, 0.1]

## One value of a weight, its slope against a central difference with this step, and how close they must be.
STEP = 0.00001
SLOPE_ABS = 0.00000001
SLOPE_REL = 0.00001

private

struct Pick
  name: String
  at: Int
end

def picks() -> List[Pick]
  [
    Pick(name: "embeddings.token_type_embeddings.weight", at: 8 + 3),
    Pick(name: "encoder.layer.0.attention.self.query.weight", at: 3),
    Pick(name: "encoder.layer.1.output.dense.weight", at: 10),
  ]
end

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/bert_peer.py", "agentica-bert-", body)
end

def ids(backend: Tensor.Backend, shape: List[Int]) -> Result[Tensor.Tensor]
  count = 1
  for size in shape
    count = count * size
  end

  Tensor.from_list(IDS.take(count).map { |i| Tensor.Values.Int(i) }, shape, Tensor.Dtype.I32, backend)
end

## The mask of ids: 0.0 where the id is the pad id, as a model sees its padding.
def padding_mask(ids: Tensor.Tensor, pad: Int) -> Result[Tensor.Tensor]
  backend = Tensor.backend(ids)
  shape = Tensor.shape(ids)
  pads = try Tensor.full(shape, Tensor.Values.Int(pad), Tensor.Dtype.I32, backend)
  ones = try Tensor.full(shape, Tensor.Values.Float(1.0), Tensor.Dtype.F32, backend)
  Tensor.sub(ones, try Tensor.to_dtype(try Tensor.eq(ids, pads), Tensor.Dtype.F32))
end

## The token types of `n` pairs as a batch of `2 n` rows: 0 for the first sentence of a pair and 1 for the second.
def side_types(n: Int, seq: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  first = try Tensor.full([n, 1, seq], Tensor.Values.Int(0), Tensor.Dtype.I32, backend)
  second = try Tensor.full([n, 1, seq], Tensor.Values.Int(1), Tensor.Dtype.I32, backend)
  Tensor.reshape(try Tensor.concat([first, second], 1), [2 * n, seq])
end

## The cosine loss of a batch of pairs (`inputs` is `[pair, side, token]`, `targets` the scores) with the weights
## `ps` of `split`: the two sides go through the model as one batch, then each pair's two embeddings are compared.
def loss(m: Bert.Bert, split: Params.Split, ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  model = try Bert.with_weights(m, try Params.join(split, ps))
  n = Tensor.shape(b.inputs)[0]
  seq = Tensor.shape(b.inputs)[2]
  width = m.config.hidden_size
  flat = try Tensor.reshape(b.inputs, [2 * n, seq])
  types = try side_types(n, seq, Tensor.backend(flat))
  pooled = try Bert.pooled(model, flat, types, try padding_mask(flat, m.config.pad_token_id))
  sides = try Tensor.reshape(pooled, [n, 2, width])
  a = try Tensor.reshape(try Tensor.slice(sides, [0, 0, 0], [n, 1, width]), [n, width])
  c = try Tensor.reshape(try Tensor.slice(sides, [0, 1, 0], [n, 2, width]), [n, width])
  Sentence.cosine_similarity_loss(a, c, b.targets)
end

## The first `pairs` pairs.
def batch(backend: Tensor.Backend, pairs: Int) -> Result[Data.Batch]
  ok(Data.Batch(inputs: try ids(backend, [pairs, 2, 5]), targets: try S.f32(SCORES.take(pairs), [pairs], backend)))
end

def scalar(t: Tensor.Tensor) -> Result[Float] = ok((try S.floats_of(t))[0])

## `ps` with value `at` of tensor `i` moved by `by`.
def moved(ps: List[Tensor.Tensor], i: Int, at: Int, by: Float) -> Result[List[Tensor.Tensor]]
  values = try S.floats_of(ps[i])
  values = values.take(at) + [values[at] + by] + values.drop(at + 1)
  out = ps.take(i) + [try S.f32(values, Tensor.shape(ps[i]))] + ps.drop(i + 1)
  ok(out)
end

def test_the_names_are_the_checkpoints_and_the_weights_go_back() -> Result[Void]
  with_peer do |dir|
    for folder in ["tiny", "legacy", "nopooler"]
      m = try Bert.open("#{dir}/#{folder}", S.REF)
      w = Bert.weights(m)
      names = Hub.names(try Hub.open("#{dir}/#{folder}/model.safetensors")).select { |n: String|
        !n.start_with?("cls.")
      }

      names = names.map { |n: String|
        n.replace("LayerNorm.gamma", "LayerNorm.weight").replace("LayerNorm.beta", "LayerNorm.bias")
      }

      try Test.eq(w.keys.sort, names.sort)
      want_prefix = ""
      if folder == "legacy"
        want_prefix = "bert."
      end

      try Test.eq(m.prefix, want_prefix)
      flat = try ids(S.REF, [4, 5])
      types = try side_types(2, 5, S.REF)
      mask = try padding_mask(flat, 0)
      own = try S.floats_of(try Bert.pooled(m, flat, types, mask, true))
      scaled: Map[String, Tensor.Tensor] = {}
      for name in w.keys
        values = (try S.floats_of(w[name])).map { |x: Float| x * 1.5 }
        scaled = scaled.put(name, try S.f32(values, Tensor.shape(w[name])))
      end

      other = try Bert.with_weights(m, scaled)
      try Test.assert(
        own != try S.floats_of(try Bert.pooled(other, flat, types, mask, true)),
        "the weights were ignored",
      )

      back = try Bert.with_weights(other, w)
      try Test.eq(try S.floats_of(try Bert.pooled(back, flat, types, mask, true)), own)
      norm = "#{m.prefix}embeddings.LayerNorm.weight"
      try S.expect_error(Bert.with_weights(m, w.delete(norm)), norm)
      try S.expect_error(Bert.with_weights(m, w.put("extra.weight", w[norm])), "extra.weight")
      table = w["#{m.prefix}embeddings.word_embeddings.weight"]
      try S.expect_error(Bert.with_weights(m, w.put(norm, table)), "#{norm}: the model holds")
    end

    ok()
  end
end

## The slope of the loss for three values (a table, attention, MLP) agrees with a central difference of the loss
## itself, in doubles on the reference. The mean pooling leaves the pooler out of the loss and the softmax makes the
## key bias of an attention free, so those have a gradient of zero; every other weight has one that is not.
def test_the_gradient_matches_a_central_difference_and_reaches_the_encoder() -> Result[Void]
  with_peer do |dir|
    m = try Bert.open("#{dir}/tiny", S.REF)
    split = try Params.split(Bert.weights(m), ->(_name: String) { true })
    b = try batch(S.REF, 1)
    both = try Trace.value_and_grad(
      split.trainable,
      ->(ps: List[Tensor.Tensor]) {
        loss(m, split, ps, Data.Batch(inputs: Trace.constant(b.inputs), targets: Trace.constant(b.targets)))
      },
    )

    grads: List[List[Float]] = []
    for g in both.grads
      grads = grads.push(try S.floats_of(g))
    end

    try Test.eq(grads.size, 5 + 16 * 2 + 2)
    for i in 0...grads.size
      largest = grads[i].map { |g: Float| g.abs }.max.unwrap!
      name = split.names[i]
      if name.start_with?("pooler.") or name.end_with?(".key.bias")
        try Test.assert(largest < SLOPE_ABS, "#{name} has a gradient of #{largest}")
      else
        try Test.assert(largest > SLOPE_ABS, "#{name} has no gradient")
      end
    end

    for p in picks()
      i = try split.names.index(p.name).to_result("no weight #{p.name}")
      up = try scalar(try loss(m, split, try moved(split.trainable, i, p.at, STEP), b))
      down = try scalar(try loss(m, split, try moved(split.trainable, i, p.at, 0.0 - STEP), b))
      try S.expect_close([grads[i][p.at]], [(up - down) / (2.0 * STEP)], SLOPE_ABS, SLOPE_REL, "#{p.name}[#{p.at}]")
    end

    ok()
  end
end

## The loss and the gradient of every weight for one pair on `backend`, by the order of `Params.split`.
def graded(dir: String, backend: Tensor.Backend) -> Result[List[List[Float]]]
  m = try Bert.open("#{dir}/tiny", backend)
  split = try Params.split(Bert.weights(m), ->(_name: String) { true })
  b = try batch(backend, 1)
  both = try Trace.value_and_grad(
    split.trainable,
    ->(ps: List[Tensor.Tensor]) {
      loss(m, split, ps, Data.Batch(inputs: Trace.constant(b.inputs), targets: Trace.constant(b.targets)))
    },
  )

  out: List[List[Float]] = [try S.floats_of(both.value)]
  for g in both.grads
    out = out.push(try S.floats_of(g))
  end

  ok(out)
end

## libtorch computes the loss and every gradient of the model as the reference does, in `F32`: the attention backward
## runs on its `batch_matmul`.
def test_the_gradient_agrees_on_libtorch() -> Result[Void]
  with_peer do |dir|
    return ok() if (try S.torch_backends()).empty?
    want = try graded(dir, S.REF)
    got = try graded(dir, S.TORCH)
    try Test.eq(got.size, want.size)
    for i in 0...want.size
      try S.expect_close(got[i], want[i], 0.00001, 0.001, "output #{i} on libtorch")
    end

    ok()
  end
end

## `Train.fit_batches` with AdamW: the loss falls over three steps, and every weight with a gradient moved.
def fits(dir: String, backend: Tensor.Backend) -> Result[Void]
  m = try Bert.open("#{dir}/tiny", backend)
  split = try Params.split(Bert.weights(m), ->(_name: String) { true })
  before = split.trainable.map { |t: Tensor.Tensor| S.floats_of(t).unwrap! }
  fit = try Train.fit_batches(
    ->(ps: List[Tensor.Tensor], b: Data.Batch) { loss(m, split, ps, b) },
    split.trainable,
    ->(_e: Int, _i: Int) { batch(backend, 2) },
    1,
    Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.01, weight_decay: 0.0)),
    3,
  )

  try Test.assert(fit.history[2] < fit.history[0], "the loss went from #{fit.history[0]} to #{fit.history[2]}")
  for i in 0...split.names.size
    name = split.names[i]
    moved = try S.floats_of(fit.params[i]) != before[i]
    if !name.start_with?("pooler.") and !name.end_with?(".key.bias")
      try Test.assert(moved, "#{name} did not move")
    end
  end

  ok()
end

def test_fit_batches_lowers_the_loss_and_moves_the_encoder() -> Result[Void]
  with_peer do |dir|
    return ok() if (try S.ggml_backends()).empty?
    fits(dir, S.GGML)
  end
end

def test_fit_batches_lowers_the_loss_and_moves_the_encoder_on_libtorch() -> Result[Void]
  with_peer do |dir|
    return ok() if (try S.torch_backends()).empty?
    fits(dir, S.TORCH)
  end
end
