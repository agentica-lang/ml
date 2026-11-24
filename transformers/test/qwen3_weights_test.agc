## The weights of a Qwen3 by name (`Qwen3.weights`, `Qwen3.with_weights`) on the tiny models of `qwen_peer.py`: the
## names are the checkpoint's, a model rebuilt from its own weights gives the same logits, and `Train.fit` trains the
## last block alone while every other weight keeps its bits. Run from `transformers/`.

import Data from "pkg/train/data"
import Hub from "pkg/hub/hub"
import Loss from "pkg/train/loss"
import Optim from "pkg/train/optim"
import Params from "pkg/train/params"
import Qwen3 from "../qwen3"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Train from "pkg/train/train"

IDS = [[1, 5, 2, 7, 3, 12], [4, 4, 9, 0, 6, 11]]
LAST = "model.layers.1."

private

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/qwen_peer.py", "agentica-qwen-", body)
end

def bits(t: Tensor.Tensor) -> Result[List[String]] = ok((try S.floats_of(t)).map { |x: Float| "#{Float.to_bits(x)}" })

def test_the_names_are_the_checkpoints_and_the_weights_go_back() -> Result[Void]
  with_peer do |dir|
    for folder in ["tiny", "untied", "qwen2"]
      m = try Qwen3.open("#{dir}/#{folder}", S.REF)
      w = Qwen3.weights(m)
      names = Hub.names(try Qwen3.checkpoint("#{dir}/#{folder}"))
      if m.config.tie_word_embeddings
        names = names.select { |name: String| name != "lm_head.weight" }
      end

      try Test.eq(w.keys.sort, names)
      back = try Qwen3.with_weights(m, w)
      try Test.eq(try bits(try Qwen3.run(back, IDS)), try bits(try Qwen3.run(m, IDS)))
      try S.expect_error(Qwen3.with_weights(m, w.delete("model.norm.weight")), "model.norm.weight")
    end

    ok()
  end
end

## The first five tokens of each row, or with `shift` the five that follow them.
def tokens(shift: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  values: List[Tensor.Values] = []
  for row in IDS
    for id in row.drop(shift).take(5)
      values = values.push(Tensor.Values.Int(id))
    end
  end

  Tensor.from_list(values, [2, 5], Tensor.Dtype.I32, backend)
end

## The cross-entropy of the next token at every position.
def loss(m: Qwen3.Qwen3, split: Params.Split, ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  model = try Qwen3.with_weights(m, try Params.join(split, ps))
  logits = try Qwen3.logits(model, b.inputs)
  Loss.cross_entropy(try Tensor.reshape(logits, [10, 13]), try Tensor.reshape(b.targets, [10]))
end

def trains_the_last_block(dir: String, backend: Tensor.Backend) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  split = try Params.split(Qwen3.weights(m), ->(name: String) { name.start_with?(LAST) })
  try Test.eq(split.names.size, 11)
  before: Map[String, List[String]] = {}
  for name in split.frozen.keys
    before = before.put(name, try bits(split.frozen[name]))
  end

  start = try bits(split.trainable[0])
  fit = try Train.fit(
    ->(ps: List[Tensor.Tensor], b: Data.Batch) { loss(m, split, ps, b) },
    split.trainable,
    Data.Loader(
      dataset: try Data.dataset(try tokens(0, backend), try tokens(1, backend)),
      batch_size: 2,
      shuffle: false,
    ),
    Optim.Optimizer.Adam(Optim.Adam(lr: 0.01)),
    3,
  )

  try Test.assert(fit.history[2] < fit.history[0], "the loss went from #{fit.history[0]} to #{fit.history[2]}")
  try Test.assert(start != try bits(fit.params[0]), "#{split.names[0]} did not move")
  after = try Params.join(split, fit.params)
  for name in split.frozen.keys
    try Test.eq(try bits(after[name]), before[name])
  end

  trained = try Qwen3.with_weights(m, after)
  Test.eq(Tensor.shape(try Qwen3.run(trained, IDS)), [2, 6, 13])
end

def test_fit_trains_the_last_block_and_the_rest_keeps_its_bits() -> Result[Void]
  with_peer do |dir|
    for backend in [S.REF].concat(try S.ggml_backends())
      try trains_the_last_block(dir, backend)
    end

    ok()
  end
end
