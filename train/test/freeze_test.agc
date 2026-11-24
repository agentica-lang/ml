## Training part of a model (`params.agc`): a three-layer MLP whose first layer is frozen, against the PyTorch peer
## `freeze_golden.py` (`requires_grad = False` on the same layer, Adam, the same rows), whose printed run is
## `fixtures/freeze_golden.json`. The frozen weights keep their bits, the trained ones follow the peer, the optimizer
## state holds the trainable tensors alone, and the trainable part saved to a safetensors file reads back as it was.
## Run from `train/`.

import Data from "../data"
import Hub from "pkg/hub/hub"
import Loss from "../loss"
import Nn from "pkg/nn/nn"
import Optim from "../optim"
import Params from "../params"
import Prop from "pkg/prop/prop"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"
import Write from "pkg/hub/safetensors_write"

## The reference computes in doubles and the peer in `F32`.
ABS = 0.00001
REL = 0.0001

struct Named
  shape: List[Int]
  values: List[Float]
end

## What the peer printed: the weights before and after, the rows, the mean loss of every epoch, and the names its
## optimizer holds.
struct Golden
  frozen: String
  state: List[String]
  inputs: List[Float]
  targets: List[Float]
  start: Map[String, Named]
  history: List[Float]
  trained: Map[String, Named]
end

## What a run answers: the split, the values of the frozen weights before it, and the fit.
struct Run
  split: Params.Split
  before: Map[String, List[String]]
  fit: Train.Fit
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/freeze_golden.json"), Golden)

def weights(g: Golden, backend: Tensor.Backend) -> Result[Map[String, Tensor.Tensor]]
  out: Map[String, Tensor.Tensor] = {}
  for name in g.start.keys
    out = out.put(name, try S.tensor(g.start[name].values, g.start[name].shape, Tensor.Dtype.F32, backend))
  end

  ok(out)
end

def layer(w: Map[String, Tensor.Tensor], prefix: String) -> Nn.Dense
  Nn.Dense(w: w["#{prefix}.weight"], b: some(w["#{prefix}.bias"]), out_in: true)
end

def model(w: Map[String, Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  first = try Tensor.relu(try Nn.dense(layer(w, "body.0"), x))
  second = try Tensor.relu(try Nn.dense(layer(w, "body.2"), first))
  Nn.dense(layer(w, "head"), second)
end

def bits(t: Tensor.Tensor) -> Result[List[String]] = ok((try S.floats_of(t)).map { |x: Float| "#{Float.to_bits(x)}" })

def train(g: Golden, backend: Tensor.Backend) -> Result[Run]
  split = try Params.split(try weights(g, backend), ->(name: String) { not name.start_with?(g.frozen) })
  before: Map[String, List[String]] = {}
  for name in split.frozen.keys
    before = before.put(name, try bits(split.frozen[name]))
  end

  rows = try Data.dataset(
    try S.tensor(g.inputs, [8, 3], Tensor.Dtype.F32, backend),
    try S.tensor(g.targets, [8, 2], Tensor.Dtype.F32, backend),
  )

  fit = try Train.fit(
    ->(ps: List[Tensor.Tensor], b: Data.Batch) {
      Loss.mse(try model(try Params.join(split, ps), b.inputs), b.targets)
    },
    split.trainable,
    Data.Loader(dataset: rows, batch_size: 4, shuffle: false),
    Optim.Optimizer.Adam(Optim.Adam(lr: 0.01)),
    g.history.size,
  )

  ok(Run(split: split, before: before, fit: fit))
end

def follows_the_peer(g: Golden, backend: Tensor.Backend) -> Result[Void]
  run = try train(g, backend)
  try Test.eq(run.split.names, g.state.sort)
  try Test.eq(run.split.frozen.keys.sort, ["body.0.bias", "body.0.weight"])
  try S.expect_close(run.fit.history, g.history, ABS, REL, "loss history on #{backend}")
  after = try Params.join(run.split, run.fit.params)
  try Test.eq(after.keys.sort, g.trained.keys.sort)
  for name in run.split.frozen.keys
    try Test.eq(try bits(after[name]), run.before[name])
    try S.expect_close(try S.floats_of(after[name]), g.trained[name].values, 0.0, 0.0, "#{name} on #{backend}")
  end

  moved = false
  for name in run.split.names
    try S.expect_close(try S.floats_of(after[name]), g.trained[name].values, ABS, REL, "#{name} on #{backend}")
    moved = moved or g.trained[name].values != g.start[name].values
  end

  Test.assert(moved, "the peer trained nothing")
end

def test_frozen_weights_keep_their_bits_and_the_trained_follow_pytorch() -> Result[Void]
  follows_the_peer(try golden(), S.REF)
end

def test_the_same_run_on_libtorch_and_ggml() -> Result[Void]
  g = try golden()
  backends = try S.ggml_backends()
  if Tensor.available(S.TORCH)
    backends = backends.push(S.TORCH)
  end

  for backend in backends
    try follows_the_peer(g, backend)
  end

  ok()
end

## `fit` makes the optimizer's state from the trainable list alone: one first and one second moment for each
## parameter the peer's optimizer holds, none for a frozen weight.
def test_the_optimizer_state_holds_the_trainable_tensors_alone() -> Result[Void]
  g = try golden()
  split = try Params.split(try weights(g, S.REF), ->(name: String) { not name.start_with?(g.frozen) })
  state = try Optim.init(Optim.Optimizer.Adam(Optim.Adam(lr: 0.01)), split.trainable)
  try Test.eq(state.first.size, g.state.size)
  try Test.eq(state.second.size, g.state.size)
  for i in 0...split.names.size
    try Test.eq(Tensor.shape(state.first[i]), g.start[split.names[i]].shape)
  end

  ok()
end

## The trainable part alone goes to a safetensors file, and the file read back into the starting weights gives the
## trained model, bit for bit on ggml; the reference holds doubles, which the `F32` file rounds.
def test_the_trainable_part_saves_and_loads() -> Result[Void]
  g = try golden()
  dir = "/tmp/agentica-train-freeze-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = saves_and_loads(g, "#{dir}/trained.safetensors")
  _ = Path.remove(dir, 5000)
  result
end

def saves_and_loads(g: Golden, path: String) -> Result[Void]
  backends = [S.REF].concat(try S.ggml_backends())
  for backend in backends
    run = try train(g, backend)
    try Write.write(path, try Params.named(run.split, run.fit.params))
    c = try Hub.open(path)
    try Test.eq(Hub.names(c), run.split.names)
    loaded = try Params.update(try weights(g, backend), try Hub.tensors(c, backend))
    after = try Params.join(run.split, run.fit.params)
    for name in after.keys
      got = try S.floats_of(loaded[name])
      if backend == S.REF
        try S.expect_close(got, try S.floats_of(after[name]), 0.0, 0.000001, name)
      else
        try Test.eq(try bits(loaded[name]), try bits(after[name]))
      end
    end
  end

  ok()
end

def test_a_bad_selection_or_update_is_an_error() -> Result[Void]
  g = try golden()
  w = try weights(g, S.REF)
  try Test.error?(Params.split(w, ->(name: String) { name.start_with?("encoder.") }))
  split = try Params.split(w, ->(name: String) { name.start_with?("head.") })
  try Test.error?(Params.named(split, split.trainable.take(1)))
  try Test.error?(Params.join(split, []))
  try Test.error?(Params.update(w, {"tail.weight": w["head.bias"]}))
  Test.error?(Params.update(w, {"head.weight": w["head.bias"]}))
end

## Any selection splits the names in two, in sorted order, and `join` gives every weight back.
def test_a_split_parts_the_names_and_join_gives_them_back() -> Result[Void]
  Prop.check(
    Prop.pair(Prop.list(Prop.int(0, 20), 12), Prop.int(0, 21)),
    Prop.Config(runs: 50),
  ) do |p: Prop.Pair[List[Int], Int]|
    w: Map[String, Tensor.Tensor] = {}
    for n in p.first
      w = w.put("w#{n}", try S.tensor([n.to_f], [1]))
    end

    low = p.first.select { |n: Int| n < p.second }.map { |n: Int| "w#{n}" }
    chosen = w.keys.sort.select { |name: String| low.include?(name) }
    split = Params.split(w, ->(name: String) { chosen.include?(name) })
    return Prop.eq(split.ok?, false) if chosen.empty?
    s = try split
    try Prop.eq(s.names, chosen)
    try Prop.eq((s.names + s.frozen.keys).sort, w.keys.sort)
    back = try Params.join(s, s.trainable)
    for name in w.keys
      try Prop.eq(try S.floats_of(back[name]), try S.floats_of(w[name]))
    end

    Prop.eq(back.size, w.size)
  end
end
