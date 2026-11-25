## `Lora` against the PyTorch + PEFT peer `lora_golden.py` (`fixtures/lora_golden.json`): a two-layer MLP whose two
## layers have an adapter. The output of the adapted model, the gradients of every `A` and `B`, and a short AdamW run
## on the adapter alone follow PEFT, and the weights of the model keep their bits. Run from `train/`.

import Data from "../data"
import Lora from "../lora"
import Loss from "../loss"
import Nn from "pkg/nn/nn"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "../train"

## The reference computes in doubles and the peer in `F32`.
ABS = 0.00001
REL = 0.0001

CONFIG = Lora.Config(targets: ["first", "second"], r: 2, alpha: 4.0)
ADAPTED = ["first.weight", "second.weight"]

## What the peer printed. `start`, `gradients` and `trained` are `A` then `B` of each layer, as `Lora.params`.
struct Golden
  shapes: Map[String, List[Int]]
  weights: Map[String, List[Float]]
  adapter_shapes: List[List[Int]]
  start: List[List[Float]]
  inputs: List[Float]
  targets: List[Float]
  output: List[Float]
  loss: Float
  gradients: List[List[Float]]
  history: List[Float]
  trained: List[List[Float]]
  fresh: Fresh
end

## What PEFT starts a layer with: the largest `|A|` of its `[r, in]` and whether `B` is all zero.
struct Fresh
  inputs: Int
  r: Int
  a_max: Float
  b_zero: Bool
end

struct Setup
  weights: Map[String, Tensor.Tensor]
  adapter: Lora.Adapter
  batch: Data.Batch
end

private

def setup(backend: Tensor.Backend) -> Result[Setup]
  g: Golden = try JSON.decode(try File.read_text("test/fixtures/lora_golden.json"), Golden)
  weights: Map[String, Tensor.Tensor] = {}
  for name in g.weights.keys
    weights = weights.put(name, try S.tensor(g.weights[name], g.shapes[name], Tensor.Dtype.F32, backend))
  end

  params: List[Tensor.Tensor] = []
  for i in 0...g.start.size
    params = params.push(try S.tensor(g.start[i], g.adapter_shapes[i], Tensor.Dtype.F32, backend))
  end

  adapter = try Lora.with_params(try Lora.init(weights, CONFIG, 1, backend), params)
  batch = Data.Batch(
    inputs: try S.tensor(g.inputs, [6, 3], Tensor.Dtype.F32, backend),
    targets: try S.tensor(g.targets, [6, 2], Tensor.Dtype.F32, backend),
  )

  ok(Setup(weights: weights, adapter: adapter, batch: batch))
end

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/lora_golden.json"), Golden)

def layer(w: Map[String, Tensor.Tensor], prefix: String) -> Nn.Dense
  Nn.Dense(w: w["#{prefix}.weight"], b: some(w["#{prefix}.bias"]), out_in: true)
end

def model(w: Map[String, Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  Nn.dense(layer(w, "second"), try Tensor.relu(try Nn.dense(layer(w, "first"), x)))
end

## The loss of the model with `ps` as the adapter's tensors.
def loss(s: Setup, ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  w = try Lora.applied(s.weights, try Lora.with_params(s.adapter, ps))
  Loss.mse(try model(w, b.inputs), b.targets)
end

def bits(t: Tensor.Tensor) -> Result[List[String]] = ok((try S.floats_of(t)).map { |x: Float| "#{Float.to_bits(x)}" })

## The reference, ggml where it is built, and libtorch where it is built. CI builds ggml and not libtorch, so the
## libtorch cases are local only.
def backends() -> Result[List[Tensor.Backend]]
  out = [S.REF] + try S.ggml_backends()
  return ok(out.push(S.TORCH)) if Tensor.available(S.TORCH)
  ok(out)
end

def test_a_new_adapter_takes_the_targets_and_changes_nothing() -> Result[Void]
  s = try setup(S.REF)
  made = try Lora.init(s.weights, Lora.Config(targets: ["second"], r: 3), 7, S.REF)
  try Test.eq(made.names, ["second.weight"])
  try Test.eq(Tensor.shape(made.a[0]), [3, 4])
  try Test.eq(Tensor.shape(made.b[0]), [2, 3])
  a = try S.floats_of(made.a[0])
  try Test.assert(a.all? { |x: Float| x.abs <= 0.5 }, "A lies within 1 / sqrt(in)")
  try Test.assert(a.max.unwrap! > a.min.unwrap!, "A is not one value")
  with = try Lora.applied(s.weights, made)
  try Test.eq(with.keys.sort, s.weights.keys.sort)
  for name in s.weights.keys
    try Test.eq(try bits(with[name]), try bits(s.weights[name]))
  end

  ok()
end

def test_a_new_adapter_starts_as_peft_does() -> Result[Void]
  g = try golden()
  weights = {"layer.weight": try Tensor.zeros([g.fresh.inputs, g.fresh.inputs], Tensor.Dtype.F32, S.REF)}
  made = try Lora.init(weights, Lora.Config(targets: ["layer"], r: g.fresh.r), 3, S.REF)
  bound = 1.0 / Math.sqrt(g.fresh.inputs.to_f)
  top = (try S.floats_of(made.a[0])).map { |x: Float| x.abs }.max.unwrap!
  try Test.assert((top - g.fresh.a_max).abs < 0.05 * bound, "the largest |A| is #{top}, PEFT's #{g.fresh.a_max}")
  try Test.assert(top > 0.5 * bound and top <= bound, "|A| fills 0...#{bound}")
  try Test.assert(g.fresh.b_zero, "PEFT starts B at zero")
  try Test.assert((try S.floats_of(made.b[0])).all? { |x: Float| x == 0.0 }, "B starts at zero")
  ok()
end

def adapted(weights: Map[String, Tensor.Tensor], targets: List[String]) -> Result[List[String]]
  ok((try Lora.init(weights, Lora.Config(targets: targets), 1, S.REF)).names)
end

def test_targets_match_by_the_end_of_the_name_and_in_any_order() -> Result[Void]
  w = try Tensor.zeros([2, 2], Tensor.Dtype.F32, S.REF)
  weights = {
    "m.layers.0.q_proj.weight": w,
    "m.layers.0.proj.weight": w,
    "m.layers.1.q_proj.weight": w,
    "m.layers.1.q_proj.bias": w,
    "q_proj_extra.weight": w,
  }

  try Test.eq(try adapted(weights, ["q_proj"]), ["m.layers.0.q_proj.weight", "m.layers.1.q_proj.weight"])
  try Test.eq(try adapted(weights, ["proj"]), ["m.layers.0.proj.weight"])
  both = ["m.layers.0.proj.weight", "m.layers.0.q_proj.weight", "m.layers.1.q_proj.weight"]
  try Test.eq(try adapted(weights, ["q_proj", "proj"]), both)
  try Test.eq(try adapted(weights, ["proj", "q_proj", "layers.0.q_proj", "proj"]), both)
  try Test.eq(try adapted(weights, ["layers.1.q_proj"]), ["m.layers.1.q_proj.weight"])
  ok()
end

def test_params_and_with_params_are_inverses() -> Result[Void]
  s = try setup(S.REF)
  params = try Lora.params(s.adapter)
  again = try Lora.params(try Lora.with_params(s.adapter, params))
  try Test.eq(again.size, 2 * s.adapter.names.size)
  for i in 0...params.size
    try Test.eq(try bits(again[i]), try bits(params[i]))
  end

  ok()
end

def test_the_adapted_model_gives_the_output_of_peft() -> Result[Void]
  g = try golden()
  s = try setup(S.REF)
  try Test.eq(s.adapter.names, ADAPTED)
  out = try model(try Lora.applied(s.weights, s.adapter), s.batch.inputs)
  S.expect_close(try S.floats_of(out), g.output, ABS, REL, "the output")
end

def test_merged_weights_are_the_applied_ones_on_every_backend() -> Result[Void]
  g = try golden()
  ref = try setup(S.REF)
  want = try Lora.applied(ref.weights, ref.adapter)
  for backend in try backends()
    s = try setup(backend)
    before = try bits(s.weights["first.weight"])
    with = try Lora.merged(s.weights, s.adapter)
    try Test.eq(with.keys.sort, s.weights.keys.sort)
    for name in ADAPTED
      try Test.eq(Tensor.backend(with[name]), backend)
      try S.expect_close(try S.floats_of(with[name]), try S.floats_of(want[name]), ABS, REL, "#{name} on #{backend}")
    end

    try Test.eq(try bits(with["first.bias"]), try bits(s.weights["first.bias"]))
    try Test.eq(try bits(s.weights["first.weight"]), before)
    out = try Trace.eval([s.batch.inputs], ->(xs: List[Tensor.Tensor]) { model(constants(with), xs[0]) })
    try S.expect_close(try S.floats_of(out), g.output, ABS, REL, "the output of the merged model on #{backend}")
  end

  ok()
end

def constants(weights: Map[String, Tensor.Tensor]) -> Map[String, Tensor.Tensor]
  out: Map[String, Tensor.Tensor] = {}
  for name in weights.keys
    out = out.put(name, Trace.constant(weights[name]))
  end

  out
end

## More adapted weights than one ggml graph answers (512): `merged` adds them in several graphs.
def test_merged_takes_more_weights_than_a_graph_answers() -> Result[Void]
  count = 600
  for backend in try backends()
    weights: Map[String, Tensor.Tensor] = {}
    for i in 0...count
      weights = weights.put(
        "layers.#{i}.proj.weight",
        try S.tensor([1.0, 2.0, 3.0, i.to_f], [2, 2], Tensor.Dtype.F32, backend),
      )
    end

    made = try Lora.init(weights, Lora.Config(targets: ["proj"], r: 1, alpha: 1.0), 1, backend)
    ones = made.b.map { |_b: Tensor.Tensor| S.tensor([1.0, 1.0], [2, 1], Tensor.Dtype.F32, backend).unwrap! }
    with = try Lora.merged(weights, made.with(b: ones))
    try Test.eq(with.size, count)
    for k in [0, 255, 256, 511, 512, count - 1]
      name = made.names[k]
      a = try S.floats_of(made.a[k])
      want = [1.0 + a[0], 2.0 + a[1], 3.0 + a[0], try name.split(".")[1].to_f + a[1]]
      try S.expect_close(try S.floats_of(with[name]), want, ABS, REL, "#{name} on #{backend}")
    end
  end

  ok()
end

## A failure in a later graph of `merged` (here the 300th weight is missing) is the error of the whole call.
def test_merged_fails_as_a_whole_when_a_later_graph_fails() -> Result[Void]
  for backend in try backends()
    weights: Map[String, Tensor.Tensor] = {}
    for i in 0...300
      weights = weights.put("layers.#{i}.proj.weight", try S.tensor([1.0, 2.0, 3.0, 4.0], [2, 2], Tensor.Dtype.F32, backend))
    end

    made = try Lora.init(weights, Lora.Config(targets: ["proj"], r: 1), 1, backend)
    last = made.names[299]
    try S.expect_error(Lora.merged(weights.delete(last), made), last)
  end

  ok()
end

def test_merged_refuses_what_it_cannot_add() -> Result[Void]
  s = try setup(S.REF)
  try Test.eq((try Lora.merged(s.weights, s.adapter.with(names: [], a: [], b: []))).keys.sort, s.weights.keys.sort)
  try Test.assert(!Lora.merged(s.weights.delete("first.weight"), s.adapter).ok?, "a weight the model lacks")
  half = try Tensor.zeros([4, 3], Tensor.Dtype.F16, S.REF)
  try Test.assert(!Lora.merged(s.weights.put("first.weight", half), s.adapter).ok?, "an F16 weight")
  for backend in try S.ggml_backends()
    there = try setup(backend)
    try Test.assert(!Lora.merged(s.weights, there.adapter).ok?, "weights on the reference, adapter on #{backend}")
    try Test.assert(!Lora.merged(there.weights, s.adapter).ok?, "weights on #{backend}, adapter on the reference")
  end

  ok()
end

def test_the_gradients_of_a_and_b_are_those_of_peft() -> Result[Void]
  g = try golden()
  for backend in try backends()
    s = try setup(backend)
    rows = Data.Batch(inputs: Trace.constant(s.batch.inputs), targets: Trace.constant(s.batch.targets))
    both = try Trace.value_and_grad(try Lora.params(s.adapter), ->(ps: List[Tensor.Tensor]) { loss(s, ps, rows) })
    try S.expect_close(try S.floats_of(both.value), [g.loss], ABS, REL, "the loss on #{backend}")
    for i in 0...g.gradients.size
      try S.expect_close(try S.floats_of(both.grads[i]), g.gradients[i], ABS, REL, "gradient #{i} on #{backend}")
    end
  end

  ok()
end

def test_a_fit_of_the_adapter_follows_peft_and_keeps_the_model() -> Result[Void]
  g = try golden()
  for backend in try backends()
    s = try setup(backend)
    before: Map[String, List[String]] = {}
    for name in s.weights.keys
      before = before.put(name, try bits(s.weights[name]))
    end

    rows = try Data.dataset(s.batch.inputs, s.batch.targets)
    fit = try Train.fit(
      ->(ps: List[Tensor.Tensor], b: Data.Batch) { loss(s, ps, b) },
      try Lora.params(s.adapter),
      Data.Loader(dataset: rows, batch_size: 6, shuffle: false),
      Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.05, weight_decay: 0.0)),
      g.history.size,
    )

    try S.expect_close(fit.history, g.history, ABS, REL, "the loss history on #{backend}")
    trained = try Lora.with_params(s.adapter, fit.params)
    got = try Lora.params(trained)
    for i in 0...g.trained.size
      try S.expect_close(try S.floats_of(got[i]), g.trained[i], ABS, REL, "tensor #{i} on #{backend}")
    end

    for name in s.weights.keys
      try Test.eq(try bits(s.weights[name]), before[name])
    end
  end

  Test.assert(g.history[g.history.size - 1] < g.history[0], "the peer lowered the loss")
end

def test_what_does_not_fit_is_an_error() -> Result[Void]
  s = try setup(S.REF)
  try Test.assert(!Lora.init(s.weights, Lora.Config(targets: ["third"]), 1, S.REF).ok?, "a target of no layer")
  try Test.assert(!Lora.init(s.weights, Lora.Config(targets: []), 1, S.REF).ok?, "no target")
  try Test.assert(!Lora.init(s.weights, Lora.Config(targets: ["first"], r: 0), 1, S.REF).ok?, "rank 0")
  dropped = s.adapter.with(config: CONFIG.with(dropout: 0.1))
  try Test.assert(!Lora.init(s.weights, Lora.Config(targets: ["first"], alpha: 0.0), 1, S.REF).ok?, "alpha 0")
  vector = {"norm.weight": try Tensor.zeros([4], Tensor.Dtype.F32, S.REF)}
  try Test.assert(!Lora.init(vector, Lora.Config(targets: ["norm"]), 1, S.REF).ok?, "a rank-1 target")
  half = try Tensor.zeros([2, 2], Tensor.Dtype.F16, S.REF)
  try Test.assert(!Lora.init({"layer.weight": half}, Lora.Config(targets: ["layer"]), 1, S.REF).ok?, "an F16 weight")
  try Test.assert(!Lora.applied(s.weights.put("first.weight", half), s.adapter).ok?, "an F16 weight to apply to")
  try Test.assert(!Lora.params(dropped).ok?, "a dropout above 0 is not trained")
  try Test.assert(Lora.applied(s.weights, dropped).ok?, "and it runs")
  try Test.assert(!Lora.with_params(s.adapter, s.adapter.a).ok?, "half the tensors")
  other = try Lora.init({"first.weight": s.weights["second.weight"]}, Lora.Config(targets: ["first"]), 1, S.REF)
  try Test.assert(!Lora.applied(s.weights, other).ok?, "an adapter of another shape")
  Test.assert(!Lora.applied(s.weights.delete("first.weight"), s.adapter).ok?, "a weight the model lacks")
end
