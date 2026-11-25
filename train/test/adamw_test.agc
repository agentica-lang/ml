## `Optim.AdamW` against the PyTorch peer `adamw_golden.py` (`fixtures/adamw_golden.json`): two short runs of an MLP
## with `torch.optim.AdamW`, where the weights decay and the biases do not, one with a constant learning rate and one
## with a linear warmup. On ggml the decay is an input of the step, as the learning rate is. Run from `train/`.

import Data from "../data"
import Loss from "../loss"
import Lower from "pkg/tensor/src/lower"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"

## The reference computes in doubles and the peer in `F32`.
ABS = 0.00001
REL = 0.0001

struct Run
  history: List[Float]
  trained: List[List[Float]]
end

struct Golden
  shapes: List[List[Int]]
  start: List[List[Float]]
  inputs: List[Float]
  targets: List[Float]
  constant: Run
  warmup: Run
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/adamw_golden.json"), Golden)

def adamw(schedule: Optim.Schedule) -> Optim.Optimizer
  Optim.Optimizer.AdamW(
    Optim.AdamW(lr: 0.05, weight_decay: 0.3, decay: [true, false, true, false], schedule: schedule),
  )
end

def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  h = try Tensor.relu(try Tensor.linear(b.inputs, ps[0], some(ps[1])))
  Loss.mse(try Tensor.linear(h, ps[2], some(ps[3])), b.targets)
end

def train(g: Golden, optimizer: Optim.Optimizer, backend: Tensor.Backend) -> Result[Train.Fit]
  weights: List[Tensor.Tensor] = []
  for i in 0...g.shapes.size
    weights = weights.push(try S.tensor(g.start[i], g.shapes[i], Tensor.Dtype.F32, backend))
  end

  rows = try Data.dataset(
    try S.tensor(g.inputs, [8, 3], Tensor.Dtype.F32, backend),
    try S.tensor(g.targets, [8, 2], Tensor.Dtype.F32, backend),
  )

  Train.fit(&loss, weights, Data.Loader(dataset: rows, batch_size: 2, shuffle: false), optimizer, 3)
end

def follows(g: Golden, optimizer: Optim.Optimizer, want: Run, backend: Tensor.Backend) -> Result[Void]
  fit = try train(g, optimizer, backend)
  try S.expect_close(fit.history, want.history, ABS, REL, "loss history on #{backend}")
  for i in 0...want.trained.size
    try S.expect_close(try S.floats_of(fit.params[i]), want.trained[i], ABS, REL, "weight #{i} on #{backend}")
  end

  ok()
end

def test_a_fit_with_adamw_follows_pytorch_on_the_reference() -> Result[Void]
  g = try golden()
  try follows(g, adamw(Optim.Schedule.Constant), g.constant, S.REF)
  follows(g, adamw(Optim.Schedule.WarmupLinear(2, 10)), g.warmup, S.REF)
end

def test_the_same_fits_on_libtorch_ggml_and_xla() -> Result[Void]
  g = try golden()
  backends = try S.compiled_backends()
  if Tensor.available(S.TORCH)
    backends = backends.push(S.TORCH)
  else
    puts("adamw_test: libtorch is not available, its run is skipped")
  end

  for backend in backends
    try follows(g, adamw(Optim.Schedule.Constant), g.constant, backend)
    try follows(g, adamw(Optim.Schedule.WarmupLinear(2, 10)), g.warmup, backend)
  end

  ok()
end

## The decay follows the learning rate, so it is an input of the step too: a warmup compiles at most one graph.
def test_a_decay_that_changes_does_not_compile_a_graph_for_each_step() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends()
    before = try Lower.compiles()
    try train(g, adamw(Optim.Schedule.WarmupLinear(2, 10)), backend)
    try Test.assert(try Lower.compiles() - before <= 1, "the steps of AdamW compiled more than one graph")
  end

  ok()
end
