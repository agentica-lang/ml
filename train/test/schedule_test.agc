## Learning-rate schedules (`Optim.Schedule`), against the PyTorch peer `schedule_golden.py`: the factor of every step of
## `torch.optim.lr_scheduler.LambdaLR` with the Hugging Face lambdas (`fixtures/schedule_golden.json`), and two short
## runs of an MLP trained with a schedule (SGD with a linear warmup, Adam with a cosine one). On ggml the schedule is
## inputs of the step, so it does not compile a graph for each step. Run from `train/`.

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
BASE = 0.1

struct Run
  history: List[Float]
  trained: List[List[Float]]
end

struct Golden
  linear: List[Float]
  cosine: List[Float]
  linear_no_warmup: List[Float]
  cosine_all_warmup: List[Float]
  shapes: List[List[Int]]
  start: List[List[Float]]
  inputs: List[Float]
  targets: List[Float]
  sgd_linear: Run
  adam_cosine: Run
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/schedule_golden.json"), Golden)

## The learning rate of `state` at step `t`.
def rate_at(optimizer: Optim.Optimizer, t: Int) -> Result[Float]
  state = try Optim.init(optimizer, [try S.tensor([0.0], [1])])
  ok(Optim.learning_rate(state.with(steps: t)))
end

def rates(optimizer: Optim.Optimizer, n: Int) -> Result[List[Float]]
  out: List[Float] = []
  for t in 0...n
    out = out.push(try rate_at(optimizer, t))
  end

  ok(out)
end

def test_the_learning_rate_of_a_step_follows_lambdalr() -> Result[Void]
  g = try golden()
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: BASE, schedule: Optim.Schedule.WarmupLinear(2, 10)))
  adam = Optim.Optimizer.Adam(Optim.Adam(lr: BASE, schedule: Optim.Schedule.WarmupCosine(3, 12)))
  try S.expect_close(try rates(sgd, g.linear.size), g.linear, 0.000000000000001, 0.000000000001, "linear")
  try S.expect_close(try rates(adam, g.cosine.size), g.cosine, 0.000000000000001, 0.000000000001, "cosine")
  edge_linear = Optim.Optimizer.Sgd(Optim.Sgd(lr: BASE, schedule: Optim.Schedule.WarmupLinear(0, 4)))
  edge_cosine = Optim.Optimizer.Adam(Optim.Adam(lr: BASE, schedule: Optim.Schedule.WarmupCosine(4, 4)))
  try S.expect_close(
    try rates(edge_linear, g.linear_no_warmup.size),
    g.linear_no_warmup,
    0.000000000000001,
    0.000000000001,
    "no warmup",
  )

  try S.expect_close(
    try rates(edge_cosine, g.cosine_all_warmup.size),
    g.cosine_all_warmup,
    0.000000000000001,
    0.000000000001,
    "all warmup",
  )

  constant = Optim.Optimizer.Sgd(Optim.Sgd(lr: BASE))
  S.expect_close(try rates(constant, 4), [BASE, BASE, BASE, BASE], 0.0, 0.0, "constant")
end

def test_a_schedule_needs_a_warmup_inside_its_total() -> Result[Void]
  p = [try S.tensor([0.0], [1])]
  for schedule in [
    Optim.Schedule.WarmupLinear(5, 3),
    Optim.Schedule.WarmupLinear(-1, 3),
    Optim.Schedule.WarmupCosine(0, 0),
  ]
    try Test.error?(Optim.init(Optim.Optimizer.Sgd(Optim.Sgd(lr: BASE, schedule: schedule)), p))
    try Test.error?(Optim.init(Optim.Optimizer.Adam(Optim.Adam(lr: BASE, schedule: schedule)), p))
  end

  try Optim.init(Optim.Optimizer.Sgd(Optim.Sgd(lr: BASE, schedule: Optim.Schedule.WarmupLinear(0, 1))), p)
  ok()
end

def weights(g: Golden, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for i in 0...g.shapes.size
    out = out.push(try S.tensor(g.start[i], g.shapes[i], Tensor.Dtype.F32, backend))
  end

  ok(out)
end

def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  h = try Tensor.relu(try Tensor.linear(b.inputs, ps[0], some(ps[1])))
  Loss.mse(try Tensor.linear(h, ps[2], some(ps[3])), b.targets)
end

def train(g: Golden, optimizer: Optim.Optimizer, backend: Tensor.Backend) -> Result[Train.Fit]
  rows = try Data.dataset(
    try S.tensor(g.inputs, [8, 3], Tensor.Dtype.F32, backend),
    try S.tensor(g.targets, [8, 2], Tensor.Dtype.F32, backend),
  )

  Train.fit(&loss, try weights(g, backend), Data.Loader(dataset: rows, batch_size: 2, shuffle: false), optimizer, 3)
end

def follows(g: Golden, optimizer: Optim.Optimizer, want: Run, backend: Tensor.Backend) -> Result[Void]
  fit = try train(g, optimizer, backend)
  try S.expect_close(fit.history, want.history, ABS, REL, "loss history on #{backend}")
  for i in 0...want.trained.size
    try S.expect_close(try S.floats_of(fit.params[i]), want.trained[i], ABS, REL, "weight #{i} on #{backend}")
  end

  ok()
end

def sgd_linear() -> Optim.Optimizer
  Optim.Optimizer.Sgd(Optim.Sgd(lr: BASE, schedule: Optim.Schedule.WarmupLinear(2, 10)))
end

def adam_cosine() -> Optim.Optimizer
  Optim.Optimizer.Adam(Optim.Adam(lr: 0.05, schedule: Optim.Schedule.WarmupCosine(3, 12)))
end

def test_a_fit_with_a_schedule_follows_pytorch_on_the_reference() -> Result[Void]
  g = try golden()
  try follows(g, sgd_linear(), g.sgd_linear, S.REF)
  follows(g, adam_cosine(), g.adam_cosine, S.REF)
end

def test_the_same_fits_on_libtorch_and_ggml() -> Result[Void]
  g = try golden()
  backends = try S.ggml_backends()
  if Tensor.available(S.TORCH)
    backends = backends.push(S.TORCH)
  else
    puts("schedule_test: libtorch is not available, its run is skipped")
  end

  for backend in backends
    try follows(g, sgd_linear(), g.sgd_linear, backend)
    try follows(g, adam_cosine(), g.adam_cosine, backend)
  end

  ok()
end

## The learning rate is an input of the step: a run of schedules that starts at 0 (a zero constant lowers differently)
## and changes at every step compiles at most the one graph of the batch shape.
def test_a_schedule_does_not_compile_a_graph_for_each_step() -> Result[Void]
  g = try golden()
  for backend in try S.ggml_backends()
    before = try Lower.compiles()
    try train(g, adam_cosine(), backend)
    try Test.assert(try Lower.compiles() - before <= 1, "the steps of a schedule compiled more than one graph")
  end

  ok()
end
