## `Loop`: a trainer run as a pipe gives what `Train.fit` and `Train.fit_batches` give with the same arguments, on the
## reference and on every ggml backend, and the short optimizer constructors are the structs they name. Run from `train/`.

import Data from "../data"
import Loop from "../loop"
import Loss from "../loss"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"

INPUTS = [1.0, 2.0, -1.0, 0.5, 0.25, -2.0, 1.5, 1.0]
TARGETS = [0.5, -1.0, 2.0, 1.0]

private

def adam() -> Optim.Optimizer = Optim.adam(lr: 0.05)

def weights(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([
    try S.tensor([0.3, -0.2], [2, 1], Tensor.Dtype.F32, backend),
    try S.tensor([0.1], [1], Tensor.Dtype.F32, backend),
  ])
end

def linear(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.mse(try Tensor.linear(b.inputs, ps[0], some(ps[1])), b.targets)
end

def rows(backend: Tensor.Backend) -> Result[Data.Dataset]
  Data.dataset(
    try S.tensor(INPUTS, [4, 2], Tensor.Dtype.F32, backend),
    try S.tensor(TARGETS, [4, 1], Tensor.Dtype.F32, backend),
  )
end

def pair(backend: Tensor.Backend, i: Int) -> Result[Data.Batch]
  ok(
    Data.Batch(
      inputs: try S.tensor(INPUTS.drop(4 * i).take(4), [2, 2], Tensor.Dtype.F32, backend),
      targets: try S.tensor(TARGETS.drop(2 * i).take(2), [2, 1], Tensor.Dtype.F32, backend),
    ),
  )
end

def same(got: Train.Fit, want: Train.Fit, what: String) -> Result[Void]
  try S.expect_close(got.history, want.history, 0.000001, 0.00001, "#{what}: history")
  for i in 0...want.params.size
    try S.expect_close(
      try S.floats_of(got.params[i]),
      try S.floats_of(want.params[i]),
      0.000001,
      0.00001,
      "#{what}: parameter #{i}",
    )
  end

  ok()
end

def backends() -> Result[List[Tensor.Backend]] = ok([S.REF] + try S.compiled_backends())

## `Loop.run` is `Train.fit`: the same loss history and the same parameters.
def test_run_is_fit() -> Result[Void]
  for backend in try backends()
    loader = Data.Loader(dataset: try rows(backend), batch_size: 2, shuffle: false)
    want = try Train.fit(&linear, try weights(backend), loader, adam(), 3)
    got = Loop.trainer(&linear, try weights(backend), adam()) |> try Loop.run(loader, epochs: 3)
    try same(got, want, "run on #{backend}")
  end

  ok()
end

## `Loop.run_batches` is `Train.fit_batches`, with the batches a function makes.
def test_run_batches_is_fit_batches() -> Result[Void]
  for backend in try backends()
    batch = ->(_epoch: Int, i: Int) { pair(backend, i) }
    want = try Train.fit_batches(&linear, try weights(backend), batch, 2, adam(), 2)
    got = Loop.trainer(&linear, try weights(backend), adam()) |> try Loop.run_batches(batch, 2, epochs: 2)
    try same(got, want, "run_batches on #{backend}")
  end

  ok()
end

## The short constructors are the structs with the same knobs, and the knobs they leave out keep their defaults.
def test_the_short_optimizer_constructors_are_the_structs() -> Result[Void]
  decay = [true, false]
  try Test.eq(Optim.sgd(0.1), Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1)))
  try Test.eq(Optim.sgd(0.1, momentum: 0.9), Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1, momentum: 0.9)))
  try Test.eq(Optim.adam(), Optim.Optimizer.Adam(Optim.Adam()))
  try Test.eq(Optim.adam(lr: 0.05), Optim.Optimizer.Adam(Optim.Adam(lr: 0.05)))
  warm = Optim.Schedule.WarmupLinear(2, 10)
  try Test.eq(
    Optim.adamw(lr: 0.0001, weight_decay: 0.0, decay: decay, schedule: warm),
    Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.0001, weight_decay: 0.0, decay: decay, schedule: warm)),
  )

  Test.eq(Optim.adamw(), Optim.Optimizer.AdamW(Optim.AdamW()))
end
