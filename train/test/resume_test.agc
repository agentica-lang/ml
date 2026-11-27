## An optimizer state saved and loaded, and a fit that starts from it: 2 epochs, a saved state and 2 more epochs give
## the parameters and the losses of 4 epochs in one run, with a schedule that goes on from its step. Run from `train/`.

import Data from "../data"
import Loss from "../loss"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"

INPUTS = [1.0, 2.0, -1.0, 0.5, 0.25, -2.0, 1.5, 1.0]
TARGETS = [0.5, -1.0, 2.0, 1.0]

private

def adam() -> Optim.Optimizer = Optim.adam(lr: 0.1, schedule: Optim.Schedule.WarmupLinear(2, 8))

def weights(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([
    try S.tensor([0.3, -0.2], [2, 1], Tensor.Dtype.F32, backend),
    try S.tensor([0.1], [1], Tensor.Dtype.F32, backend),
  ])
end

def linear(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.mse(try Tensor.linear(b.inputs, ps[0], some(ps[1])), b.targets)
end

def loader(backend: Tensor.Backend) -> Result[Data.Loader]
  rows = Data.dataset(
    try S.tensor(INPUTS, [4, 2], Tensor.Dtype.F32, backend),
    try S.tensor(TARGETS, [4, 1], Tensor.Dtype.F32, backend),
  )

  ok(Data.Loader(dataset: try rows, batch_size: 2, shuffle: false))
end

def backends() -> Result[List[Tensor.Backend]] = ok([S.REF] + try S.compiled_backends())

def with_folder(body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-train-resume-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = body.call(dir)
  _ = Path.remove(dir, 5000)
  result
end

## A fit of `epochs` epochs; `saved` is called with the state of the end of the last epoch.
def run_fit(
  backend: Tensor.Backend,
  params: List[Tensor.Tensor],
  epochs: Int,
  start: Optional[Optim.State] = none,
  saved: Optional[Fn[Optim.State, Result[Void]]] = none,
) -> Result[Train.Fit]
  on_epoch = ->(e: Train.Epoch) {
    return ok(Train.Control.Continue) if e.epoch != epochs

    case saved
      some(f) -> try f.call(e.state)
      none -> nil
    end

    ok(Train.Control.Continue)
  }

  Train.fit(&linear, params, try loader(backend), adam(), epochs, some(on_epoch), start: start)
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

## Two epochs, the state saved, two more epochs from it: the run of four epochs, on the reference and on ggml.
def test_a_resumed_fit_is_the_run_it_resumes() -> Result[Void]
  for backend in try backends()
    whole = try run_fit(backend, try weights(backend), 4)
    try with_folder(
      ->(dir: String) {
        path = "#{dir}/optim.safetensors"
        first = try run_fit(backend, try weights(backend), 2, saved: some(->(s: Optim.State) { Optim.save(s, path) }))
        second = try run_fit(backend, first.params, 2, start: some(try Optim.load(adam(), first.params, path)))
        try same(second, Train.Fit(params: whole.params, history: whole.history.drop(2)), "resumed on #{backend}")

        forgot = try run_fit(backend, first.params, 2)
        Test.assert(
          try S.floats_of(forgot.params[0]) != try S.floats_of(whole.params[0]),
          "a fit without the state gave the same weights on #{backend}",
        )
      },
    )
  end

  ok()
end

## The state of a file is the state that was saved: its step count and its moments.
def test_the_state_round_trips() -> Result[Void]
  for backend in try backends()
    moments: Channel[List[Float]] = Channel.new(2)
    try with_folder(
      ->(dir: String) {
        path = "#{dir}/optim.safetensors"
        saving = ->(s: Optim.State) { record(s, moments, path) }

        fit = try run_fit(backend, try weights(backend), 2, saved: some(saving))
        loaded = try Optim.load(adam(), fit.params, path)
        try Test.eq(loaded.steps, 4)
        try S.expect_close(
          try S.floats_of(loaded.first[0]),
          try moments.recv(0),
          0.000001,
          0.00001,
          "first moment on #{backend}",
        )

        S.expect_close(
          try S.floats_of(loaded.second[1]),
          try moments.recv(0),
          0.000001,
          0.00001,
          "second moment on #{backend}",
        )
      },
    )
  end

  ok()
end

## The first moment of the first parameter and the second of the second go to `moments`, and the state to `path`.
def record(s: Optim.State, moments: Channel[List[Float]], path: String) -> Result[Void]
  _ = moments.send(try S.floats_of(s.first[0]), 0)
  _ = moments.send(try S.floats_of(s.second[1]), 0)
  Optim.save(s, path)
end

## A state of another optimizer, or of other parameters, is refused with an error that says why.
def test_a_state_that_does_not_fit_is_an_error() -> Result[Void]
  with_folder(
    ->(dir: String) {
      path = "#{dir}/optim.safetensors"
      fit = try run_fit(S.REF, try weights(S.REF), 1, saved: some(->(s: Optim.State) { Optim.save(s, path) }))
      try S.expect_error(Optim.load(Optim.sgd(lr: 0.1, momentum: 0.9), fit.params, path), "adam")
      try S.expect_error(Optim.load(adam(), [fit.params[0]], path), "4 moments")
      other = [try S.tensor([0.0, 0.0, 0.0], [3, 1], Tensor.Dtype.F32, S.REF), fit.params[1]]
      try S.expect_error(Optim.load(adam(), other, path), "first.0")
      plain = try Optim.init(Optim.sgd(lr: 0.1), fit.params)
      S.expect_error(run_fit(S.REF, fit.params, 1, start: some(plain)), "adam")
    },
  )
end
