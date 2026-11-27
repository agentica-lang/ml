## An optimizer state saved and loaded, and a fit that starts from it: 2 epochs, a saved state and 2 more epochs give
## the parameters and the losses of 4 epochs in one run, with a schedule that goes on from its step, for every optimizer.
## Run from `train/`.

import Data from "../data"
import Loss from "../loss"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"
import Write from "pkg/hub/safetensors_write"

INPUTS = [1.0, 2.0, -1.0, 0.5, 0.25, -2.0, 1.5, 1.0]
TARGETS = [0.5, -1.0, 2.0, 1.0]

private

def schedule() -> Optim.Schedule = Optim.Schedule.WarmupLinear(2, 8)

def adam() -> Optim.Optimizer = Optim.adam(lr: 0.1, schedule: schedule())

## Plain SGD (no moments), SGD with momentum, Adam and AdamW with a decay mask.
def optimizers() -> List[Optim.Optimizer]
  [
    Optim.sgd(lr: 0.1, schedule: schedule()),
    Optim.sgd(lr: 0.05, momentum: 0.9, schedule: schedule()),
    adam(),
    Optim.adamw(lr: 0.1, weight_decay: 0.1, decay: [true, false], schedule: schedule()),
  ]
end

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

## A fit of `epochs` epochs of `optimizer`; `saved` is called with the state of the end of the last epoch.
def run_fit(
  backend: Tensor.Backend,
  optimizer: Optim.Optimizer,
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

  Train.fit(&linear, params, try loader(backend), optimizer, epochs, some(on_epoch), start: start)
end

def saving(path: String) -> Optional[Fn[Optim.State, Result[Void]]] = some(->(s: Optim.State) { Optim.save(s, path) })

def same(got: Train.Fit, want: Train.Fit, what: String, tolerance: List[Float]) -> Result[Void]
  try S.expect_close(got.history, want.history, tolerance[0], tolerance[1], "#{what}: history")
  for i in 0...want.params.size
    try S.expect_close(
      try S.floats_of(got.params[i]),
      try S.floats_of(want.params[i]),
      tolerance[0],
      tolerance[1],
      "#{what}: parameter #{i}",
    )
  end

  ok()
end

## Two epochs, the state saved, two more epochs from the loaded state: the run of four epochs, for every optimizer on the
## reference (to the rounding of a moment to F32), ggml and XLA; and the same without the state is another run.
def test_a_resumed_fit_is_the_run_it_resumes() -> Result[Void]
  for backend in try backends()
    tolerance = [0.000001, 0.00001]
    if backend == S.REF
      tolerance = [0.0000001, 0.000001]
    end

    for optimizer in optimizers()
      whole = try run_fit(backend, optimizer, try weights(backend), 4)
      try with_folder(
        ->(dir: String) {
          path = "#{dir}/optim.safetensors"
          first = try run_fit(backend, optimizer, try weights(backend), 2, saved: saving(path))
          start = try Optim.load(optimizer, first.params, path)
          second = try run_fit(backend, optimizer, first.params, 2, start: some(start))
          try same(second, Train.Fit(params: whole.params, history: whole.history.drop(2)), "#{backend}", tolerance)

          forgot = try run_fit(backend, optimizer, first.params, 2)
          Test.assert(
            try S.floats_of(forgot.params[0]) != try S.floats_of(whole.params[0]),
            "a fit without the state gave the same weights on #{backend}",
          )
        },
      )
    end
  end

  ok()
end

## A file written, loaded and written again holds the same bytes, for every optimizer (plain SGD's holds no tensor), and
## the step count comes back.
def test_the_state_round_trips() -> Result[Void]
  for backend in try backends()
    for optimizer in optimizers()
      try with_folder(
        ->(dir: String) {
          one = "#{dir}/one.safetensors"
          fit = try run_fit(backend, optimizer, try weights(backend), 2, saved: saving(one))
          loaded = try Optim.load(optimizer, fit.params, one)
          try Test.eq(loaded.steps, 4)
          try Optim.save(loaded, "#{dir}/two.safetensors")
          try Test.eq(try bytes(one), try bytes("#{dir}/two.safetensors"))
          again = try Optim.load(optimizer, fit.params, "#{dir}/two.safetensors")
          Test.eq(again.steps, 4)
        },
      )
    end
  end

  ok()
end

def bytes(path: String) -> Result[Bytes]
  File.open(path) do |f|
    parts: List[Bytes] = []
    while true
      chunk = try File.read_bytes(f, 1048576)
      break if chunk.empty?
      parts = parts.push(chunk)
    end

    ok(Bytes.concat(parts))
  end
end

## A resume may change the learning rate or the schedule: the step count is kept and the new rate is read at it, so the
## run is not the one with the old rate.
def test_a_resume_may_change_the_learning_rate() -> Result[Void]
  with_folder(
    ->(dir: String) {
      path = "#{dir}/optim.safetensors"
      first = try run_fit(S.REF, adam(), try weights(S.REF), 2, saved: saving(path))
      slower = Optim.adam(lr: 0.05, schedule: schedule())
      start = try Optim.load(slower, first.params, path)
      try Test.eq(start.steps, 4)
      try S.expect_close([Optim.learning_rate(start)], [(0.05 * 4.0) / 6.0], 0.0000001, 0.0, "the rate at step 4")
      same_rate = try run_fit(S.REF, adam(), first.params, 2, start: some(try Optim.load(adam(), first.params, path)))
      other_rate = try run_fit(S.REF, slower, first.params, 2, start: some(start))
      Test.assert(
        try S.floats_of(same_rate.params[0]) != try S.floats_of(other_rate.params[0]),
        "the learning rate of the start did not matter",
      )
    },
  )
end

## A state of another optimizer, other parameters, a bad schedule or decay list, or another dtype is refused with an error
## that says why; the file's own faults too.
def test_a_state_that_does_not_fit_is_an_error() -> Result[Void]
  with_folder(
    ->(dir: String) {
      path = "#{dir}/optim.safetensors"
      fit = try run_fit(S.REF, adam(), try weights(S.REF), 1, saved: saving(path))
      try S.expect_error(Optim.load(Optim.sgd(lr: 0.1, momentum: 0.9), fit.params, path), "adam")
      try S.expect_error(Optim.load(adam(), [fit.params[0]], path), "no moment")
      other = [try S.tensor([0.0, 0.0, 0.0], [3, 1], Tensor.Dtype.F32, S.REF), fit.params[1]]
      try S.expect_error(Optim.load(adam(), other, path), "first.0")
      half = [try Tensor.zeros([2, 1], Tensor.Dtype.F16, S.REF), fit.params[1]]
      try S.expect_error(Optim.load(adam(), half, path), "dtype")
      bad = Optim.adam(lr: 0.1, schedule: Optim.Schedule.WarmupLinear(9, 8))
      try S.expect_error(Optim.load(bad, fit.params, path), "warmup")
      wrong_decay = Optim.adamw(decay: [true], schedule: schedule())
      state = try Optim.init(adam(), fit.params)
      try S.expect_error(Optim.resume(wrong_decay, state, fit.params), "decay")
      try S.expect_error(Optim.resume(bad, state, fit.params), "warmup")
      try S.expect_error(Optim.resume(Optim.sgd(lr: 0.1), state, fit.params), "adam")
      try S.expect_error(Optim.resume(adam(), state, half), "dtype")
      S.expect_error(Optim.resume(adam(), state, [fit.params[0]]), "moments")
    },
  )
end

## A file that is not a saved state: no metadata, a foreign tensor, a step count below 0.
def test_a_file_that_is_not_a_state_is_an_error() -> Result[Void]
  with_folder(
    ->(dir: String) {
      params = try weights(S.REF)
      moments = {
        "first.0": try S.tensor([0.0, 0.0], [2, 1], Tensor.Dtype.F32, S.REF),
        "first.1": try S.tensor([0.0], [1], Tensor.Dtype.F32, S.REF),
        "second.0": try S.tensor([0.0, 0.0], [2, 1], Tensor.Dtype.F32, S.REF),
        "second.1": try S.tensor([0.0], [1], Tensor.Dtype.F32, S.REF),
      }

      try Write.write("#{dir}/bare.safetensors", moments)
      try S.expect_error(Optim.load(adam(), params, "#{dir}/bare.safetensors"), "metadata")
      try Write.write("#{dir}/negative.safetensors", moments, {"steps": "-1", "optimizer": "adam"})
      try S.expect_error(Optim.load(adam(), params, "#{dir}/negative.safetensors"), "step count")
      foreign = moments.put("extra", try S.tensor([0.0], [1], Tensor.Dtype.F32, S.REF))
      try Write.write("#{dir}/foreign.safetensors", foreign, {"steps": "1", "optimizer": "adam"})
      S.expect_error(Optim.load(adam(), params, "#{dir}/foreign.safetensors"), "extra")
    },
  )
end

## A start state that does not fit frees its moments, as the run does, on libtorch and ggml, where a tensor holds native
## memory: the fit is refused and the moments are gone.
def test_a_refused_start_is_freed() -> Result[Void]
  for backend in (try backends()).drop(1)
    start = try Optim.init(adam(), try weights(backend))
    refused = run_fit(backend, Optim.sgd(lr: 0.1), try weights(backend), 1, start: some(start))
    try S.expect_error(refused, "adam")
    gone = Data.Batch(inputs: start.first[0], targets: start.second[0])
    try Test.assert(Data.freed?(gone), "the moments of a refused start are not freed on #{backend}")
    fewer = try Optim.init(adam(), try weights(backend))
    short = Train.fit_batches(
      &linear,
      try weights(backend),
      ->(_e: Int, _i: Int) { error("not called") },
      0,
      adam(),
      1,
      start: some(fewer),
    )

    try S.expect_error(short, "at least one batch")
    try Test.assert(
      Data.freed?(Data.Batch(inputs: fewer.first[0], targets: fewer.second[0])),
      "the moments are not freed on an early error on #{backend}",
    )
  end

  ok()
end
