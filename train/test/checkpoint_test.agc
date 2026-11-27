## `Loop.checkpoint`: a run writes its weights, its optimizer state and where it was to a folder, atomically, and keeps
## the last few. A checkpoint that fails half way leaves the earlier ones readable. Run from `train/`.

import Data from "../data"
import Loop from "../loop"
import Loss from "../loss"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"
import Write from "pkg/hub/safetensors_write"
import Xla from "pkg/tensor/src/xla"

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

  ok(Data.Loader(dataset: try rows, batch_size: 2, shuffle: false, seed: 7))
end

def backends() -> Result[List[Tensor.Backend]] = ok([S.REF] + try S.compiled_backends())

def with_folder(body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-train-checkpoint-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = body.call(dir)
  _ = Path.remove(dir, 5000)
  result
end

def trained(backend: Tensor.Backend, dir: String, epochs: Int, every: Int, keep: Int) -> Result[Train.Fit]
  (
    Loop.trainer(&linear, try weights(backend), adam())
    |> Loop.checkpoint(dir, every: every, keep: keep)
    |> Loop.run(try loader(backend), epochs: epochs)
  )
end

## Three epochs of two steps, a checkpoint every 2 steps and the last 2 kept: the checkpoints of steps 4 and 6. The one
## of step 4 holds the weights and the state of a run of two epochs, and where it was: epoch 2, its last batch, the
## data seed.
def test_a_run_keeps_its_last_checkpoints() -> Result[Void]
  for backend in try backends()
    try with_folder(
      ->(dir: String) {
        try trained(backend, dir, 3, 2, 2)
        try Test.eq(try Loop.checkpoints(dir), ["step-0000000004", "step-0000000006"])
        like = try weights(backend)
        saved = try Loop.read_checkpoint(dir, "step-0000000004", adam(), like)
        try Test.eq([saved.step, saved.epoch, saved.batch, saved.seed], [4, 2, 1, 7])
        try Test.eq(saved.state.steps, 4)
        two = try Train.fit(&linear, try weights(backend), try loader(backend), adam(), 2)
        for i in 0...2
          try S.expect_close(
            try S.floats_of(saved.params[i]),
            try S.floats_of(two.params[i]),
            0.000001,
            0.00001,
            "weights of the step 4 checkpoint on #{backend}",
          )
        end

        Test.eq(try Loop.checkpoints(dir), ["step-0000000004", "step-0000000006"])
      },
    )
  end

  ok()
end

## A checkpoint at the end of each epoch is named by its step, and says it was written there.
def test_a_checkpoint_at_the_end_of_an_epoch() -> Result[Void]
  with_folder(
    ->(dir: String) {
      r = (
        Loop.trainer(&linear, try weights(S.REF), adam())
        |> Loop.checkpoint(dir, every: 1, event: Loop.Event.EpochCompleted)
        |> Loop.run(try loader(S.REF), epochs: 2)
      )

      try r
      try Test.eq(try Loop.checkpoints(dir), ["step-0000000002", "step-0000000004"])
      saved = try Loop.read_checkpoint(dir, "step-0000000004", adam(), try weights(S.REF))
      try Test.eq(saved.event, "epoch")
      Test.eq([saved.epoch, saved.step], [2, 4])
    },
  )
end

## A checkpoint that fails after everything was written, at the last step (the name is taken by a file), is an error
## that ends the run, leaves the earlier checkpoint readable and no temporary folder behind.
def test_a_checkpoint_that_fails_half_way_leaves_the_previous_one() -> Result[Void]
  with_folder(
    ->(dir: String) {
      try File.write_text("#{dir}/step-0000000004", "in the way")
      try S.expect_error(trained(S.REF, dir, 3, 2, 3), "step-0000000004")
      try Test.eq(try Loop.checkpoints(dir), ["step-0000000002"])
      saved = try Loop.read_checkpoint(dir, "step-0000000002", adam(), try weights(S.REF))
      try Test.eq(saved.step, 2)
      Test.eq(try Dir.list(dir), ["step-0000000002", "step-0000000004"])
    },
  )
end

## What a process killed while writing leaves is a temporary folder with some of the files: it is not a checkpoint, the
## readers do not see it, and the next checkpoint removes it.
def test_a_killed_write_is_not_a_checkpoint() -> Result[Void]
  with_folder(
    ->(dir: String) {
      try trained(S.REF, dir, 1, 2, 3)
      partial = "#{dir}/.tmp-step-0000000004"
      try Dir.mkdir(partial)
      try File.write_text("#{partial}/model.safetensors", "half")
      try Test.eq(try Loop.checkpoints(dir), ["step-0000000002"])
      saved = try Loop.read_checkpoint(dir, "step-0000000002", adam(), try weights(S.REF))
      try Test.eq(saved.step, 2)
      try trained(S.REF, dir, 2, 2, 3)
      Test.eq(try Dir.list(dir), ["step-0000000002", "step-0000000004"])
    },
  )
end

def test_keep_is_one_or_more() -> Result[Void]
  with_folder(
    ->(dir: String) {
      r = trained(S.REF, dir, 1, 1, 0)
      try S.expect_error(r, "keep")
      Test.eq(try Dir.list(dir), [])
    },
  )
end

## A checkpoint that does not load, because the optimizer is another or a moment is of another shape, frees the tensors
## it had read (XLA counts its live buffers).
def test_a_failed_load_frees_what_it_read() -> Result[Void]
  for backend in try S.xla_backends()
    try with_folder(
      ->(dir: String) {
        try trained(backend, dir, 1, 2, 1)
        like = try weights(backend)
        before = Xla.buffers()
        try S.expect_error(
          Loop.read_checkpoint(dir, "step-0000000002", Optim.sgd(lr: 0.1, momentum: 0.9), like),
          "adam",
        )

        try Test.eq(Xla.buffers(), before)
        bad = Optim.adam(lr: 0.1, schedule: Optim.Schedule.WarmupLinear(9, 8))
        try S.expect_error(Loop.read_checkpoint(dir, "step-0000000002", bad, like), "warmup")
        try Test.eq(Xla.buffers(), before)
        moments = {
          "first.0": try S.tensor([0.0, 0.0], [2, 1], Tensor.Dtype.F32, backend),
          "first.1": try S.tensor([0.0], [1], Tensor.Dtype.F32, backend),
          "second.0": try S.tensor([0.0, 0.0], [2, 1], Tensor.Dtype.F32, backend),
          "second.1": try S.tensor([0.0, 0.0], [2], Tensor.Dtype.F32, backend),
        }

        try Write.write("#{dir}/odd.safetensors", moments, {"steps": "1", "optimizer": "adam"})
        for t in moments.values
          Data.free(t)
        end

        counted = Xla.buffers()
        try S.expect_error(Optim.load(adam(), like, "#{dir}/odd.safetensors"), "second.1")
        Test.eq(Xla.buffers(), counted)
      },
    )
  end

  ok()
end
