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

## The values of `t` as the bits of an `F32`, which is what a checkpoint holds.
def bits(t: Tensor.Tensor) -> Result[List[Int]]
  ok((try S.floats_of(t)).map { |x: Float| Float32.to_bits(x.to_float32).to_i })
end

def rewrite(dir: String, name: String, file: String, text: String) -> Result[Void]
  File.write_text("#{dir}/#{name}/#{file}", text)
end

## Three epochs of two steps, a checkpoint every 2 steps and the last 2 kept: the checkpoints of steps 4 and 6. The one
## of step 4 holds, exactly, the weights of a run of two epochs (as `F32`), its position, the seed and the loss of the
## step; written again it is the same bytes; and a fit that starts from its state goes on at step 5.
def test_a_run_keeps_its_last_checkpoints() -> Result[Void]
  for backend in try backends()
    try with_folder(
      ->(dir: String) {
        losses: Channel[Float] = Channel.new(8)
        keep_loss = ->(info: Loop.Info) {
          _ = losses.send(info.loss, 0)
          ok()
        }

        r = (
          Loop.trainer(&linear, try weights(backend), adam())
          |> Loop.handle(Loop.Event.StepCompleted, keep_loss)
          |> Loop.checkpoint(dir, every: 2, keep: 2)
          |> Loop.run(try loader(backend), epochs: 3)
        )

        try r
        try Test.eq(try Loop.checkpoints(dir), ["step-0000000004", "step-0000000006"])
        saved = try Loop.read_checkpoint(dir, "step-0000000004", adam(), try weights(backend))
        try Test.eq([saved.step, saved.epoch, saved.batch, saved.seed], [4, 2, 1, 7])
        try Test.eq(saved.event, "step")
        try Test.eq(saved.state.steps, 4)
        stepped = [try losses.recv(0), try losses.recv(0), try losses.recv(0), try losses.recv(0)]
        try Test.eq(saved.loss, stepped[3])
        two = try Train.fit(&linear, try weights(backend), try loader(backend), adam(), 2)
        for i in 0...2
          try Test.eq(try bits(saved.params[i]), try bits(two.params[i]))
        end

        folder = "#{dir}/step-0000000004"
        again = "#{dir}/again"
        try Dir.mkdir(again)
        try Write.write("#{again}/model.safetensors", {"param.0": saved.params[0], "param.1": saved.params[1]})
        try Test.eq(try bytes("#{folder}/model.safetensors"), try bytes("#{again}/model.safetensors"))
        try Optim.save(saved.state, "#{again}/optim.safetensors")
        try Test.eq(try bytes("#{folder}/optim.safetensors"), try bytes("#{again}/optim.safetensors"))

        first: Channel[Int] = Channel.new(1)
        note = ->(s: Train.StepInfo) {
          _ = first.send(s.step, 0)
          ok(Train.Control.Continue)
        }

        _ = try Train.fit(
          &linear,
          saved.params,
          try loader(backend),
          adam(),
          1,
          on_step: some(note),
          start: some(saved.state),
        )

        Test.eq(try first.recv(0), 5)
      },
    )
  end

  ok()
end

## A checkpoint every 2 epochs of 4: the steps 4 and 8, written at the end of an epoch, and said so.
def test_a_checkpoint_every_two_epochs() -> Result[Void]
  with_folder(
    ->(dir: String) {
      r = (
        Loop.trainer(&linear, try weights(S.REF), adam())
        |> Loop.checkpoint(dir, every: 2, event: Loop.Event.EpochCompleted)
        |> Loop.run(try loader(S.REF), epochs: 4)
      )

      try r
      try Test.eq(try Loop.checkpoints(dir), ["step-0000000004", "step-0000000008"])
      saved = try Loop.read_checkpoint(dir, "step-0000000008", adam(), try weights(S.REF))
      try Test.eq(saved.event, "epoch")
      Test.eq([saved.epoch, saved.batch, saved.step], [4, 0, 8])
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
## readers do not see it, and the next run removes it at its start. Only `.tmp-step-*` is the run's to remove: another
## `.tmp-*` folder, a `notes` folder and a `step-*` folder that is not `step-` and ten digits are left alone, and never
## listed or pruned.
def test_only_its_own_folders_are_swept_and_pruned() -> Result[Void]
  with_folder(
    ->(dir: String) {
      for name in [
        ".tmp-step-0000000004", ".tmp-step-old-0000000001", ".tmp-keep", "notes", "step-12", "step-000000000x",
      ]
        try Dir.mkdir("#{dir}/#{name}")
        try File.write_text("#{dir}/#{name}/#{"checkpoint.json"}", "{}")
      end

      try Test.eq(try Loop.checkpoints(dir), [])
      try trained(S.REF, dir, 3, 2, 1)
      try Test.eq(try Loop.checkpoints(dir), ["step-0000000006"])
      Test.eq(try Dir.list(dir), [".tmp-keep", "notes", "step-0000000006", "step-000000000x", "step-12"])
    },
  )
end

## One folder serves one run: a second run that reaches a step that has a checkpoint stops there with an error that says
## so, and the checkpoint is as it was.
def test_a_second_run_in_the_same_folder_is_an_error() -> Result[Void]
  with_folder(
    ->(dir: String) {
      try trained(S.REF, dir, 1, 2, 3)
      before = try bytes("#{dir}/step-0000000002/model.safetensors")
      try S.expect_error(trained(S.REF, dir, 1, 2, 3), "step-0000000002 exists in #{dir}")
      Test.eq(try bytes("#{dir}/step-0000000002/model.safetensors"), before)
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

## A checkpoint that cannot be read is an error that names it: a corrupt or incomplete `checkpoint.json`, a different
## number of weights, a weight of another shape, a weight that is not there. A loss of NaN comes back as NaN.
def test_a_checkpoint_that_cannot_be_read_is_an_error() -> Result[Void]
  with_folder(
    ->(dir: String) {
      try trained(S.REF, dir, 1, 2, 1)
      name = "step-0000000002"
      like = try weights(S.REF)
      meta = try File.read_text("#{dir}/#{name}/checkpoint.json")
      try rewrite(dir, name, "checkpoint.json", "not json")
      try S.expect_error(Loop.read_checkpoint(dir, name, adam(), like), "#{dir}/#{name}/checkpoint.json")
      try rewrite(dir, name, "checkpoint.json", "{\"event\": \"step\"}")
      try S.expect_error(Loop.read_checkpoint(dir, name, adam(), like), "#{dir}/#{name}/checkpoint.json")
      try Path.remove("#{dir}/#{name}/checkpoint.json")
      try S.expect_error(Loop.read_checkpoint(dir, name, adam(), like), "checkpoint.json")
      try rewrite(dir, name, "checkpoint.json", meta)
      try S.expect_error(Loop.read_checkpoint(dir, name, adam(), [like[0]]), "2 weights, 1 are needed")
      wide = [try S.tensor([0.0, 0.0, 0.0], [3, 1], Tensor.Dtype.F32, S.REF), like[1]]
      try S.expect_error(Loop.read_checkpoint(dir, name, adam(), wide), "param.0")
      try Write.write("#{dir}/#{name}/model.safetensors", {"param.0": like[0], "param.7": like[1]})

      try S.expect_error(Loop.read_checkpoint(dir, name, adam(), like), "param.1")
      nan = (
        "{\"event\": \"step\", \"epoch\": 1, \"batch\": 0, \"step\": 2, \"seed\": 7, \"loss_bits\": 9221120237041090560}"
      )

      try rewrite(dir, name, "checkpoint.json", nan)
      try Write.write("#{dir}/#{name}/model.safetensors", {"param.0": like[0], "param.1": like[1]})
      saved = try Loop.read_checkpoint(dir, name, adam(), like)
      Test.assert(saved.loss != saved.loss, "the loss is #{saved.loss}")
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
