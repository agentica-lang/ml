## The StableHLO programs of `tools/stablehlo_cases.agc`: the emitted text against the snapshots in
## `fixtures/stablehlo/`, and the reference backend against IREE's outputs for those programs (`NAME.golden.json`,
## made by `make iree-goldens`; IREE is not run here). The fit step is a program of every step: the goldens are three
## runs of it, each fed the outputs of the one before, and `fit` must reach the same parameters and losses. Run from
## `train/`.

import Cases from "../tools/stablehlo_cases"
import Data from "../data"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "../train"

FIXTURES = "test/fixtures/stablehlo"
## The programs run in `F32`, the reference in doubles.
ABS = 0.00001
REL = 0.0001

struct Run
  outputs: List[Cases.Arg]
end

struct Golden
  iree: String
  mlir_sha256: String
  args: List[Cases.Arg]
  runs: List[Run]
end

private

def test_the_emitted_text_is_the_snapshot() -> Result[Void]
  for c in try Cases.cases()
    try Test.eq(c.text, try File.read_text("#{FIXTURES}/#{c.name}.mlir"))
  end

  ok()
end

def test_the_mlp_forward_matches_iree() -> Result[Void]
  g = try golden("mlp_forward")
  args = try tensors(g.args)
  out = try Trace.eval(args, ->(xs: List[Tensor.Tensor]) { Cases.logits(xs.drop(1), xs[0], &Tensor.gelu) })
  S.expect_close(try S.floats_of(out), g.runs[0].outputs[0].data, ABS, REL, "logits")
end

## Step 1 and step 3 of `fit`, from the parameters and the batch the program ran on.
def test_the_fit_step_matches_iree_at_steps_one_and_three() -> Result[Void]
  g = try golden("fit_step")
  try Test.eq(g.runs.size, Cases.STEPS)
  one = try fit(g, 1)
  try expect_step(g, 0, one)
  three = try fit(g, Cases.STEPS)
  try expect_step(g, Cases.STEPS - 1, three)
  for i in 0...Cases.STEPS
    try S.expect_close([three.history[i]], g.runs[i].outputs[0].data, ABS, REL, "loss of step #{i + 1}")
  end

  ok()
end

## The inputs after the moments are those of `Optim.step_inputs`: an SGD rate is a rank-0 input, AdamW's two rows of 7
## numbers are rank-1 ones.
def test_the_step_inputs_of_each_optimizer_follow_the_moments() -> Result[Void]
  sgd = try module_of(Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1, momentum: 0.9)))
  try Test.assert(sgd.text.include?("%arg4: tensor<3x5xf32>, %arg5: tensor<5xf32>"), "SGD velocity")
  try Test.assert(sgd.text.include?("%arg8: tensor<f32>, %arg9: tensor<4x3xf32>"), "an SGD rate is rank 0")
  decay = try module_of(Optim.Optimizer.AdamW(Optim.AdamW(lr: 0.1)))
  Test.assert(decay.text.include?("%arg12: tensor<7xf32>, %arg13: tensor<7xf32>, %arg14: tensor<4x3xf32>"), "AdamW rows")
end

# The golden of `name`, if it is the one IREE made for the text committed as `name.mlir`.
def golden(name: String) -> Result[Golden]
  g = try JSON.decode(try File.read_text("#{FIXTURES}/#{name}.golden.json"), Golden)
  text = try File.read_text("#{FIXTURES}/#{name}.mlir")
  if OpenSSL.Digest.sha256_hexdigest(text.bytes) != g.mlir_sha256
    return error("#{name}.golden.json is not IREE's output for #{name}.mlir: run make iree-goldens")
  end

  ok(g)
end

# The fit step of the cases over `optimizer`.
def module_of(optimizer: Optim.Optimizer) -> Result[Trace.Module]
  params = [
    try Tensor.floats(Cases.W1, [3, 5]),
    try Tensor.floats(Cases.B1, [5]),
    try Tensor.floats(Cases.W2_FIT, [5, Cases.CLASSES]),
    try Tensor.floats(Cases.B2_FIT, [Cases.CLASSES]),
  ]

  ids = Cases.LABELS.map { |c| Tensor.Values.Int(c) }
  batch = Data.Batch(
    inputs: try Tensor.floats(Cases.ROWS, [4, 3]),
    targets: try Tensor.from_list(ids, [4], Tensor.Dtype.I32, Tensor.Backend.Reference),
  )

  Train.stablehlo(&Cases.loss, params, batch, optimizer)
end

def tensors(args: List[Cases.Arg]) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for a in args
    out = out.push(try S.tensor(a.data, a.shape, Tensor.Dtype.F32))
  end

  ok(out)
end

## `fit` for `epochs` full-batch epochs from the parameters and the batch of the golden's arguments.
def fit(g: Golden, epochs: Int) -> Result[Train.Fit]
  params = try tensors(g.args.take(4))
  labels = g.args[g.args.size - 1].data.map { |c| Tensor.Values.Int(c.floor) }
  data = try Data.dataset(
    try S.tensor(g.args[g.args.size - 2].data, g.args[g.args.size - 2].shape, Tensor.Dtype.F32),
    try Tensor.from_list(labels, [4], Tensor.Dtype.I32, S.REF),
  )

  Train.fit(&Cases.loss, params, Data.Loader(dataset: data, batch_size: 4, shuffle: false), Cases.optimizer(), epochs)
end

## The parameters `fit` ended with are the ones outputs `1..4` of run `step` hold.
def expect_step(g: Golden, step: Int, fit: Train.Fit) -> Result[Void]
  for i in 0...4
    want = g.runs[step].outputs[i + 1].data
    try S.expect_close(try S.floats_of(fit.params[i]), want, ABS, REL, "parameter #{i} after step #{step + 1}")
  end

  ok()
end
