## The exit criterion of `train/` (roadmap, phase 4): a relu MLP from `nn/`
## learns XOR on the reference, and the same seeded run on libtorch follows the
## reference's loss curve within `F32` drift. Where libtorch is not installed,
## the second test asserts that absence instead. On ggml, where each step is one
## compiled graph, the run follows it within the same drift. `nn` is a test
## dependency, so the model lives here, in a test file.

import Data from "../data"
import Loss from "../loss"
import Nn from "pkg/nn/nn"
import Optim from "../optim"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "../train"

## The XOR run: a 2-8-1 relu MLP from `nn/`, made from seed 1, trained by
## Adam (`lr` 0.2) on the four XOR rows for 20 full-batch epochs with binary
## cross-entropy. `HISTORY` is its loss per epoch on the reference: the golden
## the reference reproduces and libtorch follows. An epoch costs about 100,000
## VM steps on the reference, 70,000 on libtorch.
EPOCHS = 20
HISTORY = [
  0.668955092647425,
  0.6489128214903377,
  0.5862082263575338,
  0.525739794815681,
  0.4731294383849625,
  0.42645841227892767,
  0.34001898287146354,
  0.2814594298613612,
  0.22671094650797502,
  0.18799976020761108,
  0.1469886765196035,
  0.11305588210587045,
  0.08508408201158438,
  0.06389667138419038,
  0.05135472401537906,
  0.04180874658056082,
  0.03518219015236515,
  0.02932017078944909,
  0.024213729307330817,
  0.020488959894194907,
]

private

def test_an_mlp_learns_xor() -> Result[Void]
  seen: Channel[Train.Epoch] = Channel.new(EPOCHS)
  fit = try xor(S.REF, some(->(e: Train.Epoch) { report(seen, e) }))
  try S.expect_close(fit.history, HISTORY, 0.0, 0.0000001, "loss history")
  try Test.assert(fit.history[EPOCHS - 1] < 0.05, "the loss ends at #{fit.history[EPOCHS - 1]}")
  for i in 0...EPOCHS
    e = try seen.recv(0)
    try Test.eq(e.epoch, i + 1)
    try Test.eq(e.loss, fit.history[i])
  end

  classifies(fit, S.REF)
end

def test_the_xor_run_on_libtorch_follows_the_reference() -> Result[Void]
  return absent() if !Tensor.available(S.TORCH)
  fit = try xor(S.TORCH)
  try S.expect_close(fit.history, HISTORY, 0.000001, 0.00001, "loss history")
  classifies(fit, S.TORCH)
end

def test_the_xor_run_on_ggml_and_xla_follows_the_reference() -> Result[Void]
  for backend in try S.compiled_backends()
    fit = try xor(backend)
    try S.expect_close(fit.history, HISTORY, 0.000001, 0.00001, "loss history on #{backend}")
    try classifies(fit, backend)
  end

  ok()
end

## A hundred parameters with Adam are 301 outputs of one ggml graph or XLA program (the loss, the parameters and two
## moments each): the step runs there and follows the reference.
def test_a_step_of_a_hundred_parameters_runs_on_ggml_and_xla() -> Result[Void]
  want = try many(S.REF)
  for backend in try S.compiled_backends()
    got = try many(backend)
    try S.expect_close(got.history, want.history, 0.000001, 0.00001, "loss history on #{backend}")
    for i in [0, 57, 99]
      close = S.expect_close(
        try S.floats_of(got.params[i]),
        try S.floats_of(want.params[i]),
        0.000001,
        0.0001,
        "p#{i}",
      )

      try close
    end
  end

  Test.eq(want.params.size, 100)
end

## Two full-batch Adam epochs on `sum_i mean((x p_i - y)^2)` over a hundred `[2]` parameters.
def many(backend: Tensor.Backend) -> Result[Train.Fit]
  params: List[Tensor.Tensor] = []
  for i in 0...100
    params = params.push(try S.tensor([0.01 * i.to_f, 1.0 - 0.02 * i.to_f], [2], Tensor.Dtype.F32, backend))
  end

  rows = try Data.dataset(
    try S.tensor([1.0, 2.0, -1.0, 0.5], [2, 2], Tensor.Dtype.F32, backend),
    try S.tensor([0.5, -1.0, 2.0, 1.0], [2, 2], Tensor.Dtype.F32, backend),
  )

  Train.fit(
    &many_loss,
    params,
    Data.Loader(dataset: rows, batch_size: 2, shuffle: false),
    Optim.Optimizer.Adam(Optim.Adam(lr: 0.05)),
    2,
  )
end

def many_loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  total = try Loss.mse(try Tensor.mul(b.inputs, try Tensor.broadcast_to(ps[0], [2, 2])), b.targets)
  for p in ps.drop(1)
    total = try Tensor.add(
      total,
      try Loss.mse(try Tensor.mul(b.inputs, try Tensor.broadcast_to(p, [2, 2])), b.targets),
    )
  end

  ok(total)
end

## The trained model, traced so that it runs on ggml too, gives the XOR of
## each row.
def classifies(fit: Train.Fit, backend: Tensor.Backend) -> Result[Void]
  args = fit.params + [(try rows(backend)).inputs]
  logits = try S.floats_of(try Trace.eval(args, ->(xs: List[Tensor.Tensor]) { model(xs.take(4), xs[4]) }))
  Test.eq(logits.map { |z| z > 0.0 }, [false, true, true, false])
end

def report(seen: Channel[Train.Epoch], e: Train.Epoch) -> Void
  _ = seen.send(e, 0)
end

## `on_step` gets the step after each step, counted across epochs (the loader has one batch, so the step is the epoch),
## and an error from it stops the fit at that step, naming the epoch and the batch.
def test_on_step_sees_every_step_and_its_error_stops_the_fit() -> Result[Void]
  seen: Channel[String] = Channel.new(16)
  on_step = ->(s) {
    _ = seen.send("#{s.epoch}/#{s.batch}/#{s.step}", 0)
    return error("stop") if s.step == 3

    ok()
  }

  try S.expect_error(xor(S.REF, on_step: some(on_step)), "epoch 3, batch 0: stop")
  try Test.eq([try seen.recv(0), try seen.recv(0), try seen.recv(0)], ["1/0/1", "2/0/2", "3/0/3"])
  Test.error?(seen.recv(0))
end

## Without the native module, a tensor on libtorch is an error, not a halt.
def absent() -> Result[Void]
  case S.tensor([1.0], [1], Tensor.Dtype.F32, S.TORCH)
    ok(_) -> error("a torch tensor was made without the native module")
    error(_) -> ok()
  end
end

def xor(
  backend: Tensor.Backend,
  on_epoch: Optional[Fn[Train.Epoch, Void]] = none,
  on_step: Optional[Fn[Train.StepInfo, Result[Void]]] = none,
) -> Result[Train.Fit]
  up = try Nn.init_dense(2, 8, 1, backend)
  down = try Nn.init_dense(8, 1, 2, backend)
  params = [up.w, up.b.unwrap!, down.w, down.b.unwrap!]
  adam = Optim.Optimizer.Adam(Optim.Adam(lr: 0.2))
  Train.fit(
    &loss,
    params,
    Data.Loader(dataset: try rows(backend), batch_size: 4, shuffle: false),
    adam,
    EPOCHS,
    on_epoch,
    none,
    on_step,
  )
end

## The four XOR rows, one full batch through the loader's prefetch task.
def rows(backend: Tensor.Backend) -> Result[Data.Dataset]
  Data.dataset(
    try S.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2], Tensor.Dtype.F32, backend),
    try S.tensor([0.0, 1.0, 1.0, 0.0], [4, 1], Tensor.Dtype.F32, backend),
  )
end

## The logits of the MLP whose parameters are `ps`.
def model(ps: List[Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  up = Nn.Dense(w: ps[0], b: some(ps[1]))
  down = Nn.Dense(w: ps[2], b: some(ps[3]))
  Nn.mlp(Nn.Mlp(up: up, down: down, activation: Nn.Activation.Relu), x)
end

def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Loss.binary_cross_entropy(try model(ps, b.inputs), b.targets)
end
