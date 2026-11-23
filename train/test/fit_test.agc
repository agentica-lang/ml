## The exit criterion of `train/` on the reference (roadmap, phase 4): a relu
## MLP from `nn/` learns XOR. One test per file: the run takes most of the
## file's step budget.

import S from "support"
import Train from "../train"

private

def test_an_mlp_learns_xor() -> Result[Void]
  seen: Channel[Train.Epoch] = Channel.new(S.XOR_EPOCHS)
  fit = try S.xor(S.REF, some(->(e: Train.Epoch) { report(seen, e) }))
  try S.expect_close(fit.history, S.XOR_HISTORY, 0.0, 0.000000001, "loss history")
  try Test.assert(fit.history[S.XOR_EPOCHS - 1] < 0.05, "the loss ends at #{fit.history[S.XOR_EPOCHS - 1]}")
  for i in 0...S.XOR_EPOCHS
    e = try seen.recv(0)
    try Test.eq(e.epoch, i + 1)
    try Test.eq(e.loss, fit.history[i])
  end

  logits = try S.floats_of(try S.xor_model(fit.params, (try S.xor_batch(S.REF)).inputs))
  Test.eq(logits.map { |z| z > 0.0 }, [false, true, true, false])
end

def report(seen: Channel[Train.Epoch], e: Train.Epoch) -> Void
  _ = seen.send(e, 0)
end
