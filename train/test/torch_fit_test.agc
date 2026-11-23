## The exit criterion of `train/` on libtorch (roadmap, phase 4): the XOR run
## of `fit_test`, made from the same seed, follows the reference's loss curve
## within `F32` drift. Where libtorch is not installed, the test asserts that
## absence instead.

import S from "support"
import Tensor from "pkg/tensor/tensor"

private

def test_the_xor_run_on_libtorch_follows_the_reference() -> Result[Void]
  return absent() if !Tensor.available(S.TORCH)
  fit = try S.xor(S.TORCH)
  try S.expect_close(fit.history, S.XOR_HISTORY, 0.000001, 0.00001, "loss history")
  logits = try S.floats_of(try S.xor_model(fit.params, (try S.xor_batch(S.TORCH)).inputs))
  Test.eq(logits.map { |z| z > 0.0 }, [false, true, true, false])
end

## Without the native module, a tensor on libtorch is an error, not a halt.
def absent() -> Result[Void]
  case S.tensor([1.0], [1], Tensor.Dtype.F32, S.TORCH)
    ok(_) -> error("a torch tensor was made without the native module")
    error(_) -> ok()
  end
end
