## The same model, made from one seed on the reference and on libtorch, gives
## the same output within `F32` rounding. Where libtorch is not installed, the
## test asserts that absence instead (`tensor/README.md`, "CI").

import Init from "../init"
import Nn from "../nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## A block is a few dozen `F32` ops deep; each rounds by about 6e-8, and the
## gaps seen are near 1e-7 relative.
ABS = 0.000001
REL = 0.00001

private

def test_an_mlp_agrees_with_the_reference() -> Result[Void]
  return absent() if !Tensor.available(S.TORCH)
  data = Init.uniform_values(12, 1.0, 21)
  for act in [Nn.Activation.Relu, Nn.Activation.Silu]
    want = try Nn.mlp(try Nn.init_mlp(4, 8, act, 4, S.REF), try S.f32(data, [3, 4]))
    got = try Nn.mlp(try Nn.init_mlp(4, 8, act, 4, S.TORCH), try S.f32(data, [3, 4], S.TORCH))
    try Test.eq(Tensor.shape(got), [3, 4])
    try S.expect_close(try S.floats_of(got), try S.floats_of(want), ABS, REL, "mlp")
  end

  ok()
end

def test_an_encoder_block_agrees_with_the_reference() -> Result[Void]
  return absent() if !Tensor.available(S.TORCH)
  data = Init.uniform_values(24, 1.0, 22)
  want = try Nn.encoder_block(
    try Nn.init_encoder_block(4, 2, 8, Nn.Activation.Silu, 6, S.REF),
    try S.f32(data, [2, 3, 4]),
  )

  got = try on_torch(data)
  try Test.eq(Tensor.shape(got), [2, 3, 4])
  S.expect_close(try S.floats_of(got), try S.floats_of(want), ABS, REL, "encoder_block")
end

## The block made and run in one arena, the way a caller runs a step.
def on_torch(data: List[Float]) -> Result[Tensor.Tensor]
  Native.arena do
    block = try Nn.init_encoder_block(4, 2, 8, Nn.Activation.Silu, 6, S.TORCH)
    Nn.encoder_block(block, try S.f32(data, [2, 3, 4], S.TORCH))
  end
end

## Without the native module, making a layer on libtorch is an error, not a
## halt.
def absent() -> Result[Void]
  case Nn.init_dense(2, 2, 0, S.TORCH)
    ok(_) -> error("a torch layer was made without the native module")
    error(e) -> Test.assert(e.message.include?("backend not available"), "init_dense said \"#{e.message}\"")
  end
end
