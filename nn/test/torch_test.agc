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

def test_a_dense_with_out_in_agrees_with_the_reference() -> Result[Void]
  return absent() if !Tensor.available(S.TORCH)
  x = Init.uniform_values(12, 1.0, 84)
  w = Init.uniform_values(15, 1.0, 85)
  b = Init.uniform_values(5, 1.0, 86)
  want = try Nn.dense(
    Nn.Dense(w: try S.f32(w, [5, 3]), b: some(try S.f32(b, [5])), out_in: true),
    try S.f32(x, [4, 3]),
  )

  got = try Nn.dense(
    Nn.Dense(w: try S.f32(w, [5, 3], S.TORCH), b: some(try S.f32(b, [5], S.TORCH)), out_in: true),
    try S.f32(x, [4, 3], S.TORCH),
  )

  S.expect_close(try S.floats_of(got), try S.floats_of(want), ABS, REL, "out_in")
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

def test_a_decoder_block_agrees_with_the_reference() -> Result[Void]
  return absent() if !Tensor.available(S.TORCH)
  x = Init.uniform_values(24, 1.0, 23)
  memory = Init.uniform_values(40, 1.0, 24)
  block = try Nn.init_decoder_block(4, 2, 8, Nn.Activation.Silu, 7, S.REF)
  want = try Nn.decoder_block(block, try S.f32(x, [2, 3, 4]), try S.f32(memory, [2, 5, 4]))
  got: Result[List[Float]] = Native.arena do
    on = try Nn.init_decoder_block(4, 2, 8, Nn.Activation.Silu, 7, S.TORCH)
    S.floats_of(try Nn.decoder_block(on, try S.f32(x, [2, 3, 4], S.TORCH), try S.f32(memory, [2, 5, 4], S.TORCH)))
  end

  S.expect_close(try got, try S.floats_of(want), ABS, REL, "decoder_block")
end

## A Qwen-like block (grouped-query attention, rope, query and key norms, RMS
## norms, a gated MLP, causal) with random norm scales.
def test_a_qwen_like_block_agrees_with_the_reference() -> Result[Void]
  return absent() if !Tensor.available(S.TORCH)
  data = Init.uniform_values(96, 1.0, 25)
  options = Nn.BlockOptions(
    kv_heads: some(2),
    rope: some(Nn.Rope()),
    qk_norm: true,
    bias: false,
    rms: true,
    gated: true,
    eps: 0.000001,
  )

  start = try Nn.init_encoder_block(16, 4, 24, Nn.Activation.Silu, 8, S.REF, options)
  block = try S.jitter_encoder_block(start, 40)
  want = try Nn.encoder_block(block, try S.f32(data, [2, 3, 16]), none, true)
  got: Result[List[Float]] = Native.arena do
    on = try S.map_encoder_block(block, ->(t: Tensor.Tensor) { S.move(t, S.TORCH) })
    S.floats_of(try Nn.encoder_block(on, try S.f32(data, [2, 3, 16], S.TORCH), none, true))
  end

  S.expect_close(try got, try S.floats_of(want), ABS, REL, "qwen-like block")
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
    error(_) -> ok()
  end
end
