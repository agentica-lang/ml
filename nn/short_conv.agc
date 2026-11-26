## The gated short convolution of LFM3, over `[tokens, width]` of one sequence, with the `taps - 1` rows of the
## convolution's input that the next call needs (`nn/DESIGN.md`).

import Delta from "delta"
import Nn from "nn"
import Tensor from "pkg/tensor/tensor"

## `input` projects to three parts of `width` (`out_in`, no bias): `b`, `c` and `x`, in that order, as the checkpoint
## stores `in_proj`. `conv` is the depthwise convolution `[width, taps]` (a checkpoint has `[width, 1, taps]`), `output`
## projects the result back. The layer is `output(c * conv(b * x))`.
struct ShortConv
  input: Nn.Dense
  conv: Tensor.Tensor
  output: Nn.Dense
end

## The answers `[tokens, width]` of `short_conv` and the state after the last token: `[taps - 1, width]`, the last rows
## of the convolution's input (`b * x`), as `Delta.DeltaState.conv`.
struct ShortOut
  out: Tensor.Tensor
  state: Tensor.Tensor
end

## The empty state of `p`: zeros, for the first call.
def short_start(p: ShortConv, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  shape = Tensor.shape(p.conv)
  return error("conv needs [width, taps], got #{shape}") if shape.size != 2
  Tensor.zeros([shape[1] - 1, shape[0]], Tensor.Dtype.F32, backend)
end

## The layer over `x` `[tokens, width]`, after the `state` of the tokens before it. With `padding` (`Delta.padding`) the
## rows of `x` after its `real` ones are padding: their answers are not meaningful and the state is the one after the
## real rows.
def short_conv(
  p: ShortConv,
  x: Tensor.Tensor,
  state: Tensor.Tensor,
  padding: Optional[Delta.Padding] = none,
) -> Result[ShortOut]
  shape = Tensor.shape(x)
  return error("short_conv needs x [tokens, width], got #{shape}") if shape.size != 2
  width = shape[1]
  projected = try Nn.dense(p.input, x)
  if Tensor.shape(projected) != [shape[0], 3 * width]
    return error("short_conv needs an input projection to #{3 * width}, got #{Tensor.shape(projected)}")
  end

  b = try Tensor.slice(projected, [0, 0], [shape[0], width])
  c = try Tensor.slice(projected, [0, width], [shape[0], 2 * width])
  v = try Tensor.slice(projected, [0, 2 * width], [shape[0], 3 * width])
  mixed = try Delta.causal_conv1d(try Tensor.mul(b, v), p.conv, state, padding.map { |pad| pad.keep })
  out = try Nn.dense(p.output, try Tensor.mul(c, mixed.out))
  ok(ShortOut(out: out, state: mixed.state))
end
