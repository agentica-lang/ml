## Composite checks: whole blocks of the §3.2 vocabulary computed both ways,
## which is where a per-op match is not enough (an op can agree on its own and
## still disagree once its output shape feeds the next one).

import Ref from "pkg/tensor/tensor"
import Support from "support"
import Torch from "../torch"

private

## `linear(x, w1) + b1` -> `relu` -> `linear(h, w2) + b2`, on libtorch, in one
## arena for the whole block: the intermediates die at the end of the step.
def mlp_block(
  x: Torch.Tensor,
  w1: Torch.Tensor,
  b1: Torch.Tensor,
  w2: Torch.Tensor,
  b2: Torch.Tensor,
) -> Result[Torch.Tensor]
  Native.arena do
    xw = try Torch.matmul(Support.CPU, x, w1)
    biased = try Torch.add(Support.CPU, xw, b1)
    hidden = try Torch.relu(Support.CPU, biased)
    hw = try Torch.matmul(Support.CPU, hidden, w2)
    Torch.add(Support.CPU, hw, b2)
  end
end

## The same block over the reference backend, the oracle.
def ref_mlp_block(x: Ref.Tensor, w1: Ref.Tensor, b1: Ref.Tensor, w2: Ref.Tensor, b2: Ref.Tensor) -> Result[Ref.Tensor]
  xw = try Ref.matmul(x, w1)
  biased = try Ref.add(xw, b1)
  hidden = try Ref.relu(biased)
  hw = try Ref.matmul(hidden, w2)
  Ref.add(hw, b2)
end

## One attention head's scores `q . kᵀ`, scaled by `scale` (a `full` tensor:
## M2 has no `div` primitive, and a per-element scale is what `mul` takes).
def torch_scores(q: Torch.Tensor, k: Torch.Tensor, scale: Torch.Tensor) -> Result[Torch.Tensor]
  Native.arena do
    transposed = try Torch.transpose(Support.CPU, k)
    qk = try Torch.matmul(Support.CPU, q, transposed)
    Torch.mul(Support.CPU, qk, scale)
  end
end

## The same scores over the reference backend.
def ref_scores(q: Ref.Tensor, k: Ref.Tensor, scale: Ref.Tensor) -> Result[Ref.Tensor]
  transposed = try Ref.transpose(k)
  qk = try Ref.matmul(q, transposed)
  Ref.mul(qk, scale)
end

def test_a_two_layer_mlp_block_agrees_with_the_reference() -> Result[Void]
  return Support.absent() if !Support.available()

  x = try Support.torch_f32([1.0, -2.0, 3.0, 0.5, 1.5, -0.5], [2, 3])
  w1 = try Support.torch_f32([0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.125, -0.5, 0.75, 1.0, -1.25, 0.5], [3, 4])
  # No implicit broadcasting (`tensor/DESIGN.md` §2): the bias arrives widened,
  # the way a caller would pass `broadcast_to(b1, [2, 4])`.
  b1 = try Support.torch_f32([0.1, -0.2, 0.3, 0.0, 0.1, -0.2, 0.3, 0.0], [2, 4])
  w2 = try Support.torch_f32([1.0, -1.0, 0.5, 0.5, -0.5, 2.0, 1.5, -1.5], [4, 2])
  b2 = try Support.torch_f32([0.25, -0.25, 0.25, -0.25], [2, 2])

  ours = try mlp_block(x, w1, b1, w2, b2)
  try Test.eq(Torch.shape(ours), [2, 2])

  rx = try Support.ref_f32([1.0, -2.0, 3.0, 0.5, 1.5, -0.5], [2, 3])
  rw1 = try Support.ref_f32([0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.125, -0.5, 0.75, 1.0, -1.25, 0.5], [3, 4])
  rb1 = try Support.ref_f32([0.1, -0.2, 0.3, 0.0, 0.1, -0.2, 0.3, 0.0], [2, 4])
  rw2 = try Support.ref_f32([1.0, -1.0, 0.5, 0.5, -0.5, 2.0, 1.5, -1.5], [4, 2])
  rb2 = try Support.ref_f32([0.25, -0.25, 0.25, -0.25], [2, 2])
  Support.agrees(ours, try ref_mlp_block(rx, rw1, rb1, rw2, rb2), "mlp block")
end

def test_attention_scores_agree_with_the_reference() -> Result[Void]
  return Support.absent() if !Support.available()

  # Two 3-wide keys, two 3-wide queries, scaled by 1/sqrt(3).
  q = try Support.torch_f32([1.0, 0.0, -1.0, 0.5, 0.5, 0.0], [2, 3])
  k = try Support.torch_f32([1.0, 1.0, 0.0, 0.0, -1.0, 2.0], [2, 3])
  scale = try Torch.full(Support.CPU, [2, 2], Torch.Values.Float(0.5773502691896258), Torch.Dtype.F32)

  ours = try torch_scores(q, k, scale)
  try Test.eq(Torch.shape(ours), [2, 2])

  rq = try Support.ref_f32([1.0, 0.0, -1.0, 0.5, 0.5, 0.0], [2, 3])
  rk = try Support.ref_f32([1.0, 1.0, 0.0, 0.0, -1.0, 2.0], [2, 3])
  rscale = try Ref.full([2, 2], Ref.Values.Float(0.5773502691896258), Ref.Dtype.F32, Ref.Backend.Reference)
  Support.agrees(ours, try ref_scores(rq, rk, rscale), "attention scores")
end

def test_a_chain_that_breaks_midway_is_an_error() -> Result[Void]
  return Support.absent() if !Support.available()

  # The second matmul cannot work: the first one's inner dimension rules it
  # out, so the block reports an error rather than a broadcast result.
  ours = try Support.torch_f32([1.0, 2.0, 3.0, 4.0], [2, 2])
  other = try Support.torch_f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  Test.error?(mlp_block(ours, other, ours, ours, ours))
end
