## Composite checks: whole blocks of the §3.2 vocabulary through the dispatch,
## computed both ways. A per-op match is not enough — an op can agree on its
## own and still disagree once its output shape feeds the next one.

import Conform from "torch_support"
import Tensor from "../tensor"

private

## `linear(x, w1) + b1` -> `relu` -> `linear(h, w2) + b2`, in one arena for the
## whole block: the intermediates die at the end of the step.
def mlp_block(
  x: Tensor.Tensor,
  w1: Tensor.Tensor,
  b1: Tensor.Tensor,
  w2: Tensor.Tensor,
  b2: Tensor.Tensor,
) -> Result[Tensor.Tensor]
  Native.arena do
    xw = try Tensor.matmul(x, w1)
    biased = try Tensor.add(xw, b1)
    hidden = try Tensor.relu(biased)
    hw = try Tensor.matmul(hidden, w2)
    Tensor.add(hw, b2)
  end
end

## One attention head's scores `q . kᵀ`, scaled by `scale` (a `full` tensor:
## the primitive set has no `div` yet, and a per-element scale is what `mul`
## takes).
def scores(q: Tensor.Tensor, k: Tensor.Tensor, scale: Tensor.Tensor) -> Result[Tensor.Tensor]
  Native.arena do
    transposed = try Tensor.transpose(k)
    qk = try Tensor.matmul(q, transposed)
    Tensor.mul(qk, scale)
  end
end

def test_a_two_layer_mlp_block_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  x = [1.0, -2.0, 3.0, 0.5, 1.5, -0.5]
  # `[2, 4]` and `[2, 2]` biases: no implicit broadcasting
  # (`tensor/DESIGN.md` §2), the way a caller would pass
  # `broadcast_to(b1, [2, 4])`.
  w1 = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.125, -0.5, 0.75, 1.0, -1.25, 0.5]
  b1 = [0.1, -0.2, 0.3, 0.0, 0.1, -0.2, 0.3, 0.0]
  w2 = [1.0, -1.0, 0.5, 0.5, -0.5, 2.0, 1.5, -1.5]
  b2 = [0.25, -0.25, 0.25, -0.25]

  ours = try mlp_block(
    try Conform.on_torch(x, [2, 3]),
    try Conform.on_torch(w1, [3, 4]),
    try Conform.on_torch(b1, [2, 4]),
    try Conform.on_torch(w2, [4, 2]),
    try Conform.on_torch(b2, [2, 2]),
  )

  try Test.eq(Tensor.shape(ours), [2, 2])
  Conform.agrees(
    ours,
    try mlp_block(
      try Conform.on_ref(x, [2, 3]),
      try Conform.on_ref(w1, [3, 4]),
      try Conform.on_ref(b1, [2, 4]),
      try Conform.on_ref(w2, [4, 2]),
      try Conform.on_ref(b2, [2, 2]),
    ),
    "mlp block",
  )
end

def test_attention_scores_agree_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # Two 3-wide keys and two 3-wide queries, scaled by 1/sqrt(3).
  q = [1.0, 0.0, -1.0, 0.5, 0.5, 0.0]
  k = [1.0, 1.0, 0.0, 0.0, -1.0, 2.0]
  scale_value = Tensor.Values.Float(0.5773502691896258)

  ours_scale = try Tensor.full([2, 2], scale_value, Tensor.Dtype.F32, Conform.TORCH)
  ours = try scores(try Conform.on_torch(q, [2, 3]), try Conform.on_torch(k, [2, 3]), ours_scale)
  try Test.eq(Tensor.shape(ours), [2, 2])

  theirs_scale = try Tensor.full([2, 2], scale_value, Tensor.Dtype.F32, Conform.REF)
  Conform.agrees(
    ours,
    try scores(try Conform.on_ref(q, [2, 3]), try Conform.on_ref(k, [2, 3]), theirs_scale),
    "attention scores",
  )
end

def test_a_chain_that_breaks_midway_is_an_error() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # The second matmul cannot work: the first one's inner dimension rules it
  # out, so the block reports an error rather than a broadcast result.
  square = try Conform.on_torch([1.0, 2.0, 3.0, 4.0], [2, 2])
  wide = try Conform.on_torch([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  Test.error?(mlp_block(square, wide, square, square, square))
end
