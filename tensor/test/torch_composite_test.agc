## Composite checks: whole blocks of the §3.2 vocabulary through the dispatch,
## computed both ways. A per-op match is not enough — an op can agree on its
## own and still disagree once its output shape feeds the next one.

import Conform from "torch_support"
import Support from "support"
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

## One attention head's scores `q . kᵀ / sqrt(d)`: the scale is widened by
## `broadcast_to` and applied with `div`, which is the shape a transformer
## block takes now that the primitive set has a division on both backends.
def scores(q: Tensor.Tensor, k: Tensor.Tensor, scale: Tensor.Tensor) -> Result[Tensor.Tensor]
  Native.arena do
    transposed = try Tensor.transpose(k)
    qk = try Tensor.matmul(q, transposed)
    Tensor.div(qk, try Tensor.broadcast_to(scale, qk.shape))
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

  ours_scale = try Tensor.full([], scale_value, Tensor.Dtype.F32, Conform.TORCH)
  ours = try scores(try Conform.on_torch(q, [2, 3]), try Conform.on_torch(k, [2, 3]), ours_scale)
  try Test.eq(Tensor.shape(ours), [2, 2])

  theirs_scale = try Tensor.full([], scale_value, Tensor.Dtype.F32, Conform.REF)
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

def test_softmax_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [0.5, -1.0, 2.0, 1.0, 1.0, 1.0, -3.0, 0.25, 4.0]
  ours = try Tensor.softmax(try Conform.on_torch(data, [3, 3]), 1)
  try Test.eq(Tensor.shape(ours), [3, 3])
  Conform.agrees(ours, try Tensor.softmax(try Conform.on_ref(data, [3, 3]), 1), "softmax over the last axis")
end

def test_softmax_over_the_first_axis_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 0.0, -1.0]
  Conform.agrees(
    try Tensor.softmax(try Conform.on_torch(data, [3, 2]), 0),
    try Tensor.softmax(try Conform.on_ref(data, [3, 2]), 0),
    "softmax over axis 0",
  )
end

def test_softmax_of_a_large_shift_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # The max-subtraction is what keeps `exp` inside `F32`: without it the two
  # power-of-ten rows below and their shifted versions saturate to `+inf` and
  # `0.0`. `div` is the op that closes the block.
  data = [30.0, 20.0, 10.0, -30.0, -20.0, -10.0]
  ours = try Tensor.softmax(try Conform.on_torch(data, [2, 3]), 1)
  row_sums = try Tensor.sum(ours, some([1]), false)
  try Conform.expect_floats(row_sums, [1.0, 1.0], "softmax rows sum to one")
  Conform.agrees(ours, try Tensor.softmax(try Conform.on_ref(data, [2, 3]), 1), "softmax of a large shift")
end

def test_layer_norm_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 8.0, 5.0, 7.0, -2.0, 0.5, 0.25, 1.5, 3.0]
  ours = try Tensor.layer_norm(try Conform.on_torch(data, [3, 4]), 0.00001)
  try Test.eq(Tensor.shape(ours), [3, 4])
  Conform.agrees(ours, try Tensor.layer_norm(try Conform.on_ref(data, [3, 4]), 0.00001), "layer_norm")
end

def test_layer_norm_has_zero_mean_on_the_torch_backend() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 8.0, 5.0]
  normalised = try Tensor.layer_norm(try Conform.on_torch(data, [2, 3]), 0.000001)
  try Conform.expect_floats(try Tensor.mean(normalised, some([1]), false), [0.0, 0.0], "zero mean")
  ok()
end

def test_rms_norm_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, -0.5, 4.0, 2.5]
  Conform.agrees(
    try Tensor.rms_norm(try Conform.on_torch(data, [2, 3]), 0.000001),
    try Tensor.rms_norm(try Conform.on_ref(data, [2, 3]), 0.000001),
    "rms_norm",
  )
end

def test_silu_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-2.0, -0.5, 0.0, 0.5, 2.0, 6.0]
  Conform.agrees(
    try Tensor.silu(try Conform.on_torch(data, [2, 3])),
    try Tensor.silu(try Conform.on_ref(data, [2, 3])),
    "silu",
  )
end

def test_linear_with_a_bias_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  x = [1.0, -2.0, 3.0, 0.5, 1.5, -0.5]
  w = [0.5, -1.0, 2.0, 0.25, -0.75, 1.5]
  b = [0.1, -0.2]
  Conform.agrees(
    try Tensor.linear(
      try Conform.on_torch(x, [2, 3]),
      try Conform.on_torch(w, [3, 2]),
      some(try Conform.on_torch(b, [2])),
    ),
    try Tensor.linear(try Conform.on_ref(x, [2, 3]), try Conform.on_ref(w, [3, 2]), some(try Conform.on_ref(b, [2]))),
    "linear with a bias",
  )
end

def test_dot_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  a = [1.0, 2.0, 3.0]
  b = [-1.0, 0.5, 2.0]
  ours = try Tensor.dot(try Conform.on_torch(a, [3]), try Conform.on_torch(b, [3]))
  try Test.eq(Tensor.shape(ours), [])
  Conform.agrees(ours, try Tensor.dot(try Conform.on_ref(a, [3]), try Conform.on_ref(b, [3])), "dot")
end

## `softmax(q . kᵀ / sqrt(d)) . v` for one head, the block every attention
## layer is made of.
def head_context(q: Tensor.Tensor, k: Tensor.Tensor, v: Tensor.Tensor, scale: Tensor.Tensor) -> Result[Tensor.Tensor]
  Native.arena do
    probs = try Tensor.softmax(try scores(q, k, scale), 1)
    Tensor.matmul(probs, v)
  end
end

def test_attention_softmax_then_context_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  q = [1.0, 0.0, -1.0, 0.5, 0.5, 0.0]
  k = [1.0, 1.0, 0.0, 0.0, -1.0, 2.0]
  v = [0.5, -0.5, 1.0, 2.0]
  scale = Tensor.Values.Float(0.5773502691896258)

  ours = try head_context(
    try Conform.on_torch(q, [2, 3]),
    try Conform.on_torch(k, [2, 3]),
    try Conform.on_torch(v, [2, 2]),
    try Tensor.full([], scale, Tensor.Dtype.F32, Conform.TORCH),
  )

  try Test.eq(Tensor.shape(ours), [2, 2])
  Conform.agrees(
    ours,
    try head_context(
      try Conform.on_ref(q, [2, 3]),
      try Conform.on_ref(k, [2, 3]),
      try Conform.on_ref(v, [2, 2]),
      try Tensor.full([], scale, Tensor.Dtype.F32, Conform.REF),
    ),
    "attention context",
  )
end

def test_attention_agrees_with_the_reference_over_grouped_heads_masks_and_the_causal_flag() -> Result[Void]
  return Conform.absent() if !Conform.available()

  for flags in [
    [false, false, false],
    [true, false, false],
    [false, true, false],
    [false, false, true],
    [true, false, true],
  ]
    ours = try attend_on(Conform.TORCH, flags)
    try Test.eq(Tensor.shape(ours), [2, 4, 3, 3])
    try Conform.agrees(ours, try attend_on(Conform.REF, flags), "attention #{flags}")
  end

  ok()
end

def test_permute_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  x = Support.ramp(6, 2 * 3 * 5 * 6)
  ours = try Tensor.permute(try Tensor.floats(x, [2, 3, 5, 6], Conform.TORCH), [3, 1, 0, 2])
  try Test.eq(Tensor.shape(ours), [6, 3, 2, 5])
  Conform.agrees(ours, try Tensor.permute(try Support.f32(x, [2, 3, 5, 6]), [3, 1, 0, 2]), "permute")
end

def test_rope_agrees_with_the_reference_in_both_layouts() -> Result[Void]
  return Conform.absent() if !Conform.available()

  for neox in [true, false]
    for inverse in [false, true]
      for dims in [6, 4, 2]
        try rope_agrees(neox, inverse, dims)
      end
    end
  end

  ok()
end

def test_multi_head_attention_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  q = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
  k = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
  v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  ours = try Tensor.multi_head_attention(
    try Conform.on_torch(q, [1, 2, 4]),
    try Conform.on_torch(k, [1, 2, 4]),
    try Conform.on_torch(v, [1, 2, 4]),
    2,
    none,
  )

  try Test.eq(Tensor.shape(ours), [1, 2, 4])
  Conform.agrees(
    ours,
    try Tensor.multi_head_attention(
      try Conform.on_ref(q, [1, 2, 4]),
      try Conform.on_ref(k, [1, 2, 4]),
      try Conform.on_ref(v, [1, 2, 4]),
      2,
      none,
    ),
    "multi_head_attention",
  )
end

def test_multi_head_attention_with_a_mask_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  q = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
  k = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]
  v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  bias = [0.0, -100.0, -100.0, 0.0, 0.0, -100.0, -100.0, 0.0]
  ours = try Tensor.multi_head_attention(
    try Conform.on_torch(q, [1, 2, 4]),
    try Conform.on_torch(k, [1, 2, 4]),
    try Conform.on_torch(v, [1, 2, 4]),
    2,
    some(try Conform.on_torch(bias, [1, 2, 2, 2])),
  )

  try Test.eq(Tensor.shape(ours), [1, 2, 4])
  Conform.agrees(
    ours,
    try Tensor.multi_head_attention(
      try Conform.on_ref(q, [1, 2, 4]),
      try Conform.on_ref(k, [1, 2, 4]),
      try Conform.on_ref(v, [1, 2, 4]),
      2,
      some(try Conform.on_ref(bias, [1, 2, 2, 2])),
    ),
    "masked multi_head_attention",
  )
end

def test_multi_head_attention_over_two_heads_and_two_batches_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # A `[2, 3, 4]` batch: four heads of width one, three keys per batch, so the
  # head split, the `stack`/`concat` merge and the batch axis all run.
  q = [
    1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.25, 0.75, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0,
    1.0, 1.0,
  ]

  v = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
    21.0, 22.0, 23.0, 24.0,
  ]

  ours = try Tensor.multi_head_attention(
    try Conform.on_torch(q, [2, 3, 4]),
    try Conform.on_torch(q, [2, 3, 4]),
    try Conform.on_torch(v, [2, 3, 4]),
    4,
    none,
  )

  try Test.eq(Tensor.shape(ours), [2, 3, 4])
  Conform.agrees(
    ours,
    try Tensor.multi_head_attention(
      try Conform.on_ref(q, [2, 3, 4]),
      try Conform.on_ref(q, [2, 3, 4]),
      try Conform.on_ref(v, [2, 3, 4]),
      4,
      none,
    ),
    "multi_head_attention over two batches",
  )
end

def test_greedy_step_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  logits = [0.5, 2.0, -1.0, 2.0, 0.25]
  ours = try Tensor.greedy_step(try Conform.on_torch(logits, [5]))
  try Test.eq(Tensor.dtype(ours), Tensor.Dtype.I64)
  Conform.agrees(ours, try Tensor.greedy_step(try Conform.on_ref(logits, [5])), "greedy_step")
end

def test_greedy_step_with_a_temperature_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  logits = [0.5, 2.0, -1.0, 2.0, 0.25]
  Conform.agrees(
    try Tensor.greedy_step(try Conform.on_torch(logits, [5]), some(0.5)),
    try Tensor.greedy_step(try Conform.on_ref(logits, [5]), some(0.5)),
    "greedy_step with a temperature",
  )
end

def test_greedy_step_refuses_a_non_positive_temperature_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  logits = [0.5, 2.0]
  try Test.error?(Tensor.greedy_step(try Conform.on_torch(logits, [2]), some(0.0)))
  Test.error?(Tensor.greedy_step(try Conform.on_ref(logits, [2]), some(0.0)))
end

## 1100 queries over 1100 keys are 1 210 000 scores, past `List.repeat`'s cap
## of 1 048 576: the scale goes on the query, so Whisper's 1500 encoder
## positions run. Every key is the same, so each output row is the mean value.
def test_attention_runs_past_a_million_scores() -> Result[Void]
  return Conform.absent() if !Conform.available()

  n = 1100
  q = try Conform.on_torch((0...n).map { |i| (i % 7).to_f }, [1, n, 1])
  k = try Conform.on_torch(List.repeat(1.0, n), [1, n, 1])
  v = try Conform.on_torch((0...n).map { |i| (i % 2).to_f }, [1, n, 1])
  out = try Conform.floats_of(try Tensor.multi_head_attention(q, k, v, 1))
  try Test.eq(out.size, n)
  Test.assert((out[0] - 0.5).abs < 0.0001 and (out[n - 1] - 0.5).abs < 0.0001, "got #{out[0]} and #{out[n - 1]}")
end

## `attention` over grouped heads (`[2, 4, 3, 4]` queries over two kv heads of
## five keys), with the mask `flags` pick: `[per-head, shared, causal]`.
def attend_on(backend: Tensor.Backend, flags: List[Bool]) -> Result[Tensor.Tensor]
  q = try Tensor.floats(Support.ramp(1, 2 * 4 * 3 * 4), [2, 4, 3, 4], backend)
  k = try Tensor.floats(Support.ramp(2, 2 * 2 * 5 * 4), [2, 2, 5, 4], backend)
  v = try Tensor.floats(Support.ramp(3, 2 * 2 * 5 * 3), [2, 2, 5, 3], backend)
  mask: Optional[Tensor.Tensor] = none
  if flags[0]
    mask = some(try Tensor.floats(Support.ramp(4, 2 * 1 * 3 * 5), [2, 1, 3, 5], backend))
  end

  if flags[1]
    mask = some(try Tensor.floats(Support.ramp(5, 3 * 5), [3, 5], backend))
  end

  Tensor.attention(q, k, v, mask, none, flags[2])
end

## `rope` of a `[2, 3, 5, 6]` tensor at positions up to 100, on libtorch and
## on the reference.
def rope_agrees(neox: Bool, inverse: Bool, dims: Int) -> Result[Void]
  x = Support.ramp(7, 2 * 3 * 5 * 6)
  at = [0, 3, 7, 8, 100]
  got = try Tensor.rope(
    try Tensor.floats(x, [2, 3, 5, 6], Conform.TORCH),
    try Conform.on_torch_i64(at, [5]),
    1000.0,
    some(dims),
    neox,
    inverse,
  )

  want = try Tensor.rope(
    try Support.f32(x, [2, 3, 5, 6]),
    try Conform.on_ref_i64(at, [5]),
    1000.0,
    some(dims),
    neox,
    inverse,
  )

  Conform.agrees(got, want, "rope neox #{neox}, inverse #{inverse}, dims #{dims}")
end
