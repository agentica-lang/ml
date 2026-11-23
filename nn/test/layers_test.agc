## Every layer on the reference backend against the naive model of
## `support.agc` or values worked out by hand.

import Init from "../init"
import Nn from "../nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## The reference computes in doubles, as the naive model does, so only the
## order of the sums separates them.
ABS = 0.000000001
REL = 0.000000001

private

def test_init_is_deterministic_and_bounded() -> Result[Void]
  a = try S.floats_of(try Init.glorot_uniform([4, 6], 7, S.REF))
  try Test.eq(a, try S.floats_of(try Init.glorot_uniform([4, 6], 7, S.REF)))
  try Test.assert(
    a != try S.floats_of(try Init.glorot_uniform([4, 6], 8, S.REF)),
    "seeds 7 and 8 drew the same weights",
  )

  limit = Math.sqrt(0.6)
  try Test.assert(a.all? { |v| v.abs <= limit }, "a weight is past the Glorot limit #{limit}: #{a}")
  try Test.assert(a.any? { |v| v < 0.0 } and a.any? { |v| v > 0.0 }, "the weights have one sign: #{a}")
  attn = try Nn.init_attention(4, 2, 7, S.REF)
  try Test.assert(
    try S.floats_of(attn.query.w) != try S.floats_of(attn.key.w),
    "two projections of one layer drew the same weights",
  )

  # Seed 0 starts at 48271 and steps to 48271^2 mod (2^31 - 1) = 182605794.
  try Test.assert((Init.uniform_values(1, 1.0, 0)[0] - -0.8299351017130236).abs <= 0.000000000001, "seed 0 moved")
  case Init.uniform([2], -1.0, 0, S.REF)
    ok(_) -> error("a negative limit made a tensor")
    error(_) -> ok()
  end
end

def test_dense_matches_the_naive_product_on_rank_2_and_3() -> Result[Void]
  p = try Nn.init_dense(3, 2, 1, S.REF)
  p = p.with(b: some(try S.f32([0.5, -0.25], [2])))
  data = [1.0, -2.0, 0.5, 0.0, 3.0, -1.0, 0.25, 1.5, -0.75, 2.0, -0.5, 1.0]
  want = try S.dense(p, S.item(data, 0, 4, 3))
  flat = want.flat_map { |row| row }
  try S.expect_close(try S.floats_of(try Nn.dense(p, try S.f32(data, [4, 3]))), flat, ABS, REL, "rank 2")
  y = try Nn.dense(p, try S.f32(data, [2, 2, 3]))
  try Test.eq(Tensor.shape(y), [2, 2, 2])
  S.expect_close(try S.floats_of(y), flat, ABS, REL, "rank 3")
end

def test_layer_norm_scales_and_shifts_the_normalized_rows() -> Result[Void]
  p = Nn.LayerNorm(gamma: try S.f32([1.0, 2.0, 0.5], [3]), beta: try S.f32([0.0, -1.0, 1.0], [3]), eps: 0.00001)
  data = [1.0, 2.0, 3.0, -1.0, 0.0, 4.0]
  want = try S.layer_norm(p, [data.take(3), data.drop(3)])
  got = try Nn.layer_norm(p, try S.f32(data, [2, 3]))
  S.expect_close(try S.floats_of(got), want[0] + want[1], ABS, REL, "layer_norm")
end

def test_embedding_and_positional_add_the_table_rows() -> Result[Void]
  table = Nn.Embedding(table: try S.f32([0.0, 1.0, 10.0, 11.0, 20.0, 21.0], [3, 2]))
  ids = try Tensor.from_list([Tensor.Values.Int(2), Tensor.Values.Int(0)], [2], Tensor.Dtype.I64, S.REF)
  rows = try Nn.embedding(table, ids)
  try Test.eq(try S.floats_of(rows), [20.0, 21.0, 0.0, 1.0])
  try Test.eq(try S.floats_of(try Nn.positional(table, rows)), [20.0, 22.0, 10.0, 12.0])
  batched = try Nn.positional(table, try Tensor.reshape(rows, [1, 2, 2]))
  try Test.eq(try S.floats_of(batched), [20.0, 22.0, 10.0, 12.0])
  case Nn.positional(table, try S.f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [4, 2]))
    ok(_) -> error("positional ran past the end of its table")
    error(_) -> ok()
  end
end

def test_conv1d_adds_the_bias_per_output_channel_when_there_is_one() -> Result[Void]
  weight = try S.f32([1.0, 2.0, -1.0, 0.5, 0.5, 0.5], [2, 1, 3])
  p = Nn.Conv1d(weight: weight, bias: some(try S.f32([10.0, -1.0], [2])), stride: 2, padding: 1)
  # Padded [0, 1, 2, 3, 4, 0]: windows [0, 1, 2] and [2, 3, 4].
  want = [10.0, 14.0, 0.5, 3.5]
  try Test.eq(try S.floats_of(try Nn.conv1d(p, try S.f32([1.0, 2.0, 3.0, 4.0], [1, 4]))), want)
  batched = try Nn.conv1d(p, try S.f32([1.0, 2.0, 3.0, 4.0], [1, 1, 4]))
  try Test.eq(Tensor.shape(batched), [1, 2, 2])
  try Test.eq(try S.floats_of(batched), want)
  bare = try Nn.conv1d(p.with(bias: none), try S.f32([1.0, 2.0, 3.0, 4.0], [1, 4]))
  Test.eq(try S.floats_of(bare), [0.0, 4.0, 1.5, 4.5])
end

def test_activations_match_their_values_by_hand() -> Result[Void]
  x = try S.f32([1.0, -2.0, 0.0], [3])
  try Test.eq(try S.floats_of(try Nn.activate(Nn.Activation.Relu, x)), [1.0, 0.0, 0.0])
  # silu(x) = x * sigmoid(x); gelu is the tanh approximation, the exact one
  # is Python's `x * (1 + math.erf(x / sqrt(2))) / 2`.
  silu = [0.7310585786300049, -0.2384058440442351, 0.0]
  try S.expect_close(try S.floats_of(try Nn.activate(Nn.Activation.Silu, x)), silu, ABS, REL, "silu")
  gelu = [0.8411919906082768, -0.04540230591222494, 0.0]
  try S.expect_close(try S.floats_of(try Nn.activate(Nn.Activation.Gelu, x)), gelu, ABS, REL, "gelu")
  exact = [0.8413447460685429, -0.04550026389635842, 0.0]
  S.expect_close(try S.floats_of(try Nn.activate(Nn.Activation.GeluErf, x)), exact, ABS, REL, "exact gelu")
end

def test_self_attention_matches_the_naive_heads_per_batch_item() -> Result[Void]
  p = try Nn.init_attention(4, 2, 3, S.REF)
  data = Init.uniform_values(24, 1.0, 11)
  got = try Nn.self_attention(p, try S.f32(data, [2, 3, 4]))
  try Test.eq(Tensor.shape(got), [2, 3, 4])
  want: List[Float] = []
  for b in 0...2
    want = want + (try S.self_attention(p, S.item(data, b, 3, 4))).flat_map { |row| row }
  end

  S.expect_close(try S.floats_of(got), want, ABS, REL, "self_attention")
end

def test_masked_cross_attention_matches_the_naive_heads() -> Result[Void]
  p = try Nn.init_attention(4, 2, 17, S.REF)
  queries = Init.uniform_values(16, 1.0, 12)
  memory = Init.uniform_values(24, 1.0, 13)
  # The second query does not see the last key.
  bias = [[0.0, 0.0, 0.0], [0.0, 0.0, -1000000000.0]]
  mask = try S.f32(bias.flat_map { |row| row }, [2, 3])
  got = try Nn.attention(p, try S.f32(queries, [2, 2, 4]), try S.f32(memory, [2, 3, 4]), some(mask))
  try Test.eq(Tensor.shape(got), [2, 2, 4])
  want: List[Float] = []
  for b in 0...2
    one = try S.attention(p, S.item(queries, b, 2, 4), S.item(memory, b, 3, 4), some(bias))
    want = want + one.flat_map { |row| row }
  end

  S.expect_close(try S.floats_of(got), want, ABS, REL, "masked cross attention")
end

def test_mlp_and_encoder_block_match_the_naive_model() -> Result[Void]
  data = Init.uniform_values(24, 1.0, 5)
  for act in [Nn.Activation.Relu, Nn.Activation.Silu]
    m = try Nn.init_mlp(4, 8, act, 9, S.REF)
    want = (try S.mlp(m, S.item(data, 0, 6, 4))).flat_map { |row| row }
    try S.expect_close(try S.floats_of(try Nn.mlp(m, try S.f32(data, [6, 4]))), want, ABS, REL, "mlp")
    block = try Nn.init_encoder_block(4, 2, 8, act, 13, S.REF)
    got = try Nn.encoder_block(block, try S.f32(data, [2, 3, 4]))
    want_block: List[Float] = []
    for b in 0...2
      want_block = want_block + (try S.encoder_block(block, S.item(data, b, 3, 4))).flat_map { |row| row }
    end

    try S.expect_close(try S.floats_of(got), want_block, ABS, REL, "encoder")
  end

  ok()
end

def test_the_causal_mask_blocks_the_keys_after_each_query() -> Result[Void]
  m = Nn.MASKED
  got = try S.floats_of(try Nn.causal_mask(3, Tensor.Dtype.F32, S.REF))
  try Test.eq(got, [0.0, m, m, 0.0, 0.0, m, 0.0, 0.0, 0.0])
  Test.error?(Nn.causal_mask(0, Tensor.Dtype.F32, S.REF))
end

def test_a_decoder_block_over_a_longer_memory_matches_the_naive_model() -> Result[Void]
  x = Init.uniform_values(24, 1.0, 31)
  memory = Init.uniform_values(40, 1.0, 32)
  for act in [Nn.Activation.Relu, Nn.Activation.Silu]
    block = try Nn.init_decoder_block(4, 2, 8, act, 19, S.REF)
    got = try Nn.decoder_block(block, try S.f32(x, [2, 3, 4]), try S.f32(memory, [2, 5, 4]))
    try Test.eq(Tensor.shape(got), [2, 3, 4])
    want: List[Float] = []
    for b in 0...2
      one = try S.decoder_block(block, S.item(x, b, 3, 4), S.item(memory, b, 5, 4))
      want = want + one.flat_map { |row| row }
    end

    try S.expect_close(try S.floats_of(got), want, ABS, REL, "decoder_block")
  end

  Test.error?(
    Nn.decoder_block(
      try Nn.init_decoder_block(4, 2, 8, Nn.Activation.Relu, 19, S.REF),
      try S.f32(x, [6, 4]),
      try S.f32(memory, [10, 4]),
    ),
  )
end

def test_a_later_token_does_not_change_the_earlier_decoder_outputs() -> Result[Void]
  block = try Nn.init_decoder_block(4, 2, 8, Nn.Activation.Relu, 23, S.REF)
  memory = try S.f32(Init.uniform_values(8, 1.0, 34), [1, 2, 4])
  x = Init.uniform_values(12, 1.0, 33)
  moved = x.take(8) + [5.0, -3.0, 2.0, 7.0]
  a = try S.floats_of(try Nn.decoder_block(block, try S.f32(x, [1, 3, 4]), memory))
  b = try S.floats_of(try Nn.decoder_block(block, try S.f32(moved, [1, 3, 4]), memory))
  try Test.eq(b.take(8), a.take(8))
  Test.assert(b.drop(8) != a.drop(8), "the moved token did not change its own output")
end
