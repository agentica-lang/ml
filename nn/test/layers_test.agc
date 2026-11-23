## Every layer on the reference backend against the naive model of
## `support.agc` or values worked out by hand.

import Init from "../init"
import Nn from "../nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"

## Gaps the doubles of the reference leave against the naive model.
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

  case Init.uniform([2], -1.0, 0, S.REF)
    ok(_) -> error("a negative limit made a tensor")
    error(_) -> ok()
  end
end

def test_dense_matches_the_naive_product_on_rank_2_and_3() -> Result[Void]
  p = try Nn.init_dense(3, 2, 1, S.REF)
  p = p.with(b: some(try S.f32([0.5, -0.25], [2])))
  data = [1.0, -2.0, 0.5, 0.0, 3.0, -1.0]
  want = try S.dense(p, [data.take(3), data.drop(3)])
  flat = want.flat_map { |row| row }
  try S.expect_close(try S.floats_of(try Nn.dense(p, try S.f32(data, [2, 3]))), flat, ABS, REL, "rank 2")
  y = try Nn.dense(p, try S.f32(data, [1, 2, 3]))
  try Test.eq(Tensor.shape(y), [1, 2, 2])
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

def test_activations_match_their_values_by_hand() -> Result[Void]
  x = try S.f32([1.0, -2.0, 0.0], [3])
  try Test.eq(try S.floats_of(try Nn.activate(Nn.Activation.Relu, x)), [1.0, 0.0, 0.0])
  # silu(x) = x * sigmoid(x); gelu is the tanh approximation.
  silu = [0.7310585786300049, -0.2384058440442351, 0.0]
  try S.expect_close(try S.floats_of(try Nn.activate(Nn.Activation.Silu, x)), silu, ABS, REL, "silu")
  gelu = [0.8411919906082768, -0.04540230591222494, 0.0]
  S.expect_close(try S.floats_of(try Nn.activate(Nn.Activation.Gelu, x)), gelu, ABS, REL, "gelu")
end

def test_self_attention_matches_the_naive_heads() -> Result[Void]
  p = try Nn.init_attention(4, 2, 3, S.REF)
  data = Init.uniform_values(12, 1.0, 11)
  x = [data.take(4), data.drop(4).take(4), data.drop(8)]
  want = try S.self_attention(p, x)
  got = try Nn.self_attention(p, try S.f32(data, [1, 3, 4]))
  try Test.eq(Tensor.shape(got), [1, 3, 4])
  S.expect_close(try S.floats_of(got), want[0] + want[1] + want[2], ABS, REL, "self_attention")
end

def test_mlp_and_encoder_block_match_the_naive_model() -> Result[Void]
  data = Init.uniform_values(12, 1.0, 5)
  x = [data.take(4), data.drop(4).take(4), data.drop(8)]
  for act in [Nn.Activation.Relu, Nn.Activation.Silu]
    m = try Nn.init_mlp(4, 8, act, 9, S.REF)
    want = try S.mlp(m, x)
    got = try Nn.mlp(m, try S.f32(data, [3, 4]))
    try S.expect_close(try S.floats_of(got), want[0] + want[1] + want[2], ABS, REL, "mlp")
    block = try Nn.init_encoder_block(4, 2, 8, act, 13, S.REF)
    want_block = try S.encoder_block(block, x)
    got_block = try Nn.encoder_block(block, try S.f32(data, [1, 3, 4]))
    try S.expect_close(try S.floats_of(got_block), want_block[0] + want_block[1] + want_block[2], ABS, REL, "encoder")
  end

  ok()
end
