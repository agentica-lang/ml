## Derived ops: hand-computed values plus checks against the primitives.

import Support from "support"
import Tensor from "../tensor"

private

def test_softmax_matches_the_hand_values() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0], [3])
  Support.expect_floats(try Tensor.softmax(t, 0), [0.09003057317038045, 0.24472847105479764, 0.6652409557748219])
end

def test_softmax_sums_to_one_per_row() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 1.0, 1.0, 1.0], [2, 3])
  s = try Tensor.softmax(t, 1)
  Support.expect_floats(try Tensor.sum(s, some([1]), false), [1.0, 1.0])
end

def test_softmax_matches_a_manual_composition() -> Result[Void]
  t = try Support.f32([0.5, -1.0, 2.0, 0.0], [2, 2])
  got = try Tensor.softmax(t, 1)
  m = try Tensor.max(t, some([1]), true)
  shifted = try Tensor.sub(t, try Tensor.broadcast_to(m, [2, 2]))
  e = try Tensor.exp(shifted)
  total = try Tensor.sum(e, some([1]), true)
  manual = try Tensor.div(e, try Tensor.broadcast_to(total, [2, 2]))
  try Test.eq(Tensor.shape(got), Tensor.shape(manual))
  Support.expect_floats(try Tensor.sub(got, manual), [0.0, 0.0, 0.0, 0.0])
end

def test_softmax_rejects_an_axis_out_of_range() -> Result[Void]
  t = try Support.f32([1.0], [1])
  Test.error?(Tensor.softmax(t, 1))
end

def test_layer_norm_matches_the_hand_values() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0], [3])
  Support.expect_floats(try Tensor.layer_norm(t, 0.00001), [-1.2247356859083902, 0.0, 1.2247356859083902])
end

def test_layer_norm_has_zero_mean_and_unit_variance() -> Result[Void]
  t = try Support.f32([1.0, 4.0, 2.0, 8.0, 5.0, 7.0], [2, 3])
  n = try Tensor.layer_norm(t, 0.000001)
  try Support.expect_floats(try Tensor.mean(n, some([1]), false), [0.0, 0.0])
  squares = try Support.floats_of(try Tensor.mean(try Tensor.mul(n, n), some([1]), false))
  for s in squares
    try Test.assert((s - 1.0).abs <= 0.00001, "unit variance up to eps, got #{s}")
  end

  Test.assert(true, "zero mean and almost-unit variance")
end

def test_layer_norm_rejects_a_non_positive_eps() -> Result[Void]
  t = try Support.f32([1.0], [1])
  Test.error?(Tensor.layer_norm(t, 0.0))
end

def test_rms_norm_matches_the_hand_values() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0], [3])
  Support.expect_floats(try Tensor.rms_norm(t, 0.000001), [0.4629100002887783, 0.9258200005775566, 1.388730000866335])
end

def test_rms_norm_matches_a_manual_composition() -> Result[Void]
  t = try Support.f32([2.0, -1.0, 0.5], [3])
  got = try Tensor.rms_norm(t, 0.000001)
  ms = try Tensor.mean(try Tensor.mul(t, t), some([0]), true)
  denom = try Tensor.sqrt(
    try Tensor.add(
      try Tensor.broadcast_to(ms, [3]),
      try Tensor.full([3], Tensor.Values.Float(0.000001), Tensor.Dtype.F32, Tensor.Backend.Reference),
    ),
  )

  manual = try Tensor.div(t, denom)
  Support.expect_floats(try Tensor.sub(got, manual), [0.0, 0.0, 0.0])
end

def test_gelu_is_the_tanh_approximation() -> Result[Void]
  t = try Support.f32([1.0, -1.0, 0.5], [3])
  got = try Tensor.gelu(t)
  want = try Tensor.gelu_tanh_approx(t)
  Support.expect_floats(try Tensor.sub(got, want), [0.0, 0.0, 0.0])
end

def test_silu_matches_the_hand_values() -> Result[Void]
  t = try Support.f32([1.0, -2.0, 0.0], [3])
  Support.expect_floats(try Tensor.silu(t), [0.7310585786300049, -0.2384058440442351, 0.0])
end

def test_dot_is_the_rank_one_product() -> Result[Void]
  a = try Support.f32([1.0, 2.0, 3.0], [3])
  b = try Support.f32([4.0, 5.0, 6.0], [3])
  c = try Tensor.dot(a, b)
  try Test.eq(Tensor.shape(c), [])
  Support.expect_floats(c, [32.0])
end

def test_dot_rejects_lengths_that_differ() -> Result[Void]
  a = try Support.f32([1.0], [1])
  b = try Support.f32([1.0, 2.0], [2])
  Test.error?(Tensor.dot(a, b))
end

def test_linear_applies_weights_and_bias() -> Result[Void]
  x = try Support.f32([1.0, 2.0, 3.0, 4.0], [2, 2])
  w = try Support.f32([1.0, 0.0, 0.0, 1.0], [2, 2])
  plain = try Tensor.linear(x, w, none)
  try Support.expect_floats(plain, [1.0, 2.0, 3.0, 4.0])
  b = try Support.f32([10.0, 20.0], [2])
  biased = try Tensor.linear(x, w, some(b))
  Support.expect_floats(biased, [11.0, 22.0, 13.0, 24.0])
end

def test_linear_rejects_a_bias_of_another_width() -> Result[Void]
  x = try Support.f32([1.0, 2.0], [1, 2])
  w = try Support.f32([1.0, 0.0, 0.0, 1.0], [2, 2])
  b = try Support.f32([1.0, 2.0, 3.0], [3])
  Test.error?(Tensor.linear(x, w, some(b)))
end

def test_embedding_gathers_rows() -> Result[Void]
  table = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
  ids = try Support.i64([2, 0], [2])
  out = try Tensor.embedding(ids, table)
  try Test.eq(Tensor.shape(out), [2, 2])
  Support.expect_floats(out, [5.0, 6.0, 1.0, 2.0])
end

def test_embedding_rejects_float_ids() -> Result[Void]
  table = try Support.f32([1.0, 2.0], [1, 2])
  ids = try Support.f32([0.0], [1])
  Test.error?(Tensor.embedding(ids, table))
end

def test_greedy_step_takes_the_argmax() -> Result[Void]
  logits = try Support.f32([1.0, 5.0, 3.0], [3])
  tok = try Tensor.greedy_step(logits, none)
  try Test.eq(Tensor.dtype(tok), Tensor.Dtype.I64)
  Test.eq(try Tensor.to_list(tok), [Tensor.Values.Int(1)])
end

def test_greedy_step_scales_by_temperature() -> Result[Void]
  logits = try Support.f32([2.0, 1.0, 0.0], [3])
  tok = try Tensor.greedy_step(logits, some(0.5))
  Test.eq(try Tensor.to_list(tok), [Tensor.Values.Int(0)])
end

def test_greedy_step_rejects_a_denormal_temperature() -> Result[Void]
  logits = try Support.f32([1.0, 2.0], [2])
  case Tensor.greedy_step(logits, some(5.0e-324))
    ok(_) -> Test.assert(false, "a denormal temperature must be an error")
    error(e) -> Test.assert(e.message.include?("reciprocal"), "wrong error: #{e.message}")
  end
end

def test_greedy_step_accepts_the_smallest_finite_reciprocal() -> Result[Void]
  logits = try Support.f32([0.0, 1.0], [2])
  tok = try Tensor.greedy_step(logits, some(1.0e-308))
  Test.eq(try Tensor.to_list(tok), [Tensor.Values.Int(1)])
end

def test_greedy_step_rejects_a_non_finite_temperature() -> Result[Void]
  logits = try Support.f32([2.0, 1.0], [2])
  inf = 1.0e308 * 10.0
  try Test.error?(Tensor.greedy_step(logits, some(inf)))
  case Tensor.greedy_step(logits, some(inf - inf))
    ok(_) -> Test.assert(false, "a NaN temperature must be an error")
    error(e) -> Test.assert(e.message.include?("finite temperature"), "wrong error: #{e.message}")
  end
end

def test_greedy_step_rejects_a_non_positive_temperature() -> Result[Void]
  logits = try Support.f32([1.0, 2.0], [2])
  Test.error?(Tensor.greedy_step(logits, some(0.0)))
end

def test_dropout_is_the_identity_in_inference() -> Result[Void]
  t = try Support.f32([1.0, -2.0], [2])
  u = try Tensor.dropout(t, 0.5)
  Test.eq(try Tensor.to_list(u), try Tensor.to_list(t))
end

def test_dropout_rejects_a_probability_outside_zero_to_one() -> Result[Void]
  t = try Support.f32([1.0], [1])
  try Test.error?(Tensor.dropout(t, 1.5))
  Test.error?(Tensor.dropout(t, -0.1))
end
