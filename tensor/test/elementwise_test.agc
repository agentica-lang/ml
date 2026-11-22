## Elementwise ops against hand-computed values, plus the error paths.

import Support from "support"
import Tensor from "../tensor"

private

def test_add_is_elementwise() -> Result[Void]
  a = try Support.f32([1.0, 2.0, 3.0, 4.0], [2, 2])
  b = try Support.f32([10.0, 20.0, 30.0, 40.0], [2, 2])
  Support.expect_floats(try Tensor.add(a, b), [11.0, 22.0, 33.0, 44.0])
end

def test_sub_is_elementwise() -> Result[Void]
  a = try Support.f32([10.0, 5.0], [2])
  b = try Support.f32([3.0, 7.0], [2])
  Support.expect_floats(try Tensor.sub(a, b), [7.0, -2.0])
end

def test_mul_is_elementwise() -> Result[Void]
  a = try Support.f32([2.0, 3.0], [2])
  b = try Support.f32([4.0, 5.0], [2])
  Support.expect_floats(try Tensor.mul(a, b), [8.0, 15.0])
end

def test_div_is_elementwise() -> Result[Void]
  a = try Support.f32([7.0, 9.0], [2])
  b = try Support.f32([2.0, 3.0], [2])
  Support.expect_floats(try Tensor.div(a, b), [3.5, 3.0])
end

def test_div_by_zero_is_an_error() -> Result[Void]
  a = try Support.f32([1.0, 2.0], [2])
  b = try Support.f32([1.0, 0.0], [2])
  Test.error?(Tensor.div(a, b))
end

def test_add_rejects_two_shapes() -> Result[Void]
  a = try Support.f32([1.0, 2.0], [2])
  b = try Support.f32([1.0, 2.0], [1, 2])
  Test.error?(Tensor.add(a, b))
end

def test_add_rejects_two_dtypes() -> Result[Void]
  a = try Support.f32([1.0], [1])
  b = try Support.i64([1], [1])
  Test.error?(Tensor.add(a, b))
end

def test_add_rejects_int_tensors() -> Result[Void]
  a = try Support.i64([1], [1])
  b = try Support.i64([2], [1])
  Test.error?(Tensor.add(a, b))
end

def test_neg_flips_the_sign() -> Result[Void]
  a = try Support.f32([1.0, -2.5], [2])
  Support.expect_floats(try Tensor.neg(a), [-1.0, 2.5])
end

def test_exp_saturates_at_the_boundary() -> Result[Void]
  big = try Support.floats_of(try Tensor.exp(try Support.f32([708.0], [1])))
  try Test.assert((big[0] - 3.023383144276055e307).abs / 3.023383144276055e307 <= 0.000000001, "exp(708) drifts")
  saturated = try Tensor.to_list(try Tensor.exp(try Support.f32([709.0], [1])))
  Test.eq(saturated, [Tensor.Values.Float(1.0e308 * 10.0)])
end

def test_exp_of_nan_is_an_error() -> Result[Void]
  saturated = try Tensor.exp(try Support.f32([1000.0], [1]))
  nan = try Tensor.sub(saturated, saturated)
  try Test.error?(Tensor.exp(nan))
  try Test.error?(Tensor.tanh(nan))
  Test.error?(Tensor.log(saturated))
end

def test_log_of_a_saturated_exp_is_an_error() -> Result[Void]
  saturated = try Tensor.exp(try Support.f32([1000.0, 0.0], [2]))
  Test.error?(Tensor.log(saturated))
end

def test_sqrt_pow_sigmoid_gelu_reject_non_finite_inputs() -> Result[Void]
  saturated = try Tensor.exp(try Support.f32([1000.0], [1]))
  nan = try Tensor.sub(saturated, saturated)
  try Test.error?(Tensor.sqrt(saturated))
  try Test.error?(Tensor.pow(saturated, try Support.f32([1.0], [1])))
  try Test.error?(Tensor.sigmoid(nan))
  Test.error?(Tensor.gelu_tanh_approx(nan))
end

def test_exp_matches_the_hand_values() -> Result[Void]
  a = try Support.f32([1.0, -1.0, 0.0, 2.0, -3.0], [5])
  Support.expect_floats(
    try Tensor.exp(a),
    [2.718281828459045, 0.36787944117144233, 1.0, 7.38905609893065, 0.049787068367863944],
  )
end

def test_log_matches_the_hand_values() -> Result[Void]
  a = try Support.f32([2.0, 10.0, 0.5], [3])
  Support.expect_floats(try Tensor.log(a), [0.6931471805599453, 2.302585092994046, -0.6931471805599453])
end

def test_log_rejects_zero_and_negatives() -> Result[Void]
  zero = try Support.f32([0.0], [1])
  try Test.error?(Tensor.log(zero))
  neg = try Support.f32([-1.0], [1])
  Test.error?(Tensor.log(neg))
end

def test_sqrt_matches_the_hand_values() -> Result[Void]
  a = try Support.f32([2.0, 0.25, 0.0], [3])
  Support.expect_floats(try Tensor.sqrt(a), [1.4142135623730951, 0.5, 0.0])
end

def test_sqrt_rejects_negatives() -> Result[Void]
  neg = try Support.f32([-4.0], [1])
  Test.error?(Tensor.sqrt(neg))
end

def test_pow_matches_the_hand_values() -> Result[Void]
  base = try Support.f32([2.0, 2.0], [2])
  exponent = try Support.f32([3.0, 0.5], [2])
  Support.expect_floats(try Tensor.pow(base, exponent), [8.0, 1.4142135623730951])
end

def test_pow_rejects_non_positive_bases() -> Result[Void]
  base = try Support.f32([0.0], [1])
  exponent = try Support.f32([2.0], [1])
  Test.error?(Tensor.pow(base, exponent))
end

def test_relu_clamps_at_zero() -> Result[Void]
  a = try Support.f32([-1.5, 0.0, 2.5], [3])
  Support.expect_floats(try Tensor.relu(a), [0.0, 0.0, 2.5])
end

def test_sigmoid_matches_the_hand_values() -> Result[Void]
  a = try Support.f32([0.0, 1.0, -1.0, 2.0], [4])
  Support.expect_floats(try Tensor.sigmoid(a), [0.5, 0.7310585786300049, 0.2689414213699951, 0.8807970779778823])
end

def test_tanh_matches_the_hand_values() -> Result[Void]
  a = try Support.f32([0.0, 1.0, -1.0, 0.5], [4])
  Support.expect_floats(try Tensor.tanh(a), [0.0, 0.7615941559557649, -0.7615941559557649, 0.46211715726000974])
end

def test_gelu_tanh_approx_matches_the_hand_values() -> Result[Void]
  a = try Support.f32([0.0, 1.0, -1.0, 0.5, 2.0], [5])
  Support.expect_floats(
    try Tensor.gelu_tanh_approx(a),
    [0.0, 0.8411919906082768, -0.15880800939172324, 0.34571400982514394, 1.954597694087775],
  )
end

def test_clamp_bounds_both_sides() -> Result[Void]
  a = try Support.f32([-5.0, 0.5, 7.0], [3])
  Support.expect_floats(try Tensor.clamp(a, 0.0, 1.0), [0.0, 0.5, 1.0])
end

def test_clamp_rejects_an_empty_interval() -> Result[Void]
  a = try Support.f32([0.5], [1])
  Test.error?(Tensor.clamp(a, 1.0, 0.0))
end

def test_eq_compares_floats_to_a_bool_mask() -> Result[Void]
  a = try Support.f32([1.0, 2.0], [2])
  b = try Support.f32([1.0, 3.0], [2])
  t = try Tensor.eq(a, b)
  try Test.eq(Tensor.dtype(t), Tensor.Dtype.Bool)
  Test.eq(try Tensor.to_list(t), [Tensor.Values.Bool(true), Tensor.Values.Bool(false)])
end

def test_lt_compares_ints_to_a_bool_mask() -> Result[Void]
  a = try Support.i64([1, 5], [2])
  b = try Support.i64([2, 4], [2])
  t = try Tensor.lt(a, b)
  Test.eq(try Tensor.to_list(t), [Tensor.Values.Bool(true), Tensor.Values.Bool(false)])
end

def test_eq_rejects_bool_tensors() -> Result[Void]
  a = try Support.bools([true], [1])
  b = try Support.bools([false], [1])
  Test.error?(Tensor.eq(a, b))
end

def test_where_picks_per_element() -> Result[Void]
  cond = try Support.bools([true, false, true], [3])
  a = try Support.f32([1.0, 2.0, 3.0], [3])
  b = try Support.f32([10.0, 20.0, 30.0], [3])
  Support.expect_floats(try Tensor.where(cond, a, b), [1.0, 20.0, 3.0])
end

def test_where_rejects_a_non_bool_condition() -> Result[Void]
  cond = try Support.i64([1, 0], [2])
  a = try Support.f32([1.0, 2.0], [2])
  b = try Support.f32([3.0, 4.0], [2])
  Test.error?(Tensor.where(cond, a, b))
end

def test_where_rejects_a_condition_of_another_shape() -> Result[Void]
  cond = try Support.bools([true], [1])
  a = try Support.f32([1.0, 2.0], [2])
  b = try Support.f32([3.0, 4.0], [2])
  Test.error?(Tensor.where(cond, a, b))
end
