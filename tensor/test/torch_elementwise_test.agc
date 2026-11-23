## Elementwise ops through the dispatch: the same call on both backends,
## compared, plus the error paths an eager backend would compute through
## (libtorch broadcasts where the reference does not) and the one place the two
## backends legitimately disagree inside `F32`.

import Conform from "torch_support"
import Tensor from "../tensor"

private

def test_add_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0]
  other = [10.0, 20.0, 30.0, 40.0]
  Conform.agrees(
    try Tensor.add(try Conform.on_torch(data, [2, 2]), try Conform.on_torch(other, [2, 2])),
    try Tensor.add(try Conform.on_ref(data, [2, 2]), try Conform.on_ref(other, [2, 2])),
    "add",
  )
end

def test_mul_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [0.5, -2.0, 3.25]
  other = [4.0, 8.0, -2.0]
  Conform.agrees(
    try Tensor.mul(try Conform.on_torch(data, [3]), try Conform.on_torch(other, [3])),
    try Tensor.mul(try Conform.on_ref(data, [3]), try Conform.on_ref(other, [3])),
    "mul",
  )
end

def test_relu_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-2.0, 0.0, 1.5]
  Conform.agrees(
    try Tensor.relu(try Conform.on_torch(data, [3])),
    try Tensor.relu(try Conform.on_ref(data, [3])),
    "relu",
  )
end

def test_exp_agrees_with_the_reference_inside_the_float32_range() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-8.0, -1.0, 0.0, 0.5, 1.0, 7.0]
  Conform.agrees(
    try Tensor.exp(try Conform.on_torch(data, [2, 3])),
    try Tensor.exp(try Conform.on_ref(data, [2, 3])),
    "exp",
  )
end

def test_exp_saturation_is_a_documented_difference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # `F32` overflows above ~88.7 and the reference saturates at 1e308 in
  # doubles, so the same finite input gives `+inf` on one side and a large
  # finite value on the other. The conformance comparison must not call that
  # agreement: this is the check that keeps `Conform.close?` honest at
  # infinity (a tolerance would say `inf <= inf`).
  ours = try Tensor.exp(try Conform.on_torch([100.0], [1]))
  theirs = try Tensor.exp(try Conform.on_ref([100.0], [1]))
  got = try Conform.floats_of(ours)
  want = try Conform.floats_of(theirs)
  try Test.assert(!Conform.finite?(got[0]), "exp(100) on torch must saturate to +inf")
  try Test.assert(Conform.finite?(want[0]), "exp(100) on the reference stays finite")
  Test.error?(Conform.agrees(ours, theirs, "exp saturation"))
end

def test_log_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [0.5, 1.0, 2.0, 10.0, 1024.0]
  Conform.agrees(try Tensor.log(try Conform.on_torch(data, [5])), try Tensor.log(try Conform.on_ref(data, [5])), "log")
end

def test_add_rejects_two_shapes_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # libtorch broadcasts [2] against [1, 2] on its own; the wrapper must not
  # let it, and the reference has no implicit broadcasting either.
  try Test.error?(Tensor.add(try Conform.on_torch([1.0, 2.0], [2]), try Conform.on_torch([1.0, 2.0], [1, 2])))
  Test.error?(Tensor.add(try Conform.on_ref([1.0, 2.0], [2]), try Conform.on_ref([1.0, 2.0], [1, 2])))
end

def test_add_rejects_two_dtypes_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.add(try Conform.on_torch([1.0], [1]), try Conform.on_torch_i64([1], [1])))
  Test.error?(Tensor.add(try Conform.on_ref([1.0], [1]), try Conform.on_ref_i64([1], [1])))
end

def test_float_ops_reject_int_tensors_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  ours = try Conform.on_torch_i64([1, 2], [2])
  theirs = try Conform.on_ref_i64([1, 2], [2])
  try Test.error?(Tensor.add(ours, ours))
  try Test.error?(Tensor.add(theirs, theirs))
  try Test.error?(Tensor.relu(ours))
  try Test.error?(Tensor.relu(theirs))
  try Test.error?(Tensor.sum(ours))
  Test.error?(Tensor.sum(theirs))
end

def test_exp_of_a_saturated_value_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # exp(1000) saturates on both sides: a finite input with a non-finite
  # result, which is allowed. Feeding that back in must be an `error`.
  ours = try Tensor.exp(try Conform.on_torch([1000.0], [1]))
  theirs = try Tensor.exp(try Conform.on_ref([1000.0], [1]))
  try Test.error?(Tensor.exp(ours))
  try Test.error?(Tensor.log(ours))
  try Test.error?(Tensor.exp(theirs))
  Test.error?(Tensor.log(theirs))
end

def test_log_of_a_non_positive_value_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  zero = [1.0, 0.0]
  negative = [1.0, -2.0]
  try Test.error?(Tensor.log(try Conform.on_torch(zero, [2])))
  try Test.error?(Tensor.log(try Conform.on_torch(negative, [2])))
  try Test.error?(Tensor.log(try Conform.on_ref(zero, [2])))
  Test.error?(Tensor.log(try Conform.on_ref(negative, [2])))
end

def test_sub_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0]
  other = [10.0, 20.0, 30.0, 40.0]
  Conform.agrees(
    try Tensor.sub(try Conform.on_torch(data, [2, 2]), try Conform.on_torch(other, [2, 2])),
    try Tensor.sub(try Conform.on_ref(data, [2, 2]), try Conform.on_ref(other, [2, 2])),
    "sub",
  )
end

def test_div_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, -2.0, 9.0, 0.5]
  other = [4.0, 8.0, 3.0, 0.125]
  Conform.agrees(
    try Tensor.div(try Conform.on_torch(data, [2, 2]), try Conform.on_torch(other, [2, 2])),
    try Tensor.div(try Conform.on_ref(data, [2, 2]), try Conform.on_ref(other, [2, 2])),
    "div",
  )
end

def test_div_by_zero_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0]
  zero = [1.0, 0.0]
  try Test.error?(Tensor.div(try Conform.on_torch(data, [2]), try Conform.on_torch(zero, [2])))
  Test.error?(Tensor.div(try Conform.on_ref(data, [2]), try Conform.on_ref(zero, [2])))
end

def test_div_rejects_two_shapes_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # libtorch would broadcast [2, 2] against [2]; the wrapper does not let it.
  square = try Conform.on_torch([1.0, 2.0, 3.0, 4.0], [2, 2])
  row = try Conform.on_torch([1.0, 2.0], [2])
  try Test.error?(Tensor.div(square, row))
  Test.error?(Tensor.div(try Conform.on_ref([1.0, 2.0, 3.0, 4.0], [2, 2]), try Conform.on_ref([1.0, 2.0], [2])))
end

def test_neg_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-1.5, 0.0, 2.25]
  Conform.agrees(try Tensor.neg(try Conform.on_torch(data, [3])), try Tensor.neg(try Conform.on_ref(data, [3])), "neg")
end

def test_sqrt_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [0.0, 0.25, 2.0, 9.0]
  Conform.agrees(
    try Tensor.sqrt(try Conform.on_torch(data, [2, 2])),
    try Tensor.sqrt(try Conform.on_ref(data, [2, 2])),
    "sqrt",
  )
end

def test_sqrt_of_a_negative_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, -0.5]
  try Test.error?(Tensor.sqrt(try Conform.on_torch(data, [2])))
  Test.error?(Tensor.sqrt(try Conform.on_ref(data, [2])))
end

def test_pow_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  base = [2.0, 9.0, 4.0, 1.5]
  exponent = [2.0, 0.5, 0.0, -1.0]
  Conform.agrees(
    try Tensor.pow(try Conform.on_torch(base, [2, 2]), try Conform.on_torch(exponent, [2, 2])),
    try Tensor.pow(try Conform.on_ref(base, [2, 2]), try Conform.on_ref(exponent, [2, 2])),
    "pow",
  )
end

def test_pow_of_a_non_positive_base_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  exponent = [2.0, 2.0]
  zero = [1.0, 0.0]
  negative = [1.0, -2.0]
  try Test.error?(Tensor.pow(try Conform.on_torch(zero, [2]), try Conform.on_torch(exponent, [2])))
  try Test.error?(Tensor.pow(try Conform.on_torch(negative, [2]), try Conform.on_torch(exponent, [2])))
  try Test.error?(Tensor.pow(try Conform.on_ref(zero, [2]), try Conform.on_ref(exponent, [2])))
  Test.error?(Tensor.pow(try Conform.on_ref(negative, [2]), try Conform.on_ref(exponent, [2])))
end

def test_sigmoid_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-6.0, -1.0, 0.0, 1.0, 4.0, 9.0]
  Conform.agrees(
    try Tensor.sigmoid(try Conform.on_torch(data, [2, 3])),
    try Tensor.sigmoid(try Conform.on_ref(data, [2, 3])),
    "sigmoid",
  )
end

def test_tanh_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-6.0, -1.0, 0.0, 1.0, 4.0, 9.0]
  Conform.agrees(
    try Tensor.tanh(try Conform.on_torch(data, [2, 3])),
    try Tensor.tanh(try Conform.on_ref(data, [2, 3])),
    "tanh",
  )
end

def test_gelu_tanh_approx_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-6.0, -1.0, 0.0, 0.5, 4.0, 9.0]
  try Conform.agrees(
    try Tensor.gelu_tanh_approx(try Conform.on_torch(data, [2, 3])),
    try Tensor.gelu_tanh_approx(try Conform.on_ref(data, [2, 3])),
    "gelu_tanh_approx",
  )

  Test.error?(Tensor.gelu_tanh_approx(try Tensor.exp(try Conform.on_torch([1000.0], [1]))))
end

def test_gelu_erf_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-6.0, -1.0, 0.0, 0.5, 4.0, 9.0]
  try Conform.agrees(
    try Tensor.gelu_erf(try Conform.on_torch(data, [2, 3])),
    try Tensor.gelu_erf(try Conform.on_ref(data, [2, 3])),
    "gelu_erf",
  )

  Test.error?(Tensor.gelu_erf(try Tensor.exp(try Conform.on_torch([1000.0], [1]))))
end

def test_sigmoid_and_tanh_refuse_a_non_finite_input_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # A finite input that saturates `F32` is legitimate; feeding the `+inf` back
  # in is not: the guard sits at every transcendental entry point.
  ours = try Tensor.exp(try Conform.on_torch([1000.0], [1]))
  theirs = try Tensor.exp(try Conform.on_ref([1000.0], [1]))
  try Test.error?(Tensor.sigmoid(ours))
  try Test.error?(Tensor.tanh(ours))
  try Test.error?(Tensor.sigmoid(theirs))
  Test.error?(Tensor.tanh(theirs))
end

def test_eq_and_lt_agree_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0]
  other = [1.0, 0.0, 3.0, 5.0]
  ours_eq = try Tensor.eq(try Conform.on_torch(data, [4]), try Conform.on_torch(other, [4]))
  theirs_eq = try Tensor.eq(try Conform.on_ref(data, [4]), try Conform.on_ref(other, [4]))
  try Test.eq(Tensor.dtype(ours_eq), Tensor.Dtype.Bool)
  try Conform.agrees(ours_eq, theirs_eq, "eq")
  Conform.agrees(
    try Tensor.lt(try Conform.on_torch(data, [4]), try Conform.on_torch(other, [4])),
    try Tensor.lt(try Conform.on_ref(data, [4]), try Conform.on_ref(other, [4])),
    "lt",
  )
end

def test_eq_agrees_on_integer_tensors() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1, 2, 3]
  other = [1, 7, 3]
  Conform.agrees(
    try Tensor.eq(try Conform.on_torch_i64(data, [3]), try Conform.on_torch_i64(other, [3])),
    try Tensor.eq(try Conform.on_ref_i64(data, [3]), try Conform.on_ref_i64(other, [3])),
    "eq on I64",
  )
end

def test_where_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  left = [1.0, 2.0, 3.0, 4.0]
  right = [10.0, 20.0, 30.0, 40.0]
  threshold = [2.5, 2.5, 2.5, 2.5]

  ours_cond = try Tensor.lt(try Conform.on_torch(left, [4]), try Conform.on_torch(threshold, [4]))
  ours = try Tensor.where(ours_cond, try Conform.on_torch(left, [4]), try Conform.on_torch(right, [4]))

  theirs_cond = try Tensor.lt(try Conform.on_ref(left, [4]), try Conform.on_ref(threshold, [4]))
  theirs = try Tensor.where(theirs_cond, try Conform.on_ref(left, [4]), try Conform.on_ref(right, [4]))
  Conform.agrees(ours, theirs, "where")
end

def test_where_needs_a_bool_condition_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0]
  try Test.error?(
    Tensor.where(try Conform.on_torch(data, [2]), try Conform.on_torch(data, [2]), try Conform.on_torch(data, [2])),
  )

  Test.error?(
    Tensor.where(try Conform.on_ref(data, [2]), try Conform.on_ref(data, [2]), try Conform.on_ref(data, [2])),
  )
end

def test_clamp_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [-3.0, -0.5, 0.0, 2.0, 7.0]
  Conform.agrees(
    try Tensor.clamp(try Conform.on_torch(data, [5]), -1.0, 2.5),
    try Tensor.clamp(try Conform.on_ref(data, [5]), -1.0, 2.5),
    "clamp",
  )
end

def test_clamp_with_lo_above_hi_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0]
  try Test.error?(Tensor.clamp(try Conform.on_torch(data, [1]), 2.0, 1.0))
  Test.error?(Tensor.clamp(try Conform.on_ref(data, [1]), 2.0, 1.0))
end
