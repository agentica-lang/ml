## Half weights on XLA (`Tensor.from_file(..., dtype: F16)` and `Tensor.from_list(..., F16)`): the buffer holds 16 bits per
## value, the load converts F32, F16 and BF16 ranges and refuses a value beyond 65504, naming its element, and a compiled
## program reads a half weight as the weight of `linear` and `matmul` and the table of `embedding`, widened to `F32` inside
## the program.
##
## The products are compared with the reference on `F32` weights. Weights that are multiples of 1/8 are the same in F16, so
## the two agree to the rounding of an `F32` sum (`Support.TOL`). Weights that F16 rounds (a tenth) differ from the `F32`
## ones by at most one F16 step of 2^-11 of each weight, which bounds a product to `STEP` of the sum of its terms.
##
## Without the plugin (`tools/install_xla.agc`) the checks pass by asserting nothing, unless `TENSOR_REQUIRE_XLA` is
## set (the `xla` job of CI sets it), and then they fail.

import Files from "ggml_file_support"
import Support from "xla_support"
import Tensor from "../tensor"
import Trace from "../trace"

## One F16 mantissa step: the relative error of rounding a weight to F16.
STEP = 0.00048828125

## Three padding bytes, then 3 F32 (1.5, -2.25, 0.0) at 3, 5 F16 (1.0, -2.0, 0.5, the smallest subnormal, 65504.0)
## at 15, and 3 BF16 (1.0, -3.0, 0.5) at 25.
BYTES = [
  9, 9, 9, 0, 0, 192, 63, 0, 0, 16, 192, 0, 0, 0, 0, 0, 60, 0, 192, 0, 56, 1, 0, 255, 123, 128, 63, 64, 192, 0, 63,
]

private

def with_bytes(data: List[Int], body: Fn[String, Result[Void]]) -> Result[Void]
  Files.with_file("xla-half", try Bytes.from_list(data), body)
end

def floats(t: Tensor.Tensor) -> Result[List[Float]] = Support.values(t)

def half(path: String, offset: Int, shape: List[Int], format: Tensor.FloatFormat) -> Result[Tensor.Tensor]
  Tensor.from_file(path, offset, shape, format, Support.XLA, Tensor.Dtype.F16)
end

def test_a_half_load_converts_each_format() -> Result[Void]
  return ok() if !try Support.present()
  with_bytes(BYTES) do |path|
    f32 = try half(path, 3, [3, 1], Tensor.FloatFormat.F32)
    try Test.eq(Tensor.dtype(f32), Tensor.Dtype.F16)
    try Test.eq(try floats(f32), [1.5, -2.25, 0.0])
    f16 = try half(path, 15, [5], Tensor.FloatFormat.F16)
    try Test.eq(try floats(f16), [1.0, -2.0, 0.5, 1.0 / 16777216.0, 65504.0])
    bf16 = try half(path, 25, [1, 3], Tensor.FloatFormat.BF16)
    try Test.eq(try floats(bf16), [1.0, -3.0, 0.5])
    bytes = try Tensor.from_bytes(try Bytes.from_list([0, 60, 0, 192]), [2], Tensor.FloatFormat.F16, Support.XLA)
    try Test.eq(try floats(bytes), [1.0, -2.0])
    ok()
  end
end

## F32 70000.0 at element 1, after 1.0, is named; 65504.0 is the largest value a half holds.
def test_a_value_beyond_f16_is_an_error_naming_its_element() -> Result[Void]
  return ok() if !try Support.present()
  try with_bytes([0, 0, 128, 63, 0, 184, 136, 71]) do |path|
    r = half(path, 0, [2], Tensor.FloatFormat.F32)
    try Test.assert(!r.ok?, "70000.0 was loaded as F16")
    try Test.assert(r.message.include?("outside what F16 holds"), r.message)
    Test.assert(r.message.include?("element 1 "), r.message)
  end

  with_bytes([0, 224, 127, 71, 0, 224, 127, 199]) do |path|
    Test.eq(try floats(try half(path, 0, [2], Tensor.FloatFormat.F32)), [65504.0, -65504.0])
  end
end

## A half value that is not a multiple of an F16 step rounds to nearest even, both ways.
def test_f16_values_made_from_a_list_round_to_nearest_even() -> Result[Void]
  return ok() if !try Support.present()
  values = [0.1, -0.1, 1.0009765625, 1.00146484375, 0.00000005960464477539063, 1.0 / 33554432.0]
  t = try Tensor.from_list(values.map { |x: Float| Tensor.Values.Float(x) }, [6], Tensor.Dtype.F16, Support.XLA)
  Test.eq(
    try floats(t),
    [0.0999755859375, -0.0999755859375, 1.0009765625, 1.001953125, 0.00000005960464477539063, 0.0],
  )
end

## A list value beyond 65504 (or an infinity) is an error naming its element, as in a file; 65520 would round to an
## infinity, so it is refused too.
def test_a_list_value_beyond_f16_is_an_error_naming_its_element() -> Result[Void]
  return ok() if !try Support.present()
  for bad in [70000.0, 65520.0, -70000.0, Float.from_bits(9218868437227405312)]
    r = Tensor.from_list(values_of([1.0, bad]), [2], Tensor.Dtype.F16, Support.XLA)
    try Test.assert(!r.ok?, "#{bad} was held as F16")
    try Test.assert(r.message.include?("outside what F16 holds"), r.message)
    try Test.assert(r.message.include?("element 1 "), r.message)
  end

  ok()
end

## What F16 holds comes back exactly, with the sign of a zero; a NaN stays a NaN; a subnormal tie goes to the even
## neighbour (3 * 2^-25 is half way between 1 and 2 of the smallest steps, so it goes up); and a value just above a tie
## rounds up (one rounding from the double: through an `F32` it would land on the tie and round down).
def test_f16_values_round_trip_and_round_once() -> Result[Void]
  return ok() if !try Support.present()
  exact = [1.0, -2.0, 65504.0, -0.0]
  t = try Tensor.from_list(values_of(exact), [4], Tensor.Dtype.F16, Support.XLA)
  got = try floats(t)
  try Test.eq(got, exact)
  try Test.eq(Float.to_bits(got[3]), Float.to_bits(-0.0))
  nan = try Tensor.from_list(values_of([Float.from_bits(9221120237041090560)]), [1], Tensor.Dtype.F16, Support.XLA)
  back = (try floats(nan))[0]
  try Test.assert(back != back, "a NaN came back as #{back}")
  ties = [3.0 / 33554432.0, 1.0 + 1.0 / 2048.0 + 1.0 / 1099511627776.0, 1.0 + 1.0 / 2048.0]
  rounded = try Tensor.from_list(values_of(ties), [3], Tensor.Dtype.F16, Support.XLA)
  Test.eq(try floats(rounded), [1.0 / 8388608.0, 1.0009765625, 1.0])
end

## 12 weights, each a multiple of 1/8: row `j` of the `[4, 3]` weight is `weights()[3 j...3 j + 3]`.
def weights() -> List[Float]
  out: List[Float] = []
  for i in 0...12
    out = out.push(((i * 5) % 11 - 5).to_f / 8.0)
  end

  out
end

def weight_bytes() -> List[Int] = weights().flat_map { |x: Float| Files.f32_bytes(x) }

def activations() -> List[Float] = [0.5, -1.25, 2.0, 0.75, 1.5, -0.25]

def values_of(xs: List[Float]) -> List[Tensor.Values] = xs.map { |x: Float| Tensor.Values.Float(x) }

## A product with the `[4, 3]` weight read as `[out, in]`, one with it transposed to `[3, 4]`, and the rows of the weight at
## `ids`, widened to `F32`.
def model(w: Tensor.Tensor, x: Tensor.Tensor, ids: Tensor.Tensor) -> Result[List[Tensor.Tensor]]
  rows = try Tensor.embedding(ids, w)
  back = try Tensor.matmul(x, try Tensor.transpose(w))
  ok([try Tensor.linear(x, w, none, true), back, try Tensor.to_dtype(rows, Tensor.Dtype.F32)])
end

def inputs(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  x = try Tensor.from_list(values_of(activations()), [2, 3], Tensor.Dtype.F32, backend)
  ids = try Tensor.from_list([3, 0, 2].map { |i: Int| Tensor.Values.Int(i) }, [3], Tensor.Dtype.I32, backend)
  ok([x, ids])
end

def test_a_half_weight_computes_as_the_f32_weight_it_holds() -> Result[Void]
  return ok() if !try Support.present()
  with_bytes(weight_bytes()) do |path|
    shared = try inputs(Support.XLA)
    w = try half(path, 0, [4, 3], Tensor.FloatFormat.F32)
    compiled = try Trace.compile(
      [Trace.spec_of(shared[0]), Trace.spec_of(shared[1]), Trace.spec_of(w)],
      ->(xs: List[Tensor.Tensor]) { model(xs[2], xs[0], xs[1]) },
    )

    got = try Trace.run(compiled, [shared[0], shared[1], w])
    on_reference = try inputs(Support.REF)
    f32 = try Tensor.from_file(path, 0, [4, 3], Tensor.FloatFormat.F32, Support.REF)
    Support.close(got, try model(f32, on_reference[0], on_reference[1]))
  end
end

## A weight that F16 rounds: the product differs from the `F32` one by `STEP` of the sum of its terms at most.
def test_a_rounded_half_weight_stays_within_one_f16_step_of_the_f32_product() -> Result[Void]
  return ok() if !try Support.present()
  values = Support.uniform(12, 3, 1.0)
  w16 = try Tensor.from_list(values_of(values), [4, 3], Tensor.Dtype.F16, Support.XLA)
  shared = try inputs(Support.XLA)
  compiled = try Trace.compile(
    [Trace.spec_of(shared[0]), Trace.spec_of(w16)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.linear(xs[0], xs[1], none, true)]) },
  )

  got = try floats((try Trace.run(compiled, [shared[0], w16]))[0])
  on_reference = try inputs(Support.REF)
  f32 = try Tensor.from_list(values_of(values), [4, 3], Tensor.Dtype.F32, Support.REF)
  want = try floats(try Tensor.linear(on_reference[0], f32, none, true))
  for r in 0...2
    for c in 0...4
      bound = 0.0
      for k in 0...3
        bound = bound + (activations()[3 * r + k] * values[3 * c + k]).abs * STEP
      end

      gap = (got[4 * r + c] - want[4 * r + c]).abs
      try Test.assert(gap <= bound + 0.000001, "[#{r}, #{c}]: gap #{gap} is above #{bound}")
    end
  end

  ok()
end

## A half weight is for inference: a gradient over one is an error, as on ggml.
def test_a_gradient_over_an_f16_weight_is_an_error() -> Result[Void]
  return ok() if !try Support.present()
  w = try Tensor.from_list(values_of([1.0, 2.0]), [2], Tensor.Dtype.F16, Support.XLA)
  r = Trace.value_and_grad(
    [w],
    ->(xs: List[Tensor.Tensor]) { Tensor.sum(try Tensor.to_dtype(xs[0], Tensor.Dtype.F32), none, false) },
  )

  try Test.assert(!r.ok?, "a gradient was taken over an F16 weight")
  Test.assert(r.message.include?("no gradient is taken"), r.message)
end
