## BF16 half weights on XLA (`Tensor.from_file(..., dtype: BF16)` and `Tensor.from_list(..., BF16)`): a BF16 file keeps every 16-bit word, including the values F16 loses; a list rounds to
## nearest even once, from the double, and refuses a value beyond the largest BF16; a compiled program widens a BF16
## input with `to_dtype`.
##
## Without the plugin (`tools/install_xla.agc`) the checks pass by asserting nothing, unless `TENSOR_REQUIRE_XLA` is
## set (the `xla` job of CI sets it), and then they fail.

import Files from "ggml_file_support"
import Support from "xla_support"
import Tensor from "../tensor"
import Trace from "../trace"

## The largest finite BF16.
MAX = 3.3895313892515355e38

## Little-endian BF16 words: 1.0, -0.0, 2^-30 (under F16's smallest), 131072.0 (over F16's largest), an infinity, a NaN
## with a payload, and the smallest BF16 subnormal, 2^-133.
WORDS = [128, 63, 0, 128, 128, 48, 0, 72, 128, 127, 193, 127, 1, 0]

private

def floats(t: Tensor.Tensor) -> Result[List[Float]] = Support.values(t)

def values_of(xs: List[Float]) -> List[Tensor.Values] = xs.map { |x: Float| Tensor.Values.Float(x) }

def expect_words(t: Tensor.Tensor) -> Result[Void]
  try Test.eq(Tensor.dtype(t), Tensor.Dtype.BF16)
  got = try floats(t)
  try Test.eq(got.take(4), [1.0, -0.0, 1.0 / 1073741824.0, 131072.0])
  try Test.eq(Float.to_bits(got[1]), Float.to_bits(-0.0))
  try Test.eq(got[4], Float.from_bits(9218868437227405312))
  try Test.assert(got[5] != got[5], "a NaN came back as #{got[5]}")
  Test.eq(got[6], 9.183549615799121e-41)
end

def test_a_bf16_file_keeps_every_word() -> Result[Void]
  return ok() if !try Support.present()
  Files.with_file("xla-bf16", try Bytes.from_list(WORDS)) do |path|
    expect_words(try Tensor.from_file(path, 0, [7], Tensor.FloatFormat.BF16, Support.XLA, Tensor.Dtype.BF16))
  end
end

## Ties go to the even neighbour; a value just above a tie rounds up (one rounding from the double: through an `F32` it
## would land on the tie and round down); the largest BF16 is held.
def test_bf16_values_made_from_a_list_round_to_nearest_even_once() -> Result[Void]
  return ok() if !try Support.present()
  values = [1.0 + 1.0 / 256.0, 1.0 + 3.0 / 256.0, 1.0 + 1.0 / 256.0 + 1.0 / 1099511627776.0, 0.1, -0.1, MAX]
  t = try Tensor.from_list(values_of(values), [6], Tensor.Dtype.BF16, Support.XLA)
  Test.eq(try floats(t), [1.0, 1.015625, 1.0078125, 0.10009765625, -0.10009765625, MAX])
end

def test_a_list_value_beyond_bf16_is_an_error_naming_its_element() -> Result[Void]
  return ok() if !try Support.present()
  for bad in [3.4e38, -3.4e38, Float.from_bits(9218868437227405312)]
    r = Tensor.from_list(values_of([1.0, bad]), [2], Tensor.Dtype.BF16, Support.XLA)
    try Test.assert(!r.ok?, "#{bad} was held as BF16")
    try Test.assert(r.message.include?("outside what BF16 holds"), r.message)
    try Test.assert(r.message.include?("element 1 "), r.message)
  end

  ok()
end

## An F32 file rounds to BF16 and refuses a value beyond the largest one, naming its element.
def test_an_f32_file_rounds_to_bf16_and_refuses_what_it_cannot_hold() -> Result[Void]
  return ok() if !try Support.present()
  bytes = [0.1, -2.5].flat_map { |x: Float| Files.f32_bytes(x) }
  try Files.with_file("xla-bf16", try Bytes.from_list(bytes)) do |path|
    t = try Tensor.from_file(path, 0, [2], Tensor.FloatFormat.F32, Support.XLA, Tensor.Dtype.BF16)
    Test.eq(try floats(t), [0.10009765625, -2.5])
  end

  big = [1.0, 3.4e38].flat_map { |x: Float| Files.f32_bytes(x) }
  Files.with_file("xla-bf16", try Bytes.from_list(big)) do |path|
    r = Tensor.from_file(path, 0, [2], Tensor.FloatFormat.F32, Support.XLA, Tensor.Dtype.BF16)
    try Test.assert(!r.ok?, "3.4e38 was loaded as BF16")
    try Test.assert(r.message.include?("outside what BF16 holds"), r.message)
    Test.assert(r.message.include?("element 1 "), r.message)
  end
end

## The CPU gate of the plan: a compiled program takes a BF16 input and widens it to `F32` exactly.
def test_a_compiled_program_widens_a_bf16_input() -> Result[Void]
  return ok() if !try Support.present()
  values = [1.0, -0.5, 131072.0, 1.0 / 1073741824.0]
  x = try Tensor.from_list(values_of(values), [2, 2], Tensor.Dtype.BF16, Support.XLA)
  compiled = try Trace.compile(
    [Trace.spec_of(x)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.to_dtype(xs[0], Tensor.Dtype.F32)]) },
  )

  got = (try Trace.run(compiled, [x]))[0]
  try Test.eq(Tensor.dtype(got), Tensor.Dtype.F32)
  Test.eq(try floats(got), values)
end

def test_only_xla_reads_a_file_as_bf16() -> Result[Void]
  Files.with_file("xla-bf16", try Bytes.from_list([128, 63])) do |path|
    for backend in [Tensor.Backend.Reference, Tensor.Backend.Ggml(Tensor.Device.Cpu)]
      r = Tensor.from_file(path, 0, [1], Tensor.FloatFormat.BF16, backend, Tensor.Dtype.BF16)
      try Test.assert(!r.ok?, "a backend other than XLA read a file as BF16")
      try Test.assert(r.message.include?("BF16"), r.message)
    end

    ok()
  end
end
