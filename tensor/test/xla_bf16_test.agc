## BF16 half weights on XLA (`Tensor.from_file(..., dtype: BF16)` and `Tensor.from_list(..., BF16)`): a BF16 file keeps every 16-bit word, including the values F16 loses; a list rounds to
## nearest even once, from the double, and refuses a value beyond the largest BF16; a compiled program widens a BF16
## input with `to_dtype`; and a compiled product, lookup and attention read BF16 weights.
##
## The parity tests write the weights once as BF16 bytes: XLA loads them as `BF16` and the reference loads the same bytes
## as `F32` (an exact widening), so both compute on the same weights and agree to the rounding of an `F32` sum
## (`Support.TOL`).
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
  backends = try bf16_backends()
  Files.with_file("bf16", try Bytes.from_list(WORDS)) do |path|
    for backend in backends
      try expect_words(try Tensor.from_file(path, 0, [7], Tensor.FloatFormat.BF16, backend, Tensor.Dtype.BF16))
    end

    ok()
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

def test_the_reference_rejects_a_bf16_destination() -> Result[Void]
  Files.with_file("xla-bf16", try Bytes.from_list([128, 63])) do |path|
    r = Tensor.from_file(path, 0, [1], Tensor.FloatFormat.BF16, Support.REF, Tensor.Dtype.BF16)
    try Test.assert(!r.ok?, "the reference read a file as BF16")
    Test.assert(r.message.include?("BF16"), r.message)
  end
end

## The bytes of `xs` as BF16 words, each the top half of its `F32` (the low bits dropped): weights BF16 holds exactly.
def bf16_bytes(xs: List[Float]) -> List[Int] = xs.flat_map { |x: Float| Files.f32_bytes(x).drop(2) }

def bf16_backends() -> Result[List[Tensor.Backend]]
  try Tensor.vulkan_fp16(false)
  backends = [
    Tensor.Backend.Ggml(Tensor.Device.Cpu),
    Tensor.Backend.Ggml(Tensor.Device.Vulkan(0)),
    Tensor.Backend.Torch(Tensor.Device.Cpu),
  ].select { |backend| Tensor.available(backend) }
  if try Support.present()
    backends = backends.push(Support.XLA)
  end

  ok(backends)
end

def same_weights(
  backend: Tensor.Backend,
  count: Int,
  shape: List[Int],
  seed: Int,
  body: Fn[Tensor.Tensor, Tensor.Tensor, Result[Void]],
) -> Result[Void]
  Files.with_file("xla-bf16", try Bytes.from_list(bf16_bytes(Support.uniform(count, seed, 1.0)))) do |path|
    half = try Tensor.from_file(path, 0, shape, Tensor.FloatFormat.BF16, backend, Tensor.Dtype.BF16)
    wide = try Tensor.from_file(path, 0, shape, Tensor.FloatFormat.BF16, Support.REF)
    body.call(half, wide)
  end
end

## A product with the `[4, 3]` weight read as `[out, in]`, one with it transposed, and the rows of the weight at `ids`,
## widened to `F32`.
def model(w: Tensor.Tensor, x: Tensor.Tensor, ids: Tensor.Tensor) -> Result[List[Tensor.Tensor]]
  rows = try Tensor.embedding(ids, w)
  back = try Tensor.matmul(x, try Tensor.transpose(w))
  ok([try Tensor.linear(x, w, none, true), back, try Tensor.to_dtype(rows, Tensor.Dtype.F32)])
end

def inputs(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  x = try Support.floats(Support.uniform(6, 7, 2.0), [2, 3], backend)
  ids = try Tensor.from_list([3, 0, 2].map { |i: Int| Tensor.Values.Int(i) }, [3], Tensor.Dtype.I32, backend)
  ok([x, ids])
end

def test_a_bf16_weight_computes_as_the_f32_weight_it_holds() -> Result[Void]
  return ok() if !try Support.present()
  same_weights(Support.XLA, 12, [4, 3], 3) do |half, wide|
    shared = try inputs(Support.XLA)
    compiled = try Trace.compile(
      [Trace.spec_of(shared[0]), Trace.spec_of(shared[1]), Trace.spec_of(half)],
      ->(xs: List[Tensor.Tensor]) { model(xs[2], xs[0], xs[1]) },
    )

    got = try Trace.run(compiled, [shared[0], shared[1], half])
    on_reference = try inputs(Support.REF)
    Support.close(got, try model(wide, on_reference[0], on_reference[1]))
  end
end

def attend(q: Tensor.Tensor, k: Tensor.Tensor, v: Tensor.Tensor) -> Result[List[Tensor.Tensor]]
  ok([try Tensor.attention(q, k, v), try Tensor.attention(q, k, v, causal: true)])
end

## Grouped-query attention with BF16 keys and values and an `F32` query.
def test_attention_reads_bf16_keys_and_values() -> Result[Void]
  for backend in try bf16_backends()
    try attention_parity(backend)
  end

  ok()
end

def attention_parity(backend: Tensor.Backend) -> Result[Void]
  same_weights(backend, 24, [1, 1, 3, 8], 5) do |k_half, k_wide|
    same_weights(backend, 24, [1, 1, 3, 8], 11) do |v_half, v_wide|
      q = try Support.floats(Support.uniform(32, 13, 1.0), [1, 2, 2, 8], backend)
      compiled = try Trace.compile(
        [Trace.spec_of(q), Trace.spec_of(k_half), Trace.spec_of(v_half)],
        ->(xs: List[Tensor.Tensor]) { attend(xs[0], xs[1], xs[2]) },
      )

      got = try Trace.run(compiled, [q, k_half, v_half])
      q_ref = try Support.floats(Support.uniform(32, 13, 1.0), [1, 2, 2, 8], Support.REF)
      Support.close(got, try attend(q_ref, k_wide, v_wide))
    end
  end
end
