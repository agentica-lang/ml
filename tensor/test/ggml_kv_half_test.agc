## An F16 cache on ggml, the key-value cache of Qwen3 (`transformers/generate.agc`, decoded by `transformers/kv_decoder.agc`): `to_dtype` makes it, `set_rows` writes F32 rows
## into it in place, `attention` reads it as the key and the value (flash attention, and the products when a mask of
## arbitrary values keeps it off, or only its first rows through a slice), and `to_dtype` widens it. The F16 rounding of
## what it holds is bounded.
##
## Without the ggml module this passes by asserting nothing, unless `TENSOR_REQUIRE_GGML` is set.

import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
REF = Tensor.Backend.Reference
## One F16 mantissa step is 2^-11 of a value: the rounding of a value is at most that, relative.
HALF = 0.00049

private

def floats(xs: List[Float], shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, backend)
end

## Values that F16 cannot hold exactly.
def odd(count: Int, seed: Int) -> List[Float] = (0...count).map { |i| ((i * seed) % 19 - 9).to_f * 0.1371 + 0.0123 }

def halved(x: Tensor.Tensor) -> Result[Tensor.Tensor]
  out = try Trace.run(
    try Trace.compile(
      [Trace.spec_of(x)],
      ->(xs: List[Tensor.Tensor]) { ok([try Tensor.to_dtype(xs[0], Tensor.Dtype.F16)]) },
    ),
    [x],
  )

  ok(out[0])
end

def widened(x: Tensor.Tensor) -> Result[Tensor.Tensor]
  out = try Trace.run(
    try Trace.compile(
      [Trace.spec_of(x)],
      ->(xs: List[Tensor.Tensor]) { ok([try Tensor.to_dtype(xs[0], Tensor.Dtype.F32)]) },
    ),
    [x],
  )

  ok(out[0])
end

def close(got: List[Float], want: List[Float], tolerance: Float, what: String) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...got.size
    try Test.assert(
      (got[i] - want[i]).abs <= want[i].abs * tolerance + 0.000001,
      "#{what} #{i}: #{got[i]} for #{want[i]}",
    )
  end

  ok()
end

def test_an_f16_cache_is_written_in_place_and_read_by_attention() -> Result[Void]
  return ok() if !try Support.ggml_present()

  k = try floats(odd(64, 5), [1, 2, 4, 8], CPU)
  v = try floats(odd(64, 3), [1, 2, 4, 8], CPU)
  cache_k = try halved(k)
  cache_v = try halved(v)
  try Test.eq(Tensor.dtype(cache_k), Tensor.Dtype.F16)
  fresh_k = try floats(odd(16, 7), [1, 2, 1, 8], CPU)
  fresh_v = try floats(odd(16, 11), [1, 2, 1, 8], CPU)
  row = try Tensor.from_list([Tensor.Values.Int(2)], [1], Tensor.Dtype.I32, CPU)
  q = try floats(odd(16, 13), [1, 2, 1, 8], CPU)
  mask = try floats([0.0, -0.5, 0.25, -3.0], [1, 4], CPU)
  args = [cache_k, cache_v, fresh_k, fresh_v, row, q, mask]
  out = try Trace.run(
    try Trace.compile(
      args.map { |t| Trace.spec_of(t) },
      ->(xs: List[Tensor.Tensor]) {
        kk = try Tensor.set_rows(xs[0], xs[2], xs[4])
        vv = try Tensor.set_rows(xs[1], xs[3], xs[4])
        ok([
          kk,
          vv,
          try Tensor.attention(xs[5], kk, vv, none, none, false),
          try Tensor.attention(xs[5], kk, vv, some(xs[6]), none, false),
        ])
      },
    ),
    args,
  )

  # The cache given is the cache returned, F16 still.
  try Test.eq(Tensor.dtype(out[0]), Tensor.Dtype.F16)
  wide_k = try Support.floats_of(try widened(out[0]))
  wide_v = try Support.floats_of(try widened(out[1]))
  written = try Tensor.set_rows(
    try floats(odd(64, 5), [1, 2, 4, 8], REF),
    try floats(odd(16, 7), [1, 2, 1, 8], REF),
    try Tensor.from_list([Tensor.Values.Int(2)], [1], Tensor.Dtype.I32, REF),
  )

  try close(wide_k, try Support.floats_of(written), HALF, "key")
  bias = try floats([0.0, -0.5, 0.25, -3.0], [1, 4], REF)
  unmasked: Optional[Tensor.Tensor] = none
  for masked in [unmasked, some(bias)]
    want = try Tensor.attention(
      try floats(odd(16, 13), [1, 2, 1, 8], REF),
      try floats(wide_k, [1, 2, 4, 8], REF),
      try floats(wide_v, [1, 2, 4, 8], REF),
      masked,
      none,
      false,
    )

    got = out[2]
    if masked.some?
      got = out[3]
    end

    try close(try Support.floats_of(got), try Support.floats_of(want), 0.00001, "attention")
  end

  ok()
end

## An attention reads the first rows of an F16 cache where they are: a slice from the first element, of the cache
## that a `set_rows` wrote. With no mask it is flash attention, which reads the rows through the view of the cache, and
## with a mask of arbitrary values the products, which read a copy of them. Both answer what the reference answers
## over those rows of each head, and a slice goes nowhere but into an attention.
def test_an_attention_reads_the_first_rows_of_an_f16_cache() -> Result[Void]
  return ok() if !try Support.ggml_present()

  cache_k = try halved(try floats(odd(64, 5), [1, 2, 4, 8], CPU))
  cache_v = try halved(try floats(odd(64, 3), [1, 2, 4, 8], CPU))
  fresh_k = try floats(odd(16, 7), [1, 2, 1, 8], CPU)
  fresh_v = try floats(odd(16, 11), [1, 2, 1, 8], CPU)
  row = try Tensor.from_list([Tensor.Values.Int(1)], [1], Tensor.Dtype.I32, CPU)
  q = try floats(odd(16, 13), [1, 2, 1, 8], CPU)
  mask = try floats([0.0, -0.5], [1, 2], CPU)
  args = [cache_k, cache_v, fresh_k, fresh_v, row, q, mask]
  out = try Trace.run(
    try Trace.compile(
      args.map { |t| Trace.spec_of(t) },
      ->(xs: List[Tensor.Tensor]) {
        kk = try Tensor.set_rows(xs[0], xs[2], xs[4])
        vv = try Tensor.set_rows(xs[1], xs[3], xs[4])
        near_k = try Tensor.slice(kk, [0, 0, 0, 0], [1, 2, 2, 8])
        near_v = try Tensor.slice(vv, [0, 0, 0, 0], [1, 2, 2, 8])
        ok([
          kk,
          vv,
          try Tensor.attention(xs[5], near_k, near_v, none, none, false),
          try Tensor.attention(xs[5], near_k, near_v, some(xs[6]), none, false),
        ])
      },
    ),
    args,
  )

  wide_k = try floats(try Support.floats_of(try widened(out[0])), [1, 2, 4, 8], REF)
  wide_v = try floats(try Support.floats_of(try widened(out[1])), [1, 2, 4, 8], REF)
  unmasked: Optional[Tensor.Tensor] = none
  for masked in [unmasked, some(try floats([0.0, -0.5], [1, 2], REF))]
    want = try Tensor.attention(
      try floats(odd(16, 13), [1, 2, 1, 8], REF),
      try Tensor.slice(wide_k, [0, 0, 0, 0], [1, 2, 2, 8]),
      try Tensor.slice(wide_v, [0, 0, 0, 0], [1, 2, 2, 8]),
      masked,
      none,
      false,
    )

    got = out[2]
    if masked.some?
      got = out[3]
    end

    try close(try Support.floats_of(got), try Support.floats_of(want), 0.00001, "attention")
  end

  cache = try halved(try floats(odd(64, 5), [1, 2, 4, 8], CPU))
  refused([cache], "F16 operand") do |x|
    near = try Tensor.slice(x[0], [0, 0, 0, 0], [1, 2, 2, 8])
    Tensor.to_dtype(near, Tensor.Dtype.F32)
  end
end

def test_an_f16_cache_goes_nowhere_else() -> Result[Void]
  return ok() if !try Support.ggml_present()

  cache = try halved(try floats(odd(8, 5), [2, 4], CPU))
  rows = try floats(odd(4, 7), [1, 4], CPU)
  half_rows = try halved(rows)
  row = try Tensor.from_list([Tensor.Values.Int(1)], [1], Tensor.Dtype.I32, CPU)
  try refused([cache], "add") do |x| Tensor.add(x[0], x[0]) end
  try refused([cache], "output 0") do |x| Tensor.transpose(x[0]) end
  try refused([cache], "reshape") do |x| Tensor.reshape(x[0], [4, 2]) end
  # an F16 query, F16 rows for set_rows, a cache that is not an input, and a weight (a constant) widened
  q = try halved(try floats(odd(8, 3), [1, 1, 1, 8], CPU))
  k = try halved(try floats(odd(64, 5), [1, 1, 8, 8], CPU))
  try refused([q, k], "(attention)") do |x| Tensor.attention(x[0], x[1], x[1], none, none, false) end
  try refused([cache, half_rows, row], "F16 operand") do |x| Tensor.set_rows(x[0], x[1], x[2]) end
  try refused([rows, rows, row], "operand") do |x|
    Tensor.set_rows(try Tensor.to_dtype(try Tensor.reshape(x[0], [1, 4]), Tensor.Dtype.F16), x[1], x[2])
  end

  refused([rows], "operand") do |_x| Tensor.to_dtype(Trace.constant(cache), Tensor.Dtype.F32) end
end

## The function of `args` that does not run, and says `text`.
def refused(args: List[Tensor.Tensor], text: String, f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]) -> Result[Void]
  compiled = Trace.compile(args.map { |t| Trace.spec_of(t) }, ->(xs: List[Tensor.Tensor]) { ok([try f.call(xs)]) })
  ran = case compiled
    ok(c) -> Trace.run(c, args)
    error(e) -> error(e.message)
  end

  case ran
    ok(_) -> error("ran where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end
