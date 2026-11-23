## Laws of the shape and host-crossing ops, on every available backend:
## `concat` is associative along both axes (and the three-way call is the same
## as the folded pair, which is how libtorch joins a list), and `from_list` then
## `to_list` returns the values it was given, for every dtype.

import Conform from "torch_support"
import Prop from "pkg/prop/prop"
import Support from "prop_support"
import Tensor from "../tensor"

private

## A `[rows, 2]` tensor, or a `[2, cols]` one, from up to eight values: the
## odd one out is dropped, and no values is a tensor with a zero axis.
def spec_of(values: List[Int], by_rows: Bool) -> Support.Spec
  pairs = values.size / 2
  data = values.take(pairs * 2).map do |n: Int| n.to_f end
  return Support.Spec(shape: [pairs, 2], data: data) if by_rows

  Support.Spec(shape: [2, pairs], data: data)
end

def pieces(by_rows: Bool) -> Prop.Gen[Support.Spec]
  Prop.map(Prop.list(Prop.int(-9, 9), 8)) do |values: List[Int]| spec_of(values, by_rows) end
end

## What `concat` of two pieces answers, read back as a `Spec`.
def joined(backend: Tensor.Backend, axis: Int, a: Support.Spec, b: Support.Spec) -> Support.Spec
  t = Tensor.concat([Support.tensor(backend, a).unwrap!, Support.tensor(backend, b).unwrap!], axis).unwrap!
  Support.Spec(shape: Tensor.shape(t), data: Support.numbers(t).unwrap!)
end

def concat_is_associative(axis: Int) -> Result[Void]
  for backend in Support.backends()
    result = Prop.associative(pieces(axis == 0), Prop.Config(runs: Support.RUNS)) do |a: Support.Spec, b: Support.Spec|
      joined(backend, axis, a, b)
    end

    try result
  end

  ok()
end

def test_concat_along_the_first_axis_is_associative() -> Result[Void] = concat_is_associative(0)

def test_concat_along_the_second_axis_is_associative() -> Result[Void] = concat_is_associative(1)

def test_concat_of_three_is_the_fold_of_pairs() -> Result[Void]
  gen = Prop.pair(pieces(true), Prop.pair(pieces(true), pieces(true)))
  for backend in Support.backends()
    result = Prop.check(
      gen,
      Prop.Config(runs: Support.RUNS),
    ) do |p: Prop.Pair[Support.Spec, Prop.Pair[Support.Spec, Support.Spec]]|
      ts = [p.first, p.second.first, p.second.second].map do |s: Support.Spec| Support.tensor(backend, s).unwrap! end
      at_once = try Tensor.concat(ts, 0)
      folded = try Tensor.concat([try Tensor.concat([ts[0], ts[1]], 0), ts[2]], 0)
      Support.same(at_once, folded, "concat of three on #{backend}")
    end

    try result
  end

  ok()
end

def test_the_torch_side_of_the_shape_laws_is_compared_or_absent() -> Result[Void]
  return Conform.absent() if !Conform.available()

  Test.eq(Support.backends(), [Conform.REF, Conform.TORCH])
end

## `from_list` then `to_list` gives the values back, as the dtype, in a rank-1
## tensor, whatever the backend.
def round_trips(dtype: Tensor.Dtype, values: Prop.Gen[Tensor.Values]) -> Result[Void]
  for backend in Support.backends()
    encode = ->(xs: List[Tensor.Values]) { Tensor.from_list(xs, [xs.size], dtype, backend).unwrap! }
    result = Prop.roundtrip(Prop.list(values, 6), encode, Prop.Config(runs: Support.RUNS)) do |t: Tensor.Tensor|
      read_back(t, dtype)
    end

    try result
  end

  ok()
end

def read_back(t: Tensor.Tensor, dtype: Tensor.Dtype) -> Result[List[Tensor.Values]]
  return error("dtype #{Tensor.dtype(t)}, expected #{dtype}") if Tensor.dtype(t) != dtype

  Tensor.to_list(t)
end

def floats(scale: Int, bound: Int) -> Prop.Gen[Tensor.Values]
  Prop.map(Prop.int(-bound, bound)) do |n: Int| Tensor.Values.Float(n.to_f / scale.to_f) end
end

def ints(low: Int, high: Int) -> Prop.Gen[Tensor.Values]
  Prop.map(Prop.int(low, high)) do |n: Int| Tensor.Values.Int(n) end
end

def test_f32_values_round_trip() -> Result[Void] = round_trips(Tensor.Dtype.F32, floats(1024, 1048576))

def test_f16_values_round_trip() -> Result[Void] = round_trips(Tensor.Dtype.F16, floats(16, 1024))

def test_f64_values_round_trip() -> Result[Void]
  anything = Prop.map(Prop.float(-1000000.0, 1000000.0)) do |x: Float| Tensor.Values.Float(x) end
  round_trips(Tensor.Dtype.F64, anything)
end

def test_i8_values_round_trip() -> Result[Void] = round_trips(Tensor.Dtype.I8, ints(-128, 127))

def test_i16_values_round_trip() -> Result[Void] = round_trips(Tensor.Dtype.I16, ints(-32768, 32767))

def test_i32_values_round_trip() -> Result[Void] = round_trips(Tensor.Dtype.I32, ints(-2147483648, 2147483647))

## The boundary carries doubles, so an `I64` is exact up to 2^53.
def test_i64_values_round_trip() -> Result[Void]
  round_trips(Tensor.Dtype.I64, ints(-9007199254740992, 9007199254740992))
end

def test_u8_values_round_trip() -> Result[Void] = round_trips(Tensor.Dtype.U8, ints(0, 255))

def test_bool_values_round_trip() -> Result[Void]
  flags = Prop.map(Prop.bool()) do |flag: Bool| Tensor.Values.Bool(flag) end
  round_trips(Tensor.Dtype.Bool, flags)
end
