## Laws of the shape and host-crossing ops, on every available backend:
## `concat` is associative along both axes, joins the data the way its shape says,
## and the three-way call is the same as the folded pair (which is how libtorch
## joins a list); `from_list` then `to_list` returns the values it was given, for
## every dtype, limits included.

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
def joined(backend: Tensor.Backend, axis: Int, a: Support.Spec, b: Support.Spec) -> Result[Support.Spec]
  t = try Tensor.concat([try Support.tensor(backend, a), try Support.tensor(backend, b)], axis)
  ok(Support.Spec(shape: Tensor.shape(t), data: try Support.numbers(t)))
end

def concat_is_associative(axis: Int) -> Result[Void]
  piece = pieces(axis == 0)
  for backend in Support.backends()
    try Prop.check(
      Prop.pair(piece, Prop.pair(piece, piece)),
      Prop.Config(runs: Support.RUNS),
    ) do |p: Prop.Pair[Support.Spec, Prop.Pair[Support.Spec, Support.Spec]]|
      left = try joined(backend, axis, try joined(backend, axis, p.first, p.second.first), p.second.second)
      right = try joined(backend, axis, p.first, try joined(backend, axis, p.second.first, p.second.second))
      Prop.eq(left, right)
    end
  end

  ok()
end

def test_concat_along_the_first_axis_is_associative() -> Result[Void] = concat_is_associative(0)

def test_concat_along_the_second_axis_is_associative() -> Result[Void] = concat_is_associative(1)

## Associativity alone holds for a `concat` that answers its first argument, so
## this is the value model: `[pa, 2]` pieces stack their rows, and `[2, ca]`
## pieces put the rows of `b` after the rows of `a`, row by row.
def concat_is_the_joined_data(axis: Int) -> Result[Void]
  piece = pieces(axis == 0)
  for backend in Support.backends()
    try Prop.check(
      Prop.pair(piece, piece),
      Prop.Config(runs: Support.RUNS),
    ) do |p: Prop.Pair[Support.Spec, Support.Spec]|
      got = try joined(backend, axis, p.first, p.second)
      shape = [p.first.shape[0] + p.second.shape[0], 2]
      want = p.first.data + p.second.data
      if axis == 1
        shape = [2, p.first.shape[1] + p.second.shape[1]]
        want = row(p.first, 0) + row(p.second, 0) + row(p.first, 1) + row(p.second, 1)
      end

      Support.holds(try Support.tensor(backend, got), shape, want, "concat on axis #{axis} on #{backend}", 0.0)
    end
  end

  ok()
end

## Row `i` of a `[2, cols]` piece.
def row(piece: Support.Spec, i: Int) -> List[Float]
  cols = piece.shape[1]
  piece.data.drop(i * cols).take(cols)
end

def test_concat_along_the_first_axis_stacks_the_rows() -> Result[Void] = concat_is_the_joined_data(0)

def test_concat_along_the_second_axis_extends_each_row() -> Result[Void] = concat_is_the_joined_data(1)

def test_concat_of_three_is_the_fold_of_pairs() -> Result[Void]
  gen = Prop.pair(pieces(true), Prop.pair(pieces(true), pieces(true)))
  for backend in Support.backends()
    try Prop.check(
      gen,
      Prop.Config(runs: Support.RUNS),
    ) do |p: Prop.Pair[Support.Spec, Prop.Pair[Support.Spec, Support.Spec]]|
      specs = [p.first, p.second.first, p.second.second]
      ts: List[Tensor.Tensor] = []
      for spec in specs
        ts = ts.push(try Support.tensor(backend, spec))
      end

      at_once = try Tensor.concat(ts, 0)
      folded = try Tensor.concat([try Tensor.concat([ts[0], ts[1]], 0), ts[2]], 0)
      Support.same(at_once, folded, "concat of three on #{backend}")
    end
  end

  ok()
end

def test_the_torch_side_of_the_shape_laws_is_compared_or_absent() -> Result[Void]
  Support.torch_is_compared_or_absent()
end

## `from_list` then `to_list` gives the values back, as the dtype, in a rank-1
## tensor, whatever the backend.
def round_trips(dtype: Tensor.Dtype, values: Prop.Gen[Tensor.Values]) -> Result[Void]
  for backend in Support.backends()
    encode = ->(xs: List[Tensor.Values]) { Tensor.from_list(xs, [xs.size], dtype, backend) }
    try Prop.roundtrip(Prop.list(values, 6), encode, Prop.Config(runs: Support.RUNS)) do |made: Result[Tensor.Tensor]|
      read_back(try made, dtype)
    end
  end

  ok()
end

def read_back(t: Tensor.Tensor, dtype: Tensor.Dtype) -> Result[List[Tensor.Values]]
  return error("dtype #{Tensor.dtype(t)}, expected #{dtype}") if Tensor.dtype(t) != dtype

  Tensor.to_list(t)
end

## One case in three is an `edge`, the rest come from `any`: a random value
## rarely lands on the limit of a dtype, and the limits are where it breaks.
def mixed(edge: Prop.Gen[Tensor.Values], any: Prop.Gen[Tensor.Values]) -> Prop.Gen[Tensor.Values]
  both = Prop.pair(edge, any)
  Prop.map(Prop.pair(both, Prop.int(0, 2))) do |p: Prop.Pair[Prop.Pair[Tensor.Values, Tensor.Values], Int]|
    pick(p.first, p.second)
  end
end

def pick(both: Prop.Pair[Tensor.Values, Tensor.Values], draw: Int) -> Tensor.Values
  return both.first if draw == 0

  both.second
end

## Values a dtype can hold, in `low` ..= `high`, and the ones next to its limits.
def ints(low: Int, high: Int) -> Prop.Gen[Tensor.Values]
  limits = [low, low + 1, -1, 0, 1, high - 1, high].select do |n: Int| n >= low and n <= high end
  as_values = ->(n: Int) { Tensor.Values.Int(n) }
  mixed(Prop.map(Prop.one_of(limits), as_values), Prop.map(Prop.int(low, high), as_values))
end

## Multiples of `1 / scale` that the dtype holds exactly, and its `extremes`: the
## largest finite value and the smallest normal one, of both signs, and zero.
def floats(scale: Int, bound: Int, largest: Float, smallest: Float) -> Prop.Gen[Tensor.Values]
  as_values = ->(x: Float) { Tensor.Values.Float(x) }
  extremes = [0.0, largest, -largest, smallest, -smallest]
  exact = Prop.map(Prop.int(-bound, bound)) do |n: Int| n.to_f / scale.to_f end
  mixed(Prop.map(Prop.one_of(extremes), as_values), Prop.map(exact, as_values))
end

def test_f32_values_round_trip() -> Result[Void]
  round_trips(Tensor.Dtype.F32, floats(1024, 1048576, 3.4028234663852886e38, 1.1754943508222875e-38))
end

def test_f16_values_round_trip() -> Result[Void]
  round_trips(Tensor.Dtype.F16, floats(16, 1024, 65504.0, 0.00006103515625))
end

def test_f64_values_round_trip() -> Result[Void]
  anything = Prop.map(Prop.float(-1000000.0, 1000000.0)) do |x: Float| Tensor.Values.Float(x) end
  extremes = Prop.one_of([1.7976931348623157e308, -1.7976931348623157e308, 2.2250738585072014e-308])
  round_trips(Tensor.Dtype.F64, mixed(Prop.map(extremes) do |x: Float| Tensor.Values.Float(x) end, anything))
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
