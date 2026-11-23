## Shared helpers of the property tests (not a test file itself). Only a
## `*_test.agc` file may import `prop` (it is a test-group dependency), so the
## generators live in the test files and this module holds what has no `Prop`
## in it.
##
## The inputs are small and exactly representable: values are multiples of 1/8
## with at most 7 significant bits, so a sum or a product of them is exact in
## `F32` as well as in doubles, and a gap between two backends on such an op is a
## wrong answer, not rounding. A tensor is a `Spec` (shape plus row-major data),
## which JSON encodes, so `Prop` can print the shrunk counterexample.

import Conform from "torch_support"
import Tensor from "../tensor"

## Cases a property runs on its backend; kept low so the whole suite stays in
## seconds.
RUNS = 60

## The shape and row-major values of one tensor.
struct Spec
  shape: List[Int]
  data: List[Float]
end

## The backends a comparison runs on: always the reference, and libtorch when
## the native module is loaded. Where it is not, a test of the torch side
## asserts that absence through `Conform.absent()`.
def backends() -> List[Tensor.Backend]
  return [Conform.REF, Conform.TORCH] if Conform.available()

  [Conform.REF]
end

## `data` repeated up to `count` values: the generated list sets the pattern,
## the shape sets the size.
def cycle(data: List[Float], count: Int) -> List[Float]
  out: List[Float] = []
  for i in 0...count
    out = out.push(data[i % data.size])
  end

  out
end

def volume(shape: List[Int]) -> Int = shape.reduce(1) do |acc: Int, d: Int| acc * d end

## `spec` as an `F32` tensor on `backend`.
def tensor(backend: Tensor.Backend, spec: Spec) -> Result[Tensor.Tensor]
  Tensor.floats(spec.data, spec.shape, backend)
end

## Every value of `t` as a `Float`, whatever its dtype (an index tensor reads as
## whole numbers).
def numbers(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(x) -> out = out.push(x)
      Tensor.Values.Int(i) -> out = out.push(i.to_f)
      Tensor.Values.Bool(flag) -> out = out.push(flag_number(flag))
    end
  end

  ok(out)
end

def flag_number(flag: Bool) -> Float
  return 1.0 if flag

  0.0
end

## `Conform.close?` with the relative tolerance `rel`, and two NaNs are equal
## (an op that answers NaN on both sides agrees). `rel` of 0.0 asks for exact
## equality, for the ops whose inputs make them exact. Below `floor` in absolute
## terms two values agree whatever `rel` says.
def near?(a: Float, b: Float, rel: Float, floor: Float = Conform.ABS) -> Bool
  return true if a != a and b != b
  return a == b if rel == 0.0 or !Conform.finite?(a) or !Conform.finite?(b)

  gap = (a - b).abs
  gap <= floor or gap <= rel * (a.abs + b.abs)
end

## What a test of the torch side proves: libtorch is among `backends()` where it
## is built, and where it is not, its absence (`Conform.absent()`).
def torch_is_compared_or_absent() -> Result[Void]
  return Conform.absent() if !Conform.available()

  Test.eq(backends(), [Conform.REF, Conform.TORCH])
end

## `t` has `shape` and holds `want`, each value `near?` it. A plain `error` (never
## `Test.eq`), so `Prop.check` can shrink the input that caused it.
def holds(
  t: Tensor.Tensor,
  shape: List[Int],
  want: List[Float],
  what: String,
  rel: Float = Conform.REL,
  floor: Float = Conform.ABS,
) -> Result[Void]
  return error("#{what}: shape #{Tensor.shape(t)}, expected #{shape}") if Tensor.shape(t) != shape

  got = try numbers(t)
  return error("#{what}: #{got.size} values, expected #{want.size}") if got.size != want.size

  for i in 0...want.size
    return error("#{what}: value #{i} is #{got[i]}, expected #{want[i]}") if !near?(got[i], want[i], rel, floor)
  end

  ok()
end

## `got` and `want` have the same shape, dtype and values (within the
## tolerance), as a plain `error`.
def same(
  got: Tensor.Tensor,
  want: Tensor.Tensor,
  what: String,
  rel: Float = Conform.REL,
  floor: Float = Conform.ABS,
) -> Result[Void]
  if Tensor.dtype(got) != Tensor.dtype(want)
    return error("#{what}: dtype #{Tensor.dtype(got)}, expected #{Tensor.dtype(want)}")
  end

  holds(got, Tensor.shape(want), try numbers(want), what, rel, floor)
end

## The row-major coordinates of offset `n` in a block of `dims`.
def coords(n: Int, dims: List[Int]) -> List[Int]
  out = List.repeat(0, dims.size)
  rest = n
  i = dims.size - 1
  while i >= 0
    out = out.set(i, rest % dims[i])
    rest = rest / dims[i]
    i -= 1
  end

  out
end

## The row-major offset of `at` in a block of `dims`.
def offset(at: List[Int], dims: List[Int]) -> Int
  total = 0
  for i in 0...dims.size
    total = total * dims[i] + at[i]
  end

  total
end
