## Shared helpers of the `tensor/` unit tests (not a test file itself).

import Tensor from "../tensor"

## Two floats count as equal below this absolute gap.
TOL = 0.000000001

## True when the ggml CPU backend is there; with `TENSOR_REQUIRE_GGML` set, a missing one is an error, never a pass
## (as in `ggml_compile_test.agc`).
def ggml_present() -> Result[Bool]
  return ok(true) if Tensor.available(Tensor.Backend.Ggml(Tensor.Device.Cpu))
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok(false) if !flag.ok?
  return ok(false) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end

## `data` as an `F32` tensor of `shape` on the reference backend.
def f32(data: List[Float], shape: List[Int]) -> Result[Tensor.Tensor] = Tensor.floats(data, shape)

## `data` as an `I64` tensor of `shape` on the reference backend.
def i64(data: List[Int], shape: List[Int]) -> Result[Tensor.Tensor]
  ints = data.map do |x: Int| Tensor.Values.Int(x) end
  Tensor.from_list(ints, shape, Tensor.Dtype.I64, Tensor.Backend.Reference)
end

## `data` as a `Bool` tensor of `shape` on the reference backend.
def bools(data: List[Bool], shape: List[Int]) -> Result[Tensor.Tensor]
  flags = data.map do |x: Bool| Tensor.Values.Bool(x) end
  Tensor.from_list(flags, shape, Tensor.Dtype.Bool, Tensor.Backend.Reference)
end

## The row-major values of a float tensor.
def floats_of(t: Tensor.Tensor) -> Result[List[Float]]
  raw = try Tensor.to_list(t)
  out: List[Float] = []
  for v in raw
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end

## The row-major values of an integer tensor.
def ints_of(t: Tensor.Tensor) -> Result[List[Int]]
  raw = try Tensor.to_list(t)
  out: List[Int] = []
  for v in raw
    case v
      Tensor.Values.Int(i) -> out = out.push(i)
      _ -> return error("an integer tensor holds Int values")
    end
  end

  ok(out)
end

## `t` holds `expected` within `TOL`.
def expect_floats(t: Tensor.Tensor, expected: List[Float]) -> Result[Void]
  got = try floats_of(t)
  try Test.eq(got.size, expected.size)
  worst = 0.0
  for i in 0...got.size
    gap = (got[i] - expected[i]).abs
    if gap > worst
      worst = gap
    end
  end

  Test.assert(worst <= TOL, "values differ by #{worst}, got #{got}")
end

## `a` and `b` agree elementwise within `TOL`.
def expect_close(a: Float, b: Float, what: String) -> Result[Void]
  Test.assert((a - b).abs <= TOL, "#{what}: expected #{b}, got #{a}")
end
