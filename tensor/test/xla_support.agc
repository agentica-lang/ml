## Shared helpers of the tests that run on `Backend.Xla` (not a test file itself).
##
## Without the plugin (`tools/install_xla.agc`) the checks pass by asserting nothing, unless `TENSOR_REQUIRE_XLA` is
## set (the `xla` job of CI sets it), and then they fail.

import Tensor from "../tensor"

XLA = Tensor.Backend.Xla(Tensor.Device.Cpu)
REF = Tensor.Backend.Reference
## The largest gap allowed, relative to the largest reference value of a tensor.
TOL = 0.000001

## True when the plugin is there; with `TENSOR_REQUIRE_XLA` set, a missing one is an error, never a pass.
def present() -> Result[Bool]
  return ok(true) if Tensor.available(XLA)
  flag = System.env("TENSOR_REQUIRE_XLA")
  return ok(false) if !flag.ok?
  return ok(false) if try flag == ""
  error("the XLA backend is required but not available (TENSOR_REQUIRE_XLA is set)")
end

## `count` floats in `[-scale, scale)` from `seed`.
def uniform(count: Int, seed: Int, scale: Float) -> List[Float]
  s = seed
  out: List[Float] = []
  for _ in 0...count
    s = (s * 1103515245 + 12345) % 2147483648
    out = out.push(((s.to_f / 2147483648.0) * 2.0 - 1.0) * scale)
  end

  out
end

def floats(xs: List[Float], shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, backend)
end

## Every tensor of `got` within `TOL` of the one of `want`, relative to its largest value.
def close(got: List[Tensor.Tensor], want: List[Tensor.Tensor]) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...got.size
    try Test.eq(Tensor.shape(got[i]), Tensor.shape(want[i]))
    a = try values(got[i])
    b = try values(want[i])
    worst = 0.0
    scale = 0.000001
    for k in 0...b.size
      worst = [worst, (a[k] - b[k]).abs].max.unwrap!
      scale = [scale, b[k].abs].max.unwrap!
    end

    return error("output #{i}: gap #{worst / scale} is above #{TOL}") if worst / scale > TOL
  end

  ok()
end

## The values of `t` as floats (an integer or a boolean too, for the comparison of `close`).
def values(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(x) -> out = out.push(x)
      Tensor.Values.Int(i) -> out = out.push(i.to_f)
      Tensor.Values.Bool(b) ->
        if b
          out = out.push(1.0)
        else
          out = out.push(0.0)
        end
    end
  end

  ok(out)
end

## The class ids `xs` as an `I32` tensor.
def labels(xs: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Int(x) }, [xs.size], Tensor.Dtype.I32, backend)
end
