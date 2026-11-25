## Shared helpers of the `vision/` tests (not a test file itself).

import Image from "pkg/image/image"
import Tensor from "pkg/tensor/tensor"

REF = Tensor.Backend.Reference
GGML = Tensor.Backend.Ggml(Tensor.Device.Cpu)

## The backends that run here: the reference, and ggml's CPU when it is built (`TENSOR_REQUIRE_GGML` makes a missing
## one an error, as CI sets it).
def backends() -> Result[List[Tensor.Backend]]
  return ok([REF, GGML]) if Tensor.available(GGML)
  flag = System.env("TENSOR_REQUIRE_GGML")
  return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)") if flag.ok? and try flag != ""
  ok([REF])
end

## A new folder for `body`, removed when it ends.
def with_dir(body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-vision-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = body.call(dir)
  _ = Path.remove(dir, 5000)
  result
end

## A `U8` image of `width` by `height` with `bands` bands, `pixels` row-major and interleaved.
def u8(pixels: List[Int], width: Int, height: Int, bands: Int) -> Result[Image.Image]
  Image.from_pixels(try Bytes.from_list(pixels), width, height, bands, Image.Format.U8)
end

## The `U8` samples of an image.
def samples_of(image: Image.Image) -> Result[List[Int]] = ok((try Image.to_pixels(image)).to_a)

def floats_of(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end

def ints_of(t: Tensor.Tensor) -> Result[List[Int]]
  out: List[Int] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Int(i) -> out = out.push(i)
      _ -> return error("an integer tensor holds Int values")
    end
  end

  ok(out)
end

## `got` is `expected` within `tol` at each place.
def close(got: List[Float], expected: List[Float], tol: Float) -> Result[Void]
  try Test.eq(got.size, expected.size)
  for i in 0...got.size
    try Test.assert((got[i] - expected[i]).abs <= tol, "place #{i}: expected #{expected[i]}, got #{got[i]}")
  end

  ok()
end
