## Shared helpers of the `train/` tests (not a test file itself).

import Tensor from "pkg/tensor/tensor"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)
GGML = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

## The ggml backends this machine has: its CPU backend when the module is
## built, and Vulkan device 0 when ggml sees one. With `TENSOR_REQUIRE_GGML`
## set to anything but "" (CI sets it) a missing module is an error.
def ggml_backends() -> Result[List[Tensor.Backend]]
  try Tensor.vulkan_fp16(false)
  if !Tensor.available(GGML)
    flag = System.env("TENSOR_REQUIRE_GGML")
    return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)") if flag.ok? and try flag != ""
    return ok([])
  end

  return ok([GGML, VULKAN]) if Tensor.available(VULKAN)
  ok([GGML])
end

def tensor(
  data: List[Float],
  shape: List[Int],
  dtype: Tensor.Dtype = Tensor.Dtype.F64,
  backend: Tensor.Backend = REF,
) -> Result[Tensor.Tensor]
  Tensor.from_list(data.map { |x| Tensor.Values.Float(x) }, shape, dtype, backend)
end

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

## `got` matches `want` within `abs + rel * |want|`, element by element.
def expect_close(got: List[Float], want: List[Float], abs: Float, rel: Float, what: String) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    gap = (got[i] - want[i]).abs
    return error("#{what}: value #{i} is #{got[i]}, expected #{want[i]}") if gap > abs + rel * want[i].abs
  end

  ok()
end

## `r` is an error whose message holds `phrase`.
def expect_error[T](r: Result[T], phrase: String) -> Result[Void]
  case r
    ok(_) -> error("expected an error with \"#{phrase}\", got a value")
    error(e) -> Test.assert(e.message.include?(phrase), "\"#{e.message}\" does not hold \"#{phrase}\"")
  end
end
