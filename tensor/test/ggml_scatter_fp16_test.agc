## `scatter_add` on Vulkan in the fp16 default, on its own file because the precision belongs to the process: the
## masked matmul would round the updates to F16 (1e5 becomes infinity), so the lowering keeps `get_rows_back` there,
## and the answer is ggml's CPU one to the bit.
##
## Local only: without the module or a Vulkan device it passes and says so on stderr, unless `TENSOR_REQUIRE_VULKAN`
## is set.

import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

private

def sums(backend: Tensor.Backend) -> Result[List[Tensor.Values]]
  ids = try Tensor.from_list([Tensor.Values.Int(0)], [1], Tensor.Dtype.I32, backend)
  updates = try Tensor.from_list(
    [1e-9, 100000.0, 0.1].map { |x| Tensor.Values.Float(x) },
    [1, 3],
    Tensor.Dtype.F32,
    backend,
  )

  Tensor.to_list(try Trace.eval([ids, updates]) do |a| Tensor.scatter_add(a[0], a[1], 3) end)
end

def test_scatter_add_keeps_get_rows_back_in_fp16_mode() -> Result[Void]
  if !Tensor.available(VULKAN)
    flag = System.env("TENSOR_REQUIRE_VULKAN")
    return error("a Vulkan device is required (TENSOR_REQUIRE_VULKAN is set)") if flag.ok? and try flag != ""
    IO.warn("SKIP: no ggml module or no Vulkan device 0")
    return ok()
  end

  Test.eq(try sums(VULKAN), try sums(CPU))
end
