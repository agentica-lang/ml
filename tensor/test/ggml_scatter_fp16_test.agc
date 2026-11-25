## `scatter_add` on Vulkan in the fp16 default, on its own file because the precision belongs to the process: the
## masked matmul would round the updates to F16 (1e5 becomes infinity, 1e-9 zero), so the lowering keeps
## `get_rows_back` there, and the answer is ggml's CPU one to the bit. The matmul rounds only when its mask chunk has
## more than 8 columns (ggml-Vulkan's mat-vec kernel for up to 8 is always float), hence the 16 indices.
##
## Local only: without the module or a Vulkan device it passes and says so on stderr, unless `TENSOR_REQUIRE_VULKAN`
## is set.

import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

private

## 16 updates of [1e-9, 1e5, 0.1] into 32 rows, two for each of 8 indices: the masked matmul would have 16 columns,
## too many for ggml-Vulkan's mat-vec kernel (up to 8), which is always float, so it reaches the fp16 matmul.
def sums(backend: Tensor.Backend) -> Result[List[Tensor.Values]]
  ids = try Tensor.from_list((0...16).map { |i| Tensor.Values.Int(i % 8) }, [16], Tensor.Dtype.I32, backend)
  values = [1e-9, 100000.0, 0.1]
  updates = try Tensor.from_list(
    (0...48).map { |i| Tensor.Values.Float(values[i % 3]) },
    [16, 3],
    Tensor.Dtype.F32,
    backend,
  )

  Tensor.to_list(try Trace.eval([ids, updates]) do |a| Tensor.scatter_add(a[0], a[1], 32) end)
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
