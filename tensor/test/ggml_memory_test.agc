## `Tensor.device_memory`: the free and total bytes of the device a ggml backend runs on, and the errors of a backend that
## has none. Like `ggml_test.agc`, the checks that need the native module or a Vulkan device pass without them.

import Tensor from "../tensor"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

private

## The memory of `backend` is a positive total and a free part of it.
def holds_memory(backend: Tensor.Backend) -> Result[Void]
  memory = try Tensor.device_memory(backend)
  try Test.assert(memory.total > 0, "total #{memory.total} on #{backend}")
  try Test.assert(memory.free > 0 and memory.free <= memory.total, "free #{memory.free} of #{memory.total}")
  ok()
end

## `backend` has no memory to report, and the error names the call.
def refused(backend: Tensor.Backend) -> Result[Void]
  case Tensor.device_memory(backend)
    ok(m) -> error("device_memory of #{backend} gave #{m.total} bytes where an error was due")
    error(e) -> Test.assert(e.message.include?("device_memory"), "the error said \"#{e.message}\"")
  end
end

def test_the_cpu_backend_reports_the_host_memory() -> Result[Void]
  return ok() if !Tensor.available(CPU)

  holds_memory(CPU)
end

def test_a_vulkan_device_reports_its_memory() -> Result[Void]
  return ok() if !Tensor.available(VULKAN)

  holds_memory(VULKAN)
end

def test_a_backend_without_a_ggml_device_is_an_error() -> Result[Void]
  try refused(Tensor.Backend.Reference)
  refused(Tensor.Backend.Ggml(Tensor.Device.Vulkan(7)))
end
