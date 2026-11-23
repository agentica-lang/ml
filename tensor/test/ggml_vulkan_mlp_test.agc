## The benchmark's model (`vulkan_mlp.agc`) compiled on Vulkan device 0 with
## F32 `mul_mat` against ggml's CPU backend: the loss and every gradient
## within `TOL` of the largest value of their tensor.
##
## `TOL` is 1.5e-6, set on the Vega iGPU of this repository's maintainer
## (AMD Raven, RADV), about twice the 6e-7 the spike measured there
## (spike/ggml-graph/README.md). Another device may round differently:
## `TENSOR_VULKAN_TOL` overrides it.
##
## Local only: without a Vulkan device it passes and says so on stderr.

import Model from "vulkan_mlp"
import Tensor from "../tensor"

VULKAN = Tensor.Device.Vulkan(0)
TOL = 0.0000015

private

## `TOL`, or the float in `TENSOR_VULKAN_TOL`.
def tolerance() -> Result[Float]
  set = System.env("TENSOR_VULKAN_TOL")
  return ok(TOL) if !set.ok?
  text = try set
  return ok(TOL) if text == ""
  text.to_f
end

def test_the_benchmark_model_agrees_between_cpu_and_vulkan() -> Result[Void]
  if !Tensor.available(Tensor.Backend.Ggml(VULKAN))
    IO.warn("SKIP: no Vulkan device 0; the 784-512-512-10 parity runs where there is one")
    return ok()
  end

  tol = try tolerance()
  c = try Model.compare(VULKAN)
  for i in 0...c.gaps.size
    return error("output #{i}: gap #{c.gaps[i]} is above #{tol}") if c.gaps[i] > tol
  end

  ok()
end
