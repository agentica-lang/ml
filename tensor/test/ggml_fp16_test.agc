## The fp16 default of Vulkan (`Tensor.vulkan_fp16`), on its own file because the choice
## belongs to the process and the first ggml call fixes it: with nothing chosen,
## the benchmark model (`vulkan_mlp.agc`) on Vulkan device 0 still agrees with
## ggml's CPU backend within 0.1, but further than F32 does, which shows fp16
## `mul_mat` took effect. The exact opt-out is `ggml_vulkan_mlp_test.agc`.
##
## Local only: without the module or a Vulkan device it passes and says so on
## stderr. On a build without Vulkan the choice changes nothing.

import Model from "vulkan_mlp"
import Tensor from "../tensor"

VULKAN = Tensor.Device.Vulkan(0)
## The F32 bound of `ggml_vulkan_mlp_test.agc`: fp16 must be past it.
F32_TOL = 0.0000015
FP16_TOL = 0.1

private

def test_fp16_mul_mat_is_the_default_and_changes_the_gradients_within_its_bound() -> Result[Void]
  if !Tensor.available(Tensor.Backend.Ggml(VULKAN))
    IO.warn("SKIP: no ggml module or no Vulkan device 0; the fp16 check runs where there is one")
    return ok()
  end

  flag = System.env("GGML_VK_DISABLE_F16")
  try Test.assert(!flag.ok? or try flag == "", "GGML_VK_DISABLE_F16 is set: fp16 is the default")
  c = try Model.compare(VULKAN)
  worst = 0.0
  for g in c.gaps
    return error("a gap of #{g} is above the fp16 bound #{FP16_TOL}") if g > FP16_TOL
    if g > worst
      worst = g
    end
  end

  Test.assert(worst > F32_TOL, "the largest gap #{worst} is within F32's #{F32_TOL}: fp16 did not engage")
end
