## `Vision.Kernel`: a file that imports only `vision` names the resize kernel of a step.

import S from "support"
import Vision from "../vision"

def test_a_resize_step_takes_the_kernel_from_vision() -> Result[Void]
  wide = try S.u8(List.repeat(1, 8 * 4), 8, 4, 1)
  for kernel in [Vision.Kernel.Nearest, Vision.Kernel.Linear, Vision.Kernel.Cubic]
    small = try Vision.apply(wide, [Vision.Step.Resize(2, kernel)])
    try Test.eq(List.size(try S.samples_of(small)), 8)
  end

  ok()
end
