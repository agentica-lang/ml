## `reshape` against the reference backend, plus the volume rule that keeps a
## native reshape from reinterpreting memory the wrapper did not expect.

import Ref from "pkg/tensor/tensor"
import Support from "support"
import Torch from "../torch"

private

def test_reshape_agrees_with_the_reference() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Torch.reshape(Support.CPU, try Support.torch_f32(data, [2, 3]), [3, 2])
  try Test.eq(Torch.shape(ours), [3, 2])
  Support.agrees(ours, try Ref.reshape(try Support.ref_f32(data, [2, 3]), [3, 2]), "reshape")
end

def test_reshape_to_a_rank_zero_tensor_agrees() -> Result[Void]
  return Support.absent() if !Support.available()

  ours = try Torch.reshape(Support.CPU, try Support.torch_f32([4.5], [1]), [])
  try Test.eq(Torch.shape(ours), [])
  Support.agrees(ours, try Ref.reshape(try Support.ref_f32([4.5], [1]), []), "reshape to rank 0")
end

def test_reshape_to_a_bigger_rank_agrees() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1.0, 2.0, 3.0, 4.0]
  ours = try Torch.reshape(Support.CPU, try Support.torch_f32(data, [4]), [2, 1, 2])
  Support.agrees(ours, try Ref.reshape(try Support.ref_f32(data, [4]), [2, 1, 2]), "reshape to rank 3")
end

def test_reshape_with_a_wrong_volume_is_an_error() -> Result[Void]
  return Support.absent() if !Support.available()

  a = try Support.torch_f32([1.0, 2.0, 3.0], [3])
  try Test.error?(Torch.reshape(Support.CPU, a, [2, 2]))
  Test.error?(Ref.reshape(try Support.ref_f32([1.0, 2.0, 3.0], [3]), [2, 2]))
end

def test_reshape_with_a_negative_dimension_is_an_error() -> Result[Void]
  return Support.absent() if !Support.available()

  a = try Support.torch_f32([1.0, 2.0], [2])
  Test.error?(Torch.reshape(Support.CPU, a, [2, -1]))
end
