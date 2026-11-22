## `matmul` and `transpose` against the reference backend. `tensor/DESIGN.md`
## §3.2 builds attention out of these two, so their rank, inner-dimension and
## axis-order rules are the ones a composite check leans on.

import Ref from "pkg/tensor/tensor"
import Support from "support"
import Torch from "../torch"

private

def test_matmul_agrees_with_the_reference() -> Result[Void]
  return Support.absent() if !Support.available()

  a = try Support.torch_f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  b = try Support.torch_f32([1.0, 0.0, 0.0, 1.0, 1.0, -1.0], [3, 2])
  ra = try Support.ref_f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  rb = try Support.ref_f32([1.0, 0.0, 0.0, 1.0, 1.0, -1.0], [3, 2])
  ours = try Torch.matmul(Support.CPU, a, b)
  try Test.eq(Torch.shape(ours), [2, 2])
  Support.agrees(ours, try Ref.matmul(ra, rb), "matmul")
end

def test_transpose_of_a_rank_two_tensor_agrees() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Torch.transpose(Support.CPU, try Support.torch_f32(data, [2, 3]))
  try Test.eq(Torch.shape(ours), [3, 2])
  Support.agrees(ours, try Ref.transpose(try Support.ref_f32(data, [2, 3])), "transpose")
end

def test_transpose_of_a_rank_three_tensor_agrees() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  ours = try Torch.transpose(Support.CPU, try Support.torch_f32(data, [2, 2, 2]))
  try Test.eq(Torch.shape(ours), [2, 2, 2])
  Support.agrees(ours, try Ref.transpose(try Support.ref_f32(data, [2, 2, 2])), "transpose rank 3")
end

def test_transpose_below_rank_two_keeps_the_shape() -> Result[Void]
  return Support.absent() if !Support.available()

  data = [1.0, 2.0, 3.0]
  ours = try Torch.transpose(Support.CPU, try Support.torch_f32(data, [3]))
  try Test.eq(Torch.shape(ours), [3])
  Support.agrees(ours, try Ref.transpose(try Support.ref_f32(data, [3])), "transpose rank 1")
end

def test_matmul_rejects_a_rank_one_tensor() -> Result[Void]
  return Support.absent() if !Support.available()

  a = try Support.torch_f32([1.0, 2.0], [2])
  b = try Support.torch_f32([1.0, 2.0], [2])
  try Test.error?(Torch.matmul(Support.CPU, a, b))
  Test.error?(Ref.matmul(try Support.ref_f32([1.0, 2.0], [2]), try Support.ref_f32([1.0, 2.0], [2])))
end

def test_matmul_rejects_a_mismatched_inner_dimension() -> Result[Void]
  return Support.absent() if !Support.available()

  a = try Support.torch_f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  b = try Support.torch_f32([1.0, 2.0], [1, 2])
  Test.error?(Torch.matmul(Support.CPU, a, b))
end
