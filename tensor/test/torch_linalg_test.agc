## `matmul` and `transpose` through the dispatch. `tensor/DESIGN.md` §3.2
## builds attention out of these two, so their rank, inner-dimension and
## axis-order rules are the ones the composite checks lean on.

import Conform from "torch_support"
import Tensor from "../tensor"

private

def test_matmul_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  left = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  right = [1.0, 0.0, 0.0, 1.0, 1.0, -1.0]
  ours = try Tensor.matmul(try Conform.on_torch(left, [2, 3]), try Conform.on_torch(right, [3, 2]))
  try Test.eq(Tensor.shape(ours), [2, 2])
  Conform.agrees(
    ours,
    try Tensor.matmul(try Conform.on_ref(left, [2, 3]), try Conform.on_ref(right, [3, 2])),
    "matmul",
  )
end

def test_batch_matmul_agrees_with_the_reference_at_rank_three_and_four() -> Result[Void]
  return Conform.absent() if !Conform.available()

  for shapes in [[[2, 3, 4], [2, 4, 2]], [[2, 2, 3, 2], [2, 2, 2, 3]], [[1, 3, 1], [1, 1, 2]]]
    left = (0...volume(shapes[0])).to_a.map { |i| 0.25 * (i % 7).to_f - 0.5 }
    right = (0...volume(shapes[1])).to_a.map { |i| 0.5 - 0.125 * (i % 5).to_f }
    ours = try Tensor.batch_matmul(try Conform.on_torch(left, shapes[0]), try Conform.on_torch(right, shapes[1]))
    want = try Tensor.batch_matmul(try Conform.on_ref(left, shapes[0]), try Conform.on_ref(right, shapes[1]))
    try Test.eq(Tensor.shape(ours), Tensor.shape(want))
    try Conform.agrees(ours, want, "batch_matmul #{shapes}")
  end

  ok()
end

def test_batch_matmul_rejects_what_it_cannot_pair_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = (0...24).to_a.map { |i| i.to_f }
  for shapes in [[[2, 3, 4], [3, 4, 2]], [[2, 3, 4], [2, 3, 2]], [[2, 3, 4], [4, 2]], [[4], [4]]]
    left = data.take(volume(shapes[0]))
    right = data.take(volume(shapes[1]))
    try Test.error?(Tensor.batch_matmul(try Conform.on_torch(left, shapes[0]), try Conform.on_torch(right, shapes[1])))
    try Test.error?(Tensor.batch_matmul(try Conform.on_ref(left, shapes[0]), try Conform.on_ref(right, shapes[1])))
  end

  ok()
end

def volume(shape: List[Int]) -> Int
  n = 1
  for d in shape
    n = n * d
  end

  n
end

def test_transpose_of_a_rank_two_tensor_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.transpose(try Conform.on_torch(data, [2, 3]))
  try Test.eq(Tensor.shape(ours), [3, 2])
  Conform.agrees(ours, try Tensor.transpose(try Conform.on_ref(data, [2, 3])), "transpose")
end

def test_transpose_of_a_rank_three_tensor_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  ours = try Tensor.transpose(try Conform.on_torch(data, [2, 2, 2]))
  try Test.eq(Tensor.shape(ours), [2, 2, 2])
  Conform.agrees(ours, try Tensor.transpose(try Conform.on_ref(data, [2, 2, 2])), "transpose rank 3")
end

def test_transpose_below_rank_two_keeps_the_shape() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0]
  ours = try Tensor.transpose(try Conform.on_torch(data, [3]))
  try Test.eq(Tensor.shape(ours), [3])
  Conform.agrees(ours, try Tensor.transpose(try Conform.on_ref(data, [3])), "transpose rank 1")
end

def test_matmul_rejects_a_rank_one_tensor_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  try Test.error?(Tensor.matmul(try Conform.on_torch([1.0, 2.0], [2]), try Conform.on_torch([1.0, 2.0], [2])))
  Test.error?(Tensor.matmul(try Conform.on_ref([1.0, 2.0], [2]), try Conform.on_ref([1.0, 2.0], [2])))
end

def test_matmul_rejects_a_mismatched_inner_dimension_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  wide = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  narrow = [1.0, 2.0]
  try Test.error?(Tensor.matmul(try Conform.on_torch(wide, [2, 3]), try Conform.on_torch(narrow, [1, 2])))
  Test.error?(Tensor.matmul(try Conform.on_ref(wide, [2, 3]), try Conform.on_ref(narrow, [1, 2])))
end
