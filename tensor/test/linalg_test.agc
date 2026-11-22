## Linear algebra: values and error paths.

import Support from "support"
import Tensor from "../tensor"

private

def test_matmul_multiplies_two_matrices() -> Result[Void]
  a = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  b = try Support.f32([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
  Support.expect_floats(try Tensor.matmul(a, b), [58.0, 64.0, 139.0, 154.0])
end

def test_matmul_rejects_a_mismatched_inner_dimension() -> Result[Void]
  a = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  b = try Support.f32([1.0, 2.0], [2, 1])
  Test.error?(Tensor.matmul(a, b))
end

def test_matmul_rejects_non_rank_two() -> Result[Void]
  a = try Support.f32([1.0, 2.0, 3.0], [3])
  b = try Support.f32([1.0, 2.0, 3.0], [3])
  Test.error?(Tensor.matmul(a, b))
end

def test_batch_matmul_multiplies_two_batches() -> Result[Void]
  a = try Support.f32([1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0], [2, 2, 2])
  b = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])
  c = try Tensor.batch_matmul(a, b)
  try Test.eq(Tensor.shape(c), [2, 2, 2])
  Support.expect_floats(c, [1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0])
end

def test_batch_matmul_rejects_two_batch_shapes() -> Result[Void]
  a = try Support.f32([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], [3, 2, 2])
  b = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])
  Test.error?(Tensor.batch_matmul(a, b))
end

def test_transpose_swaps_rows_and_columns() -> Result[Void]
  t = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  u = try Tensor.transpose(t)
  try Test.eq(Tensor.shape(u), [3, 2])
  Support.expect_floats(u, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
end

def test_transpose_twice_is_the_identity() -> Result[Void]
  vals: List[Tensor.Values] = (1...25).to_a.map do |i: Int| Tensor.Values.Float(i.to_f) end
  t = try Tensor.from_list(vals, [2, 3, 4], Tensor.Dtype.F32, Tensor.Backend.Reference)
  twice = try Tensor.transpose(try Tensor.transpose(t))
  try Test.eq(Tensor.shape(twice), [2, 3, 4])
  Test.eq(try Tensor.to_list(twice), try Tensor.to_list(t))
end

def test_transpose_leaves_rank_one_alone() -> Result[Void]
  t = try Support.f32([1.0, 2.0], [2])
  u = try Tensor.transpose(t)
  try Test.eq(Tensor.shape(u), [2])
  Support.expect_floats(u, [1.0, 2.0])
end
