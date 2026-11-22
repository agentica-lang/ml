## `conv1d` and `avg_pool1d` through the dispatch: the same call on both
## backends, compared on shape, dtype and values.
##
## The shapes are the ones the operator admits: stride 1, stride 2, a nonzero
## padding, a 1-sample input, and a rank-2 input (no batch axis). The error
## cases are the window arithmetic and the channel count, which must be an
## `error` on both sides rather than a native call libtorch would reinterpret.

import Conform from "torch_support"
import Tensor from "../tensor"

private

def test_conv1d_with_stride_one_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
  weight = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -1.0, 0.0]
  ours = try Tensor.conv1d(try Conform.on_torch(input, [1, 2, 4]), try Conform.on_torch(weight, [2, 2, 2]))
  try Test.eq(Tensor.shape(ours), [1, 2, 3])
  Conform.agrees(
    ours,
    try Tensor.conv1d(try Conform.on_ref(input, [1, 2, 4]), try Conform.on_ref(weight, [2, 2, 2])),
    "conv1d stride 1",
  )
end

def test_conv1d_with_stride_two_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0, 3.0, 4.0, 5.0]
  weight = [1.0, 1.0]
  ours = try Tensor.conv1d(try Conform.on_torch(input, [1, 1, 5]), try Conform.on_torch(weight, [1, 1, 2]), 2, 0)
  try Test.eq(Tensor.shape(ours), [1, 1, 2])
  Conform.agrees(
    ours,
    try Tensor.conv1d(try Conform.on_ref(input, [1, 1, 5]), try Conform.on_ref(weight, [1, 1, 2]), 2, 0),
    "conv1d stride 2",
  )
end

def test_conv1d_with_padding_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0, 3.0, 4.0, 5.0]
  weight = [1.0, -1.0, 0.5]
  ours = try Tensor.conv1d(try Conform.on_torch(input, [1, 1, 5]), try Conform.on_torch(weight, [1, 1, 3]), 1, 1)
  try Test.eq(Tensor.shape(ours), [1, 1, 5])
  Conform.agrees(
    ours,
    try Tensor.conv1d(try Conform.on_ref(input, [1, 1, 5]), try Conform.on_ref(weight, [1, 1, 3]), 1, 1),
    "conv1d padding 1",
  )
end

def test_conv1d_with_stride_two_and_padding_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  weight = [0.5, 0.5, 0.5]
  ours = try Tensor.conv1d(try Conform.on_torch(input, [1, 1, 6]), try Conform.on_torch(weight, [1, 1, 3]), 2, 1)
  Conform.agrees(
    ours,
    try Tensor.conv1d(try Conform.on_ref(input, [1, 1, 6]), try Conform.on_ref(weight, [1, 1, 3]), 2, 1),
    "conv1d stride 2 with padding",
  )
end

def test_conv1d_of_a_rank_two_input_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  weight = [1.0, -1.0]
  ours = try Tensor.conv1d(try Conform.on_torch(input, [2, 3]), try Conform.on_torch(weight, [1, 2, 1]))
  try Test.eq(Tensor.shape(ours), [1, 3])
  Conform.agrees(
    ours,
    try Tensor.conv1d(try Conform.on_ref(input, [2, 3]), try Conform.on_ref(weight, [1, 2, 1])),
    "conv1d of a rank-2 input",
  )
end

def test_conv1d_of_a_one_sample_input_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [2.5, -1.5]
  weight = [4.0, 2.0]
  ours = try Tensor.conv1d(try Conform.on_torch(input, [1, 1, 2]), try Conform.on_torch(weight, [1, 1, 2]))
  try Test.eq(Tensor.shape(ours), [1, 1, 1])
  Conform.agrees(
    ours,
    try Tensor.conv1d(try Conform.on_ref(input, [1, 1, 2]), try Conform.on_ref(weight, [1, 1, 2])),
    "conv1d of one window",
  )
end

def test_conv1d_with_a_channel_mismatch_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0, 3.0, 4.0]
  weight = [1.0, 2.0, 3.0]
  try Test.error?(Tensor.conv1d(try Conform.on_torch(input, [1, 2, 2]), try Conform.on_torch(weight, [1, 3, 1])))
  Test.error?(Tensor.conv1d(try Conform.on_ref(input, [1, 2, 2]), try Conform.on_ref(weight, [1, 3, 1])))
end

def test_conv1d_with_a_window_wider_than_the_input_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0]
  weight = [1.0, 1.0, 1.0]
  try Test.error?(Tensor.conv1d(try Conform.on_torch(input, [1, 1, 2]), try Conform.on_torch(weight, [1, 1, 3])))
  Test.error?(Tensor.conv1d(try Conform.on_ref(input, [1, 1, 2]), try Conform.on_ref(weight, [1, 1, 3])))
end

def test_conv1d_with_a_bad_stride_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0]
  weight = [1.0]
  try Test.error?(Tensor.conv1d(try Conform.on_torch(input, [1, 1, 2]), try Conform.on_torch(weight, [1, 1, 1]), 0, 0))
  Test.error?(Tensor.conv1d(try Conform.on_ref(input, [1, 1, 2]), try Conform.on_ref(weight, [1, 1, 1]), 0, 0))
end

def test_avg_pool1d_with_stride_one_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.avg_pool1d(try Conform.on_torch(input, [1, 1, 6]), 3, 1, 0)
  try Test.eq(Tensor.shape(ours), [1, 1, 4])
  Conform.agrees(ours, try Tensor.avg_pool1d(try Conform.on_ref(input, [1, 1, 6]), 3, 1, 0), "avg_pool1d stride 1")
end

def test_avg_pool1d_with_stride_two_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.avg_pool1d(try Conform.on_torch(input, [1, 1, 6]), 2, 2, 0)
  try Test.eq(Tensor.shape(ours), [1, 1, 3])
  Conform.agrees(ours, try Tensor.avg_pool1d(try Conform.on_ref(input, [1, 1, 6]), 2, 2, 0), "avg_pool1d stride 2")
end

def test_avg_pool1d_with_padding_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0, 3.0, 4.0]
  ours = try Tensor.avg_pool1d(try Conform.on_torch(input, [1, 1, 4]), 3, 1, 1)
  try Test.eq(Tensor.shape(ours), [1, 1, 4])
  Conform.agrees(ours, try Tensor.avg_pool1d(try Conform.on_ref(input, [1, 1, 4]), 3, 1, 1), "avg_pool1d padding 1")
end

def test_avg_pool1d_of_a_rank_two_input_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  ours = try Tensor.avg_pool1d(try Conform.on_torch(input, [2, 3]), 2, 1, 0)
  try Test.eq(Tensor.shape(ours), [2, 2])
  Conform.agrees(
    ours,
    try Tensor.avg_pool1d(try Conform.on_ref(input, [2, 3]), 2, 1, 0),
    "avg_pool1d of a rank-2 input",
  )
end

def test_avg_pool1d_of_a_one_sample_input_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [3.25, -1.75]
  ours = try Tensor.avg_pool1d(try Conform.on_torch(input, [1, 2, 1]), 1, 1, 0)
  try Test.eq(Tensor.shape(ours), [1, 2, 1])
  Conform.agrees(ours, try Tensor.avg_pool1d(try Conform.on_ref(input, [1, 2, 1]), 1, 1, 0), "avg_pool1d one position")
end

def test_avg_pool1d_with_padding_wider_than_half_the_kernel_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0, 3.0, 4.0]
  try Test.error?(Tensor.avg_pool1d(try Conform.on_torch(input, [1, 1, 4]), 2, 1, 2))
  Test.error?(Tensor.avg_pool1d(try Conform.on_ref(input, [1, 1, 4]), 2, 1, 2))
end

def test_avg_pool1d_with_a_bad_kernel_is_an_error_on_both_backends() -> Result[Void]
  return Conform.absent() if !Conform.available()

  input = [1.0, 2.0]
  try Test.error?(Tensor.avg_pool1d(try Conform.on_torch(input, [1, 1, 2]), 0, 1, 0))
  Test.error?(Tensor.avg_pool1d(try Conform.on_ref(input, [1, 1, 2]), 0, 1, 0))
end

def test_conv1d_of_a_batch_agrees() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # Two samples of one channel and three positions: the batch axis must not
  # become a second channel.
  input = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
  weight = [1.0, 1.0]
  ours = try Tensor.conv1d(try Conform.on_torch(input, [2, 1, 3]), try Conform.on_torch(weight, [1, 1, 2]))
  try Test.eq(Tensor.shape(ours), [2, 1, 2])
  Conform.agrees(
    ours,
    try Tensor.conv1d(try Conform.on_ref(input, [2, 1, 3]), try Conform.on_ref(weight, [1, 1, 2])),
    "conv1d of a batch of two",
  )
end
