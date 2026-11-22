## `conv1d` and `avg_pool1d` on the reference backend: hand-computed windows
## over stride, padding and the 1-sample edge, plus the window arithmetic and
## the errors a caller can reach through the public API.
##
## The reference is the oracle (`tensor/DESIGN.md` §11), so these values are
## worked out by hand here and `torch_convolution_test.agc` compares the
## backend against them through the same calls.

import Support from "support"
import Tensor from "../tensor"

private

def test_conv1d_matches_the_hand_values() -> Result[Void]
  # One sample, two channels of four positions, two kernels of width two:
  # out[0] = ch0[o] + ch1[o+1], out[1] = 0.5*(ch0[o] + ch0[o+1]) - ch1[o].
  input = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 2, 4])
  weight = try Support.f32([1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -1.0, 0.0], [2, 2, 2])
  out = try Tensor.conv1d(input, weight)
  try Test.eq(Tensor.shape(out), [1, 2, 3])
  Support.expect_floats(out, [7.0, 9.0, 11.0, -3.5, -3.5, -3.5])
end

def test_conv1d_with_stride_two_skips_positions() -> Result[Void]
  input = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 5])
  weight = try Support.f32([1.0, 1.0], [1, 1, 2])
  out = try Tensor.conv1d(input, weight, 2, 0)
  # (5 + 0 - 2) / 2 + 1 = 2 windows: positions 0..1 and 2..3.
  try Test.eq(Tensor.shape(out), [1, 1, 2])
  Support.expect_floats(out, [3.0, 7.0])
end

def test_conv1d_with_padding_includes_the_zeros() -> Result[Void]
  input = try Support.f32([1.0, 2.0, 3.0], [1, 1, 3])
  weight = try Support.f32([1.0, 1.0, 1.0], [1, 1, 3])
  out = try Tensor.conv1d(input, weight, 1, 1)
  try Test.eq(Tensor.shape(out), [1, 1, 3])
  Support.expect_floats(out, [3.0, 6.0, 5.0])
end

def test_conv1d_of_a_rank_two_input_has_no_batch_axis() -> Result[Void]
  input = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  weight = try Support.f32([1.0, -1.0], [1, 2, 1])
  out = try Tensor.conv1d(input, weight)
  try Test.eq(Tensor.shape(out), [1, 3])
  Support.expect_floats(out, [-3.0, -3.0, -3.0])
end

def test_conv1d_of_a_one_position_input_is_one_window() -> Result[Void]
  # The 1-sample edge: a single position and a kernel of one is one window.
  input = try Support.f32([2.5], [1, 1, 1])
  weight = try Support.f32([4.0], [1, 1, 1])
  out = try Tensor.conv1d(input, weight)
  try Test.eq(Tensor.shape(out), [1, 1, 1])
  Support.expect_floats(out, [10.0])
end

def test_conv1d_rejects_a_channel_count_mismatch() -> Result[Void]
  input = try Support.f32([1.0, 2.0, 3.0, 4.0], [1, 2, 2])
  weight = try Support.f32([1.0, 2.0, 3.0], [1, 3, 1])
  Test.error?(Tensor.conv1d(input, weight))
end

def test_conv1d_rejects_a_window_wider_than_the_input() -> Result[Void]
  input = try Support.f32([1.0, 2.0], [1, 1, 2])
  weight = try Support.f32([1.0, 1.0, 1.0], [1, 1, 3])
  Test.error?(Tensor.conv1d(input, weight))
end

def test_conv1d_rejects_a_bad_window() -> Result[Void]
  input = try Support.f32([1.0, 2.0], [1, 1, 2])
  weight = try Support.f32([1.0], [1, 1, 1])
  try Test.error?(Tensor.conv1d(input, weight, 0, 0))
  try Test.error?(Tensor.conv1d(input, weight, 1, -1))
  Test.error?(Tensor.conv1d(try Support.f32([], [1, 1, 0]), weight, 1, 0))
end

def test_conv1d_rejects_a_weight_that_is_not_rank_three() -> Result[Void]
  input = try Support.f32([1.0, 2.0], [1, 1, 2])
  weight = try Support.f32([1.0], [1])
  Test.error?(Tensor.conv1d(input, weight))
end

def test_conv1d_rejects_a_rank_one_input() -> Result[Void]
  input = try Support.f32([1.0, 2.0], [2])
  weight = try Support.f32([1.0], [1, 1, 1])
  Test.error?(Tensor.conv1d(input, weight))
end

def test_conv1d_rejects_an_int_dtype() -> Result[Void]
  input = try Support.i64([1, 2], [1, 2])
  weight = try Support.i64([1], [1, 1, 1])
  Test.error?(Tensor.conv1d(input, weight))
end

def test_avg_pool1d_matches_the_hand_values() -> Result[Void]
  input = try Support.f32([1.0, 2.0, 3.0, 4.0], [1, 1, 4])
  out = try Tensor.avg_pool1d(input, 2, 2, 0)
  try Test.eq(Tensor.shape(out), [1, 1, 2])
  Support.expect_floats(out, [1.5, 3.5])
end

def test_avg_pool1d_overlapping_windows_match_the_hand_values() -> Result[Void]
  input = try Support.f32([1.0, 2.0, 3.0, 4.0], [1, 1, 4])
  out = try Tensor.avg_pool1d(input, 2, 1, 0)
  try Test.eq(Tensor.shape(out), [1, 1, 3])
  Support.expect_floats(out, [1.5, 2.5, 3.5])
end

def test_avg_pool1d_with_padding_divides_by_the_kernel() -> Result[Void]
  # `count_include_pad`: a window that hangs off the edge averages over the
  # zeros it padded with, so the divisor is always the kernel.
  input = try Support.f32([1.0, 2.0, 3.0, 4.0], [1, 1, 4])
  out = try Tensor.avg_pool1d(input, 3, 1, 1)
  try Test.eq(Tensor.shape(out), [1, 1, 4])
  Support.expect_floats(out, [1.0, 2.0, 3.0, 7.0 / 3.0])
end

def test_avg_pool1d_of_a_rank_two_input_has_no_batch_axis() -> Result[Void]
  input = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  out = try Tensor.avg_pool1d(input, 2, 1, 0)
  try Test.eq(Tensor.shape(out), [2, 2])
  Support.expect_floats(out, [1.5, 2.5, 4.5, 5.5])
end

def test_avg_pool1d_of_a_one_position_input_is_one_window() -> Result[Void]
  input = try Support.f32([3.0], [1, 1, 1])
  out = try Tensor.avg_pool1d(input, 1, 1, 0)
  try Test.eq(Tensor.shape(out), [1, 1, 1])
  Support.expect_floats(out, [3.0])
end

def test_avg_pool1d_rejects_padding_wider_than_half_the_kernel() -> Result[Void]
  input = try Support.f32([1.0, 2.0, 3.0, 4.0], [1, 1, 4])
  Test.error?(Tensor.avg_pool1d(input, 2, 1, 2))
end

def test_avg_pool1d_rejects_a_bad_window() -> Result[Void]
  input = try Support.f32([1.0, 2.0], [1, 1, 2])
  try Test.error?(Tensor.avg_pool1d(input, 0, 1, 0))
  try Test.error?(Tensor.avg_pool1d(input, 1, 0, 0))
  try Test.error?(Tensor.avg_pool1d(input, 1, 1, -1))
  Test.error?(Tensor.avg_pool1d(input, 3, 1, 0))
end

def test_avg_pool1d_rejects_a_rank_one_input() -> Result[Void]
  input = try Support.f32([1.0, 2.0], [2])
  Test.error?(Tensor.avg_pool1d(input, 1, 1, 0))
end

def test_avg_pool1d_rejects_an_int_dtype() -> Result[Void]
  Test.error?(Tensor.avg_pool1d(try Support.i64([1, 2], [1, 2]), 1, 1, 0))
end
