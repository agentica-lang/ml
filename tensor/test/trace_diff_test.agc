## `Trace.diff`: the first node where two backends disagree, found on real
## disagreements (the precision cast of libtorch, an exact tolerance on ggml)
## and across the slices a big graph is run in.

import Conform from "torch_support"
import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
REF = Tensor.Backend.Reference

private

def test_diff_names_the_precision_cast_that_libtorch_rounds() -> Result[Void]
  return Conform.absent() if !Conform.available()

  args = [try Support.f32([0.1, 0.2], [2]), try Support.f32([0.0, 0.0], [2])]
  d = try Trace.diff(args, REF, Conform.TORCH) do |xs|
    sum = try Tensor.add(xs[0], xs[1])
    Tensor.to_dtype(try Tensor.to_dtype(sum, Tensor.Dtype.F16), Tensor.Dtype.F32)
  end

  try Test.eq(d.nodes, 3)
  case d.first
    some(node) ->
      try Test.eq(node.op, "to_dtype")
      try Test.eq(node.shape, [2])
      try Test.eq(node.inputs, [[2]])
      try Test.eq(node.dtype, Tensor.Dtype.F16)
      try Test.assert(node.max_abs > 0.00004 and node.max_abs < 0.00006, "max_abs #{node.max_abs}")
      Test.assert(node.max_rel > 0.0002 and node.max_rel < 0.0003, "max_rel #{node.max_rel}")
    none -> Test.assert(false, "diff found no disagreement")
  end
end

def test_diff_with_a_looser_tolerance_finds_nothing() -> Result[Void]
  return Conform.absent() if !Conform.available()

  args = [try Support.f32([0.1, 0.2], [2])]
  d = try Trace.diff(args, REF, Conform.TORCH, rtol: 0.001) do |xs|
    Tensor.to_dtype(try Tensor.to_dtype(xs[0], Tensor.Dtype.F16), Tensor.Dtype.F32)
  end

  try Test.eq(d.first, none)
  Test.eq(d.worst.size, 2)
end

def test_diff_with_an_exact_tolerance_names_the_first_inexact_node_on_ggml() -> Result[Void]
  return ok() if !try Support.ggml_present()

  args = [try Support.f32([0.5, 0.25, 0.75], [3]), try Support.f32([0.25, 0.5, 0.125], [3])]
  d = try Trace.diff(args, REF, CPU, rtol: 0.0, atol: 0.0) do |xs| Tensor.exp(try Tensor.add(xs[0], xs[1])) end

  try Test.eq(d.nodes, 2)
  case d.first
    some(node) ->
      try Test.eq(node.op, "exp")
      try Test.eq(node.inputs, [[3]])
      Test.assert(node.max_abs > 0.0 and node.max_abs < 0.000001, "max_abs #{node.max_abs}")
    none -> Test.assert(false, "diff found no disagreement")
  end
end

def test_diff_runs_a_big_graph_in_slices() -> Result[Void]
  return ok() if !try Support.ggml_present()

  args = [try Support.f32([0.25, 0.5], [2]), try Support.f32([0.125, 0.0625], [2])]
  d = try Trace.diff(args, REF, CPU) do |xs|
    acc = xs[0]
    for _ in 0...450
      acc = try Tensor.add(acc, xs[1])
    end

    ok(acc)
  end

  try Test.eq(d.nodes, 450)
  try Test.eq(d.first, none)
  Test.eq(d.worst.size, 5)
end

def test_diff_refuses_a_constant_that_lives_on_one_backend() -> Result[Void]
  return ok() if !try Support.ggml_present()

  table = try Support.f32([1.0, 2.0], [2])
  Test.error?(
    Trace.diff([try Support.f32([0.5, 0.5], [2])], REF, CPU) do |xs| Tensor.add(xs[0], Trace.constant(table)) end,
  )
end

def test_diff_compares_integers_exactly() -> Result[Void]
  return Conform.absent() if !Conform.available()

  # F32 holds 51000.9999999 as 51001.0 on libtorch: the cast to I32 gives 51001 there and 51000 on the reference,
  # a gap that a relative tolerance of 1e-4 would let through.
  d = try Trace.diff([try Support.f32([51000.9999999], [1])], REF, Conform.TORCH) do |xs|
    Tensor.to_dtype(xs[0], Tensor.Dtype.I32)
  end

  case d.first
    some(node) ->
      try Test.eq(node.op, "to_dtype")
      try Test.eq(node.dtype, Tensor.Dtype.I32)
      Test.eq(node.max_abs, 1.0)
    none -> Test.assert(false, "an integer gap of 1 passed")
  end
end

def test_diff_agrees_on_a_nan_that_both_sides_have() -> Result[Void]
  return Conform.absent() if !Conform.available()

  infinity = 1.0e308 * 10.0
  d = try Trace.diff([try Support.f32([infinity - infinity, 1.0], [2])], REF, Conform.TORCH) do |xs|
    Tensor.add(xs[0], xs[0])
  end

  Test.eq(d.first, none)
end

def test_diff_refuses_a_tolerance_that_is_not_finite_or_is_negative() -> Result[Void]
  args = [try Support.f32([1.0], [1])]
  try Test.error?(Trace.diff(args, REF, REF, rtol: -1.0) do |xs| Tensor.neg(xs[0]) end)
  Test.error?(Trace.diff(args, REF, REF, atol: 1.0e308 * 10.0) do |xs| Tensor.neg(xs[0]) end)
end

def test_diff_indexes_a_divergence_planted_in_the_second_slice() -> Result[Void]
  return ok() if !try Support.ggml_present()

  args = [try Support.f32([0.25, 0.5], [2]), try Support.f32([0.125, 0.0625], [2])]
  d = try Trace.diff(args, REF, CPU, rtol: 0.0, atol: 0.0) do |xs|
    acc = xs[0]
    for _ in 0...250
      acc = try Tensor.add(acc, xs[1])
    end

    acc = try Tensor.exp(acc)
    for _ in 0...100
      acc = try Tensor.add(acc, xs[1])
    end

    ok(acc)
  end

  try Test.eq(d.nodes, 351)
  case d.first
    some(node) -> Test.eq(node.op, "exp")
    none -> Test.assert(false, "diff found no disagreement")
  end
end
