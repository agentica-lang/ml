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

## True when the ggml CPU backend is there; with `TENSOR_REQUIRE_GGML` set, a
## missing one is an error, never a pass (as in `ggml_compile_test.agc`).
def ggml_present() -> Result[Bool]
  return ok(true) if Tensor.available(CPU)
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok(false) if !flag.ok?
  return ok(false) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end

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
  d = try Trace.diff_within(args, REF, Conform.TORCH, 0.001, 0.000001) do |xs|
    Tensor.to_dtype(try Tensor.to_dtype(xs[0], Tensor.Dtype.F16), Tensor.Dtype.F32)
  end

  try Test.eq(d.first, none)
  Test.eq(d.worst.size, 2)
end

def test_diff_with_an_exact_tolerance_names_the_first_inexact_node_on_ggml() -> Result[Void]
  return ok() if !try ggml_present()

  args = [try Support.f32([0.5, 0.25, 0.75], [3]), try Support.f32([0.25, 0.5, 0.125], [3])]
  d = try Trace.diff_within(args, REF, CPU, 0.0, 0.0) do |xs| Tensor.exp(try Tensor.add(xs[0], xs[1])) end

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
  return ok() if !try ggml_present()

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
  return ok() if !try ggml_present()

  table = try Support.f32([1.0, 2.0], [2])
  Test.error?(
    Trace.diff([try Support.f32([0.5, 0.5], [2])], REF, CPU) do |xs| Tensor.add(xs[0], Trace.constant(table)) end,
  )
end
