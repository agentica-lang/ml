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

## `layer_norm` is one node of a trace: ggml's norm and the reference's composition agree on it within the default
## tolerance, over a vector, a single column and a rank 3 tensor, with a big eps that a dropped eps would show.
def test_diff_of_a_layer_norm_on_ggml_finds_nothing() -> Result[Void]
  return ok() if !try Support.ggml_present()

  for shape in [[5], [2, 1], [2, 2, 3]]
    n = shape.reduce(1) do |acc, d| acc * d end
    data = (0...n).map { |i| 0.7 * (i % 5).to_f - 1.1 + 0.05 * i.to_f }
    args = [try Support.f32(data, shape)]
    d = try Trace.diff(args, REF, CPU) do |xs| Tensor.layer_norm(xs[0], 0.1) end
    try Test.eq(d.nodes, 1)
    case d.first
      some(node) -> return error("#{shape}: the norm differs from the reference: #{node.max_abs}")
      none -> nil
    end
  end

  ok()
end

## A max over one row of 1024 floats or more is found with argmax on ggml (the pool of ggml-vulkan walks a row in one
## thread); a max over many rows is not. Both agree with the reference, with the largest value present twice.
def test_diff_of_a_long_row_max_finds_nothing_on_cpu_and_vulkan() -> Result[Void]
  return ok() if !try Support.ggml_present()

  backends = [CPU]
  vulkan = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
  if Tensor.available(vulkan)
    backends = backends.push(vulkan)
  end

  for backend in backends
    for shape in [[1500], [8192], [3, 2000], [1, 3000]]
      n = shape.reduce(1) do |acc, d| acc * d end
      ramp = Support.ramp(3, n)
      data = ramp.take(7) + [9.5] + ramp.drop(8).take(n - 9) + [9.5]
      args = [try Support.f32(data, shape)]
      d = try Trace.diff(args, REF, backend) do |xs| Tensor.max(xs[0], some([shape.size - 1]), false) end
      case d.first
        some(node) -> return error("#{shape} on #{backend}: max differs from the reference: #{node.max_abs}")
        none -> nil
      end
    end
  end

  ok()
end

## A long row is cut into k pieces of m floats (m the ceiling of its square root), the last piece overlapping the
## others when m does not divide it: the largest value is found wherever it lies, at either end of a piece or only in
## the overlapping tail (1500 = 38 pieces of 39 and a tail; 1521 = 39 * 39; 151936, a vocabulary).
def test_a_long_row_max_finds_the_largest_at_the_seams_of_its_pieces() -> Result[Void]
  return ok() if !try Support.ggml_present()

  backends = [CPU]
  vulkan = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
  if Tensor.available(vulkan)
    backends = backends.push(vulkan)
  end

  for n in [1500, 1521, 151936]
    m = Math.sqrt(n.to_f).ceil
    k = n / m
    ramp = Support.ramp(5, n)
    seams = [0, m - 1, m, k * m - 1, k * m, n - m, n - 1].uniq.select { |i| i < n }
    for at in seams
      data = ramp.set(at, 3.0)
      args = [try Support.f32(data, [n])]
      for backend in backends
        d = try Trace.diff(args, REF, backend) do |xs| Tensor.max(xs[0], some([0]), false) end
        case d.first
          some(node) -> return error("#{n} floats, the largest at #{at} on #{backend}: #{node.max_abs}")
          none -> nil
        end
      end
    end
  end

  ok()
end

## `silu(gate) * up`, an MLP's gate, is one swiglu on ggml, either way round, and agrees with the reference; the silu
## alone is still there when something else reads it.
def test_silu_times_up_on_ggml_matches_the_reference() -> Result[Void]
  return ok() if !try Support.ggml_present()

  backends = [CPU]
  vulkan = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
  if Tensor.available(vulkan)
    backends = backends.push(vulkan)
  end

  args = [
    try Support.f32(Support.ramp(7, 24).map { |x| 4.0 * x }, [2, 12]),
    try Support.f32(Support.ramp(8, 24), [2, 12]),
  ]

  for backend in backends
    for d in [
      try Trace.diff(args, REF, backend) do |xs| Tensor.mul(try Tensor.silu(xs[0]), xs[1]) end,
      try Trace.diff(args, REF, backend) do |xs| Tensor.mul(xs[1], try Tensor.silu(xs[0])) end,
      try Trace.diff(args, REF, backend) do |xs|
        gate = try Tensor.silu(xs[0])
        Tensor.add(try Tensor.mul(gate, xs[1]), gate)
      end,
    ]
      case d.first
        some(node) -> return error("on #{backend}: #{node.op} differs from the reference by #{node.max_abs}")
        none -> nil
      end
    end
  end

  ok()
end

## `rms_norm` is one node of a trace, like `layer_norm`, and its gradient on ggml is the reference's.
def test_rms_norm_on_ggml_matches_the_reference() -> Result[Void]
  return ok() if !try Support.ggml_present()

  for shape in [[5], [2, 1], [2, 2, 3]]
    n = shape.reduce(1) do |acc, d| acc * d end
    data = (0...n).map { |i| 0.7 * (i % 5).to_f - 1.1 + 0.05 * i.to_f }
    d = try Trace.diff([try Support.f32(data, shape)], REF, CPU) do |xs| Tensor.rms_norm(xs[0], 0.1) end
    try Test.eq(d.nodes, 1)
    case d.first
      some(node) -> return error("#{shape}: the rms norm differs from the reference: #{node.max_abs}")
      none -> nil
    end
  end

  have = try Support.floats_of(try normed_grad(CPU, true))
  exact = try Support.floats_of(try normed_grad(REF, true))
  for j in 0...exact.size
    return error("gradient #{j}: #{have[j]} against #{exact[j]}") if (have[j] - exact[j]).abs > 0.0001
  end

  ok()
end

## The backward of a `layer_norm` is lowered once on ggml, and gives the reference's gradient.
def test_the_gradient_of_a_layer_norm_on_ggml_matches_the_reference() -> Result[Void]
  return ok() if !try Support.ggml_present()

  want = try normed_grad(REF, false)
  got = try normed_grad(CPU, false)
  have = try Support.floats_of(got)
  exact = try Support.floats_of(want)
  for j in 0...exact.size
    return error("gradient #{j}: #{have[j]} against #{exact[j]}") if (have[j] - exact[j]).abs > 0.0001
  end

  ok()
end

## The gradient of `sum(layer_norm(x) * w)` (`rms_norm` with `rms`) at a fixed `x` on `backend`.
def normed_grad(backend: Tensor.Backend, rms: Bool) -> Result[Tensor.Tensor]
  data = [0.5, -1.0, 2.0, 0.3, 1.2, -0.7, 0.9, 0.1, -0.4, 1.7, -1.3, 0.6]
  weights = [0.3, -0.2, 0.7, 0.1, -0.5, 0.4, 0.9, -0.1, 0.2, -0.6, 0.8, 0.05]
  x = try on(backend, data)
  w = try on(backend, weights)
  step = try Trace.value_and_grad([x]) do |xs| weighed_norm(xs[0], Trace.constant(w), rms) end

  ok(step.grads[0])
end

## `sum(norm(x) * w)`, with the rms norm when `rms`.
def weighed_norm(x: Tensor.Tensor, w: Tensor.Tensor, rms: Bool) -> Result[Tensor.Tensor]
  return Tensor.sum(try Tensor.mul(try Tensor.rms_norm(x, 0.1), w)) if rms
  Tensor.sum(try Tensor.mul(try Tensor.layer_norm(x, 0.1), w))
end

def on(backend: Tensor.Backend, data: List[Float]) -> Result[Tensor.Tensor]
  Tensor.from_list(data.map { |x| Tensor.Values.Float(x) }, [2, 6], Tensor.Dtype.F32, backend)
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
