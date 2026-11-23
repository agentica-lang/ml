## One traced training step lowered to one ggml graph computes what the step
## `Train.fit` runs on the reference backend computes (`compare.agc`), within
## F32 rounding, on ggml's CPU backend and on the iGPU with F32 `mul_mat` (the
## spike's default; fp16 is opt-in and measured by `check.agc fp16`).
##
## Without the module or a Vulkan device, a test passes and says why in its
## assertion message: the runner prints only PASS, so the reason is in the
## source, not in the output (note.md §5).
##
## ```sh
## agentica test
## ```

import Compare from "../compare"
import Lower from "../lower"
import Raw from "../ggml_graph"
import Step from "../step"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## F32 rounding through a few hundred ops, relative to each tensor's largest
## value (measured: 2.3e-7 at most).
CPU_TOLERANCE = 0.00001
## About twice the largest gap measured on the iGPU with F32 `mul_mat`
## (6e-7, on the benchmark's model).
VULKAN_TOLERANCE = 0.0000015

def test_sgd_one_graph_matches_the_reference_on_the_cpu() -> Result[Void]
  return Test.assert(!ready(), "ggmlgraph is not built: build ggml (README)") if !ready()
  Native.arena do
    gaps = try Compare.sgd(try Raw.backend_cpu(2))
    within(gaps, CPU_TOLERANCE, "cpu sgd")
  end
end

def test_sgd_two_graphs_match_the_reference_on_the_cpu() -> Result[Void]
  return Test.assert(!ready(), "ggmlgraph is not built: build ggml (README)") if !ready()
  Native.arena do
    gaps = try Compare.sgd_two(try Raw.backend_cpu(2))
    within(gaps, CPU_TOLERANCE, "cpu sgd two graphs")
  end
end

def test_two_adam_steps_match_the_reference_on_the_cpu() -> Result[Void]
  return Test.assert(!ready(), "ggmlgraph is not built: build ggml (README)") if !ready()
  Native.arena do
    gaps = try Compare.adam(try Raw.backend_cpu(2))
    within(gaps, CPU_TOLERANCE, "cpu adam")
  end
end

def test_the_igpu_step_matches_the_reference() -> Result[Void]
  return Test.assert(!ready(), "ggmlgraph is not built: build ggml (README)") if !ready()
  Native.arena do
    case Raw.backend_vulkan(false)
      ok(gpu) ->
        description = try Raw.describe(gpu)
        try Test.assert(description.include?("GPU)"), "the vulkan backend is #{description}")
        try within(try Compare.sgd(gpu), VULKAN_TOLERANCE, "vulkan sgd")
        try within(try Compare.sgd_two(gpu), VULKAN_TOLERANCE, "vulkan sgd two graphs")
        try within(try Compare.adam(gpu), VULKAN_TOLERANCE, "vulkan adam")
      error(e) -> try Test.assert(true, "no Vulkan device on this machine: #{e.message}")
    end

    ok()
  end
end

## The benchmark's model (784 → 512 → 512 → 10) at batch 256, two SGD steps,
## on the iGPU against ggml's CPU backend. The reference backend cannot be
## the oracle here: one of its steps at this size takes 16 s and ~10^9 VM
## steps, past the runner's 30 s and step budget per test. `check.agc` runs
## that comparison (ggml CPU against the reference: 2e-7).
def test_two_steps_of_the_benchmark_model_agree_between_cpu_and_igpu() -> Result[Void]
  return Test.assert(!ready(), "ggmlgraph is not built: build ggml (README)") if !ready()
  Native.arena do
    case Raw.backend_vulkan(false)
      ok(gpu) ->
        within(try Compare.cross(gpu, try Raw.backend_cpu(4)), VULKAN_TOLERANCE, "vulkan against cpu, benchmark model")
      error(e) -> Test.assert(true, "no Vulkan device on this machine: #{e.message}")
    end
  end
end

def test_a_write_that_reads_a_replaced_state_is_refused() -> Result[Void]
  return Test.assert(!ready(), "ggmlgraph is not built: build ggml (README)") if !ready()
  Native.arena do
    a = Trace.constant(Step.placeholder([2], Tensor.Dtype.F32))
    b = Trace.constant(Step.placeholder([2], Tensor.Dtype.F32))
    swap = [Lower.Write(state: a, value: b), Lower.Write(state: b, value: a)]
    step = Lower.Step(outputs: [], states: [a, b], inputs: [], writes: swap)
    case Lower.lower(try Raw.backend_cpu(1), step)
      ok(_) -> Test.assert(false, "a swap of two states compiled")
      error(e) -> Test.assert(e.message.include?("another writeback replaces"), e.message)
    end
  end
end

def test_a_nan_or_an_infinity_is_an_error_not_a_gap() -> Result[Void]
  inf = 1e300 * 1e300
  try Test.error?(Compare.rel([inf], [1.0]))
  try Test.error?(Compare.rel([1.0], [0.0 - inf]))
  try Test.error?(Compare.rel([inf - inf], [1.0]))
  Test.eq(Compare.rel([1.5], [1.0]), ok(0.5))
end

def test_the_graph_is_built_once_and_reused() -> Result[Void]
  return Test.assert(!ready(), "ggmlgraph is not built: build ggml (README)") if !ready()
  Native.arena do
    init = Step.init([8, 16, 4], 3)
    shapes = init.shapes.map { |s| Step.placeholder(s, Tensor.Dtype.F32) }
    x = Step.placeholder([5, 8], Tensor.Dtype.F32)
    y = Step.placeholder([5], Tensor.Dtype.I32)
    b = try Lower.lower(try Raw.backend_cpu(1), try Step.sgd_step(shapes, x, y, 0.5, false))
    for k in 0...init.values.size
      try Raw.set(b.program, b.states[k], init.values[k])
    end

    try Raw.batches(b.program, b.inputs[0], b.inputs[1], 4, 1, 9)
    first = try Raw.run(b.program, 0)
    last = first
    for i in 1...20
      last = try Raw.run(b.program, i)
    end

    stats = try Raw.stats(b.program)
    try Test.assert(stats[4] == 20, "the program ran #{stats[4]} times")
    Test.assert(last < first, "twenty steps on one batch did not lower the loss: #{first} then #{last}")
  end
end

def test_an_op_with_no_lowering_is_an_error_naming_it() -> Result[Void]
  return Test.assert(!ready(), "ggmlgraph is not built: build ggml (README)") if !ready()
  Native.arena do
    x = Step.placeholder([2, 3], Tensor.Dtype.F32)
    traced = Trace.constant(x)
    out = try Tensor.gelu_tanh_approx(traced)
    step = Lower.Step(outputs: [out], states: [], inputs: [traced], writes: [])
    case Lower.lower(try Raw.backend_cpu(1), step)
      ok(_) -> Test.assert(false, "gelu_tanh_approx lowered")
      error(e) -> Test.assert(e.message.include?("gelu_tanh_approx is not lowered"), e.message)
    end
  end
end

private

def ready() -> Bool = Native.load("ggmlgraph").ok?

def within(g: Compare.Gaps, tolerance: Float, name: String) -> Result[Void]
  Test.assert(
    Compare.worst(g) <= tolerance,
    "#{name}: loss gap #{g.loss}, gradients #{g.grads}, parameters #{g.params} (tolerance #{tolerance})",
  )
end
