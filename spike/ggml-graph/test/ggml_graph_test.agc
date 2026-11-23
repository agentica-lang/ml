## One traced training step lowered to one ggml graph computes what the step
## `Train.fit` runs on the reference backend computes (`compare.agc`): on
## ggml's CPU backend within F32 rounding, and on the iGPU within the fp16
## `mul_mat` ggml's Vulkan backend uses by default on this device.
##
## "No Vulkan device" and "module not built" are stated skips, never silent
## passes.
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
## value.
CPU_TOLERANCE = 0.00001
## ggml's Vulkan `mul_mat` runs in fp16 here unless GGML_VK_DISABLE_F16 is
## set (`../ggml-vulkan/README.md`).
VULKAN_TOLERANCE = 0.003

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
    case Raw.backend_vulkan()
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

def test_the_graph_is_built_once_and_reused() -> Result[Void]
  return Test.assert(!ready(), "ggmlgraph is not built: build ggml (README)") if !ready()
  Native.arena do
    init = Step.init([8, 16, 4], 3)
    shapes = init.shapes.map { |s| Step.placeholder(s, Tensor.Dtype.F32) }
    x = Step.placeholder([5, 8], Tensor.Dtype.F32)
    y = Step.placeholder([5], Tensor.Dtype.I64)
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
