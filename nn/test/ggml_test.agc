## A stack of configured blocks (grouped-query attention, rope, query and key norms, RMS norms with random
## gammas, a gated MLP, causal) on ggml (CPU, and Vulkan when it is there) against the reference, node by node
## with `Trace.diff`. Where ggml is not built, the test passes, unless `TENSOR_REQUIRE_GGML` is set.

import Init from "../init"
import Nn from "../nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

private

def test_a_configured_stack_agrees_with_the_reference_on_ggml() -> Result[Void]
  backends = try engines()
  options = Nn.BlockOptions(
    kv_heads: some(2),
    rope: some(Nn.Rope()),
    qk_norm: true,
    bias: false,
    rms: true,
    gated: true,
    eps: 0.000001,
  )

  start = try Nn.init_encoder_stack(2, 16, 4, 24, Nn.Activation.Silu, 71, S.REF, options)
  stack = try S.jitter_encoder_stack(start, 60)
  weights = S.qwen_params(stack.blocks[0]) + S.qwen_params(stack.blocks[1]) + [stack.final_norm.unwrap!.gamma]
  args = [try S.f32(Init.uniform_values(96, 1.0, 72), [2, 3, 16])] + weights
  for backend in backends
    d = try Trace.diff(
      args,
      S.REF,
      backend,
      rtol: 0.001,
      atol: 0.0001,
      f: ->(a: List[Tensor.Tensor]) { run(stack, a) },
    )

    case d.first
      some(n) -> return error("on #{backend}: node #{n.node} (#{n.op} #{n.shape}) differs by #{n.max_abs}")
      none -> try Test.assert(d.nodes > 100, "only #{d.nodes} nodes compared")
    end
  end

  ok()
end

## The stack with the weights in `a` (`a[0]` is the input), run causally.
def run(stack: Nn.EncoderStack, a: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  blocks = [S.qwen_block(stack.blocks[0], a.drop(1).take(11)), S.qwen_block(stack.blocks[1], a.drop(12).take(11))]
  final = stack.final_norm.unwrap!.with(gamma: a[23])
  Nn.encoder_stack(Nn.EncoderStack(blocks: blocks, final_norm: some(final)), a[0], none, true)
end

## The ggml backends this machine has: the CPU one, and Vulkan when it is there.
## With the CPU one missing and `TENSOR_REQUIRE_GGML` unset, none.
def engines() -> Result[List[Tensor.Backend]]
  return ok([CPU, VULKAN]) if Tensor.available(CPU) and Tensor.available(VULKAN)
  return ok([CPU]) if Tensor.available(CPU)
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok([]) if !flag.ok? or try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end
