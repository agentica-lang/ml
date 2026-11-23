## `attention`, `rope`, `permute` and `batch_matmul` on ggml (CPU, and Vulkan when it is there) against the
## reference, node by node with `Trace.diff`: the forward graphs, and the backward graph of attention that
## training runs.

import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
REF = Tensor.Backend.Reference

private

## The ggml backends this machine has: the CPU one, and Vulkan when it is there. With the CPU one missing and
## `TENSOR_REQUIRE_GGML` unset, none.
def engines() -> Result[List[Tensor.Backend]]
  return ok([]) if !try Support.ggml_present()
  return ok([CPU, VULKAN]) if Tensor.available(VULKAN)
  ok([CPU])
end

## Two backends agree on every node of `f`, which has at least `nodes` of them.
def agree(
  what: String,
  backend: Tensor.Backend,
  args: List[Tensor.Tensor],
  nodes: Int,
  f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
) -> Result[Void]
  d = try Trace.diff(args, REF, backend, f: f)
  case d.first
    some(n) -> error("#{what} on #{backend}: node #{n.node} (#{n.op} #{n.shape}) differs by #{n.max_abs}")
    none -> Test.assert(d.nodes >= nodes, "#{what}: only #{d.nodes} nodes compared")
  end
end

def f32(seed: Int, shape: List[Int]) -> Result[Tensor.Tensor]
  Support.f32(Support.ramp(seed, shape.reduce(1) do |acc, n| acc * n end), shape)
end

def positions(at: List[Int]) -> Result[Tensor.Tensor]
  Tensor.from_list(at.map { |x| Tensor.Values.Int(x) }, [at.size], Tensor.Dtype.I32, REF)
end

def test_attention_agrees_over_grouped_heads_masks_and_the_causal_flag() -> Result[Void]
  q = try f32(1, [2, 4, 3, 8])
  k = try f32(2, [2, 2, 5, 8])
  v = try f32(3, [2, 2, 5, 6])
  per_head = try f32(4, [2, 1, 3, 5])
  shared = try f32(5, [3, 5])
  for backend in try engines()
    try agree("attention", backend, [q, k, v], 1) do |a| Tensor.attention(a[0], a[1], a[2]) end

    try agree("causal attention", backend, [q, k, v], 6) do |a|
      Tensor.attention(a[0], a[1], a[2], none, none, true)
    end

    try agree("masked attention", backend, [q, k, v, shared], 1) do |a|
      Tensor.attention(a[0], a[1], a[2], some(a[3]), some(0.3))
    end

    try agree("causal attention with a mask per batch", backend, [q, k, v, per_head], 6) do |a|
      Tensor.attention(a[0], a[1], a[2], some(a[3]), none, true)
    end
  end

  ok()
end

def test_rope_permute_and_batch_matmul_agree() -> Result[Void]
  x = try f32(6, [2, 3, 5, 8])
  at = try positions([0, 1, 7, 8, 20])
  a = try f32(7, [2, 3, 4, 5])
  b = try f32(8, [2, 3, 5, 6])
  for backend in try engines()
    for neox in [true, false]
      for dims in [8, 6, 2]
        try agree("rope", backend, [x, at], 1) do |t| Tensor.rope(t[0], t[1], 1000.0, some(dims), neox) end
      end
    end

    try agree("inverse rope", backend, [x, at], 1) do |t| Tensor.rope(t[0], t[1], 500.0, none, true, true) end

    try agree("permute", backend, [x], 1) do |t| Tensor.permute(t[0], [3, 1, 0, 2]) end

    try agree("batch_matmul", backend, [a, b], 1) do |t| Tensor.batch_matmul(t[0], t[1]) end
  end

  ok()
end

def test_the_backward_of_attention_agrees() -> Result[Void]
  q = try f32(9, [1, 4, 3, 4])
  k = try f32(10, [1, 2, 5, 4])
  v = try f32(11, [1, 2, 5, 4])
  per_head = try f32(12, [1, 4, 3, 5])
  at = try positions([0, 1, 2])
  for backend in try engines()
    try agree("the gradient of attention", backend, [q, k, v], 20) do |a|
      loss = try Tensor.sum(try Tensor.attention(a[0], a[1], a[2], none, none, true))
      sum_of(try Trace.gradients(loss, a))
    end

    try agree("the gradient of a masked attention", backend, [q, k, v, per_head], 20) do |a|
      loss = try Tensor.sum(try Tensor.attention(a[0], a[1], a[2], some(a[3])))
      sum_of(try Trace.gradients(loss, [a[0], a[1], a[2]]))
    end

    try agree("the gradient of rope", backend, [q, at], 3) do |a|
      turned = try Tensor.rope(a[0], a[1])
      loss = try Tensor.sum(try Tensor.mul(turned, turned))
      sum_of(try Trace.gradients(loss, [a[0]]))
    end
  end

  ok()
end

## One scalar that reaches every gradient in `grads`, so the diff walks all of them.
def sum_of(grads: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  total = try Tensor.sum(grads[0])
  for g in grads.drop(1)
    total = try Tensor.add(total, try Tensor.sum(g))
  end

  ok(total)
end
