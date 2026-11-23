## A compiled graph without fills runs first in its process (so the shared
## allocator starts at its small size), a larger graph grows the allocator and
## frees the small buffer, a new tensor of that size likely takes the freed
## memory, and the first graph runs again: it must place its intermediates in
## the allocator's new buffer, not write over the new tensor. A file of its
## own, because the order matters and the runner shuffles a file's tests.
## Whether the freed memory is reused is up to the allocator (glibc here), so
## without the fix the test fails on this machine, not by contract.

import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)

private

def test_a_graph_without_fills_takes_turns_with_a_larger_one() -> Result[Void]
  if !Tensor.available(CPU)
    flag = System.env("TENSOR_REQUIRE_GGML")
    return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)") if flag.ok? and try flag != ""
    return ok()
  end

  x = try floats((0...8).to_a.map { |i| i.to_f }, [4, 2])
  w = try floats((0...32).to_a.map { |i| (i % 5).to_f }, [2, 16])
  _ = try Trace.eval_many([x], &twice_negated)
  _ = try Trace.value_and_grad([x, w], &loss)
  sevens = (0...16).to_a.map { |_| 7.0 }
  bystander = try floats(sevens, [16])
  got = try Trace.eval_many([x], &twice_negated)
  try Test.eq(try Tensor.to_list(got[0]), try Tensor.to_list(x))
  Test.eq(try Tensor.to_list(bystander), sevens.map { |v| Tensor.Values.Float(v) })
end

def twice_negated(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  ok([try Tensor.neg(try Tensor.neg(a[0])), try Tensor.neg(a[0])])
end

def loss(a: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  Tensor.sum(try Tensor.exp(try Tensor.relu(try Tensor.matmul(a[0], a[1]))))
end

def floats(xs: List[Float], shape: List[Int]) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, CPU)
end
