## `Tensor.top_ids`: the ids of the k largest of a vector. Descending on the reference, a set on ggml (`ggml_top_k` gives
## no order), the same set, as the node of a trace.

import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

REF = Tensor.Backend.Reference
CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)

private

def vector(backend: Tensor.Backend) -> Result[Tensor.Tensor]
  # distinct values, the largest at ids 7, 40, 3, 19, 0 in that order
  values = (0...64).map { |i| ((i * 37) % 64).to_f }
  Tensor.floats(values, [64], backend)
end

def ids_of(t: Tensor.Tensor) -> Result[List[Int]]
  out: List[Int] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Int(i) -> out = out.push(i)
      _ -> return error("ids are integers")
    end
  end

  ok(out)
end

def test_the_reference_gives_the_largest_first() -> Result[Void]
  x = try Tensor.floats([3.0, 1.0, 4.0, 1.5, 5.0], [5], REF)
  top = try Tensor.top_ids(x, 3)
  try Test.eq(Tensor.dtype(top), Tensor.Dtype.I32)
  try Test.eq(Tensor.shape(top), [3])
  try Test.eq(try ids_of(top), [4, 2, 0])
  Test.eq(try ids_of(try Tensor.top_ids(x, 5)), [4, 2, 0, 3, 1])
end

def test_bad_arguments_are_errors() -> Result[Void]
  x = try Tensor.floats([3.0, 1.0, 4.0], [3], REF)
  try Test.error?(Tensor.top_ids(x, 0))
  try Test.error?(Tensor.top_ids(x, 4))
  try Test.error?(Tensor.top_ids(try Tensor.floats([1.0, 2.0, 3.0, 4.0], [2, 2], REF), 1))
  Test.error?(Tensor.top_ids(try Tensor.from_list([Tensor.Values.Int(1)], [1], Tensor.Dtype.I32, REF), 1))
end

## In a trace on ggml the node returns the same ids, in some order.
def test_a_ggml_trace_gives_the_same_set_of_ids() -> Result[Void]
  return ok() if !try Support.ggml_present()

  want = try ids_of(try Tensor.top_ids(try vector(REF), 9))
  got = try Trace.eval([try vector(CPU)]) do |xs| Tensor.top_ids(xs[0], 9) end
  try Test.eq(Tensor.dtype(got), Tensor.Dtype.I32)
  found = try ids_of(got)
  try Test.eq(found.sort, want.sort)
  # the whole vector: every id once
  all = try Trace.eval([try vector(CPU)]) do |xs| Tensor.top_ids(xs[0], 64) end
  Test.eq((try ids_of(all)).sort, (0...64).to_a)
end
