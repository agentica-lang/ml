## `Tensor.expert_matmul`: the reference against PyTorch's `torch.matmul` per `(token, use)`
## (`expert_matmul_golden.py`, `fixtures/expert_matmul_golden.json`), `ggml_mul_mat_id` (a trace on ggml) against the
## reference, with the ids made in the graph by `top_ids`, libtorch, and what the op refuses. Run from `tensor/`.

import Conform from "torch_support"
import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

REF = Tensor.Backend.Reference
CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)

## The golden is in `float32`, the reference in doubles.
GOLDEN_TOL = 0.00001

## ggml computes in `F32` too, but sums in another order.
F32_TOL = 0.0001

struct Case
  experts: Int
  rows: Int
  inner: Int
  tokens: Int
  uses: Int
  shared: Bool
  weights: List[Float]
  x: List[Float]
  ids: List[Int]
  out: List[Float]
end

struct Golden
  cases: List[Case]
end

struct Inputs
  weights: Tensor.Tensor
  x: Tensor.Tensor
  ids: Tensor.Tensor
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/expert_matmul_golden.json"), Golden)

def ids_on(ids: List[Int], tokens: Int, uses: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(ids.map { |i| Tensor.Values.Int(i) }, [tokens, uses], Tensor.Dtype.I32, backend)
end

def inputs_of(c: Case, backend: Tensor.Backend) -> Result[Inputs]
  ok(
    Inputs(
      weights: try Tensor.floats(c.weights, [c.experts, c.rows, c.inner], backend),
      x: try Tensor.floats(c.x, [c.tokens, if_shared(c), c.inner], backend),
      ids: try ids_on(c.ids, c.tokens, c.uses, backend),
    ),
  )
end

def if_shared(c: Case) -> Int
  return 1 if c.shared
  c.uses
end

def close(got: Tensor.Tensor, want: List[Float], tol: Float, what: String) -> Result[Void]
  values = try Support.floats_of(got)
  try Test.eq(values.size, want.size)
  worst = 0.0
  for i in 0...values.size
    gap = (values[i] - want[i]).abs
    if gap > worst
      worst = gap
    end
  end

  Test.assert(worst <= tol, "#{what} differs by #{worst}")
end

def test_the_reference_matches_pytorch() -> Result[Void]
  cases = (try golden()).cases
  try Test.eq(cases.size, 6)
  for c in cases
    x = try inputs_of(c, REF)
    out = try Tensor.expert_matmul(x.weights, x.x, x.ids)
    try Test.eq(Tensor.shape(out), [c.tokens, c.uses, c.rows])
    try close(out, c.out, GOLDEN_TOL, "out")
  end

  ok()
end

## A trace on the reference runs the node as `Trace.compute` does.
def test_a_trace_on_the_reference_matches_pytorch() -> Result[Void]
  for c in (try golden()).cases
    x = try inputs_of(c, REF)
    out = try Trace.eval([x.weights, x.x, x.ids]) do |t| Tensor.expert_matmul(t[0], t[1], t[2]) end
    try close(out, c.out, GOLDEN_TOL, "out")
  end

  ok()
end

def test_ggml_matches_pytorch_as_a_trace() -> Result[Void]
  return ok() if !try Support.ggml_present()

  for c in (try golden()).cases
    x = try inputs_of(c, CPU)
    out = try Trace.eval([x.weights, x.x, x.ids]) do |t| Tensor.expert_matmul(t[0], t[1], t[2]) end
    try Test.eq(Tensor.shape(out), [c.tokens, c.uses, c.rows])
    try close(out, c.out, F32_TOL, "out")
  end

  ok()
end

## The way a router uses it: two products by the ids that `top_ids` made in the same graph from the scores of each
## token, the second by the output of the first. ggml gives the ids in no order, so the uses are summed.
def routed(backend: Tensor.Backend) -> Result[Tensor.Tensor]
  experts = 6
  tokens = 5
  Trace.eval([
    try Tensor.floats(Support.ramp(1, experts * 4 * 3), [experts, 4, 3], backend),
    try Tensor.floats(Support.ramp(2, experts * 3 * 4), [experts, 3, 4], backend),
    try Tensor.floats(Support.ramp(3, tokens * experts), [tokens, experts], backend),
    try Tensor.floats(Support.ramp(4, tokens * 3), [tokens, 1, 3], backend),
  ]) do |t|
    ids = try Tensor.top_ids(t[2], 3)
    both = try Tensor.expert_matmul(t[1], try Tensor.expert_matmul(t[0], t[3], ids), ids)
    Tensor.sum(try Tensor.permute(both, [0, 2, 1]), some([2]), false)
  end
end

def test_ggml_runs_the_ids_of_top_ids_in_the_graph() -> Result[Void]
  return ok() if !try Support.ggml_present()

  close(try routed(CPU), try Support.floats_of(try routed(REF)), F32_TOL, "out")
end

## libtorch runs the loop over its own ops. Its `F32` sums differ from the `Float64` reference by a few ulps of the inputs,
## more than `Conform.agrees` allows for a result near zero, so it is held to PyTorch's output of the fixture, as ggml is.
def test_libtorch_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  c = (try golden()).cases[0]
  ours: Result[Tensor.Tensor] = Native.arena do
    Tensor.expert_matmul(
      try Conform.on_torch(c.weights, [c.experts, c.rows, c.inner]),
      try Conform.on_torch(c.x, [c.tokens, c.uses, c.inner]),
      try Tensor.from_list(
        c.ids.map { |i| Tensor.Values.Int(i) },
        [c.tokens, c.uses],
        Tensor.Dtype.I32,
        Conform.TORCH,
      ),
    )
  end

  close(try ours, c.out, F32_TOL, "out")
end

def test_there_is_no_gradient() -> Result[Void]
  c = (try golden()).cases[0]
  x = try inputs_of(c, REF)
  Test.error?(
    Trace.grad([x.x]) do |t|
      Tensor.sum(try Tensor.expert_matmul(Trace.constant(x.weights), t[0], Trace.constant(x.ids)))
    end,
  )
end

def test_an_id_outside_the_experts_is_an_error() -> Result[Void]
  c = (try golden()).cases[0]
  x = try inputs_of(c, REF)
  try Test.error?(Tensor.expert_matmul(x.weights, x.x, try ids_on([0, 5, 2, 3, 1, 0], c.tokens, c.uses, REF)))
  try Test.error?(Tensor.expert_matmul(x.weights, x.x, try ids_on([0, 4, 2, 0 - 1, 1, 0], c.tokens, c.uses, REF)))
  return ok() if !try Support.ggml_present()

  g = try inputs_of(c, CPU)
  bad = try ids_on([0, 5, 2, 3, 1, 0], c.tokens, c.uses, CPU)
  Test.error?(Trace.eval([g.weights, g.x, bad]) do |t| Tensor.expert_matmul(t[0], t[1], t[2]) end)
end

def test_what_the_op_does_not_take_is_an_error() -> Result[Void]
  c = (try golden()).cases[0]
  x = try inputs_of(c, REF)
  # x with a width that is not the inner size, with another number of tokens, with the wrong number of uses
  try Test.error?(
    Tensor.expert_matmul(x.weights, try Tensor.zeros([c.tokens, c.uses, c.inner + 1], Tensor.Dtype.F32, REF), x.ids),
  )

  try Test.error?(
    Tensor.expert_matmul(x.weights, try Tensor.zeros([c.tokens + 1, c.uses, c.inner], Tensor.Dtype.F32, REF), x.ids),
  )

  try Test.error?(
    Tensor.expert_matmul(x.weights, try Tensor.zeros([c.tokens, c.uses + 1, c.inner], Tensor.Dtype.F32, REF), x.ids),
  )

  # weights of rank 2, ids that are not I32, ids of rank 1
  try Test.error?(Tensor.expert_matmul(try Tensor.zeros([c.rows, c.inner], Tensor.Dtype.F32, REF), x.x, x.ids))
  try Test.error?(Tensor.expert_matmul(x.weights, x.x, try Support.i64(c.ids, [c.tokens, c.uses])))
  Test.error?(Tensor.expert_matmul(x.weights, x.x, try Tensor.reshape(x.ids, [c.tokens * c.uses])))
end
