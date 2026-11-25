## `Tensor.gated_delta_rule`: the reference against the recurrent rule of Hugging Face's `modeling_qwen3_5`
## (`gated_delta_rule_golden.py`, `fixtures/gated_delta_rule_golden.json`), a state carried over two calls, a padding
## that leaves the state alone, and `ggml_gated_delta_net` (a trace on ggml) against the reference. Run from `tensor/`.

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
  tokens: Int
  heads: Int
  width: Int
  q: List[Float]
  k: List[Float]
  v: List[Float]
  g: List[Float]
  beta: List[Float]
  state: List[Float]
  out: List[Float]
  final: List[Float]
end

struct Golden
  cases: List[Case]
end

struct Inputs
  q: Tensor.Tensor
  k: Tensor.Tensor
  v: Tensor.Tensor
  g: Tensor.Tensor
  beta: Tensor.Tensor
  state: Tensor.Tensor
end

private

def inputs_of(c: Case, backend: Tensor.Backend) -> Result[Inputs]
  shape = [c.tokens, c.heads, c.width]
  ok(
    Inputs(
      q: try Tensor.floats(c.q, shape, backend),
      k: try Tensor.floats(c.k, shape, backend),
      v: try Tensor.floats(c.v, shape, backend),
      g: try Tensor.floats(c.g, [c.tokens, c.heads], backend),
      beta: try Tensor.floats(c.beta, [c.tokens, c.heads], backend),
      state: try Tensor.floats(c.state, [c.heads, c.width, c.width], backend),
    ),
  )
end

def scaled(xs: List[Float], by: Float) -> List[Float] = xs.map { |x| x * by }

def rule(x: Inputs) -> Result[Tensor.DeltaRule] = Tensor.gated_delta_rule(x.q, x.k, x.v, x.g, x.beta, x.state)

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/gated_delta_rule_golden.json"), Golden)

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

def test_the_reference_matches_hugging_face() -> Result[Void]
  cases = (try golden()).cases
  try Test.eq(cases.size, 3)
  for c in cases
    r = try rule(try inputs_of(c, REF))
    try Test.eq(Tensor.shape(r.out), [c.tokens, c.heads, c.width])
    try Test.eq(Tensor.shape(r.state), [c.heads, c.width, c.width])
    try close(r.out, c.out, GOLDEN_TOL, "out")
    try close(r.state, c.final, GOLDEN_TOL, "state")
  end

  ok()
end

## The state of the first tokens is the state the next ones start from.
def test_a_state_carries_over_two_calls() -> Result[Void]
  c = (try golden()).cases[0]
  whole = try rule(try inputs_of(c, REF))
  x = try inputs_of(c, REF)
  heads = c.heads
  width = c.width
  cut = 2
  head = try Tensor.gated_delta_rule(
    try Tensor.slice(x.q, [0, 0, 0], [cut, heads, width]),
    try Tensor.slice(x.k, [0, 0, 0], [cut, heads, width]),
    try Tensor.slice(x.v, [0, 0, 0], [cut, heads, width]),
    try Tensor.slice(x.g, [0, 0], [cut, heads]),
    try Tensor.slice(x.beta, [0, 0], [cut, heads]),
    x.state,
  )

  tail = try Tensor.gated_delta_rule(
    try Tensor.slice(x.q, [cut, 0, 0], [c.tokens, heads, width]),
    try Tensor.slice(x.k, [cut, 0, 0], [c.tokens, heads, width]),
    try Tensor.slice(x.v, [cut, 0, 0], [c.tokens, heads, width]),
    try Tensor.slice(x.g, [cut, 0], [c.tokens, heads]),
    try Tensor.slice(x.beta, [cut, 0], [c.tokens, heads]),
    head.state,
  )

  try close(tail.state, try Support.floats_of(whole.state), 0.000000001, "state")
  both = try Tensor.concat([head.out, tail.out], 0)
  close(both, try Support.floats_of(whole.out), 0.000000001, "out")
end

## The case `c` followed by `extra` tokens with `g = 0` and `beta = 0` and any `q`, `k` and `v`.
def padded(c: Case, extra: Int) -> Case
  row = c.heads * c.width
  c.with(
    tokens: c.tokens + extra,
    q: c.q + Support.ramp(1, extra * row),
    k: c.k + Support.ramp(2, extra * row),
    v: c.v + Support.ramp(3, extra * row),
    g: c.g + (0...(extra * c.heads)).map { |_i| 0.0 },
    beta: c.beta + (0...(extra * c.heads)).map { |_i| 0.0 },
  )
end

## A padded prompt: tokens with `g = 0` and `beta = 0` after the real ones change neither the answers of the real
## ones nor the state, whatever their `q`, `k` and `v` are.
def test_a_token_with_no_gate_and_no_beta_leaves_the_state_alone() -> Result[Void]
  c = (try golden()).cases[1]
  r = try rule(try inputs_of(padded(c, 3), REF))
  try close(r.state, c.final, GOLDEN_TOL, "state")
  close(try Tensor.slice(r.out, [0, 0, 0], [c.tokens, c.heads, c.width]), c.out, GOLDEN_TOL, "out")
end

## The same padding through a trace on ggml.
def test_the_padding_leaves_the_state_alone_on_ggml() -> Result[Void]
  return ok() if !try Support.ggml_present()

  c = (try golden()).cases[1]
  x = try inputs_of(padded(c, 3), CPU)
  outs = try Trace.eval_many([x.q, x.k, x.v, x.g, x.beta, x.state]) do |t|
    r = try Tensor.gated_delta_rule(t[0], t[1], t[2], t[3], t[4], t[5])
    ok([r.out, r.state])
  end

  try close(outs[1], c.final, F32_TOL, "state")
  real = try Tensor.floats((try Support.floats_of(outs[0])).take(c.out.size), [c.tokens, c.heads, c.width], REF)
  close(real, c.out, F32_TOL, "out")
end

## A trace on the reference runs the node as `Trace.compute` does, through the packed answer.
def test_a_trace_on_the_reference_matches_hugging_face() -> Result[Void]
  for c in (try golden()).cases
    x = try inputs_of(c, REF)
    outs = try Trace.eval_many([x.q, x.k, x.v, x.g, x.beta, x.state]) do |t|
      r = try Tensor.gated_delta_rule(t[0], t[1], t[2], t[3], t[4], t[5])
      ok([r.out, r.state])
    end

    try close(outs[0], c.out, GOLDEN_TOL, "out")
    try close(outs[1], c.final, GOLDEN_TOL, "state")
  end

  ok()
end

## libtorch runs the recurrence over its own ops: the packed answer is the reference's.
def test_libtorch_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  c = (try golden()).cases[1]
  shape = [c.tokens, c.heads, c.width]
  gate = [c.tokens, c.heads]
  ours: Result[Tensor.Tensor] = Native.arena do
    Tensor.gated_delta_packed(
      try Conform.on_torch(c.q, shape),
      try Conform.on_torch(c.k, shape),
      try Conform.on_torch(c.v, shape),
      try Conform.on_torch(c.g, gate),
      try Conform.on_torch(c.beta, gate),
      try Conform.on_torch(c.state, [c.heads, c.width, c.width]),
    )
  end

  theirs = try Tensor.gated_delta_packed(
    try Conform.on_ref(c.q, shape),
    try Conform.on_ref(c.k, shape),
    try Conform.on_ref(c.v, shape),
    try Conform.on_ref(c.g, gate),
    try Conform.on_ref(c.beta, gate),
    try Conform.on_ref(c.state, [c.heads, c.width, c.width]),
  )

  Conform.agrees(try ours, theirs, "gated delta rule")
end

def test_there_is_no_gradient() -> Result[Void]
  c = (try golden()).cases[0]
  x = try inputs_of(c, REF)
  Test.error?(
    Trace.grad([x.q]) do |t|
      r = try Tensor.gated_delta_rule(
        t[0],
        Trace.constant(x.k),
        Trace.constant(x.v),
        Trace.constant(x.g),
        Trace.constant(x.beta),
        Trace.constant(x.state),
      )

      Tensor.sum(r.out)
    end,
  )
end

def test_ggml_matches_the_reference_as_a_trace() -> Result[Void]
  return ok() if !try Support.ggml_present()

  for c in (try golden()).cases
    x = try inputs_of(c, CPU)
    outs = try Trace.eval_many([x.q, x.k, x.v, x.g, x.beta, x.state]) do |t|
      r = try Tensor.gated_delta_rule(t[0], t[1], t[2], t[3], t[4], t[5])
      ok([r.out, r.state])
    end

    try close(outs[0], c.out, F32_TOL, "out")
    try close(outs[1], c.final, F32_TOL, "state")
  end

  ok()
end

## A width of a real model (128) and a state carried in: the node, and the recurrence composed of the primitives in the
## same trace, which ggml runs as ordinary ops (the reference is too slow at this size).
def test_ggml_node_matches_the_composed_recurrence_at_the_width_of_a_model() -> Result[Void]
  return ok() if !try Support.ggml_present()

  tokens = 6
  heads = 2
  width = 128
  row = heads * width
  c = Case(
    tokens: tokens,
    heads: heads,
    width: width,
    q: scaled(Support.ramp(1, tokens * row), 0.09),
    k: scaled(Support.ramp(2, tokens * row), 0.09),
    v: Support.ramp(3, tokens * row),
    g: Support.ramp(4, tokens * heads).map { |x| 0.0 - x.abs },
    beta: Support.ramp(5, tokens * heads).map { |x| x.abs },
    state: scaled(Support.ramp(6, heads * width * width), 0.1),
    out: [],
    final: [],
  )

  x = try inputs_of(c, CPU)
  outs = try Trace.eval_many([x.q, x.k, x.v, x.g, x.beta, x.state]) do |t|
    node = try Tensor.gated_delta_rule(t[0], t[1], t[2], t[3], t[4], t[5])
    steps = try Tensor.gated_delta_composed(t[0], t[1], t[2], t[3], t[4], t[5])
    ok([node.out, node.state, steps.out, steps.state])
  end

  try close(outs[0], try Support.floats_of(outs[2]), F32_TOL, "out")
  close(outs[1], try Support.floats_of(outs[3]), F32_TOL, "state")
end

def test_what_the_op_does_not_take_is_an_error() -> Result[Void]
  c = (try golden()).cases[0]
  x = try inputs_of(c, REF)
  try Test.error?(
    Tensor.gated_delta_rule(x.q, x.k, x.v, x.g, x.beta, try Tensor.zeros([c.heads, c.width], Tensor.Dtype.F32, REF)),
  )

  try Test.error?(
    Tensor.gated_delta_rule(
      x.q,
      x.k,
      x.v,
      try Tensor.zeros([c.heads, c.tokens], Tensor.Dtype.F32, REF),
      x.beta,
      x.state,
    ),
  )

  try Test.error?(
    Tensor.gated_delta_rule(
      x.q,
      x.k,
      try Tensor.zeros([1, c.heads, c.width], Tensor.Dtype.F32, REF),
      x.g,
      x.beta,
      x.state,
    ),
  )

  Test.error?(
    Tensor.gated_delta_rule(
      x.q,
      x.k,
      x.v,
      x.g,
      x.beta,
      try Tensor.zeros([c.heads, c.width, c.width], Tensor.Dtype.F16, REF),
    ),
  )
end
