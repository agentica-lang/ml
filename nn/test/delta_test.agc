## `Delta.gated_delta_net`, the linear attention layer of Qwen3.5, against Hugging Face's `Qwen3_5GatedDeltaNet`
## (`delta_golden.py`, `fixtures/delta_golden.json`): the whole sequence on the reference, the same sequence in two calls
## that carry the state, and the layer on ggml (CPU) against the reference node by node. Run from `nn/`.

import Delta from "../delta"
import Init from "../init"
import Nn from "../nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## The golden is in `float32` and the chunked rule, the reference in doubles and the recurrence.
ABS = 0.0001
REL = 0.0001

struct Case
  key_heads: Int
  value_heads: Int
  width: Int
  tokens: Int
  hidden: Int
  qkv: List[Float]
  z: List[Float]
  b: List[Float]
  a: List[Float]
  conv: List[Float]
  dt_bias: List[Float]
  a_log: List[Float]
  norm: List[Float]
  output: List[Float]
  x: List[Float]
  out: List[Float]
  prefill: Int
  prefill_out: List[Float]
  decode_out: List[Float]
  conv_state: List[Float]
  recurrent_state: List[Float]
end

struct Golden
  cases: List[Case]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/delta_golden.json"), Golden)

def dense(w: List[Float], outputs: Int, inputs: Int, backend: Tensor.Backend) -> Result[Nn.Dense]
  ok(Nn.Dense(w: try S.f32(w, [outputs, inputs], backend), b: none, out_in: true))
end

## The layer of `c` on `backend`.
def layer_of(c: Case, backend: Tensor.Backend) -> Result[Delta.GatedDeltaNet]
  keys = c.key_heads * c.width
  values = c.value_heads * c.width
  channels = 2 * keys + values
  ok(
    Delta.GatedDeltaNet(
      qkv: try dense(c.qkv, channels, c.hidden, backend),
      z: try dense(c.z, values, c.hidden, backend),
      b: try dense(c.b, c.value_heads, c.hidden, backend),
      a: try dense(c.a, c.value_heads, c.hidden, backend),
      conv: try S.f32(c.conv, [channels, 4], backend),
      dt_bias: try S.f32(c.dt_bias, [c.value_heads], backend),
      a_log: try S.f32(c.a_log, [c.value_heads], backend),
      norm: try S.f32(c.norm, [c.width], backend),
      output: try dense(c.output, c.hidden, values, backend),
      key_heads: c.key_heads,
      value_heads: c.value_heads,
      key_dim: c.width,
      value_dim: c.width,
    ),
  )
end

def test_the_layer_matches_hugging_face() -> Result[Void]
  for c in (try golden()).cases
    p = try layer_of(c, S.REF)
    x = try S.f32(c.x, [c.tokens, c.hidden])
    r = try Delta.gated_delta_net(p, x, try Delta.delta_start(p, S.REF))
    try Test.eq(Tensor.shape(r.out), [c.tokens, c.hidden])
    try Test.eq(Tensor.shape(r.state.conv), [3, 2 * c.key_heads * p.key_dim + c.value_heads * p.value_dim])
    try Test.eq(Tensor.shape(r.state.recurrent), [c.value_heads, p.value_dim, p.key_dim])
    try S.expect_close(try S.floats_of(r.out), c.out, ABS, REL, "#{c.value_heads} value heads")
  end

  ok()
end

## The first tokens and then the rest, with the state of the first call: the answers of the whole sequence.
def test_the_state_carries_a_sequence_over_two_calls() -> Result[Void]
  for c in (try golden()).cases
    p = try layer_of(c, S.REF)
    x = try S.f32(c.x, [c.tokens, c.hidden])
    cut = 4
    head = try Delta.gated_delta_net(p, try Tensor.slice(x, [0, 0], [cut, c.hidden]), try Delta.delta_start(p, S.REF))
    tail = try Delta.gated_delta_net(p, try Tensor.slice(x, [cut, 0], [c.tokens, c.hidden]), head.state)
    both = try S.floats_of(head.out) + try S.floats_of(tail.out)
    try S.expect_close(both, c.out, ABS, REL, "two calls, #{c.value_heads} value heads")
  end

  ok()
end

## One token at a time, as a decode does.
def test_one_token_at_a_time_gives_the_same_answers() -> Result[Void]
  c = (try golden()).cases[1]
  p = try layer_of(c, S.REF)
  x = try S.f32(c.x, [c.tokens, c.hidden])
  state = try Delta.delta_start(p, S.REF)
  got: List[Float] = []
  for t in 0...c.tokens
    r = try Delta.gated_delta_net(p, try Tensor.slice(x, [t, 0], [t + 1, c.hidden]), state)
    got = got + try S.floats_of(r.out)
    state = r.state
  end

  S.expect_close(got, c.out, ABS, REL, "token by token")
end

def test_a_convolution_sees_only_the_past() -> Result[Void]
  x = try S.f32([1.0, 10.0, 2.0, 20.0, 3.0, 30.0], [3, 2])
  weight = try S.f32([1.0, 0.0, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0], [2, 4])
  zeros = try Tensor.zeros([3, 2], Tensor.Dtype.F32, S.REF)
  r = try Delta.causal_conv1d(x, weight, zeros)
  # channel 0 taps [1, 0, 0, 2] over [0, 0, 0, 1, 2, 3]; channel 1 taps [0.5, 0, 0, 1] over [0, 0, 0, 10, 20, 30]
  try S.expect_close(try S.floats_of(r.out), [2.0, 10.0, 4.0, 20.0, 6.0, 30.0], 0.000000001, 0.0, "out")
  try Test.eq(try S.floats_of(r.state), [1.0, 10.0, 2.0, 20.0, 3.0, 30.0])
  try Test.error?(Delta.causal_conv1d(x, weight, try Tensor.zeros([2, 2], Tensor.Dtype.F32, S.REF)))
  Test.error?(Delta.causal_conv1d(x, try S.f32([1.0, 1.0], [2, 1]), try Tensor.zeros([0, 2], Tensor.Dtype.F32, S.REF)))
end

## The whole layer on ggml (CPU): every node of the trace against the reference, with a recurrent state to start from.
def test_the_layer_agrees_with_the_reference_on_ggml() -> Result[Void]
  try Tensor.vulkan_fp16(false)
  cpu = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  if !Tensor.available(cpu)
    flag = System.env("TENSOR_REQUIRE_GGML")
    return ok() if !flag.ok? or try flag == ""
    return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
  end

  c = (try golden()).cases[1]
  p = try layer_of(c, S.REF)
  size = c.value_heads * c.width * c.width
  args = [
    try S.f32(c.x, [c.tokens, c.hidden]),
    try S.f32(
      Init.uniform_values(3 * (2 * c.key_heads * c.width + c.value_heads * c.width), 0.5, 91),
      [3, 2 * c.key_heads * c.width + c.value_heads * c.width],
    ),
    try S.f32(Init.uniform_values(size, 0.2, 92), [c.value_heads, c.width, c.width]),
    p.qkv.w,
    p.z.w,
    p.b.w,
    p.a.w,
    p.conv,
    p.dt_bias,
    p.a_log,
    p.norm,
    p.output.w,
  ]

  d = try Trace.diff(
    args,
    S.REF,
    cpu,
    rtol: 0.001,
    atol: 0.0001,
    f: ->(a: List[Tensor.Tensor]) {
      q = p.with(
        qkv: p.qkv.with(w: a[3]),
        z: p.z.with(w: a[4]),
        b: p.b.with(w: a[5]),
        a: p.a.with(w: a[6]),
        conv: a[7],
        dt_bias: a[8],
        a_log: a[9],
        norm: a[10],
        output: p.output.with(w: a[11]),
      )

      r = try Delta.gated_delta_net(q, a[0], Delta.DeltaState(conv: a[1], recurrent: a[2]))
      ok(r.out)
    },
  )

  case d.first
    some(n) -> error("node #{n.node} (#{n.op} #{n.shape}) differs by #{n.max_abs}")
    none -> Test.assert(d.nodes > 40, "only #{d.nodes} nodes compared")
  end
end

## A prefill and then single-token steps, as a decode: Hugging Face's own cached run gives the same answers, and the
## states it holds after them are ours in the other layout (`conv` `[channels, taps]` keeps one row more than ours and
## is transposed; `recurrent` is `[heads, key, value]`).
def test_prefill_then_decode_matches_the_cached_run_of_hugging_face() -> Result[Void]
  for c in (try golden()).cases
    p = try layer_of(c, S.REF)
    x = try S.f32(c.x, [c.tokens, c.hidden])
    first = try Delta.gated_delta_net(
      p,
      try Tensor.slice(x, [0, 0], [c.prefill, c.hidden]),
      try Delta.delta_start(p, S.REF),
    )

    try S.expect_close(try S.floats_of(first.out), c.prefill_out, ABS, REL, "prefill")
    state = first.state
    got: List[Float] = []
    for t in c.prefill...c.tokens
      r = try Delta.gated_delta_net(p, try Tensor.slice(x, [t, 0], [t + 1, c.hidden]), state)
      got = got + try S.floats_of(r.out)
      state = r.state
    end

    try S.expect_close(got, c.decode_out, ABS, REL, "decode")
    channels = 2 * c.key_heads * p.key_dim + c.value_heads * p.value_dim
    cached = try S.f32(c.conv_state, [channels, 4])
    kept = try Tensor.transpose(try Tensor.slice(cached, [0, 1], [channels, 4]))
    try S.expect_close(try S.floats_of(state.conv), try S.floats_of(kept), ABS, REL, "conv state")
    held = try S.f32(c.recurrent_state, [c.value_heads, p.key_dim, p.value_dim])
    swapped = try Tensor.permute(held, [0, 2, 1])
    try S.expect_close(try S.floats_of(state.recurrent), try S.floats_of(swapped), ABS, REL, "recurrent state")
  end

  ok()
end

def test_a_layer_that_the_rule_cannot_run_is_an_error() -> Result[Void]
  c = (try golden()).cases[1]
  p = try layer_of(c, S.REF)
  x = try S.f32(c.x, [c.tokens, c.hidden])
  start = try Delta.delta_start(p, S.REF)
  try Test.error?(Delta.gated_delta_net(p.with(key_heads: 0), x, start))
  try Test.error?(Delta.gated_delta_net(p.with(value_heads: 0), x, start))
  try Test.error?(Delta.gated_delta_net(p.with(key_heads: 3), x, start))
  try Test.error?(Delta.gated_delta_net(p.with(value_dim: 4), x, start))
  try Test.error?(Delta.delta_start(p.with(key_dim: 4, value_dim: 8), S.REF))
  try Test.error?(Delta.gated_delta_net(p.with(conv: try S.f32(c.conv, [c.hidden * 4 * 4])), x, start))
  Test.error?(Delta.gated_delta_net(p, try Tensor.reshape(x, [1, c.tokens, c.hidden]), start))
end
