## What LFM3 adds to `nn/`, against the official code (`lfm3_golden.py`, `fixtures/lfm3_golden.json`): the gated short
## convolution (`ShortConv.short_conv`), the gate of one number per head (`Attention.head_gate`) and the sliding-window
## mask (`Nn.window_mask`, and `Padded.padded_mask` with a window). Run from `nn/`.

import Delta from "../delta"
import Init from "../init"
import Padded from "../kv"
import Nn from "../nn"
import ShortConv from "../short_conv"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

ABS = 0.0001
REL = 0.0001

struct Conv
  tokens: Int
  hidden: Int
  taps: Int
  prefill: Int
  input: List[Float]
  conv: List[Float]
  output: List[Float]
  x: List[Float]
  out: List[Float]
  prefill_out: List[Float]
  decode_out: List[Float]
  state_shape: List[Int]
  state: List[Float]
end

struct Layer
  kind: String
  tokens: Int
  hidden: Int
  heads: Int
  kv_heads: Int
  head: Int
  theta: Float
  window: Int
  query: List[Float]
  key: List[Float]
  value: List[Float]
  output: List[Float]
  query_norm: List[Float]
  key_norm: List[Float]
  gate: List[Float]
  x: List[Float]
  out: List[Float]
  eps: Float
end

struct Golden
  short_conv: Conv
  attentions: List[Layer]
  visible: List[Int]
  window: Int
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/lfm3_golden.json"), Golden)

def dense(w: List[Float], outputs: Int, inputs: Int, backend: Tensor.Backend) -> Result[Nn.Dense]
  ok(Nn.Dense(w: try S.f32(w, [outputs, inputs], backend), b: none, out_in: true))
end

def seen(bias: Float) -> Int
  return 1 if bias == 0.0
  0
end

def conv_of(c: Conv, backend: Tensor.Backend) -> Result[ShortConv.ShortConv]
  ok(
    ShortConv.ShortConv(
      input: try dense(c.input, 3 * c.hidden, c.hidden, backend),
      conv: try S.f32(c.conv, [c.hidden, c.taps], backend),
      output: try dense(c.output, c.hidden, c.hidden, backend),
    ),
  )
end

def rms(gamma: List[Float], width: Int, eps: Float, backend: Tensor.Backend) -> Result[Nn.LayerNorm]
  ok(
    Nn.LayerNorm(
      gamma: try S.f32(gamma, [width], backend),
      beta: try Tensor.zeros([width], Tensor.Dtype.F32, backend),
      eps: eps,
      rms: true,
    ),
  )
end

def attention_of(l: Layer, backend: Tensor.Backend) -> Result[Nn.Attention]
  ok(
    Nn.Attention(
      query: try dense(l.query, l.heads * l.head, l.hidden, backend),
      key: try dense(l.key, l.kv_heads * l.head, l.hidden, backend),
      value: try dense(l.value, l.kv_heads * l.head, l.hidden, backend),
      output: try dense(l.output, l.hidden, l.heads * l.head, backend),
      heads: l.heads,
      kv_heads: some(l.kv_heads),
      rope: some(Nn.Rope(base: l.theta)),
      query_norm: some(try rms(l.query_norm, l.head, l.eps, backend)),
      key_norm: some(try rms(l.key_norm, l.head, l.eps, backend)),
      head_gate: some(try dense(l.gate, l.heads, l.hidden, backend)),
    ),
  )
end

## The layer `l` over its input, with the mask of its kind.
def run_layer(p: Nn.Attention, l: Layer, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  if l.window > 0
    mask = try Nn.window_mask(l.tokens, l.window, Tensor.Dtype.F32, Tensor.backend(x))
    return Nn.self_attention(p, x, some(mask))
  end

  Nn.self_attention(p, x, none, true)
end

def test_the_short_convolution_matches_the_official_code() -> Result[Void]
  c = (try golden()).short_conv
  p = try conv_of(c, S.REF)
  x = try S.f32(c.x, [c.tokens, c.hidden])
  r = try ShortConv.short_conv(p, x, try ShortConv.short_start(p, S.REF))
  try Test.eq(Tensor.shape(r.out), [c.tokens, c.hidden])
  try Test.eq(Tensor.shape(r.state), [c.taps - 1, c.hidden])
  S.expect_close(try S.floats_of(r.out), c.out, ABS, REL, "the whole sequence")
end

## A prefill and then single-token steps, as a decode: the answers are those of the official cached run, and the
## state is its `[width, taps]` state without the oldest column, transposed.
def test_prefill_then_decode_matches_the_cached_run_of_the_official_code() -> Result[Void]
  c = (try golden()).short_conv
  p = try conv_of(c, S.REF)
  x = try S.f32(c.x, [c.tokens, c.hidden])
  first = try ShortConv.short_conv(
    p,
    try Tensor.slice(x, [0, 0], [c.prefill, c.hidden]),
    try ShortConv.short_start(p, S.REF),
  )

  try S.expect_close(try S.floats_of(first.out), c.prefill_out, ABS, REL, "prefill")
  state = first.state
  got: List[Float] = []
  for t in c.prefill...c.tokens
    r = try ShortConv.short_conv(p, try Tensor.slice(x, [t, 0], [t + 1, c.hidden]), state)
    got = got + try S.floats_of(r.out)
    state = r.state
  end

  try S.expect_close(got, c.decode_out, ABS, REL, "decode")
  cached = try S.f32(c.state, [c.hidden, c.taps])
  kept = try Tensor.transpose(try Tensor.slice(cached, [0, 1], [c.hidden, c.taps]))
  S.expect_close(try S.floats_of(state), try S.floats_of(kept), ABS, REL, "state")
end

## A prompt padded to a longer shape answers and leaves the state as the same prompt alone does.
def test_padding_leaves_the_answers_and_the_state_of_the_real_tokens() -> Result[Void]
  c = (try golden()).short_conv
  p = try conv_of(c, S.REF)
  x = try S.f32(c.x, [c.tokens, c.hidden])
  for real in [2, 5, c.tokens]
    head = try Tensor.slice(x, [0, 0], [real, c.hidden])
    alone = try ShortConv.short_conv(p, head, try ShortConv.short_start(p, S.REF))
    rest = try Tensor.full([6, c.hidden], Tensor.Values.Float(3.0), Tensor.Dtype.F32, S.REF)
    pad = try Delta.padding(real, real + 6, c.taps, S.REF)
    padded = try ShortConv.short_conv(
      p,
      try Tensor.concat([head, rest], 0),
      try ShortConv.short_start(p, S.REF),
      some(pad),
    )

    what = "#{real} real tokens"
    try S.expect_close(
      try S.floats_of(try Tensor.slice(padded.out, [0, 0], [real, c.hidden])),
      try S.floats_of(alone.out),
      0.000000001,
      0.0,
      what,
    )

    try S.expect_close(try S.floats_of(padded.state), try S.floats_of(alone.state), 0.000000001, 0.0, what)
  end

  ok()
end

def test_the_short_convolution_agrees_with_the_reference_on_ggml() -> Result[Void]
  return ok() if !try S.ggml_present()
  c = (try golden()).short_conv
  p = try conv_of(c, S.REF)
  args = [
    try S.f32(c.x, [c.tokens, c.hidden]),
    try S.f32(Init.uniform_values((c.taps - 1) * c.hidden, 0.5, 91), [c.taps - 1, c.hidden]),
    p.input.w,
    p.conv,
    p.output.w,
  ]

  d = try Trace.diff(
    args,
    S.REF,
    S.CPU,
    rtol: 0.001,
    atol: 0.0001,
    f: ->(a: List[Tensor.Tensor]) {
      q = p.with(input: p.input.with(w: a[2]), conv: a[3], output: p.output.with(w: a[4]))
      r = try ShortConv.short_conv(q, a[0], a[1])
      ok(r.out)
    },
  )

  case d.first
    some(n) -> error("node #{n.node} (#{n.op} #{n.shape}) differs by #{n.max_abs}")
    none -> Test.assert(d.nodes > 15, "only #{d.nodes} nodes compared")
  end
end

def test_a_short_convolution_that_does_not_fit_is_an_error() -> Result[Void]
  c = (try golden()).short_conv
  p = try conv_of(c, S.REF)
  x = try S.f32(c.x, [c.tokens, c.hidden])
  start = try ShortConv.short_start(p, S.REF)
  try Test.error?(ShortConv.short_conv(p, try Tensor.reshape(x, [1, c.tokens, c.hidden]), start))
  try Test.error?(ShortConv.short_conv(p, x, try Tensor.zeros([2, c.hidden], Tensor.Dtype.F32, S.REF)))
  try Test.error?(ShortConv.short_conv(p.with(input: p.output), x, start))
  Test.error?(ShortConv.short_start(p.with(conv: try S.f32(c.conv, [c.hidden * c.taps])), S.REF))
end

## The sliding layer (a window of 3 keys, `window_mask`) and the full one (`causal`), each with its gate per head.
def test_an_attention_with_a_gate_per_head_matches_the_official_code() -> Result[Void]
  for l in (try golden()).attentions
    p = try attention_of(l, S.REF)
    x = try S.f32(l.x, [1, l.tokens, l.hidden])
    out = try run_layer(p, l, x)
    try Test.eq(Tensor.shape(out), [1, l.tokens, l.hidden])
    try S.expect_close(try S.floats_of(out), l.out, ABS, REL, l.kind)
  end

  ok()
end

def test_the_window_mask_is_the_grid_of_keys_the_official_code_shows_each_query() -> Result[Void]
  g = try golden()
  size = g.attentions[0].tokens
  got = try S.floats_of(try Nn.window_mask(size, g.window, Tensor.Dtype.F32, S.REF))
  try Test.eq(got.map { |v| seen(v) }, g.visible)
  causal = try S.floats_of(try Nn.causal_mask(size, Tensor.Dtype.F32, S.REF))
  try Test.eq(try S.floats_of(try Nn.window_mask(size, size, Tensor.Dtype.F32, S.REF)), causal)
  try Test.error?(Nn.window_mask(size, 0, Tensor.Dtype.F32, S.REF))
  Test.error?(Nn.window_mask(0, 3, Tensor.Dtype.F32, S.REF))
end

## The rows of a padded cache that a new token reads are the row of the window mask of its position.
def test_a_padded_cache_with_a_window_hides_the_keys_a_window_or_more_before() -> Result[Void]
  g = try golden()
  size = g.attentions[0].tokens
  at = (0...size).map { |i| Tensor.Values.Int(i) }
  slots = Padded.Slots(positions: try Tensor.from_list(at, [size], Tensor.Dtype.I32, S.REF), capacity: 16)
  got = try S.floats_of(try Padded.padded_mask(slots, Tensor.Dtype.F32, some(g.window)))
  want = try S.floats_of(try Nn.window_mask(16, g.window, Tensor.Dtype.F32, S.REF))
  try Test.eq(got, want.take(size * 16))
  plain = try S.floats_of(try Padded.padded_mask(slots, Tensor.Dtype.F32))
  try Test.eq(try S.floats_of(try Padded.padded_mask(slots, Tensor.Dtype.F32, some(16))), plain)
  Test.error?(Padded.padded_mask(slots, Tensor.Dtype.F32, some(0)))
end

## The sliding layer on ggml (CPU): every node against the reference, with the mask built inside the graph.
def test_the_sliding_layer_agrees_with_the_reference_on_ggml() -> Result[Void]
  return ok() if !try S.ggml_present()
  l = (try golden()).attentions[0]
  p = try attention_of(l, S.REF)
  args = [
    try S.f32(l.x, [1, l.tokens, l.hidden]),
    p.query.w,
    p.key.w,
    p.value.w,
    p.output.w,
    p.query_norm.unwrap!.gamma,
    p.key_norm.unwrap!.gamma,
    p.head_gate.unwrap!.w,
  ]

  d = try Trace.diff(
    args,
    S.REF,
    S.CPU,
    rtol: 0.001,
    atol: 0.0001,
    f: ->(a: List[Tensor.Tensor]) {
      q = p.with(
        query: p.query.with(w: a[1]),
        key: p.key.with(w: a[2]),
        value: p.value.with(w: a[3]),
        output: p.output.with(w: a[4]),
        query_norm: some(p.query_norm.unwrap!.with(gamma: a[5])),
        key_norm: some(p.key_norm.unwrap!.with(gamma: a[6])),
        head_gate: some(p.head_gate.unwrap!.with(w: a[7])),
      )

      run_layer(q, l, a[0])
    },
  )

  case d.first
    some(n) -> error("node #{n.node} (#{n.op} #{n.shape}) differs by #{n.max_abs}")
    none -> Test.assert(d.nodes > 30, "only #{d.nodes} nodes compared")
  end
end

## An attention has the gate of its query projection or a gate per head, not both, and `attend` takes the gate of a
## gated attention only.
def test_the_two_gates_do_not_mix() -> Result[Void]
  l = (try golden()).attentions[0]
  p = try attention_of(l, S.REF)
  x = try S.f32(l.x, [1, l.tokens, l.hidden])
  try Test.error?(Nn.project(p.with(gate: true), x, x))
  heads = try Nn.project(p, x, x)
  try Test.assert(heads.gate.some?, "a head gate projects a gate")
  try Test.eq(Tensor.shape(heads.gate.unwrap!), Tensor.shape(heads.query))
  try Test.error?(Nn.attend(p, heads.query, heads.key, heads.value, none, true))
  try Test.error?(Nn.attend(p.with(head_gate: none), heads.query, heads.key, heads.value, none, true, heads.gate))
  Test.error?(Nn.project(p.with(head_gate: some(p.output)), x, x))
end
