## The gated attention of Qwen3.5 (`Attention.gate`, rotary on part of each head, grouped keys) against Hugging Face's
## `Qwen3_5Attention` (`attention_golden.py`, `fixtures/attention_golden.json`), on the reference and on ggml (CPU).
## Run from `nn/`.

import Nn from "../nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## The golden is in `float32`, the reference in doubles.
ABS = 0.00001
REL = 0.00001

struct Golden
  tokens: Int
  hidden: Int
  heads: Int
  kv_heads: Int
  head_dim: Int
  rotary_dims: Int
  theta: Float
  q_proj: List[Float]
  k_proj: List[Float]
  v_proj: List[Float]
  o_proj: List[Float]
  q_norm: List[Float]
  k_norm: List[Float]
  x: List[Float]
  out: List[Float]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/attention_golden.json"), Golden)

def dense(w: List[Float], outputs: Int, inputs: Int) -> Result[Nn.Dense]
  ok(Nn.Dense(w: try S.f32(w, [outputs, inputs]), b: none, out_in: true))
end

## An RMS norm whose weight is the checkpoint's: Hugging Face scales by `1 + weight`, so the gamma is that sum.
def zero_centered(weight: List[Float], eps: Float) -> Result[Nn.LayerNorm]
  gamma = weight.map { |w| 1.0 + w }
  ok(
    Nn.LayerNorm(
      gamma: try S.f32(gamma, [weight.size]),
      beta: try S.f32(weight.map { |_w| 0.0 }, [weight.size]),
      eps: eps,
      rms: true,
    ),
  )
end

def attention_of(g: Golden) -> Result[Nn.Attention]
  width = g.heads * g.head_dim
  kv = g.kv_heads * g.head_dim
  ok(
    Nn.Attention(
      query: try dense(g.q_proj, 2 * width, g.hidden),
      key: try dense(g.k_proj, kv, g.hidden),
      value: try dense(g.v_proj, kv, g.hidden),
      output: try dense(g.o_proj, g.hidden, width),
      heads: g.heads,
      kv_heads: some(g.kv_heads),
      rope: some(Nn.Rope(base: g.theta, dims: some(g.rotary_dims))),
      query_norm: some(try zero_centered(g.q_norm, 0.000001)),
      key_norm: some(try zero_centered(g.k_norm, 0.000001)),
      gate: true,
    ),
  )
end

def test_a_gated_attention_matches_hugging_face() -> Result[Void]
  g = try golden()
  p = try attention_of(g)
  x = try S.f32(g.x, [1, g.tokens, g.hidden])
  got = try Nn.self_attention(p, x, none, true)
  try Test.eq(Tensor.shape(got), [1, g.tokens, g.hidden])
  S.expect_close(try S.floats_of(got), g.out, ABS, REL, "gated attention")
end

## The heads of a gated attention carry the gate; without `gate` the query is not split and there is none.
def test_only_a_gated_attention_has_a_gate() -> Result[Void]
  g = try golden()
  p = try attention_of(g)
  x = try S.f32(g.x, [1, g.tokens, g.hidden])
  heads = try Nn.project(p, x, x)
  try Test.eq(Tensor.shape(heads.query), [1, g.heads, g.tokens, g.head_dim])
  try Test.eq(Tensor.shape(heads.gate.unwrap!), [1, g.heads, g.tokens, g.head_dim])
  try Test.eq(Tensor.shape(heads.key), [1, g.kv_heads, g.tokens, g.head_dim])
  rows = g.heads * g.head_dim
  plain = p.with(gate: false, query: try dense(g.q_proj.take(rows * g.hidden), rows, g.hidden))
  try Test.assert((try Nn.project(plain, x, x)).gate.none?, "a plain attention has no gate")
  Test.error?(Nn.project(p.with(heads: 3), x, x))
end

## ggml (CPU) against the reference, node by node.
def test_a_gated_attention_agrees_with_the_reference_on_ggml() -> Result[Void]
  try Tensor.vulkan_fp16(false)
  cpu = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  if !Tensor.available(cpu)
    flag = System.env("TENSOR_REQUIRE_GGML")
    return ok() if !flag.ok? or try flag == ""
    return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
  end

  g = try golden()
  p = try attention_of(g)
  args = [
    try S.f32(g.x, [1, g.tokens, g.hidden]),
    p.query.w,
    p.key.w,
    p.value.w,
    p.output.w,
    p.query_norm.unwrap!.gamma,
    p.key_norm.unwrap!.gamma,
  ]

  d = try Trace.diff(
    args,
    S.REF,
    cpu,
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
      )

      Nn.self_attention(q, a[0], none, true)
    },
  )

  case d.first
    some(n) -> error("node #{n.node} (#{n.op} #{n.shape}) differs by #{n.max_abs}")
    none -> Test.assert(d.nodes > 30, "only #{d.nodes} nodes compared")
  end
end
