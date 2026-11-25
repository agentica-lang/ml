## `Moe.moe`, the mixture-of-experts block, against Hugging Face's `GraniteMoeMoE` and `Qwen3MoeSparseMoeBlock`
## (`moe_golden.py`, `fixtures/moe_golden.json`): the seven cases on the reference and on ggml (CPU), where
## `ggml_mul_mat_id` runs the experts and `ggml_top_k` picks them in no order. The cases are 2 of 6 experts, 1 of 6, 6
## of 6, a tie for the last place that does not change the answer, the gates that do not add up to 1, one token, and 40
## tokens (many more than experts). Run from `nn/`.

import Moe from "../moe"
import Nn from "../nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## The golden is in `float32`, the reference in doubles; ggml sums in another order.
ABS = 0.00001
REL = 0.00001
GGML_ABS = 0.0001

struct Case
  name: String
  experts: Int
  width: Int
  inner: Int
  tokens: Int
  top_k: Int
  normalize: Bool
  router: List[Float]
  gate_up: List[Float]
  down: List[Float]
  x: List[Float]
  out: List[Float]
end

struct Golden
  cases: List[Case]
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/moe_golden.json"), Golden)

def block_of(c: Case, backend: Tensor.Backend) -> Result[Moe.Moe]
  ok(
    Moe.Moe(
      router: Nn.Dense(w: try S.f32(c.router, [c.experts, c.width], backend), b: none, out_in: true),
      gate_up: try S.f32(c.gate_up, [c.experts, 2 * c.inner, c.width], backend),
      down: try S.f32(c.down, [c.experts, c.width, c.inner], backend),
      top_k: c.top_k,
      activation: Nn.Activation.Silu,
      normalize: c.normalize,
    ),
  )
end

def test_the_block_matches_hugging_face() -> Result[Void]
  cases = (try golden()).cases
  try Test.eq(cases.size, 7)
  for c in cases
    out = try Moe.moe(try block_of(c, S.REF), try S.f32(c.x, [c.tokens, c.width]))
    try Test.eq(Tensor.shape(out), [c.tokens, c.width])
    try S.expect_close(try S.floats_of(out), c.out, ABS, REL, c.name)
  end

  ok()
end

## The weights and the input are the leaves of a trace, as a model passes them.
def test_ggml_matches_hugging_face() -> Result[Void]
  return ok() if !try S.ggml_present()

  for c in (try golden()).cases
    p = try block_of(c, S.CPU)
    out = try Trace.eval([try S.f32(c.x, [c.tokens, c.width], S.CPU), p.router.w, p.gate_up, p.down]) do |t|
      Moe.moe(p.with(router: p.router.with(w: t[1]), gate_up: t[2], down: t[3]), t[0])
    end

    try S.expect_close(try S.floats_of(out), c.out, GGML_ABS, REL, c.name)
  end

  ok()
end

def test_the_leading_axes_fold_into_tokens() -> Result[Void]
  c = (try golden()).cases[0]
  out = try Moe.moe(try block_of(c, S.REF), try S.f32(c.x, [1, c.tokens, c.width]))
  try Test.eq(Tensor.shape(out), [1, c.tokens, c.width])
  S.expect_close(try S.floats_of(out), c.out, ABS, REL, c.name)
end

def test_a_block_that_does_not_fit_is_an_error() -> Result[Void]
  c = (try golden()).cases[0]
  p = try block_of(c, S.REF)
  x = try S.f32(c.x, [c.tokens, c.width])
  try Test.error?(Moe.moe(p.with(top_k: 0), x))
  try Test.error?(Moe.moe(p.with(top_k: c.experts + 1), x))
  # a router for other experts, a down of another inner size, an input of another width or of rank 1
  try Test.error?(Moe.moe(p.with(router: p.router.with(w: try S.f32(c.router.take(5 * c.width), [5, c.width]))), x))
  try Test.error?(Moe.moe(p.with(down: try S.f32(c.down.take(c.experts * c.width * 3), [c.experts, c.width, 3])), x))
  try Test.error?(Moe.moe(p, try S.f32(c.x.take(c.tokens * 5), [c.tokens, 5])))
  Test.error?(Moe.moe(p, try S.f32(c.x.take(c.width), [c.width])))
end
