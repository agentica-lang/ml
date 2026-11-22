## Multi-head attention against a hand case and a naive per-head
## reference, plus `top_k` and `beam_search`.
##
## The naive path below is deliberately independent of `tensor.agc`: index
## arithmetic on plain lists with its own short Taylor `exp` (good for the
## small magnitudes the seeded generator makes), so the property loop checks
## the assembly rather than repeating it.

import Support from "support"
import Tensor from "../tensor"

private

def test_attention_matches_the_hand_case() -> Result[Void]
  q = try Support.f32([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], [1, 2, 4])
  k = try Support.f32([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [1, 2, 4])
  v = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 2, 4])
  out = try Tensor.multi_head_attention(q, k, v, 2, none)
  try Test.eq(Tensor.shape(out), [1, 2, 4])
  Support.expect_floats(
    out,
    [
      2.3209538026933725, 3.3209538026933725, 5.6790461973066275, 6.6790461973066275, 3.6790461973066275,
      4.6790461973066275, 4.3209538026933725, 5.3209538026933725,
    ],
  )
end

def test_attention_applies_an_additive_mask() -> Result[Void]
  q = try Support.f32([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], [1, 2, 4])
  k = try Support.f32([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [1, 2, 4])
  v = try Support.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 2, 4])
  mask = try Support.f32([0.0, -100.0, -100.0, 0.0, 0.0, -100.0, -100.0, 0.0], [1, 2, 2, 2])
  out = try Tensor.multi_head_attention(q, k, v, 2, some(mask))
  try Test.eq(Tensor.shape(out), [1, 2, 4])
  Support.expect_floats(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
end

def test_attention_matches_a_naive_per_head_reference() -> Result[Void]
  worst = 0.0
  for seed in 0...8
    gap = try case_gap(seed)
    if gap > worst
      worst = gap
    end
  end

  Test.assert(worst <= 0.000001, "the naive reference and Tensor differ by #{worst}")
end

def test_attention_rejects_heads_that_do_not_divide_the_width() -> Result[Void]
  q = try Support.f32([1.0, 0.0, 0.0, 1.0], [1, 1, 4])
  Test.error?(Tensor.multi_head_attention(q, q, q, 3, none))
end

def test_attention_rejects_rank_two_tensors() -> Result[Void]
  q = try Support.f32([1.0, 0.0], [1, 2])
  Test.error?(Tensor.multi_head_attention(q, q, q, 1, none))
end

def test_attention_rejects_a_mask_of_another_shape() -> Result[Void]
  q = try Support.f32([1.0, 0.0, 0.0, 1.0], [1, 1, 4])
  mask = try Support.f32([0.0, 0.0], [2])
  Test.error?(Tensor.multi_head_attention(q, q, q, 1, some(mask)))
end

def test_top_k_takes_the_largest_first() -> Result[Void]
  x = try Support.f32([3.0, 1.0, 4.0, 2.0], [4])
  got = try Tensor.top_k(x, 2)
  try Test.eq(Tensor.shape(got.values), [2])
  try Test.eq(Tensor.shape(got.indices), [2])
  try Support.expect_floats(got.values, [4.0, 3.0])
  Test.eq(try Tensor.to_list(got.indices), [Tensor.Values.Int(2), Tensor.Values.Int(0)])
end

def test_top_k_with_k_equals_n_sorts_everything() -> Result[Void]
  x = try Support.f32([2.0, 3.0, 1.0], [3])
  got = try Tensor.top_k(x, 3)
  Support.expect_floats(got.values, [3.0, 2.0, 1.0])
end

def test_top_k_rejects_non_finite_values() -> Result[Void]
  saturated = try Tensor.exp(try Support.f32([1000.0, 0.0], [2]))
  try Test.error?(Tensor.top_k(saturated, 1))
  nan = try Tensor.sub(saturated, saturated)
  Test.error?(Tensor.top_k(nan, 2))
end

def test_top_k_rejects_k_outside_one_to_n() -> Result[Void]
  x = try Support.f32([1.0, 2.0], [2])
  try Test.error?(Tensor.top_k(x, 0))
  Test.error?(Tensor.top_k(x, 3))
end

def test_beam_search_keeps_the_winning_rows() -> Result[Void]
  logits = try Support.f32([0.0, 1.0, 2.0, 3.0, 0.0, 0.0], [2, 3])
  paths = try Tensor.beam_search(logits, 2)
  try Test.eq(Tensor.shape(paths), [2, 2])
  Test.eq(
    try Tensor.to_list(paths),
    [Tensor.Values.Int(2), Tensor.Values.Int(0), Tensor.Values.Int(1), Tensor.Values.Int(0)],
  )
end

def test_beam_search_with_width_one_is_greedy() -> Result[Void]
  logits = try Support.f32([0.0, 1.0, 2.0, 3.0, 0.0, 0.0], [2, 3])
  paths = try Tensor.beam_search(logits, 1)
  Test.eq(try Tensor.to_list(paths), [Tensor.Values.Int(2), Tensor.Values.Int(0)])
end

def test_beam_search_rejects_a_width_outside_one_to_vocab() -> Result[Void]
  logits = try Support.f32([0.0, 1.0, 0.0, 1.0], [2, 2])
  try Test.error?(Tensor.beam_search(logits, 0))
  Test.error?(Tensor.beam_search(logits, 3))
end

## One seeded case: build the tensors, run both paths, report the gap.
def case_gap(seed: Int) -> Result[Float]
  b = 1 + seed % 2
  sq = 1 + seed % 3
  sk = 1 + (seed + 1) % 3
  heads = 1 + seed % 2
  d = heads * 2
  dv = heads * 2
  q = try Support.f32(gen_data(seed * 3 + 1, b * sq * d), [b, sq, d])
  k = try Support.f32(gen_data(seed * 3 + 2, b * sk * d), [b, sk, d])
  v = try Support.f32(gen_data(seed * 3 + 3, b * sk * dv), [b, sk, dv])
  use_mask = seed % 2 == 1
  mask_vals = gen_mask(seed, b, heads, sq, sk)
  m: Optional[Tensor.Tensor] = none
  flattened: List[Float] = []
  if use_mask
    m = some(try Support.f32(mask_vals, [b, heads, sq, sk]))
    flattened = mask_vals
  end

  got = try Tensor.multi_head_attention(q, k, v, heads, m)

  want = naive_mha(
    try Support.floats_of(q),
    try Support.floats_of(k),
    try Support.floats_of(v),
    b,
    sq,
    sk,
    d,
    dv,
    heads,
    flattened,
  )

  have = try Support.floats_of(got)
  worst = 0.0
  for i in 0...want.size
    gap = (want[i] - have[i]).abs
    if gap > worst
      worst = gap
    end
  end

  ok(worst)
end

## `count` values in `[-1, 1]` from a deterministic generator.
def gen_data(seed: Int, count: Int) -> List[Float]
  s = seed * 7919 + 13
  out: List[Float] = []
  for _i in 0...count
    s = (s * 1103515245 + 12345).bit_and(2147483647)
    out = out.push((s.to_f / 2147483647.0) * 2.0 - 1.0)
  end

  out
end

## An additive mask with at least one kept key per row: `-100.0` drops,
## `0.0` keeps, and key `0` is always kept.
def gen_mask(seed: Int, b: Int, heads: Int, sq: Int, sk: Int) -> List[Float]
  out: List[Float] = []
  for _bb in 0...b
    for _h in 0...heads
      for _i in 0...sq
        for j in 0...sk
          drop = (j + seed) % 3 == 0 and j > 0
          if drop
            out = out.push(-100.0)
          else
            out = out.push(0.0)
          end
        end
      end
    end
  end

  out
end

## Per-head attention on plain lists: scores, an independent softmax, and a
## context per `(batch, head)`, merged in head order.
def naive_mha(
  q: List[Float],
  k: List[Float],
  v: List[Float],
  b: Int,
  sq: Int,
  sk: Int,
  d: Int,
  dv: Int,
  heads: Int,
  mask: List[Float],
) -> List[Float]
  dh = d / heads
  dvh = dv / heads
  scale = 1.0 / naive_sqrt(dh.to_f)
  out: List[Float] = []
  for bb in 0...b
    for i in 0...sq
      for h in 0...heads
        scores: List[Float] = []
        for j in 0...sk
          acc = 0.0
          for dd in 0...dh
            acc = acc + q[(bb * sq + i) * d + h * dh + dd] * k[(bb * sk + j) * d + h * dh + dd]
          end

          biased = acc * scale
          if !mask.empty?
            biased = biased + mask[(bb * heads + h) * sq * sk + i * sk + j]
          end

          scores = scores.push(biased)
        end

        probs = naive_probs(scores, mask_for(mask, bb, h, heads, sq, sk, i))
        for e in 0...dvh
          acc = 0.0
          for j in 0...sk
            acc = acc + probs[j] * v[(bb * sk + j) * dv + h * dvh + e]
          end

          out = out.push(acc)
        end
      end
    end
  end

  out
end

## The mask row of one query, or an empty list when there is no mask.
def mask_for(mask: List[Float], bb: Int, h: Int, heads: Int, sq: Int, sk: Int, i: Int) -> List[Float]
  return [] if mask.empty?
  row: List[Float] = []
  for j in 0...sk
    row = row.push(mask[((bb * heads + h) * sq + i) * sk + j])
  end

  row
end

## `softmax` with its own short Taylor `exp`: dropped positions (mask below
## `-50.0`) weigh exactly `0.0`, so the series only ever sees kept gaps.
def naive_probs(scores: List[Float], masks: List[Float]) -> List[Float]
  kept: List[Bool] = []
  for j in 0...scores.size
    keep = true
    if !masks.empty?
      keep = masks[j] > -50.0
    end

    kept = kept.push(keep)
  end

  m = 0.0
  first = true
  for j in 0...scores.size
    if kept[j] and (first or scores[j] > m)
      m = scores[j]
      first = false
    end
  end

  weights: List[Float] = []
  for j in 0...scores.size
    if kept[j]
      weights = weights.push(naive_exp(scores[j] - m))
    else
      weights = weights.push(0.0)
    end
  end

  total = 0.0
  for w in weights
    total = total + w
  end

  weights.map do |w: Float| w / total end
end

## `exp` for small magnitudes only (the test generator keeps `|x| <= 6`):
## a plain degree-40 Taylor series, no range reduction.
def naive_exp(x: Float) -> Float
  term = 1.0
  total = 1.0
  for n in 1...41
    term = (term * x) / n.to_f
    total = total + term
  end

  total
end

## `sqrt` by Newton iteration (twenty rounds from `x` itself).
def naive_sqrt(x: Float) -> Float
  guess = x
  for _n in 0...20
    guess = 0.5 * (guess + x / guess)
  end

  guess
end
