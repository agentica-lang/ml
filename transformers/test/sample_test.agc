## `Sample` against Hugging Face's logits warpers (`sample_golden.py`, `fixtures/sample_golden.json`): the probability of
## every id after the temperature, `top_k` and `top_p`, and the properties of the draws. Run from `transformers/`.

import Sample from "../sample"

ABS = 0.000000001

struct Case
  logits: List[Float]
  temperature: Float
  top_k: Int
  top_p: Float
  probabilities: List[Float]
end

struct Drawn
  logits: List[Float]
  temperature: Float
  top_k: Int
  top_p: Float
  seed: Int
  ids: List[Int]
end

struct Golden
  cases: List[Case]
  draws: List[Drawn]
end

def test_the_probabilities_are_those_of_the_warpers() -> Result[Void]
  golden = try JSON.decode(try File.read_text("test/fixtures/sample_golden.json"), Golden)
  try Test.eq(golden.cases.size, 32)
  for c in golden.cases
    ours = try Sample.probabilities(
      c.logits,
      Sample.Sampling(temperature: c.temperature, top_k: c.top_k, top_p: c.top_p),
    )

    for i in 0...c.logits.size
      try Test.assert(
        (ours[i] - c.probabilities[i]).abs <= ABS,
        "id #{i} of #{c.temperature}, #{c.top_k}, #{c.top_p}: #{ours[i]} against #{c.probabilities[i]}",
      )
    end
  end

  ok()
end

## The ids drawn, state after state, are those of the stdlib peer in `sample_golden.py` (Park-Miller and a cumulative
## scan), with `pick` over every logit and `pick_top` over the window.
def test_the_draws_are_those_of_the_peer() -> Result[Void]
  golden = try JSON.decode(try File.read_text("test/fixtures/sample_golden.json"), Golden)
  try Test.eq(golden.draws.size, 18)
  for d in golden.draws
    s = Sample.Sampling(temperature: d.temperature, top_k: d.top_k, top_p: d.top_p, seed: d.seed)
    try Test.eq(try draws(d.logits, s, d.ids.size), d.ids)
  end

  ok()
end

## `pick_top` over the largest logits draws what `pick` draws over all of them, when the window holds what can be drawn:
## the whole vocabulary here (40 ids), or the `top_k` largest.
def test_picking_among_the_largest_logits_is_picking_among_all() -> Result[Void]
  golden = try JSON.decode(try File.read_text("test/fixtures/sample_golden.json"), Golden)
  for c in golden.cases
    s = Sample.Sampling(temperature: c.temperature, top_k: c.top_k, top_p: c.top_p, seed: 3)
    k = Sample.window(s, c.logits.size)
    order = (0...c.logits.size).to_a.sort_by { |i| 0.0 - c.logits[i] }
    ids = order.take(k)
    # a tie across the edge of the window keeps both ids on the full logits and one in the window (documented)
    next if k < order.size and c.logits[order[k - 1]] == c.logits[order[k]]
    state = Sample.start(s)
    for _ in 0...4
      full = try Sample.pick(c.logits, s, state)
      top = try Sample.pick_top(ids, ids.map { |i| c.logits[i] }, s, state)
      try Test.eq(top.id, full.id)
      try Test.eq(top.state, full.state)
      state = full.state
    end
  end

  try Test.eq(Sample.window(Sample.Sampling(), 152000), 256)
  try Test.eq(Sample.window(Sample.Sampling(top_k: 40), 152000), 40)
  try Test.eq(Sample.window(Sample.Sampling(top_k: 400), 152000), 400)
  Test.eq(Sample.window(Sample.Sampling(), 100), 100)
end

## With more ids than the window (256), what lies below the 256th largest logit is never drawn, though `pick` over every
## logit draws it at a temperature that flattens the distribution.
def test_the_window_cap_drops_the_smallest_logits() -> Result[Void]
  logits = (0...300).map { |i| 0.0 - i.to_f * 0.001 }
  s = Sample.Sampling(temperature: 1.0, seed: 1)
  k = Sample.window(s, 300)
  try Test.eq(k, 256)
  ids = (0...k).to_a
  window = ids.map { |i| logits[i] }
  state = Sample.start(s)
  full_state = state
  below = 0
  for _ in 0...400
    top = try Sample.pick_top(ids, window, s, state)
    try Test.assert(top.id < 256, "pick_top drew id #{top.id}, outside the window")
    state = top.state
    full = try Sample.pick(logits, s, full_state)
    full_state = full.state
    if full.id >= 256
      below += 1
    end
  end

  try Test.assert(below > 0, "pick over all the logits never drew an id below the window")
  ok()
end

def test_temperature_zero_is_the_first_largest_and_draws_nothing() -> Result[Void]
  logits = [0.5, 2.0, 2.0, -1.0]
  greedy = Sample.Sampling(temperature: 0.0, seed: 3)
  d = try Sample.pick(logits, greedy, Sample.start(greedy))
  try Test.eq(d.id, 1)
  try Test.eq(d.state, Sample.start(greedy))
  try Test.eq(try Sample.probabilities(logits, greedy), [0.0, 1.0, 0.0, 0.0])
  # one id kept and a tiny `top_p` are greedy too (a tie at the edge of `top_k` keeps both, as Hugging Face does)
  logits = [0.5, 2.0, 1.5, -1.0]
  for s in [Sample.Sampling(top_k: 1, seed: 4), Sample.Sampling(top_p: 0.0001, seed: 4)]
    state = Sample.start(s)
    for _ in 0...5
      d = try Sample.pick(logits, s, state)
      try Test.eq(d.id, 1)
      state = d.state
    end
  end

  ok()
end

def test_a_seed_gives_the_same_draws_and_another_gives_others() -> Result[Void]
  logits = (0...50).map { |i| (i % 7).to_f * 0.3 }
  a = try draws(logits, Sample.Sampling(seed: 11), 30)
  try Test.eq(a, try draws(logits, Sample.Sampling(seed: 11), 30))
  try Test.assert(a != try draws(logits, Sample.Sampling(seed: 12), 30), "two seeds drew the same ids")
  try Test.assert(a.uniq.size > 5, "30 draws over a flat distribution gave #{a.uniq.size} ids")
  ok()
end

def test_the_frequencies_follow_the_probabilities() -> Result[Void]
  # probabilities 0.5, 0.3, 0.2 and a far tail of ids that are never drawn
  logits = [Math.log(0.5), Math.log(0.3), Math.log(0.2), -80.0]
  counts = [0, 0, 0, 0]
  for id in try draws(logits, Sample.Sampling(seed: 5), 3000)
    counts = counts.set(id, counts[id] + 1)
  end

  try Test.eq(counts[3], 0)
  wants = [0.5, 0.3, 0.2]
  for id in 0...3
    got = counts[id].to_f / 3000.0
    try Test.assert((got - wants[id]).abs < 0.04, "id #{id} came #{got} of the draws, not #{wants[id]}")
  end

  ok()
end

def test_bad_settings_and_logits_are_errors() -> Result[Void]
  logits = [1.0, 2.0]
  try Test.error?(Sample.pick(logits, Sample.Sampling(temperature: -1.0), 1))
  try Test.error?(Sample.pick(logits, Sample.Sampling(temperature: 0.0000001), 1))
  try Test.error?(Sample.check(Sample.Sampling(seed: -1)))
  try Test.error?(Sample.check(Sample.Sampling(seed: 2147483646)))
  try Sample.check(Sample.Sampling(seed: 2147483645))
  try Test.error?(Sample.pick(logits, Sample.Sampling(top_k: -1), 1))
  try Test.error?(Sample.pick(logits, Sample.Sampling(top_p: 0.0), 1))
  try Test.error?(Sample.pick(logits, Sample.Sampling(top_p: 1.5), 1))
  try Test.error?(Sample.pick([], Sample.Sampling(), 1))
  try Test.eq((try Sample.pick([1.0, Math.log(0.0)], Sample.Sampling(), 1)).id, 0)
  Test.error?(Sample.pick([1.0, Math.log(0.0) - Math.log(0.0)], Sample.Sampling(), 1))
end

private

def draws(logits: List[Float], s: Sample.Sampling, count: Int) -> Result[List[Int]]
  state = Sample.start(s)
  out: List[Int] = []
  for _ in 0...count
    d = try Sample.pick(logits, s, state)
    out = out.push(d.id)
    state = d.state
  end

  ok(out)
end
