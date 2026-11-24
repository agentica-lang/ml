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

struct Golden
  cases: List[Case]
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
