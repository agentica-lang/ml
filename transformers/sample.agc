## Choosing the next id from the logits by sampling, with the order of Hugging Face's warpers: the temperature, then
## `top_k`, then `top_p`, and a draw from what is left. The pick is on the host, over the logits that
## `Generate.step_logits` reads back; greedy decoding (`none`) stays in the graph.

## Logits more than this below the largest carry less than e^-30 (1e-13) of the mass: they are not candidates.
SPAN = 30.0

MODULUS = 2147483647

## How to sample. `temperature` 0 is greedy (the first id of the largest logit); `top_k` 0 and `top_p` 1.0 keep all
## the ids. The same `seed` gives the same ids on every backend.
struct Sampling
  temperature: Float = 1.0
  top_k: Int = 0
  top_p: Float = 1.0
  seed: Int = 0
end

## The id drawn and the state of the generator after it.
struct Drawn
  id: Int
  state: Int
end

## The state of the generator of `s`: Park-Miller's minimal standard (`x = 48271 x mod 2^31-1`, as `nn/init.agc`), never 0.
def start(s: Sampling) -> Int = s.seed % (MODULUS - 1) + 1

## The id of one draw over `logits` under `s`, from `state`. Temperature 0 does not draw: the state is the same.
def pick(logits: List[Float], s: Sampling, state: Int) -> Result[Drawn]
  try check(s)
  return error("pick needs logits") if logits.empty?
  return ok(Drawn(id: first_largest(logits), state: state)) if s.temperature == 0.0

  next_state = (state * 48271) % MODULUS
  u = next_state.to_f / MODULUS.to_f
  kept = try candidates(logits, s)
  ok(Drawn(id: draw(kept, u), state: next_state))
end

## The probability of every id (`logits.size` of them) once the temperature, `top_k` and `top_p` are applied: 0 for an
## id that is cut. For the temperature 0 the probability is 1 at the first largest.
def probabilities(logits: List[Float], s: Sampling) -> Result[List[Float]]
  try check(s)
  return error("probabilities needs logits") if logits.empty?
  out = (0...logits.size).map { |_| 0.0 }
  return ok(out.set(first_largest(logits), 1.0)) if s.temperature == 0.0

  kept = try candidates(logits, s)
  weights = weights_of(kept)
  total = weights.reduce(0.0) do |a, w| a + w end
  for i in 0...kept.size
    out = out.set(kept[i].id, weights[i] / total)
  end

  ok(out)
end

private

struct Candidate
  id: Int
  score: Float
end

def check(s: Sampling) -> Result[Void]
  return error("temperature must be 0 or more, got #{s.temperature}") if not (s.temperature >= 0.0)
  return error("top_k must be 0 or more, got #{s.top_k}") if s.top_k < 0
  return error("top_p must be above 0 and at most 1, got #{s.top_p}") if not (s.top_p > 0.0 and s.top_p <= 1.0)
  ok()
end

def first_largest(logits: List[Float]) -> Int
  best = 0
  for i in 1...logits.size
    if logits[i] > logits[best]
      best = i
    end
  end

  best
end

## The ids that can be drawn (a `-inf` logit is a masked id, a NaN is an error), largest score first (the lower id first on a tie): the scores are the logits over the
## temperature, cut to the `top_k` largest (ties of the last kept) and to the smallest set whose mass reaches `top_p`.
def candidates(logits: List[Float], s: Sampling) -> Result[List[Candidate]]
  top = logits[first_largest(logits)]
  return error("the logits are not finite") if top - top != 0.0
  floor = top - SPAN * s.temperature
  found: List[Candidate] = []
  for i in 0...logits.size
    return error("logit #{i} is NaN") if logits[i] != logits[i]
    if logits[i] >= floor
      found = found.push(Candidate(id: i, score: logits[i] / s.temperature))
    end
  end

  ordered = found.sort_by { |c| -c.score }
  if s.top_k > 0 and ordered.size > s.top_k
    edge = ordered[s.top_k - 1].score
    ordered = ordered.select { |c| c.score >= edge }
  end

  if s.top_p < 1.0
    ordered = within_mass(ordered, s.top_p)
  end

  ok(ordered)
end

## The largest candidates whose mass before each one is under `top_p` (so at least the first).
def within_mass(ordered: List[Candidate], top_p: Float) -> List[Candidate]
  weights = weights_of(ordered)
  total = weights.reduce(0.0) do |a, w| a + w end
  kept: List[Candidate] = []
  before = 0.0
  for i in 0...ordered.size
    break if before / total >= top_p
    kept = kept.push(ordered[i])
    before += weights[i]
  end

  kept
end

def weights_of(ordered: List[Candidate]) -> List[Float]
  top = ordered[0].score
  ordered.map { |c| Math.exp(c.score - top) }
end

## The first candidate whose cumulative mass passes `u` of the total.
def draw(ordered: List[Candidate], u: Float) -> Int
  weights = weights_of(ordered)
  total = weights.reduce(0.0) do |a, w| a + w end
  target = u * total
  seen = 0.0
  for i in 0...ordered.size
    seen += weights[i]
    return ordered[i].id if seen > target
  end

  ordered[ordered.size - 1].id
end
