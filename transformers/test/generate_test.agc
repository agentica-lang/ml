## Generation with a cache against the tiny Qwen3 of `qwen_peer.py`: the logits
## of every step match the peer's forward over the whole row, a prompt padded to
## a bucket (of 5, 8 and 9 tokens) and a cache that grows from 8 to 16 to 32 rows
## included, and the greedy ids are those of the forward without a cache. Run from
## `transformers/`.

import Generate from "../generate"
import Stepping from "../stepping"
import Ggml from "pkg/tensor/src/ggml"
import Lift from "../lift"
import Qwen3 from "../qwen3"
import S from "support"
import Sample from "../sample"
import Tensor from "pkg/tensor/tensor"

ABS = 0.000000001
REL = 0.000000001

# The cache on ggml is F16: its keys and values are rounded to 11 bits, so the logits of the tiny model drift by up to
# 2.4e-4 (the real Qwen3-0.6B by 7e-3 on logits of 10 to 30); the bound is that gap with a margin, and a
# relative part of 1e-3 only.
HALF_ABS = 0.0006
HALF_REL = 0.001

# The two generators of `reading` read the same F16 cache: only the order of the sums of an attention may differ.
SAME_ABS = 0.00001
SAME_REL = 0.00001

## Prompts of 5 (padded to 8), exactly 8 and 9 (padded to 16).
def test_each_step_matches_the_forward_over_the_whole_row() -> Result[Void]
  with_peer do |dir|
    for prompt in [5, 8, 9]
      try stepping(dir, S.REF, prompt, ABS, REL)
      for backend in try S.ggml_backends()
        try stepping(dir, backend, prompt, HALF_ABS, HALF_REL)
      end
    end

    ok()
  end
end

## On ggml a one-token step of the same capacity writes its row into the tensors it was given, which are the keys of the
## cache it returns (they stay open); the step that grows the capacity makes new ones and closes the old.
def test_a_step_writes_the_cache_in_place_until_the_capacity_grows() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      g = try Generate.open(try Qwen3.open("#{dir}/tiny", backend))
      row = try ints(try JSON.get(try expected(dir), "long"))
      cache = (try Generate.step_logits(g, Generate.start(), row.take(5))).cache
      prompt = cache.keys[0]
      try Test.eq(Tensor.dtype(prompt), Tensor.Dtype.F16)
      for t in [5, 6, 7]
        cache = (try Generate.step_logits(g, cache, [row[t]])).cache
        try Test.assert(!try closed?(prompt), "step #{t} closed the cache it was given")
        try Test.eq(cache.length, t + 1)
      end

      # The ninth token needs 16 rows: new tensors, and the old ones are closed.
      cache = (try Generate.step_logits(g, cache, [row[8]])).cache
      try Test.assert(try closed?(prompt), "the old capacity stayed open")
      grown = cache.keys[0]
      for t in [9, 10]
        cache = (try Generate.step_logits(g, cache, [row[t]])).cache
        try Test.assert(!try closed?(grown), "step #{t} closed the cache it was given")
      end

      Generate.finish(cache)
      try Test.assert(try closed?(grown), "finish left the cache open")
      Generate.close(g)
    end

    ok()
  end
end

def closed?(t: Tensor.Tensor) -> Result[Bool] = ok(Native.closed?(try Ggml.handle_of(t)))

## A generator of `m` whose steps read the whole capacity of the cache: its own plans, and the logits that a step
## over the rows in use must give.
def reading_all(m: Qwen3.Qwen3) -> Result[Generate.Generator] = ok((try Generate.open(m)).with(whole: true))

## How many compiled steps `g` keeps now.
def held(g: Generate.Generator) -> Result[Int]
  plans = try g.plans.recv(1000)
  try g.plans.send(plans, 0)
  ok(plans.graphs.size)
end

## On ggml a one-token step reads the rows of the cache in use, rounded up to a reach, and each reach is a compiled
## step of its own. Three sequences each cross one: 768 rows inside a capacity of 1024, 1024 rows where the capacity
## grows to 2048 (reach 1280), and 1280 rows inside it. Every step gives the logits of a step that reads the whole
## capacity, the step of a reach ends the step of the reach before it, and a later sequence at a smaller reach
## takes the step that is there, so a capacity keeps one step.
def test_a_step_reads_the_rows_in_use_and_gives_the_logits_of_the_whole_cache() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      m = try Qwen3.open("#{dir}/roomy", backend)
      g = try Generate.open(m)
      whole = try reading_all(m)
      # the prompt, a step of each reach: 768 and 1024
      try reading(g, whole, 765, 5)
      try Test.eq(Generate.compiles(g), 3)
      try Test.eq(Generate.compiles(whole), 2)
      try Test.eq(try held(g), 2)
      # nothing new up to 1024 rows; then the step that grows the capacity and the step of reach 1280
      try reading(g, whole, 1022, 4)
      try Test.eq(Generate.compiles(g), 5)
      try Test.eq(try held(g), 4)
      # the prompt of 2048, the reach of 1280 as it is, and the reach of 1536, which ends that one
      try reading(g, whole, 1278, 4)
      try Test.eq(Generate.compiles(g), 7)
      try Test.eq(try held(g), 5)
      # a second sequence at the reach that ended takes the step of the larger one
      try reading(g, whole, 1278, 1)
      try Test.eq(Generate.compiles(g), 7)
      try Test.eq(try held(g), 5)
      Generate.close(g)
      Generate.close(whole)
      Qwen3.close_model(m)
    end

    ok()
  end
end

## One sequence that grows from 760 to 1290 rows, a token at a time: it crosses the reach of 768 rows, the growth of
## the capacity from 1024 to 2048 rows and the reach of 1280, with the logits of the whole cache at every step, and
## it leaves one step per capacity.
def test_a_sequence_that_crosses_three_reaches_gives_the_logits_of_the_whole_cache() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      m = try Qwen3.open("#{dir}/roomy", backend)
      g = try Generate.open(m)
      whole = try reading_all(m)
      try reading(g, whole, 760, 530)
      # the prompt, the reaches of 768 and 1024, the growth, the reaches of 1280 and 1536
      try Test.eq(Generate.compiles(g), 6)
      try Test.eq(try held(g), 4)
      try Test.eq(Generate.compiles(whole), 4)
      Generate.close(g)
      Generate.close(whole)
      Qwen3.close_model(m)
    end

    ok()
  end
end

## A prompt of `first` ids and `steps` more, one at a time, through `g` and through `whole`: the same logits at each.
def reading(g: Generate.Generator, whole: Generate.Generator, first: Int, steps: Int) -> Result[Void]
  vocab = g.model.config.vocab_size
  on = Tensor.backend(g.model.tokens.table)
  row = (0...(first + steps)).map { |i| (i * 7 + i / 5) % vocab }
  ours = try Generate.step_logits(g, Generate.start(), row.take(first))
  theirs = try Generate.step_logits(whole, Generate.start(), row.take(first))
  try S.expect_close(ours.logits, theirs.logits, SAME_ABS, SAME_REL, "prompt of #{first} on #{on}")
  for t in first...row.size
    ours = try Generate.step_logits(g, ours.cache, [row[t]])
    theirs = try Generate.step_logits(whole, theirs.cache, [row[t]])
    try S.expect_close(ours.logits, theirs.logits, SAME_ABS, SAME_REL, "row #{t} after a prompt of #{first} on #{on}")
  end

  Generate.finish(ours.cache)
  Generate.finish(theirs.cache)
  ok()
end

def test_generate_is_the_greedy_choice_of_the_forward_on_the_reference() -> Result[Void]
  with_peer do |dir| generating(dir, "tiny", S.REF) end
end

def test_generate_is_the_greedy_choice_of_the_forward_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      try generating(dir, "tiny", backend)
    end

    ok()
  end
end

## The Qwen2 layout (no query and key norms, a bias on the query, key and value) through the cache.
def test_generate_is_the_greedy_choice_of_the_forward_for_the_qwen2_layout() -> Result[Void]
  with_peer do |dir|
    try generating(dir, "qwen2", S.REF)
    for backend in try S.ggml_backends()
      try generating(dir, "qwen2", backend)
    end

    ok()
  end
end

## A sampled run is the greedy one when only the largest id can be drawn, the same for the same seed, and not the same
## for every seed.
def test_sampling_is_greedy_when_one_id_is_left_and_seeded_otherwise_on_the_reference() -> Result[Void]
  with_peer do |dir| sampling(dir, S.REF) end
end

def test_sampling_is_greedy_when_one_id_is_left_and_seeded_otherwise_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      try sampling(dir, backend)
    end

    ok()
  end
end

def test_bad_steps_are_errors() -> Result[Void]
  with_peer do |dir|
    for backend in [S.REF] + try S.ggml_backends()
      g = try Generate.open(try Qwen3.open("#{dir}/tiny", backend))
      try Test.error?(Generate.step(g, Generate.start(), []))
      try Test.error?(Generate.step(g, Generate.start(), [1, 13]))
      try Test.error?(Generate.step(g, Generate.start(), List.repeat(1, 33)))
      first = (try Generate.step(g, Generate.start(), [1, 2, 3])).cache
      try Test.error?(Generate.step(g, first, [1, 2]))
      # 3 + 31 - 1 positions are more than the 32 of the model.
      try Test.error?(Generate.generate(g, [1, 2, 3], 31))
      Generate.close(g)
    end

    ok()
  end
end

## Ties go to the first index, as `argmax` and Hugging Face's greedy pick do.
def test_the_pick_of_a_tie_is_the_first_index() -> Result[Void]
  for backend in [S.REF] + try S.ggml_backends()
    scores = try Tensor.from_list(
      [1.0, 3.0, 2.0, 3.0, 3.0, 0.0].map { |f| Tensor.Values.Float(f) },
      [6],
      Tensor.Dtype.F32,
      backend,
    )

    got = try Lift.staged(
      0,
      ->(m: Int) { m },
      [scores],
      ->(_: Int, xs: List[Tensor.Tensor]) { ok([try Stepping.picked(xs[0])]) },
    )

    try Test.eq(try S.floats_of(got[0]), [1.0])
  end

  ok()
end

## A row of 1024 logits or more finds its maximum in pieces on ggml (1500 = 23 pieces of 64 and a tail of 28); a tie
## still goes to the first index, across two pieces, at a seam or in the tail.
def test_the_pick_of_a_tie_in_a_long_row_is_the_first_index() -> Result[Void]
  n = 1500
  base = (0...n).map { |i| (i % 13).to_f * 0.25 }
  for backend in [S.REF] + try S.ggml_backends()
    for tie in [[63, 64], [5, 1499], [1471, 1490], [1472, 1499], [0, 63, 1472]]
      scores = try S.f32(tie.reduce(base) do |acc, i| acc.set(i, 9.0) end, [n], backend)
      got = try Lift.staged(
        0,
        ->(m: Int) { m },
        [scores],
        ->(_: Int, xs: List[Tensor.Tensor]) { ok([try Stepping.picked(xs[0])]) },
      )

      try Test.eq(try S.floats_of(got[0]), [tie[0].to_f])
    end
  end

  ok()
end

private

def sampling(dir: String, backend: Tensor.Backend) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  prompt = [1, 5, 2]
  want = try Generate.generate(g, prompt, 8)
  for s in [Sample.Sampling(temperature: 0.0), Sample.Sampling(top_k: 1, seed: 9)]
    try Test.eq(try Generate.generate(g, prompt, 8, [], some(s)), want)
  end

  warm = Sample.Sampling(temperature: 1.5, seed: 21)
  first = try Generate.generate(g, prompt, 8, [], some(warm))
  try Test.eq(try Generate.generate(g, prompt, 8, [], some(warm)), first)
  differs = false
  for seed in 1...4
    if try Generate.generate(g, prompt, 8, [], some(warm.with(seed: seed))) != first
      differs = true
    end
  end

  try Test.assert(differs, "four seeds drew the same ids")
  try Test.error?(Generate.generate(g, prompt, 8, [], some(Sample.Sampling(top_p: 0.0))))
  try Test.error?(Generate.generate(g, prompt, 8, [], some(Sample.Sampling(temperature: -1.0))))
  try Test.error?(Generate.generate(g, prompt, 8, [], some(Sample.Sampling(top_k: -1, temperature: 0.0))))
  Generate.close(g)
  ok()
end

## Seven ids after a prompt of three: the cache goes from 8 to 16 rows. The ids are those of the forward without a cache.
## A second call finds the steps of the first: it compiles none.
def generating(dir: String, folder: String, backend: Tensor.Backend) -> Result[Void]
  prompt = [1, 5, 2]
  m = try Qwen3.open("#{dir}/#{folder}", backend)
  g = try Generate.open(m)
  want = try greedy(m, prompt, 7)
  try Test.eq(try Generate.generate(g, prompt, 7), want)
  made = Generate.compiles(g)
  case backend
    Tensor.Backend.Ggml(_) -> try Test.assert(made >= 3, "the steps compiled: #{made}")
    _ -> try Test.eq(made, 0)
  end

  # A stop id ends it before that id.
  try Test.eq(try Generate.generate(g, prompt, 7, [want[3]]), want.take(first_of(want, want[3])))
  try Test.eq(try Generate.generate(g, prompt, 0), [])
  try Test.eq(Generate.compiles(g), made)
  Generate.close(g)
  ok()
end

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/qwen_peer.py", "agentica-qwen-", body)
end

def expected(dir: String) -> Result[JSON.Value] = JSON.parse(try File.read_text("#{dir}/expected.json"))

def ints(v: JSON.Value) -> Result[List[Int]]
  out: List[Int] = []
  for x in try JSON.list(v)
    out = out.push(try JSON.int(x))
  end

  ok(out)
end

## The first index of `id` in `ids`.
def first_of(ids: List[Int], id: Int) -> Int
  for i in 0...ids.size
    return i if ids[i] == id
  end

  ids.size
end

## The row `long` fed as a prompt of `first` and then one id at a time (the cache goes from 8 to 16 to 32 rows), the
## logits of every step against the peer's forward over the whole row.
def stepping(dir: String, backend: Tensor.Backend, first: Int, abs: Float, rel: Float) -> Result[Void]
  g = try Generate.open(try Qwen3.open("#{dir}/tiny", backend))
  vocab = g.model.config.vocab_size
  e = try expected(dir)
  row = try ints(try JSON.get(e, "long"))
  want = try S.flat(try JSON.get(e, "tiny_long"))
  scored = try Generate.step_logits(g, Generate.start(), row.take(first))
  try S.expect_close(scored.logits, want.drop((first - 1) * vocab).take(vocab), abs, rel, "prompt of #{first}")
  cache = scored.cache
  for t in first...row.size
    scored = try Generate.step_logits(g, cache, [row[t]])
    cache = scored.cache
    try S.expect_close(scored.logits, want.drop(t * vocab).take(vocab), abs, rel, "position #{t}")
  end

  try Test.eq(cache.length, row.size)
  Generate.finish(cache)
  Generate.close(g)
  ok()
end

## `n` ids after `prompt`, each the largest logit of the forward over everything so far (no cache).
def greedy(m: Qwen3.Qwen3, prompt: List[Int], n: Int) -> Result[List[Int]]
  ids = prompt
  out: List[Int] = []
  for _ in 0...n
    last = try S.floats_of(try Qwen3.run(m, [ids], true))
    best = 0
    for i in 0...last.size
      if last[i] > last[best]
        best = i
      end
    end

    ids = ids.push(best)
    out = out.push(best)
  end

  ok(out)
end
