## Generation with a cache against the tiny Qwen3 of `qwen_peer.py`: the logits
## of every step match the peer's forward over the whole row, a prompt padded to
## a bucket (of 5, 8 and 9 tokens) and a cache that grows from 8 to 16 to 32 rows
## included, and the greedy ids are those of the forward without a cache. Run from
## `transformers/`.

import Generate from "../generate"
import Ggml from "pkg/tensor/src/ggml"
import Lift from "../lift"
import Qwen3 from "../qwen3"
import S from "support"
import Tensor from "pkg/tensor/tensor"

ABS = 0.000000001
REL = 0.000000001

F32_ABS = 0.0001
F32_REL = 0.0001

## Prompts of 5 (padded to 8), exactly 8 and 9 (padded to 16).
def test_each_step_matches_the_forward_over_the_whole_row() -> Result[Void]
  with_peer do |dir|
    for prompt in [5, 8, 9]
      try stepping(dir, S.REF, prompt, ABS, REL)
      for backend in try S.ggml_backends()
        try stepping(dir, backend, prompt, F32_ABS, F32_REL)
      end
    end

    ok()
  end
end

## On ggml a step writes into the tensors of the cache before the last one when the capacity is the same (they stay
## open, as the keys of the new cache), and closes them when it is not: the step that grows the capacity and the one
## after it close the old ones, and the second step after the growth reuses again.
def test_a_step_reuses_the_cache_before_when_the_capacity_is_settled() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      g = try Generate.open(try Qwen3.open("#{dir}/tiny", backend))
      row = try ints(try JSON.get(try expected(dir), "long"))
      cache = (try Generate.step_logits(g, Generate.start(), row.take(5))).cache
      before = cache.keys[0]
      cache = (try Generate.step_logits(g, cache, [row[5]])).cache
      try Test.assert(!try closed?(before), "the first decode step has no spare to take")
      older = cache.spare[0]
      cache = (try Generate.step_logits(g, cache, [row[6]])).cache
      try Test.assert(!try closed?(older), "a spare of the same capacity was closed")
      try Test.eq(cache.length, 7)

      # Rows 8 to 17 take the capacity from 8 to 16: the step that fills the eighth row still fits 8.
      shrunk = cache.keys[0]
      cache = (try Generate.step_logits(g, cache, [row[7]])).cache
      cache = (try Generate.step_logits(g, cache, [row[8]])).cache
      try Test.eq(Tensor.shape(cache.keys[0])[2], 16)
      try Test.assert(try closed?(shrunk), "the spare of the old capacity stayed open")
      grown = cache.keys[0]
      old = cache.spare[0]
      cache = (try Generate.step_logits(g, cache, [row[9]])).cache
      try Test.assert(try closed?(old), "a spare of the old capacity stayed open after the growth")
      try Test.assert(!try closed?(grown), "the grown cache was closed")
      reused = cache.spare[0]
      cache = (try Generate.step_logits(g, cache, [row[10]])).cache
      try Test.assert(!try closed?(reused), "the second step after the growth did not reuse")
      Generate.finish(cache)
      Generate.close(g)
    end

    ok()
  end
end

def closed?(t: Tensor.Tensor) -> Result[Bool] = ok(Native.closed?(try Ggml.handle_of(t)))

def test_generate_is_the_greedy_choice_of_the_forward_on_the_reference() -> Result[Void]
  with_peer do |dir| generating(dir, S.REF) end
end

def test_generate_is_the_greedy_choice_of_the_forward_on_ggml() -> Result[Void]
  with_peer do |dir|
    for backend in try S.ggml_backends()
      try generating(dir, backend)
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
      ->(_: Int, xs: List[Tensor.Tensor]) { ok([try Generate.picked(xs[0])]) },
    )

    try Test.eq(try S.floats_of(got[0]), [1.0])
  end

  ok()
end

private

## Seven ids after a prompt of three: the cache goes from 8 to 16 rows. The ids are those of the forward without a cache.
## A second call finds the steps of the first: it compiles none.
def generating(dir: String, backend: Tensor.Backend) -> Result[Void]
  prompt = [1, 5, 2]
  m = try Qwen3.open("#{dir}/tiny", backend)
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
  # On ggml a step keeps the cache before it to be overwritten by the next one; elsewhere there is none.
  case backend
    Tensor.Backend.Ggml(_) -> try Test.eq(cache.spare.size, 2 * g.model.config.num_hidden_layers)
    _ -> try Test.eq(cache.spare.size, 0)
  end

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
