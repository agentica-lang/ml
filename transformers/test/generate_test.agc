## Generation with a cache against the tiny Qwen3 of `qwen_peer.py`: the logits
## of every step match the peer's forward over the whole row, a prompt padded to
## a bucket and a cache that grows from 8 to 16 to 32 rows included, and the
## greedy ids are those of the forward without a cache. Run from `transformers/`.

import Generate from "../generate"
import Qwen3 from "../qwen3"
import S from "support"
import Tensor from "pkg/tensor/tensor"

ABS = 0.000000001
REL = 0.000000001

F32_ABS = 0.0001
F32_REL = 0.0001

def test_each_step_matches_the_forward_over_the_whole_row() -> Result[Void]
  with_peer do |dir|
    try stepping(dir, S.REF, 0.000000001, 0.000000001)
    for backend in try S.ggml_backends()
      try stepping(dir, backend, F32_ABS, F32_REL)
    end

    ok()
  end
end

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
      m = try Qwen3.open("#{dir}/tiny", backend)
      try Test.error?(Generate.step(m, Generate.start(), []))
      try Test.error?(Generate.step(m, Generate.start(), [1, 13]))
      try Test.error?(Generate.step(m, Generate.start(), List.repeat(1, 33)))
      first = (try Generate.step(m, Generate.start(), [1, 2, 3])).cache
      try Test.error?(Generate.step(m, first, [1, 2]))
    end

    ok()
  end
end

private

## Seven ids after a prompt of three: the cache goes from 8 to 16 rows. The ids are those of the forward without a cache.
def generating(dir: String, backend: Tensor.Backend) -> Result[Void]
  prompt = [1, 5, 2]
  m = try Qwen3.open("#{dir}/tiny", backend)
  want = try greedy(m, prompt, 7)
  got = try Generate.generate(m, prompt, 7)
  try Test.eq(got.ids, want)
  # A stop id ends it before that id, and the steps of the first call serve the next.
  stopped = try Generate.generate(m, prompt, 7, [want[3]], got.graphs)
  try Test.eq(stopped.ids, want.take(first_of(want, want[3])))
  Generate.close_graphs(stopped.graphs)
  nothing = try Generate.generate(m, prompt, 0)
  Test.eq(nothing.ids, [])
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

## The row `long` fed as a prompt of 5 and then one id at a time (the cache goes from 8 to 16 to 32 rows), the
## logits of every step against the peer's forward over the whole row.
def stepping(dir: String, backend: Tensor.Backend, abs: Float, rel: Float) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  e = try expected(dir)
  row = try ints(try JSON.get(e, "long"))
  want = try S.flat(try JSON.get(e, "tiny_long"))
  vocab = m.config.vocab_size
  scored = try Generate.step_logits(m, Generate.start(), row.take(5))
  try S.expect_close(scored.logits, want.drop(4 * vocab).take(vocab), abs, rel, "prompt")
  cache = scored.cache
  for t in 5...row.size
    scored = try Generate.step_logits(m, cache, [row[t]])
    cache = scored.cache
    try S.expect_close(scored.logits, want.drop(t * vocab).take(vocab), abs, rel, "position #{t}")
  end

  try Test.eq(cache.length, row.size)
  Generate.close_graphs(Generate.finish(cache))
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
