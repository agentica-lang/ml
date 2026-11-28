## The configurable decode batch of `generate.agc` and `serving.agc` (ml#421): ten decode-ready Qwen3 rows in one
## forward, on the reference backend and on ggml's CPU. The rows have mixed prompt lengths and positions; each row's
## ids are the standalone greedy ones, and on ggml one batch round is one actual execution of the backend (prefills
## excluded). The scheduler fills partial batches, reuses a row after a cancellation, and gives every row a turn when
## more rows are active than `max_batch`; `step_pair` stays the two-row wrapper. Run from `transformers/`.

import Generate from "../generate"
import GgmlRaw from "pkg/tensor/src/ggml_native"
import Qwen3 from "../qwen3"
import S from "support"
import Serving from "../serving"
import Tensor from "pkg/tensor/tensor"

## How long a wait for an event may take. The engine is quick; the bound only fails a stuck run.
TIMEOUT_MS = 30000

## Ten prompts of 3 to 7 ids: mixed lengths and positions within one batch.
PROMPTS = [
  [1, 5, 2],
  [4, 4, 9],
  [2, 7, 1, 3],
  [1, 2, 3, 4],
  [9, 8, 7, 6, 5],
  [1, 1, 1, 1, 1],
  [2, 3, 4, 5, 6, 7],
  [4, 4, 9, 2, 7, 1, 5],
  [1, 2, 3, 4, 5, 6, 7],
  [3, 2, 1, 5, 4, 9, 8],
]

## Two more prompts, for the twelve-row fairness case.
EXTRA = [[5, 5, 5, 5, 5, 5], [8, 1, 6, 3, 7, 2]]

## Ten rows batch to their standalone ids, and one batch round is one ggml backend execution: three rounds on the CPU
## engine and one on the reference backend (which costs the most steps, so the case keeps it short).
def test_ten_rows_batch_to_standalone_ids_and_one_ggml_execution() -> Result[Void]
  with_peer do |dir|
    want = try standalone(dir, 4)
    for backend in try S.paired_backends()
      rounds = 3
      if backend == S.REF
        rounds = 1
      end

      try ten_rows(dir, backend, want, rounds)
    end

    ok()
  end
end

## A partial batch and a cancelled row reused by a new request: the seven remaining rows and the replacement decode
## with their standalone ids, and the scheduler kept sharing graphs.
def test_a_partial_batch_and_cancelled_rows_reuse_the_slot() -> Result[Void]
  with_peer do |dir|
    for backend in try S.paired_backends()
      next if backend == S.REF

      try partial_and_cancelled(dir, backend)
    end

    ok()
  end
end

## More active rows than `max_batch`: every round runs two batches (ten rows and two), every row advances, and the
## actual ggml execution count is the twelve prefills plus the four batch executions, never twelve singleton steps.
def test_rows_beyond_the_batch_cap_take_their_turn() -> Result[Void]
  with_peer do |dir|
    for backend in try S.paired_backends()
      next if backend == S.REF

      try over_cap(dir, backend)
    end

    ok()
  end
end

## `step_pair` is still the two-row wrapper, and a batch over `Generate.MAX_BATCH` is refused before it consumes any
## cache. A backend that does not batch is refused there too, with its caches still live.
def test_step_pair_wrapper_and_batch_preflights() -> Result[Void]
  with_peer do |dir|
    try legacy_step_pair(dir)
    for backend in try S.paired_backends()
      next if backend == S.REF

      try over_cap_refusal(dir, backend)
    end

    for backend in try S.unpaired_backends()
      try unpaired_batch(dir, backend)
    end

    try Test.assert(not Generate.pairable(S.XLA), "XLA batches")
    try Test.assert(not Generate.pairable(S.TORCH), "libtorch batches")
    ok()
  end
end

private

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/qwen_peer.py", "agentica-qwen-", body)
end

## The standalone greedy ids of `PROMPTS`, `n` each, from ggml's CPU when it is built (as CI builds it), else the
## reference backend. A ggml run is the cheap one, and every backend gives the same ids here.
def standalone(dir: String, n: Int) -> Result[List[List[Int]]] = standalone_prompts(dir, PROMPTS, n)

def standalone_prompts(dir: String, prompts: List[List[Int]], n: Int) -> Result[List[List[Int]]]
  backend = S.REF
  for b in try S.ggml_backends()
    if b == S.GGML
      backend = b
    end
  end

  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  out: List[List[Int]] = []
  for p in prompts
    out = out.push(try Generate.generate(g, p, n))
  end

  Generate.close(g)
  Qwen3.close_model(m)
  ok(out)
end

## Ten prefilled rows, `rounds` batch rounds, and each row's ids against `want`.
def ten_rows(dir: String, backend: Tensor.Backend, want: List[List[Int]], rounds: Int) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  rows: List[Generate.BatchRow] = []
  ids: List[List[Int]] = []
  for p in PROMPTS
    first = try Generate.step(g, Generate.start(), p)
    rows = rows.push(Generate.BatchRow(cache: first.cache, id: first.id))
    ids = ids.push([first.id])
  end

  for _ in 0...rounds
    before = GgmlRaw.graph_runs()
    made = try Generate.step_batch(g, rows)
    after = GgmlRaw.graph_runs()
    if backend == S.GGML
      try Test.eq(after - before, 1)
    end

    rows = []
    for r in 0...made.size
      ids = ids.set(r, ids[r].push(made[r].id))
      rows = rows.push(Generate.BatchRow(cache: made[r].cache, id: made[r].id))
    end
  end

  for r in 0...ids.size
    try Test.eq(ids[r], want[r].take(rounds + 1))
  end

  for r in rows
    Generate.finish(r.cache)
  end

  Generate.close(g)
  Qwen3.close_model(m)
  ok()
end

## A direct partial batch, then the scheduler cancelling a live row and giving its slot to a new request.
def partial_and_cancelled(dir: String, backend: Tensor.Backend) -> Result[Void]
  want = try standalone(dir, 3)
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  rows: List[Generate.BatchRow] = []
  for p in PROMPTS
    first = try Generate.step(g, Generate.start(), p)
    rows = rows.push(Generate.BatchRow(cache: first.cache, id: first.id))
  end

  made = try Generate.step_batch(g, rows)
  rows = []
  for r in 0...made.size
    rows = rows.push(Generate.BatchRow(cache: made[r].cache, id: made[r].id))
  end

  # Three rows end, and the seven that remain share one graph: the batch is partial, not padded with finished rows.
  for r in 0...3
    Generate.finish(rows[r].cache)
  end

  before = GgmlRaw.graph_runs()
  kept = try Generate.step_batch(g, rows.drop(3))
  after = GgmlRaw.graph_runs()
  try Test.eq(after - before, 1)
  for r in 0...kept.size
    try Test.eq([rows[r + 3].id, kept[r].id], want[r + 3].drop(1).take(2))
    Generate.finish(kept[r].cache)
  end

  Generate.close(g)
  Qwen3.close_model(m)
  try cancelled_reuse(dir, backend)
  ok()
end

## Ten requests on `backend`, one cancelled after its first token and replaced by a new one: every answer is its
## standalone one and the replacement shares the decode rounds of the rows that remain.
def cancelled_reuse(dir: String, backend: Tensor.Backend) -> Result[Void]
  want = try standalone(dir, 3)
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 10, max_pending: 10, max_batch: 10))
  chans: List[Channel[Serving.Event]] = []
  for r in 0...10
    ch: Channel[Serving.Event] = Channel.new(32)
    chans = chans.push(ch)
    try Serving.submit(server, PROMPTS[r], 3, [], ch)
  end

  first: List[Int] = []
  for ch in chans
    first = first.push(try expect_token(ch))
  end

  try Serving.cancel(server, chans[9])
  try expect_failed(chans[9], "cancelled")
  replacement: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [7, 2, 5, 1], 2, [], replacement)
  want_new = try standalone_one(dir, [7, 2, 5, 1], 2, backend)
  for r in 0...9
    got = [first[r]] + try collect(chans[r])
    try Test.eq(got, want[r])
  end

  try Test.eq(try collect(replacement), want_new)
  runs = (try Serving.stats(server)).pair_runs
  try Serving.stop(server)
  try Test.assert(runs > 0, "the reused run shared no decode graph")
  Generate.close(g)
  Qwen3.close_model(m)
  ok()
end

## Twelve requests, `max_batch` ten: every round is two batches and every row advances; the actual ggml executions
## are twelve prefills and four batches (three ids: the prefill and two decode rounds).
def over_cap(dir: String, backend: Tensor.Backend) -> Result[Void]
  all_prompts = PROMPTS + EXTRA
  want = try standalone_prompts(dir, all_prompts, 3)
  before = GgmlRaw.graph_runs()
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 12, max_pending: 12, max_batch: 10))
  chans: List[Channel[Serving.Event]] = []
  for r in 0...10
    ch: Channel[Serving.Event] = Channel.new(32)
    chans = chans.push(ch)
    try Serving.submit(server, PROMPTS[r], 3, [], ch)
  end

  for p in EXTRA
    ch: Channel[Serving.Event] = Channel.new(32)
    chans = chans.push(ch)
    try Serving.submit(server, p, 3, [], ch)
  end

  for r in 0...chans.size
    try Test.eq(try collect(chans[r]), want[r])
  end

  runs = (try Serving.stats(server)).pair_runs
  try Serving.stop(server)
  after = GgmlRaw.graph_runs()
  Generate.close(g)
  Qwen3.close_model(m)
  # Twelve prefills, then two rounds of a batch of ten and a batch of two: four shared batch executions. The delta is
  # isolated: `standalone` ran before `before`, and this test owns the process's records while it runs.
  try Test.eq(after - before, 16)
  Test.eq(runs, 4)
end

## `step_pair` over two prefilled rows is `step_batch` over the same two rows: the ids are the standalone ones.
def legacy_step_pair(dir: String) -> Result[Void]
  want_a = try standalone_one(dir, [1, 5, 2], 3, S.REF)
  want_b = try standalone_one(dir, [4, 4, 9, 2, 7, 1], 3, S.REF)
  m = try Qwen3.open("#{dir}/tiny", S.REF)
  g = try Generate.open(m)
  a = try Generate.step(g, Generate.start(), [1, 5, 2])
  b = try Generate.step(g, Generate.start(), [4, 4, 9, 2, 7, 1])
  p = try Generate.step_pair(g, a.cache, a.id, b.cache, b.id)
  q = try Generate.step_pair(g, p.a.cache, p.a.id, p.b.cache, p.b.id)
  try Test.eq([a.id, p.a.id, q.a.id], want_a)
  try Test.eq([b.id, p.b.id, q.b.id], want_b)
  Generate.finish(q.a.cache)
  Generate.finish(q.b.cache)
  Generate.close(g)
  Qwen3.close_model(m)
  ok()
end

## `Serving.Limits` keeps `max_batch` inside 1..`Generate.MAX_BATCH`; a width over `max_active` is clamped, and the old
## `max_active: 1` call with defaults still starts. The backend capacities are the one source `batch_width` names.
def test_serving_limits_validate_max_batch() -> Result[Void]
  with_peer do |dir|
    try Test.eq(Generate.batch_width(S.REF), Generate.MAX_BATCH)
    try Test.eq(Generate.batch_width(S.GGML), Generate.MAX_BATCH)
    try Test.eq(Generate.batch_width(S.VULKAN), 2)
    try Test.eq(Generate.batch_width(S.XLA), 0)
    try Test.eq(Generate.batch_width(S.TORCH), 0)
    m = try Qwen3.open("#{dir}/tiny", S.REF)
    g = try Generate.open(m)
    try Test.error?(Serving.start(g, Serving.Limits(max_active: 10, max_pending: 4, max_batch: 11)))
    try Test.error?(Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4, max_batch: 0)))
    old = try Serving.start(g, Serving.Limits(max_active: 1, max_pending: 1))
    try Serving.stop(old)
    wide = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4, max_batch: 3))
    try Serving.stop(wide)
    Generate.close(g)
    Qwen3.close_model(m)
    ok()
  end
end

## A backend that does not batch: `step_batch` refuses two prefilled rows before reading either cache, and both still
## decode on their own to the backend's standalone ids.
def unpaired_batch(dir: String, backend: Tensor.Backend) -> Result[Void]
  try Test.assert(not Generate.pairable(backend), "#{backend} batches: this case needs a backend that steps alone")
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  a = try Generate.step(g, Generate.start(), [1, 5, 2])
  b = try Generate.step(g, Generate.start(), [4, 4, 9, 2, 7, 1])
  try S.expect_error(
    Generate.step_batch(g, [Generate.BatchRow(cache: a.cache, id: a.id), Generate.BatchRow(cache: b.cache, id: b.id)]),
    "steps its rows alone",
  )

  want_a = try Generate.generate(g, [1, 5, 2], 2)
  want_b = try Generate.generate(g, [4, 4, 9, 2, 7, 1], 2)
  next_a = try Generate.step(g, a.cache, [a.id])
  next_b = try Generate.step(g, b.cache, [b.id])
  try Test.eq([a.id, next_a.id], want_a)
  try Test.eq([b.id, next_b.id], want_b)
  Generate.finish(next_a.cache)
  Generate.finish(next_b.cache)
  Generate.close(g)
  Qwen3.close_model(m)
  ok()
end

## A batch over `Generate.MAX_BATCH` is refused before it reads a cache: the eleven rows still decode on their own.
def over_cap_refusal(dir: String, backend: Tensor.Backend) -> Result[Void]
  want = try standalone(dir, 2)
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  rows: List[Generate.BatchRow] = []
  ids: List[List[Int]] = []
  for p in PROMPTS
    first = try Generate.step(g, Generate.start(), p)
    rows = rows.push(Generate.BatchRow(cache: first.cache, id: first.id))
    ids = ids.push([first.id])
  end

  solo = try Generate.step(g, Generate.start(), PROMPTS[0])
  single = try Generate.step_batch(g, [Generate.BatchRow(cache: solo.cache, id: solo.id)])
  try Test.eq([solo.id, single[0].id], want[0])
  Generate.finish(single[0].cache)
  extra = try Generate.step(g, Generate.start(), [1, 2, 3])
  try S.expect_error(Generate.step_batch(g, []), "at least one decode")
  too_many = rows + [Generate.BatchRow(cache: extra.cache, id: extra.id)]
  try S.expect_error(Generate.step_batch(g, too_many), "at most #{Generate.MAX_BATCH}")
  for r in 0...rows.size
    one = try Generate.step(g, rows[r].cache, [rows[r].id])
    try Test.eq(ids[r] + [one.id], want[r])
    Generate.finish(one.cache)
  end

  Generate.finish(extra.cache)
  Generate.close(g)
  Qwen3.close_model(m)
  ok()
end

## The standalone greedy ids of one prompt on `backend`: the backend the case compares with.
def standalone_one(dir: String, prompt: List[Int], n: Int, backend: Tensor.Backend) -> Result[List[Int]]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  out = try Generate.generate(g, prompt, n)
  Generate.close(g)
  Qwen3.close_model(m)
  ok(out)
end

## The next token on `reply`, within `TIMEOUT_MS`.
def expect_token(reply: Channel[Serving.Event]) -> Result[Int]
  case reply.recv(TIMEOUT_MS)
    ok(ev) ->
      case ev
        Serving.Event.Token(id) -> ok(id)
        Serving.Event.Done -> error("the sequence ended before its next token")
        Serving.Event.Failed(e) -> error("the sequence failed: #{e}")
      end

    error(e) -> error("no event in #{TIMEOUT_MS}ms: #{e.message}")
  end
end

## The `Failed` on `reply` within `TIMEOUT_MS`, after any tokens already buffered.
def expect_failed(reply: Channel[Serving.Event], phrase: String) -> Result[Void]
  while true
    case reply.recv(TIMEOUT_MS)
      ok(ev) ->
        case ev
          Serving.Event.Failed(e) ->
            try Test.assert(e.include?(phrase), "\"#{e}\" does not hold \"#{phrase}\"")
            return ok()
          Serving.Event.Done -> return error("the sequence ended with Done, not a failure")
          Serving.Event.Token(_) -> nil
        end

      error(e) -> return error("no event in #{TIMEOUT_MS}ms: #{e.message}")
    end
  end

  ok()
end

## Every token on `reply` until `Done`.
def collect(reply: Channel[Serving.Event]) -> Result[List[Int]]
  out: List[Int] = []
  going = true
  while going
    case reply.recv(TIMEOUT_MS)
      ok(ev) ->
        case ev
          Serving.Event.Token(id) -> out = out.push(id)
          Serving.Event.Done -> going = false
          Serving.Event.Failed(e) -> return error("the sequence failed: #{e}")
        end

      error(e) -> return error("no event in #{TIMEOUT_MS}ms: #{e.message}")
    end
  end

  ok(out)
end
