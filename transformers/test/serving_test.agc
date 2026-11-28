## The scheduler of `serving.agc` against the tiny Qwen3 of `qwen_peer.py`, on the reference backend and on ggml
## (`standalone` computes the greedy ids on the backend that costs the fewest steps, see there): B decodes while A
## provably still runs, both answers match the standalone greedy ids, a stop id truncates without becoming a token,
## cancels and bad steps are isolated with their caches freed, a slow and a disconnected consumer stall nobody, a full
## queue refuses, and the stop ends the task (even a submit racing it is answered). Every wait is a channel wait with a
## generous bound, never a sleep. Run from `transformers/`.

import Generate from "../generate"
import Qwen3 from "../qwen3"
import S from "support"
import Serving from "../serving"
import Tensor from "pkg/tensor/tensor"

## How long a wait for an event may take. The reference steps are quick; the bound only fails a stuck run.
TIMEOUT_MS = 30000

## B decodes while A provably still runs: both answers match the standalone greedy ids for different prompts.
def test_interleaved_sequences_match_standalone_generation() -> Result[Void] = with_peer do |dir| interleaved(dir) end

## Cancelling A frees its cache and reports it, while B runs on and the server takes new work.
def test_cancellation_is_isolated_and_frees_the_cache() -> Result[Void] = with_peer do |dir| cancelled(dir) end

## A slow consumer (a full reply channel) and a disconnected one (a closed channel) lose their own sequences while the
## others complete untouched.
def test_slow_and_disconnected_consumers_stall_nobody() -> Result[Void] = with_peer do |dir| slow_and_gone(dir) end

## Bad steps fail their own requests, with the cache they were given already consumed, and the server keeps serving.
def test_step_failures_are_isolated() -> Result[Void] = with_peer do |dir| bad_steps(dir) end

## With one slot stepping and one waiting, the next requests are refused, either at the queue or on their own channel,
## and the admitted ones still match the standalone ids.
def test_a_full_queue_refuses_and_keeps_what_it_holds() -> Result[Void] = with_peer do |dir| saturated(dir) end

## A stop id ends the sequence without ever becoming a token, and the end still arrives.
def test_a_stop_id_ends_the_sequence_without_a_token() -> Result[Void] = with_peer do |dir| stop_id(dir) end

## The stop fails what is live and waiting, ends the task, and afterwards the server stays stopped. The sequential
## orderings around the stop are answered, never left hanging: a submit buffered before the stop hears it, and a
## submit after it is refused at once.
def test_the_stop_ends_everything_cleanly() -> Result[Void] = with_peer do |dir| shutdown(dir) end

## The second slice of ml#90: two decode rows share one graph. Mixed prompt lengths decode in pairs, both answers
## match the standalone greedy ids, and every paired round ran one shared graph (prefills excluded). The standalone
## ids come from `standalone`: the backend that costs the fewest steps, and the one this run of the test serves from.
## `generate_test.agc` pins the reference backend's ids to those same ids, so this file does not decode every prompt
## on the reference backend once more.
def test_paired_decode_matches_standalone_generation() -> Result[Void]
  with_peer do |dir|
    want_a = try standalone(dir, [1, 5, 2], 10)
    want_b = try standalone(dir, [4, 4, 9, 2, 7, 1], 8)
    try paired(dir, S.REF, want_a, want_b)
    for backend in try S.paired_backends()
      try paired(dir, backend, want_a, want_b)
    end

    ok()
  end
end

## B joins while A is in decode: the join pairs them from B's prefill round on, and neither answer changes.
def test_b_joining_mid_decode_pairs_without_changing_tokens() -> Result[Void]
  with_peer do |dir|
    want_a = try standalone(dir, [1, 5, 2], 12)
    want_b = try standalone(dir, [4, 4, 9, 2, 7, 1], 5)
    try joining(dir, S.REF, want_a, want_b)
    for backend in try S.paired_backends()
      try joining(dir, backend, want_a, want_b)
    end

    ok()
  end
end

## Cancelling A ends only A: B runs on with the same ids, C reuses the free row and pairs with B, and the count holds
## one shared graph per paired round.
def test_cancellation_keeps_the_peer_on_paired_tokens() -> Result[Void]
  with_peer do |dir|
    want_b = try standalone(dir, [4, 4, 9, 2, 7, 1], 5)
    want_c = try standalone(dir, [1, 2, 3], 4)
    try cancelled_pair(dir, S.REF, want_b, want_c)
    for backend in try S.paired_backends()
      try cancelled_pair(dir, backend, want_b, want_c)
    end

    ok()
  end
end

## One active row steps alone, through the single path: its ids match the standalone ones and no paired step ran,
## so prefills and single rows never touch the paired count.
def test_one_active_row_steps_alone() -> Result[Void]
  with_peer do |dir|
    want = try standalone(dir, [2, 7, 1], 5)
    try solo(dir, S.REF, want)
    for backend in try S.ggml_backends()
      try solo(dir, backend, want)
    end

    ok()
  end
end

## A paired step needs two decodes: an empty cache, or an id outside the vocabulary, fails the step and consumes both
## caches.
def test_step_pair_needs_two_prefilled_caches() -> Result[Void]
  with_peer do |dir|
    try pair_errors(dir, S.REF)
    for backend in try S.paired_backends()
      try pair_errors(dir, backend)
    end

    ok()
  end
end

## Only the backends `Generate.pairable` names pair: the reference backend and ggml's CPU. A Vulkan device, XLA and
## libtorch take the singleton path, whatever this machine has installed.
def test_only_the_reference_and_cpu_ggml_pair() -> Result[Void]
  try Test.assert(Generate.pairable(S.REF), "the reference backend does not pair")
  try Test.assert(Generate.pairable(S.GGML), "ggml's CPU does not pair")
  try Test.assert(not Generate.pairable(S.VULKAN), "a Vulkan device pairs")
  try Test.assert(not Generate.pairable(S.XLA), "XLA pairs")
  Test.assert(not Generate.pairable(S.TORCH), "libtorch pairs")
end

## Two decode-ready rows on a backend that does not pair (XLA and a Vulkan device, when they are installed): the
## scheduler keeps both, steps each alone, and both answers are the ids of that backend's own standalone run. Nothing is
## faked: where no such backend is installed the test says so and runs none.
def test_rows_step_alone_where_the_backend_does_not_pair() -> Result[Void]
  with_peer do |dir|
    backends = try S.unpaired_backends()
    if backends.empty?
      IO.warn("SKIP no backend without pairing is installed (XLA, Vulkan): the fallback ran nowhere")
    end

    for backend in backends
      want_a = try standalone(dir, [1, 5, 2], 4, some(backend))
      want_b = try standalone(dir, [4, 4, 9, 2, 7, 1], 3, some(backend))
      try unpaired_rows(dir, backend, want_a, want_b)
    end

    ok()
  end
end

## A stop id in a paired round ends only its row: A hears Done after its token before the stop id, B runs on with the
## same ids, and the count holds the single shared graph that ended A.
def test_a_stop_id_in_a_paired_round_ends_only_that_row() -> Result[Void]
  with_peer do |dir|
    want_a = try standalone(dir, [1, 5, 2], 2)
    want_b = try standalone(dir, [4, 4, 9, 2, 7, 1], 5)
    try Test.assert(want_a.size == 2, "the probe run has #{want_a.size} ids, needs 2 for the stop-id pick")
    try stopped_pair(dir, S.REF, want_a, want_b)
    for backend in try S.paired_backends()
      try stopped_pair(dir, backend, want_a, want_b)
    end

    ok()
  end
end

private

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/qwen_peer.py", "agentica-qwen-", body)
end

## The standalone greedy ids of `ids` for `n` tokens, from the backend that costs the fewest steps: ggml when it is
## built (as CI builds it), the reference backend otherwise, or `on` when a case names the backend it compares with.
## The ids are the backend's own, and the assertion is against them: what a service must give back is this row's greedy
## ids on the backend it runs on. `generate_test.agc` pins the reference backend's ids to these ones
## (`test_a_paired_step_matches_the_rows_decoded_alone`), so no case here pays a reference prefill and a reference
## decode per row again.
def standalone(dir: String, ids: List[Int], n: Int, on: Optional[Tensor.Backend] = none) -> Result[List[Int]]
  backends = try S.ggml_backends()
  backend = S.REF
  if backends.size > 0
    backend = backends[0]
  end

  case on
    some(b) -> backend = b
    none -> nil
  end

  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  out = try Generate.generate(g, ids, n)
  Generate.close(g)
  Qwen3.close_model(m)
  ok(out)
end

## Mixed prompt lengths (3 and 6 ids) through one paired run: 7 shared graphs for the 7 rounds both rows decode, 2
## single steps for A's tail, and the ids of the standalone greedy runs. Back-to-back submits admit both rows before
## either can finish, so the count is exact either way the first round lands: with 10 against 8, A always outlives B.
def paired(dir: String, backend: Tensor.Backend, want_a: List[Int], want_b: List[Int]) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  a_ch: Channel[Serving.Event] = Channel.new(32)
  b_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 5, 2], 10, [], a_ch)
  try Serving.submit(server, [4, 4, 9, 2, 7, 1], 8, [], b_ch)
  got_a = try collect(a_ch)
  got_b = try collect(b_ch)
  runs = (try Serving.stats(server)).pair_runs
  try Serving.stop(server)
  # Two prefills, the paired shapes and A's single tails: no round compiles its graph twice.
  try check_compiles(backend, Generate.compiles(g), 9)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_a, want_a)
  try Test.eq(got_b, want_b)
  Test.eq(runs, 7)
end

## B joins while A is in decode (A's first two tokens are consumed first): B prefills and pairs in the next round,
## 4 shared graphs while both need ids, A singles the rest, and neither answer changes.
def joining(dir: String, backend: Tensor.Backend, want_a: List[Int], want_b: List[Int]) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  a_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 5, 2], 12, [], a_ch)
  first_a = try expect_token(a_ch)
  second_a = try expect_token(a_ch)
  b_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [4, 4, 9, 2, 7, 1], 5, [], b_ch)
  got_a = [first_a, second_a] + try collect(a_ch)
  got_b = try collect(b_ch)
  runs = (try Serving.stats(server)).pair_runs
  try Serving.stop(server)
  try check_compiles(backend, Generate.compiles(g), 9)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_a, want_a)
  try Test.eq(got_b, want_b)
  Test.eq(runs, 4)
end

## A is cancelled after the first paired round: A hears it, B runs on alone with the same ids, C takes the free row
## and pairs with B for its last two rounds, and the count holds one shared graph per paired round.
def cancelled_pair(dir: String, backend: Tensor.Backend, want_b: List[Int], want_c: List[Int]) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  a_ch: Channel[Serving.Event] = Channel.new(32)
  b_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 5, 2], 8, [], a_ch)
  try Serving.submit(server, [4, 4, 9, 2, 7, 1], 5, [], b_ch)
  _ = try expect_token(a_ch)
  first_b = try expect_token(b_ch)
  _ = try expect_token(a_ch)
  second_b = try expect_token(b_ch)
  try Serving.cancel(server, a_ch)
  try expect_failed(a_ch, "cancelled")
  # The failure synchronizes with the scheduler: A is out, B decoded alone since, and C pairs from its prefill on.
  c_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 2, 3], 4, [], c_ch)
  got_b = [first_b, second_b] + try collect(b_ch)
  got_c = try collect(c_ch)
  runs = (try Serving.stats(server)).pair_runs
  try Serving.stop(server)
  try check_compiles(backend, Generate.compiles(g), 8)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_b, want_b)
  try Test.eq(got_c, want_c)
  Test.eq(runs, 3)
end

## Two rows that would pair on a CPU backend, on `backend`, which does not pair: every round steps them alone, both
## answers are the standalone ids of that backend, and no paired step ran. Before pairing existed this was the only
## path, and a backend outside `Generate.pairable` keeps it.
def unpaired_rows(dir: String, backend: Tensor.Backend, want_a: List[Int], want_b: List[Int]) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  a_ch: Channel[Serving.Event] = Channel.new(32)
  b_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 5, 2], 4, [], a_ch)
  try Serving.submit(server, [4, 4, 9, 2, 7, 1], 3, [], b_ch)
  got_a = try collect(a_ch)
  got_b = try collect(b_ch)
  runs = (try Serving.stats(server)).pair_runs
  try Serving.stop(server)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_a, want_a)
  try Test.eq(got_b, want_b)
  Test.eq(runs, 0)
end

## One row alone: the single path, the standalone ids, and no paired step (prefills never count).
def solo(dir: String, backend: Tensor.Backend, want: List[Int]) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [2, 7, 1], 5, [], ch)
  got = try collect(ch)
  runs = (try Serving.stats(server)).pair_runs
  try Serving.stop(server)
  try check_compiles(backend, Generate.compiles(g), 3)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got, want)
  Test.eq(runs, 0)
end

## A paired step needs two prefills and usable ids: anything else fails and consumes both caches.
def pair_errors(dir: String, backend: Tensor.Backend) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  try Test.error?(Generate.step_pair(g, Generate.start(), 1, Generate.start(), 2))
  a = (try Generate.step(g, Generate.start(), [1, 5, 2])).cache
  b = (try Generate.step(g, Generate.start(), [1, 5, 2])).cache
  try Test.error?(Generate.step_pair(g, a, 1, b, 99))
  Generate.close(g)
  Qwen3.close_model(m)
  ok()
end

## A's second standalone id is the stop: the first paired round ends A after its first token (the stop id is never a
## token) while B decodes on in the same graph, then B singles to its end. One shared graph ran, and neither answer
## changed.
def stopped_pair(dir: String, backend: Tensor.Backend, want_a: List[Int], want_b: List[Int]) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", backend)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  a_ch: Channel[Serving.Event] = Channel.new(32)
  b_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 5, 2], 12, [want_a[1]], a_ch)
  try Serving.submit(server, [4, 4, 9, 2, 7, 1], 5, [], b_ch)
  got_a = try collect(a_ch)
  got_b = try collect(b_ch)
  runs = (try Serving.stats(server)).pair_runs
  try Serving.stop(server)
  try check_compiles(backend, Generate.compiles(g), 8)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_a, want_a.take(1))
  try Test.eq(got_b, want_b)
  Test.eq(runs, 1)
end

## The graphs a serving run on `backend` compiled: none on the reference backend, and on ggml no more than `bound`
## (the prefill, paired and tail shapes, each once: a round that compiled again would overflow it).
def check_compiles(backend: Tensor.Backend, made: Int, bound: Int) -> Result[Void]
  case backend
    Tensor.Backend.Ggml(_) -> Test.assert(made <= bound, "the serving run compiled #{made} graphs, over #{bound}")
    _ -> Test.eq(made, 0)
  end
end

def interleaved(dir: String) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", S.REF)
  want_a = try standalone(dir, [1, 5, 2], 5)
  want_b = try standalone(dir, [4, 4, 9], 5)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  # Two slots: A cannot complete unseen, so whatever is buffered, `Done` needs more sends than could have landed.
  a_ch: Channel[Serving.Event] = Channel.new(2)
  b_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 5, 2], 5, [], a_ch)
  # B is admitted before anything is consumed, so no consumer stall lets A run alone: the only head start A can
  # get is the scheduler spinning between the two sends, bounded by the drain of one round and far short of the
  # three extra steps that would drop A (a channel of two holding at most one more). What remains is fail-only
  # flake, never a false pass.
  try Serving.submit(server, [4, 4, 9], 5, [], b_ch)
  first_a = try expect_token(a_ch)
  # B decoded while A provably still runs: its first token arrived, and A cannot be done — `Done` would be the sixth
  # send through a channel of two with one consumed, so at most three could have landed. The probe keeps what it takes.
  first_b = try expect_token(b_ch)
  extra: List[Int] = []
  case a_ch.recv(0)
    ok(ev) ->
      case ev
        Serving.Event.Done -> return error("A ended before B decoded")
        Serving.Event.Token(id) -> extra = [id]
        _ -> nil
      end

    error(_) -> nil
  end

  got_a = [first_a] + extra + try collect(a_ch)
  got_b = [first_b] + try collect(b_ch)
  try Serving.stop(server)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_a, want_a)
  Test.eq(got_b, want_b)
end

def cancelled(dir: String) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", S.REF)
  want_b = try standalone(dir, [4, 4, 9], 5)
  want_c = try standalone(dir, [1, 2, 3], 4)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  a_ch: Channel[Serving.Event] = Channel.new(32)
  b_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 5, 2], 16, [], a_ch)
  try Serving.submit(server, [4, 4, 9], 5, [], b_ch)
  _ = try expect_token(a_ch)
  first_b = try expect_token(b_ch)
  try Serving.cancel(server, a_ch)
  try expect_failed(a_ch, "cancelled")
  got_b = [first_b] + try collect(b_ch)
  c_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 2, 3], 4, [], c_ch)
  got_c = try collect(c_ch)
  try Serving.stop(server)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_b, want_b)
  Test.eq(got_c, want_c)
end

def slow_and_gone(dir: String) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", S.REF)
  want_c = try standalone(dir, [2, 7, 1], 5)
  want_d = try standalone(dir, [1, 2, 3], 4)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  # Never read: the second token cannot land, so the sequence is dropped.
  slow_ch: Channel[Serving.Event] = Channel.new(1)
  try Serving.submit(server, [1, 5, 2], 6, [], slow_ch)
  b_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [4, 4, 9], 5, [], b_ch)
  _ = try expect_token(b_ch)
  b_ch.close
  c_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [2, 7, 1], 5, [], c_ch)
  got_c = try collect(c_ch)
  d_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 2, 3], 4, [], d_ch)
  got_d = try collect(d_ch)
  left = try available(slow_ch)
  for ev in left
    case ev
      Serving.Event.Done -> return error("the slow sequence heard Done")
      _ -> nil
    end
  end

  try Serving.stop(server)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_c, want_c)
  Test.eq(got_d, want_d)
end

def bad_steps(dir: String) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", S.REF)
  want = try standalone(dir, [1, 5, 2], 5)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  empty_ch: Channel[Serving.Event] = Channel.new(8)
  try Serving.submit(server, [], 6, [], empty_ch)
  try expect_failed(empty_ch, "needs ids")
  # 33 tokens pass the 32 positions of the tiny model.
  long_ch: Channel[Serving.Event] = Channel.new(8)
  try Serving.submit(server, List.repeat(1, 33), 4, [], long_ch)
  try expect_failed(long_ch, "at most")
  # A request that needs no step is answered at once; a negative one fails; bad bounds never start.
  zero_ch: Channel[Serving.Event] = Channel.new(8)
  try Serving.submit(server, [1, 5, 2], 0, [], zero_ch)
  try Test.eq(try collect(zero_ch), [])
  neg_ch: Channel[Serving.Event] = Channel.new(8)
  try Serving.submit(server, [1, 5, 2], -1, [], neg_ch)
  try expect_failed(neg_ch, "max_new")
  idle = try Generate.open(m)
  try Test.error?(Serving.start(idle, Serving.Limits(max_active: 0, max_pending: 4)))
  try Test.error?(Serving.start(idle, Serving.Limits(max_active: 1, max_pending: -1)))
  Generate.close(idle)
  good_ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 5, 2], 5, [], good_ch)
  got = try collect(good_ch)
  try Serving.stop(server)
  Generate.close(g)
  Qwen3.close_model(m)
  Test.eq(got, want)
end

def stop_id(dir: String) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", S.REF)
  want = try standalone(dir, [1, 5, 2], 4)
  # The standalone run picks `want[2]`: fail readably when the tiny model answers with fewer than 3 ids.
  try Test.assert(want.size >= 3, "the probe run has #{want.size} ids, needs at least 3 for the stop-id pick")
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 2, max_pending: 4))
  # The stop is the third id of the run: the answer is the ids before its first occurrence.
  stop = want[2]
  expected = want.take(2)
  if want[0] == stop
    expected = []
  else
    if want[1] == stop
      expected = want.take(1)
    end
  end

  ch: Channel[Serving.Event] = Channel.new(32)
  try Serving.submit(server, [1, 5, 2], 5, [stop], ch)
  got = try collect(ch)
  try Serving.stop(server)
  Generate.close(g)
  Qwen3.close_model(m)
  Test.eq(got, expected)
end

def saturated(dir: String) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", S.REF)
  want_a = try standalone(dir, [1, 5, 2], 8)
  want_b = try standalone(dir, [4, 4, 9], 5)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 1, max_pending: 1))
  a_ch: Channel[Serving.Event] = Channel.new(64)
  b_ch: Channel[Serving.Event] = Channel.new(32)
  c_ch: Channel[Serving.Event] = Channel.new(8)
  d_ch: Channel[Serving.Event] = Channel.new(8)
  try Serving.submit(server, [1, 5, 2], 8, [], a_ch)
  first_a = try expect_token(a_ch)
  # B waits while A steps; C arrives with every slot taken and is refused on its own channel.
  try Serving.submit(server, [4, 4, 9], 5, [], b_ch)
  try Serving.submit(server, [2, 7, 1], 5, [], c_ch)
  try expect_failed(c_ch, "saturated")
  # D is refused either at the queue or on its own channel, whichever fills first.
  try expect_rejected(server, [1, 2, 3], 6, d_ch)
  got_a = [first_a] + try collect(a_ch)
  got_b = try collect(b_ch)
  try Serving.stop(server)
  Generate.close(g)
  Qwen3.close_model(m)
  try Test.eq(got_a, want_a)
  Test.eq(got_b, want_b)
end

def shutdown(dir: String) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", S.REF)
  g = try Generate.open(m)
  server = try Serving.start(g, Serving.Limits(max_active: 1, max_pending: 1))
  a_ch: Channel[Serving.Event] = Channel.new(64)
  b_ch: Channel[Serving.Event] = Channel.new(8)
  try Serving.submit(server, [1, 5, 2], 20, [], a_ch)
  _ = try expect_token(a_ch)
  # B waits while A steps; the stop fails them both and ends the task.
  try Serving.submit(server, [4, 4, 9], 5, [], b_ch)
  try Serving.stop(server)
  try expect_failed(a_ch, "the server is stopped")
  try expect_failed(b_ch, "the server is stopped")
  # The sequential orderings around the stop, not a true race: a submit buffered before the stop hears the stop,
  # and a submit after the stop is refused at once (the post-stop case is also checked on `server` below).
  g2 = try Generate.open(m)
  server2 = try Serving.start(g2, Serving.Limits(max_active: 1, max_pending: 1))
  x_ch: Channel[Serving.Event] = Channel.new(8)
  try Serving.submit(server2, [1, 5, 2], 5, [], x_ch)
  try Serving.stop(server2)
  try expect_failed(x_ch, "the server is stopped")

  Generate.close(g2)
  Generate.close(g)
  Qwen3.close_model(m)
  late: Channel[Serving.Event] = Channel.new(4)
  case Serving.submit(server, [1, 5, 2], 4, [], late)
    ok(_) -> return error("a stopped server took a request")
    error(e) -> try Test.assert(e.message.include?("stopped"), "the refusal #{e.message} does not say stopped")
  end

  Test.error?(Serving.stop(server))
end

## The next event on `reply` is a token, within `TIMEOUT_MS`.
def expect_token(reply: Channel[Serving.Event]) -> Result[Int]
  case reply.recv(TIMEOUT_MS)
    ok(ev) ->
      case ev
        Serving.Event.Token(id) -> ok(id)
        Serving.Event.Done -> error("the sequence ended before its first token")
        Serving.Event.Failed(message) -> error("the sequence failed: #{message}")
      end

    error(e) -> error("no token in #{TIMEOUT_MS}ms: #{e.message}")
  end
end

## A failure holding `needle` arrives on `reply`, after any tokens still on the way.
def expect_failed(reply: Channel[Serving.Event], needle: String) -> Result[Void]
  going = true
  while going
    case reply.recv(TIMEOUT_MS)
      ok(ev) ->
        case ev
          Serving.Event.Token(_) -> nil
          Serving.Event.Done -> return error("the sequence ended instead of failing")
          Serving.Event.Failed(message) ->
            try Test.assert(message.include?(needle), "the failure #{message} does not hold #{needle}")
            going = false
        end

      error(e) -> return error("no failure in #{TIMEOUT_MS}ms: #{e.message}")
    end
  end

  ok()
end

## Every token until `Done`: the whole answer. A failure or a wait past `TIMEOUT_MS` is an error.
def collect(reply: Channel[Serving.Event]) -> Result[List[Int]]
  seen: List[Int] = []
  going = true
  while going
    case reply.recv(TIMEOUT_MS)
      ok(ev) ->
        case ev
          Serving.Event.Token(id) -> seen = seen.push(id)
          Serving.Event.Done -> going = false
          Serving.Event.Failed(message) -> return error("the sequence failed: #{message}")
        end

      error(e) -> return error("no event in #{TIMEOUT_MS}ms: #{e.message}")
    end
  end

  ok(seen)
end

## What `reply` holds without waiting: nothing is still coming that this call should see.
def available(reply: Channel[Serving.Event]) -> Result[List[Serving.Event]]
  out: List[Serving.Event] = []
  going = true
  while going
    case reply.recv(0)
      ok(ev) -> out = out.push(ev)
      error(_) -> going = false
    end
  end

  ok(out)
end

## A submission the server must refuse: either fail-fast at the queue or a failure on its own channel.
def expect_rejected(s: Serving.Server, prompt: List[Int], max_new: Int, reply: Channel[Serving.Event]) -> Result[Void]
  case Serving.submit(s, prompt, max_new, [], reply)
    ok(_) -> expect_failed(reply, "saturated")
    error(e) -> Test.assert(e.message.include?("saturated"), "the refusal #{e.message} does not say saturated")
  end
end
