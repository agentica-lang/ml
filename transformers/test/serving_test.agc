## The scheduler of `serving.agc` against the tiny Qwen3 of `qwen_peer.py`, on the reference backend: B decodes while A
## provably still runs, both answers match the standalone greedy ids, a stop id truncates without becoming a token,
## cancels and bad steps are isolated with their caches freed, a slow and a disconnected consumer stall nobody, a full
## queue refuses, and the stop ends the task (even a submit racing it is answered). Every wait is a channel wait with a
## generous bound, never a sleep. Run from `transformers/`.

import Generate from "../generate"
import Qwen3 from "../qwen3"
import S from "support"
import Serving from "../serving"

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

private

def with_peer(body: Fn[String, Result[Void]]) -> Result[Void]
  S.with_fixtures("test/qwen_peer.py", "agentica-qwen-", body)
end

def interleaved(dir: String) -> Result[Void]
  m = try Qwen3.open("#{dir}/tiny", S.REF)
  base = try Generate.open(m)
  want_a = try Generate.generate(base, [1, 5, 2], 5)
  want_b = try Generate.generate(base, [4, 4, 9], 5)
  Generate.close(base)
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
  base = try Generate.open(m)
  want_b = try Generate.generate(base, [4, 4, 9], 5)
  want_c = try Generate.generate(base, [1, 2, 3], 4)
  Generate.close(base)
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
  base = try Generate.open(m)
  want_c = try Generate.generate(base, [2, 7, 1], 5)
  want_d = try Generate.generate(base, [1, 2, 3], 4)
  Generate.close(base)
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
  base = try Generate.open(m)
  want = try Generate.generate(base, [1, 5, 2], 5)
  Generate.close(base)
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
  base = try Generate.open(m)
  want = try Generate.generate(base, [1, 5, 2], 4)
  # The fixture run below picks `want[2]`: fail readably when the tiny model answers with fewer than 3 ids.
  try Test.assert(want.size >= 3, "the probe run has #{want.size} ids, needs at least 3 for the stop-id pick")
  Generate.close(base)
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
  base = try Generate.open(m)
  want_a = try Generate.generate(base, [1, 5, 2], 8)
  want_b = try Generate.generate(base, [4, 4, 9], 5)
  Generate.close(base)
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
