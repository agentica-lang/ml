## One long-lived task serving greedy Qwen3 decodes over `generate.agc`: the first scheduler of ml#90, and only its
## first step. Many sequences share one model and one `Generate.Generator`; the scheduler owns every live cache and
## steps the active sequences round-robin: a prompt (prefill) runs alone through `Generate.step`, and the decode-ready
## rows share one decode graph per batch through `Generate.step_batch`, up to `Limits.max_batch` rows in a batch. What
## a round picks — which rows, how long each is, where each one is — stays the scheduler's; the storage is the dense
## per-row cache it hands over; and `Generate.step_batch` runs one forward over the batch and demultiplexes it by row:
## the model's projections, attention and MLP see the rows as one batch, each row with its own positions, mask and
## slice of the cache, so batching changes no token. New requests join at round boundaries and each sequence hears its
## tokens on its own reply channel. No HTTP, no sampling, no paged KV: a prompt runs as one step; there is no chunked
## prefill. Out of scope here, for ml#92: paged KV and physical slots, token budgets, and chunked prefill.
##
## The contract, which the shape of the code keeps:
##
## - One task touches the generator. `start` lends `g` to the scheduler task, which alone calls `Generate.step` and
##   `Generate.finish` until `stop`. Ownership returns to the caller only when `stop` returns `ok()`: any error leaves
##   the server running and the generator with the scheduler (retry the stop).
## - One cache per live sequence, owned by the scheduler and finished exactly once: when the sequence completes, is
##   cancelled, loses its consumer, or the server stops. A failing step has already consumed the cache it was given
##   (`Generate.step`), so the scheduler drops that sequence without finishing anything.
## - Bounded queues. At most `Limits.max_active` sequences step, at most `Limits.max_pending` more wait, and the inbox
##   holds at most `max_pending + 1` commands: `submit` fails fast with `saturated` instead of waiting.
## - One reply channel per request: sharing a channel between live requests is not supported (their streams would
##   merge and a cancel would be ambiguous). Consumers never stall each other. Every event goes out with a zero-timeout
##   send: a full or closed reply channel loses its sequence without a terminator (its cache is finished and the round
##   moves on, even for the final `Token` and `Done`), so size `reply` for at least `max_new + 1` events and treat a
##   bounded `recv` that times out with no `Done` or `Failed` as a lost sequence.
## - Admission is FIFO at round boundaries. `cancel` (or closing the reply channel) finishes the cache and reports
##   `cancelled`. The stop fails everything with `the server is stopped`, finishes every cache, and ends the task:
##   before its final drain the scheduler closes the inbox, so a submit racing the stop fails closed instead of hanging
##   unanswered. `stop` awaits the task and closes the inbox; its control send waits boundedly and a full queue refuses
##   with a retriable `busy` error that leaves the server running.

import Generate from "generate"
import Stepping from "stepping"

## How long the idle scheduler waits for a command before looking again. A command wakes the wait at once.
IDLE_WAIT_MS = 30000

## `submit` never waits for room (it fails fast when full); `cancel` and `stop` wait this long to be heard. The
## scheduler drains the inbox every round, so the wait only bounds a pathological stall. Refusals are retriable and
## leave the server running.
SUBMIT_WAIT_MS = 0
COMMAND_WAIT_MS = 5000

## What one run is owed: at most `max_active` sequences step at once, at most `max_pending` more wait, and at most
## `max_batch` of the decode-ready ones share one graph (`Generate.MAX_BATCH` at most).
struct Limits
  max_active: Int
  max_pending: Int
  max_batch: Int = 2
end

## What a request hears: every new id in turn, then the end, or the reason it ended. `Done` carries no ids: the tokens
## before it are the whole answer (a stop id ends the sequence and is never a token). Nothing more arrives for a
## sequence lost to a full or closed channel: size `reply` for `max_new + 1` events and bound every wait.
enum Event
  Token(Int)
  Done
  Failed(String)
end

## A request: the prompt, how many new ids at most, the stop ids, and where its events go. The reply channel is this
## request's alone (sharing it between live requests is not supported), with room for its answer.
struct Submit
  prompt: List[Int]
  max_new: Int
  stop: List[Int]
  reply: Address[Event]
end

## What the scheduler hears: a request, the cancellation of the sequence on a reply channel, or the end of the run.
enum Command
  Request(Submit)
  Cancel(Address[Event])
  Stop
end

## The running scheduler: its inbox, its task, and its bounds. `stop` ends the task and closes the inbox. `pairs`
## holds the count of completed shared decode steps, which `stats` reads: one per batch of two or more rows (one
## backend execution where the backend compiles the step, an eager forward on the reference backend).
struct Server
  inbox: Channel[Command]
  task: Task[Result[Void]]
  limits: Limits
  pairs: Channel[Int]
end

## How many shared decode steps the run has completed: one per batch of two or more rows, prefills excluded. On a
## backend `Generate.pairable` leaves out it stays 0, because every row steps alone there.
struct Stats
  pair_runs: Int
end

## Starts the scheduler over `g`, which belongs to the scheduler until a `stop` that returns `ok()`. `limits` bounds
## the run: `max_active` sequences step at once and `max_pending` more wait.
def start(g: Generate.Generator, limits: Limits) -> Result[Server]
  return error("serving: max_active is #{limits.max_active}, needs 1 or more") if limits.max_active < 1

  return error("serving: max_pending is #{limits.max_pending}, needs 0 or more") if limits.max_pending < 0

  return error("serving: max_batch is #{limits.max_batch}, needs 1 or more") if limits.max_batch < 1

  if limits.max_batch > Generate.MAX_BATCH
    return error("serving: max_batch is #{limits.max_batch}, at most #{Generate.MAX_BATCH}")
  end

  if limits.max_batch > limits.max_active
    return error("serving: max_batch is #{limits.max_batch}, over max_active #{limits.max_active}")
  end

  inbox: Channel[Command] = Channel.new(limits.max_pending + 1)
  pairs: Channel[Int] = Channel.new(1)
  try pairs.send(0, COMMAND_WAIT_MS)
  ok(
    Server(
      inbox: inbox,
      task: Task.spawn(&serve, Args(g: g, limits: limits, inbox: inbox, pairs: pairs)),
      limits: limits,
      pairs: pairs,
    ),
  )
end

## Asks for `max_new` greedy ids after `prompt`, stopping before a `stop` id, with the tokens and the end on `reply`
## (one request's own channel). `ok` means the scheduler took the request, not that it runs yet: the answer, or a
## `Failed` with `saturated`, arrives on `reply`. An error means the queue is full (`saturated`) or the server stopped.
def submit(s: Server, prompt: List[Int], max_new: Int, stop: List[Int], reply: Channel[Event]) -> Result[Void]
  submit = Submit(prompt: prompt, max_new: max_new, stop: stop, reply: reply.address)
  sent = s.inbox.send(Command.Request(submit), SUBMIT_WAIT_MS)
  return ok() if sent.ok?
  return error("serving: the server is stopped") if s.inbox.closed?
  error("serving: saturated, no room for another request")
end

## Cancels the live or waiting sequence on `reply` (one request per channel: sharing a channel between live requests
## is not supported; a shared channel cancels the first live match and drops every waiting match): its cache is finished and `reply` hears `Failed` with `cancelled`. An unknown or finished request is `ok`
## with nothing sent. An error means the server stopped, or the queue is full and the cancel was not heard (retriable).
def cancel(s: Server, reply: Channel[Event]) -> Result[Void]
  sent = s.inbox.send(Command.Cancel(reply.address), COMMAND_WAIT_MS)
  return ok() if sent.ok?
  return error("serving: the server is stopped") if s.inbox.closed?
  error("serving: the server is not taking cancels")
end

## Ends the run: every live and waiting sequence is sent `Failed` with `the server is stopped` (best-effort, like all
## events: a full or closed reply channel loses it), every cache is finished,
## the task ends, and the inbox closes. The control send waits `COMMAND_WAIT_MS` for room: a full queue refuses with a
## retriable `busy` error and the server keeps running with the generator. Ownership returns to the caller only when
## this returns `ok()`: then the task has joined and the caller may close the generator and the model.
def stop(s: Server) -> Result[Void]
  case s.inbox.send(Command.Stop, COMMAND_WAIT_MS)
    ok(_) -> nil
    error(_) ->
      if s.inbox.closed?
        case Task.await(s.task, 1)
          Await.Done(r) ->
            try r
            return error("serving: the server is stopped")
          _ -> return error("serving: the inbox is closed")
        end
      end

      return error("serving: the server is busy, the queue is full; retry the stop")
  end

  done = Task.await(s.task)
  s.inbox.close
  try done
  ok()
end

## How many shared decode steps the server has completed: one per batch of two or more rows, prefills excluded. The
## count is exact when the run is idle or stopped: a step in flight has not been counted yet.
def stats(s: Server) -> Result[Stats]
  case s.pairs.recv(COMMAND_WAIT_MS)
    ok(v) ->
      put = s.pairs.send(v, COMMAND_WAIT_MS)
      return error("serving: the stats count is busy, retry the stats") if not put.ok?

      ok(Stats(pair_runs: v))
    error(_) -> error("serving: the stats count is busy, retry the stats")
  end
end

private

## What the scheduler task runs on.
struct Args
  g: Generate.Generator
  limits: Limits
  inbox: Channel[Command]
  pairs: Channel[Int]
end

## A live sequence: what it asked, the cache after its last step, the ids so far, the last id fed, and whether the
## prompt has run.
struct Live
  want: Submit
  cache: Generate.Cache
  ids: List[Int]
  last: Int
  fed: Bool
end

## The sequences of the run: the ones stepping, the ones waiting, whether the stop (or a closed inbox) ended admission,
## and where the next round starts its batches so a run wider than `max_batch` gives every row a turn.
struct State
  active: List[Live]
  pending: List[Submit]
  stopped: Bool
  cursor: Int = 0
end

## What one batch of a round leaves: the sequences after settling every row, and how many of the batch ended.
struct Batched
  active: List[Live]
  removed: Int
end

## What settling one row of a step leaves: the sequences and where the scan goes on (`next` is the row itself when the
## row was dropped, the one after it otherwise).
struct Settled
  active: List[Live]
  next: Int
  removed: Bool
end

def serve(a: Args) -> Result[Void]
  st = State(active: [], pending: [], stopped: false)
  while true
    st = drain(a, st)
    st = admit(a, st)
    if st.stopped
      # No new admission: closing the inbox turns a submit racing the stop into a `closed` error instead of an
      # unanswered `ok`, and the final drain fails what is already buffered (a close keeps buffered values).
      a.inbox.close
      st = drain(a, st)
      return ok()
    end

    if st.active.empty? and st.pending.empty?
      st = idle(a, st)
      next
    end

    st = round(a, st)
  end

  ok()
end

## Takes every waiting command without blocking: requests run now, wait, or fail saturated; cancels end their sequence;
## the stop fails everything and ends admission. A closed inbox ends admission the same way.
def drain(a: Args, st: State) -> State
  active = st.active
  pending = st.pending
  stopped = st.stopped
  going = true
  while going
    case a.inbox.recv(0)
      ok(cmd) ->
        next_state = ingest(a, cmd, active, pending, stopped)
        active = next_state.active
        pending = next_state.pending
        stopped = next_state.stopped
      error(_) ->
        if a.inbox.closed?
          cleared = halt(active, pending, true)
          active = cleared.active
          pending = cleared.pending
          stopped = cleared.stopped
        end

        going = false
    end
  end

  st.with(active: active, pending: pending, stopped: stopped)
end

## One command taken from the inbox.
def ingest(a: Args, cmd: Command, active: List[Live], pending: List[Submit], stopped: Bool) -> State
  case cmd
    Command.Request(s) -> requested(a, s, active, pending, stopped)
    Command.Cancel(addr) -> cancelled(addr, active, pending, stopped)
    Command.Stop -> halt(active, pending, true)
  end
end

## A request: answered at once when it needs no step, else live, waiting, or refused saturated.
def requested(a: Args, s: Submit, active: List[Live], pending: List[Submit], stopped: Bool) -> State
  if stopped
    _ = s.reply.send(Event.Failed("serving: the server is stopped"), 0)
    return State(active: active, pending: pending, stopped: stopped)
  end

  if s.max_new < 0
    _ = s.reply.send(Event.Failed("serving: max_new is #{s.max_new}, needs 0 or more"), 0)
    return State(active: active, pending: pending, stopped: stopped)
  end

  if s.max_new == 0
    _ = s.reply.send(Event.Done, 0)
    return State(active: active, pending: pending, stopped: stopped)
  end

  return State(active: active.push(fresh(s)), pending: pending, stopped: stopped) if active.size < a.limits.max_active

  return State(active: active, pending: pending.push(s), stopped: stopped) if pending.size < a.limits.max_pending

  _ = s.reply.send(
    Event.Failed("serving: saturated, #{a.limits.max_active} active and #{a.limits.max_pending} pending"),
    0,
  )

  State(active: active, pending: pending, stopped: stopped)
end

## The live sequence of a request: an empty cache, nothing fed yet.
def fresh(s: Submit) -> Live = Live(want: s, cache: Generate.start(), ids: [], last: 0, fed: false)

## The end of the sequence on `addr` (the first live match, and every waiting match; channels are one request's
## own, so sharing one is not supported): a live one is finished
## and hears it cancelled, a waiting one is dropped the same way, anything else is nothing to do.
def cancelled(addr: Address[Event], active: List[Live], pending: List[Submit], stopped: Bool) -> State
  for i in 0...active.size
    if active[i].want.reply == addr
      Generate.finish(active[i].cache)
      _ = addr.send(Event.Failed("serving: cancelled"), 0)
      return State(active: active.take(i) + active.drop(i + 1), pending: pending, stopped: stopped)
    end
  end

  kept: List[Submit] = []
  dropped = false
  for s in pending
    if s.reply == addr
      dropped = true
    else
      kept = kept.push(s)
    end
  end

  if dropped
    _ = addr.send(Event.Failed("serving: cancelled"), 0)
  end

  State(active: active, pending: kept, stopped: stopped)
end

## Fails every live sequence (finishing its cache) and every waiting one with the stop reason.
def halt(active: List[Live], pending: List[Submit], stopped: Bool) -> State
  for live in active
    Generate.finish(live.cache)
    _ = live.want.reply.send(Event.Failed("serving: the server is stopped"), 0)
  end

  for s in pending
    _ = s.reply.send(Event.Failed("serving: the server is stopped"), 0)
  end

  State(active: [], pending: [], stopped: stopped)
end

## Moves waiting requests into free active slots, in order. New rounds step them.
def admit(a: Args, st: State) -> State
  active = st.active
  pending = st.pending
  while active.size < a.limits.max_active
    break if pending.empty?

    active = active.push(fresh(pending[0]))
    pending = pending.drop(1)
  end

  st.with(active: active, pending: pending)
end

## One step of every active sequence, in order: each prompt first, alone through `Generate.step`, then one decode step
## per batch: where the backend batches (`Generate.pairs`), every decode-ready row shares a graph through
## `Generate.step_batch`, up to `Limits.max_batch` rows in each batch (the one step of a batch of two or more is counted
## in `Args.pairs`). The batches start at `State.cursor` and wrap, so a run whose active rows are wider than `max_batch`
## moves every row through every position and no row is skipped. A backend that does not batch — XLA, libtorch — takes
## the singleton path for every row, as it did before batching existed, so nothing is lost where the step is not there.
## A ggml device batches whether it is the CPU engine, a card or the software Vulkan device: that is a property of the
## device the model was opened on, not a mode of the scheduler. A finished sequence hears its last token and `Done` (a
## stop id is never a token); a slow or gone consumer, and a failing step, end the sequence without stalling the rest.
## A failing step already consumed its cache, so there is nothing to finish; a failing batch step consumed every cache
## of the batch, so every sequence of it ends.
def round(a: Args, st: State) -> State
  active = st.active
  i = 0
  while i < active.size
    if active[i].fed
      i += 1
      next
    end

    r = active[i]
    case Generate.step(a.g, r.cache, r.want.prompt)
      ok(p) ->
        settled = settle_row(r, p, i, active)
        active = settled.active
        i = settled.next
      error(e) -> active = fail_row(r, e.message, i, active)
    end
  end

  ready: List[Int] = []
  for k in 0...active.size
    if active[k].fed
      ready = ready.push(k)
    end
  end

  # Only where the backend batches: every row whose prompt ran is in a batch (every active row after the loop above),
  # and a batch of one is still `step_batch`, never the silent singleton path of a backend that cannot batch.
  if Generate.pairs(a.g) and !active.empty?
    start = st.cursor % active.size
    active = active.drop(start) + active.take(start)
    base = 0
    while base < active.size
      take = [a.limits.max_batch, active.size - base].min.unwrap!
      indices: List[Int] = []
      for k in 0...take
        indices = indices.push(base + k)
      end

      batched = batch_rows(a, active, indices)
      active = batched.active
      base = base + take - batched.removed
    end

    return st.with(active: active, cursor: st.cursor + a.limits.max_batch)
  end

  off = 0
  for t in ready
    r = active[t - off]
    case Generate.step(a.g, r.cache, [r.last])
      ok(p) ->
        settled = settle_row(r, p, t - off, active)
        active = settled.active
        if settled.removed
          off += 1
        end

      error(e) ->
        active = fail_row(r, e.message, t - off, active)
        off += 1
    end
  end

  st.with(active: active)
end

## Settles the row `r` at `i` with the pick `p` of its step: a stop id ends the sequence without ever becoming a token,
## `max_new` ids end it after its last token, and a slow or gone consumer ends it without stalling the rest.
def settle_row(r: Live, p: Stepping.Picked[Generate.Cache], i: Int, active: List[Live]) -> Settled
  ids = r.ids.push(p.id)
  if r.want.stop.include?(p.id)
    Generate.finish(p.cache)
    _ = r.want.reply.send(Event.Done, 0)
    return Settled(active: active.take(i) + active.drop(i + 1), next: i, removed: true)
  end

  if ids.size >= r.want.max_new
    Generate.finish(p.cache)
    kept = r.want.reply.send(Event.Token(p.id), 0)
    if kept.ok?
      _ = r.want.reply.send(Event.Done, 0)
    end

    return Settled(active: active.take(i) + active.drop(i + 1), next: i, removed: true)
  end

  sent = r.want.reply.send(Event.Token(p.id), 0)
  if sent.ok?
    return Settled(
      active: active.set(i, r.with(cache: p.cache, ids: ids, last: p.id, fed: true)),
      next: i + 1,
      removed: false,
    )
  end

  Generate.finish(p.cache)
  Settled(active: active.take(i) + active.drop(i + 1), next: i, removed: true)
end

## Fails the row `r` at `i` with `message`: its cache is already consumed, so there is nothing to finish.
def fail_row(r: Live, message: String, i: Int, active: List[Live]) -> List[Live]
  _ = r.want.reply.send(Event.Failed(message), 0)
  active.take(i) + active.drop(i + 1)
end

## One batch decode step of the rows at `indices`, on a backend `Generate.pairs` accepts: each row settles its own pick
## on its own channel, so a slow or gone consumer ends only its row and its peers run on. A failing step consumed every
## cache of the batch, so every row of it fails.
def batch_rows(a: Args, active: List[Live], indices: List[Int]) -> Batched
  reqs: List[Generate.BatchRow] = []
  for i in indices
    r = active[i]
    reqs = reqs.push(Generate.BatchRow(cache: r.cache, id: r.last))
  end

  case Generate.step_batch(a.g, reqs)
    ok(picks) ->
      # A shared batch of two or more rows, one backend execution where the backend compiles it.
      if indices.size >= 2
        note_pair(a)
      end

      off = 0
      for k in 0...indices.size
        idx = indices[k] - off
        settled = settle_row(active[idx], picks[k], idx, active)
        active = settled.active
        if settled.removed
          off += 1
        end
      end

      Batched(active: active, removed: off)
    error(e) ->
      dropped = active
      for k in 0...indices.size
        idx = indices[k] - k
        dropped = fail_row(dropped[idx], e.message, idx, dropped)
      end

      Batched(active: dropped, removed: indices.size)
  end
end

## One shared decode step completed: counts it (`Stats.pair_runs`), one per batch of two or more rows.
def note_pair(a: Args) -> Void
  case a.pairs.recv(COMMAND_WAIT_MS)
    ok(v) -> _ = a.pairs.send(v + 1, COMMAND_WAIT_MS)
    error(_) -> nil
  end
end

## Nothing to step: waits for the next command, which wakes the wait at once.
def idle(a: Args, st: State) -> State
  case a.inbox.recv(IDLE_WAIT_MS)
    ok(cmd) -> ingest(a, cmd, st.active, st.pending, st.stopped)
    error(_) ->
      return halt(st.active, st.pending, true) if a.inbox.closed?

      st
  end
end
