## One long-lived task serving greedy Qwen3 decodes over `generate.agc`: the first scheduler of ml#90, and only its
## first step. Many sequences share one model and one `Generate.Generator`; the scheduler owns every live cache and
## steps the active sequences round-robin, one decode step each per round, admitting new ones at round boundaries and
## delivering each sequence's tokens on its own reply channel. No true batched graph, no HTTP, no sampling, no paged
## KV: each step runs its own graph in turn. A prompt runs as one step; there is no chunked prefill.
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

## How long the idle scheduler waits for a command before looking again. A command wakes the wait at once.
IDLE_WAIT_MS = 30000

## `submit` never waits for room (it fails fast when full); `cancel` and `stop` wait this long to be heard. The
## scheduler drains the inbox every round, so the wait only bounds a pathological stall. Refusals are retriable and
## leave the server running.
SUBMIT_WAIT_MS = 0
COMMAND_WAIT_MS = 5000

## What one run is owed: at most `max_active` sequences step at once and at most `max_pending` more wait.
struct Limits
  max_active: Int
  max_pending: Int
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

## The running scheduler: its inbox, its task, and its bounds. `stop` ends the task and closes the inbox.
struct Server
  inbox: Channel[Command]
  task: Task[Result[Void]]
  limits: Limits
end

## Starts the scheduler over `g`, which belongs to the scheduler until a `stop` that returns `ok()`. `limits` bounds
## the run: `max_active` sequences step at once and `max_pending` more wait.
def start(g: Generate.Generator, limits: Limits) -> Result[Server]
  return error("serving: max_active is #{limits.max_active}, needs 1 or more") if limits.max_active < 1

  return error("serving: max_pending is #{limits.max_pending}, needs 0 or more") if limits.max_pending < 0

  inbox: Channel[Command] = Channel.new(limits.max_pending + 1)
  ok(Server(inbox: inbox, task: Task.spawn(&serve, Args(g: g, limits: limits, inbox: inbox)), limits: limits))
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

private

## What the scheduler task runs on.
struct Args
  g: Generate.Generator
  limits: Limits
  inbox: Channel[Command]
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

## The sequences of the run: the ones stepping, the ones waiting, and whether the stop (or a closed inbox) ended
## admission.
struct State
  active: List[Live]
  pending: List[Submit]
  stopped: Bool
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

## One step of every active sequence, in order: the prompt first, then one id. A finished sequence hears its last
## token and `Done` (a stop id is never a token); a slow or gone consumer, and a failing step, end the sequence without
## stalling the rest. A failing step already consumed its cache, so there is nothing to finish.
def round(a: Args, st: State) -> State
  active = st.active
  i = 0
  while i < active.size
    r = active[i]
    feed: List[Int] = []
    if r.fed
      feed = [r.last]
    else
      feed = r.want.prompt
    end

    case Generate.step(a.g, r.cache, feed)
      ok(p) ->
        ids = r.ids.push(p.id)
        if r.want.stop.include?(p.id)
          Generate.finish(p.cache)
          _ = r.want.reply.send(Event.Done, 0)
          active = active.take(i) + active.drop(i + 1)
          next
        end

        if ids.size >= r.want.max_new
          Generate.finish(p.cache)
          kept = r.want.reply.send(Event.Token(p.id), 0)
          if kept.ok?
            _ = r.want.reply.send(Event.Done, 0)
          end

          active = active.take(i) + active.drop(i + 1)
          next
        end

        sent = r.want.reply.send(Event.Token(p.id), 0)
        if sent.ok?
          active = active.set(i, r.with(cache: p.cache, ids: ids, last: p.id, fed: true))
          i += 1
        else
          Generate.finish(p.cache)
          active = active.take(i) + active.drop(i + 1)
        end

      error(e) ->
        _ = r.want.reply.send(Event.Failed(e.message), 0)
        active = active.take(i) + active.drop(i + 1)
    end
  end

  st.with(active: active)
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
