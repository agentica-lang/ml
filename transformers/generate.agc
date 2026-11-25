## Greedy generation with Qwen3 and a key-value cache (`transformers/DESIGN.md`),
## the design of `asr/`'s decoder (`asr/asr.agc`); the cache helpers are shared
## in `nn/kv.agc`:
##
## - The cache of a layer is `[1, kv_heads, rows, head_dim]`, the keys already
##   normalized and rotated, padded with zero rows to a capacity of 8, 16, 32
##   and so on (`capacity`), so a graph changes shape only when the capacity
##   grows. On `Ggml` it is `F16`, half the memory and of the bytes a decode
##   step reads (the logits drift by about 5e-3 on values of 10 to 30); on the
##   other backends it is `F32`. A step writes the new rows with a product by a 0/1 matrix
##   (`written`), and a mask hides the later positions and the padding.
## - On `Ggml` each (tokens, rows, capacity) is one compiled graph, kept by the
##   `Generator` of the model: the prompt, padded to a power-of-2 capacity
##   (8, 16, ...), takes only its last hidden row to the head, and every later
##   step is one token. The choice of the next id runs in the graph, so one
##   number leaves the device, not the logits. A one-token step reads the rows
##   of the cache in use, rounded up (`Padded.reach`), not the padding after
##   them; each reach is a graph, and a capacity keeps the one of its largest.
## - Elsewhere the same graph runs eagerly in an arena. On the reference
##   backend attention needs `q * k` of at most 1,048,576, which the padded
##   prompt counts (a prompt of 600 tokens is padded to 1024). libtorch is not
##   tested.
##
## Batch of one. A prompt is fed in one step (`step` at length 0), then one token
## at a time. A `Generator` belongs to one task at a time, and `close` ends it
## before `Qwen3.close_model` frees the weights its graphs read.

import Padded from "pkg/nn/kv"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Lift from "lift"
import Qwen3 from "qwen3"
import Sample from "sample"

## A model and the compiled steps it has run (on `Ggml`), which every sequence
## of the model shares: the channel holds one `Plans`, which a call takes to read
## or change it and puts back at once, and a graph goes in as soon as it is made.
struct Generator
  model: Qwen3.Qwen3
  plans: Channel[Plans]
  ## Every step reads the whole capacity of the cache, not only the rows in use: `false` from `open`. The tests set it
  ## to have the logits of the other way of reading the same cache.
  whole: Bool
end

## The compiled steps by shape, and how many compiles made them.
struct Plans
  graphs: Map[String, Trace.Compiled]
  compiles: Int
end

## What `step` keeps between calls: per layer the keys and the values of the
## `length` tokens so far, padded to the capacity. On `Ggml` a one-token step of
## the same capacity writes its row into these very tensors (they are the
## tensors of the cache it returns), so a cache that was stepped is dead.
struct Cache
  keys: List[Tensor.Tensor]
  values: List[Tensor.Tensor]
  length: Int
end

## `step`: the id with the largest logit after the last token fed (the first on a
## tie), and the cache.
struct Picked
  id: Int
  cache: Cache
end

## `generate`'s choice: the id, the cache after it and the state of the draws.
struct Drawn
  id: Int
  cache: Cache
  state: Int
end

## `step_top`: the ids of the largest logits and the logits, largest first, and the cache.
struct Candidates
  ids: List[Int]
  logits: List[Float]
  cache: Cache
end

struct Candidate
  id: Int
  logit: Float
end

## `step_logits`: the logits after the last token fed, and the cache.
struct Scored
  logits: List[Float]
  cache: Cache
end

## The generator of `m`.
def open(m: Qwen3.Qwen3) -> Result[Generator]
  cell: Channel[Plans] = Channel.new(1)
  try cell.send(Plans(graphs: {}, compiles: 0), 1000)
  ok(Generator(model: m, plans: cell, whole: false))
end

## How many steps `g` has compiled: a shape it has seen compiles no more.
def compiles(g: Generator) -> Int
  case take_plans(g)
    some(p) ->
      _ = g.plans.send(p, 0)
      p.compiles
    none -> 0
  end
end

## Ends the pins of the compiled steps of `g`, which a later call compiles again.
## Call it before `Qwen3.close_model(g.model)`.
def close(g: Generator) -> Void
  case take_plans(g)
    some(p) ->
      for key in p.graphs.keys
        Trace.close(p.graphs[key])
      end

      _ = g.plans.send(Plans(graphs: {}, compiles: p.compiles), 0)
    none -> nil
  end
end

## The empty cache.
def start() -> Cache = Cache(keys: [], values: [], length: 0)

## Feeds `ids` after the tokens of `cache`: a prompt of any length when the
## cache is empty, one id after that. The cache that `step` takes is dead
## afterwards, even when the step fails: on ggml a step of the same capacity
## writes the new row into its tensors, which are the ones of the cache it gives,
## elsewhere they are closed. Use the one it gives; never step a cache twice or
## read one after it was stepped.
def step(g: Generator, cache: Cache, ids: List[Int]) -> Result[Picked]
  s = try advance(g, cache, ids, Head.Pick)
  found = chosen(s.head, g.model.config.vocab_size)
  if not found.ok?
    finish(s.cache)
  end

  ok(Picked(id: try found, cache: s.cache))
end

## `step` with the logits of the last token fed, `[vocab_size]`, in place of the choice.
def step_logits(g: Generator, cache: Cache, ids: List[Int]) -> Result[Scored]
  s = try advance(g, cache, ids, Head.Logits)
  logits: List[Float] = []
  for v in s.head
    case v
      Tensor.Values.Float(f) -> logits = logits.push(f)
      _ ->
        finish(s.cache)
        return error("the logits are floats")
    end
  end

  ok(Scored(logits: logits, cache: s.cache))
end

## `step_logits` for the `k` largest logits only: their ids and values, largest first, and the cache. The cut is in the
## graph (`Tensor.top_ids`), so `k` values and ids cross to the host and not `vocab_size` logits. Ids with equal logits
## at the edge of the `k` are not specified. A `-inf` logit among the `k` makes the step an error on ggml (its outputs
## are checked to be finite).
def step_top(g: Generator, cache: Cache, ids: List[Int], k: Int) -> Result[Candidates]
  vocab = g.model.config.vocab_size
  if k < 1 or k > vocab
    finish(cache)
    return error("step_top needs k from 1 to #{vocab}, got #{k}")
  end

  s = try advance(g, cache, ids, Head.Top(k))
  found: List[Candidate] = []
  for i in 0...k
    case s.head[i]
      Tensor.Values.Float(logit) ->
        case s.head[k + i]
          Tensor.Values.Float(id) -> found = found.push(Candidate(id: id.to_i.unwrap_or(-1), logit: logit))
          _ -> nil
        end

      _ -> nil
    end
  end

  if found.size != k or found.any? { |c| c.id < 0 or c.id >= vocab }
    finish(s.cache)
    return error("step_top: the graph gave no ids, or ids outside the vocabulary")
  end

  ordered = found.sort_by { |c| 0.0 - c.logit }
  ok(Candidates(ids: ordered.map { |c| c.id }, logits: ordered.map { |c| c.logit }, cache: s.cache))
end

## Ends a sequence: closes the tensors of `cache`.
def finish(cache: Cache) -> Void
  Lift.close(cache.keys + cache.values)
end

## Decoding from the prompt `ids`: at most `max_new` ids, stopping before a stop id (`stop`, such as `<|im_end|>`).
## Greedy (the choice runs in the graph) unless `sampling` says how to draw: then each step reads the largest logits
## (cut in the graph) and `Sample.pick_top` draws among them, the same ids for the same seed (up to float noise). An
## unusable `sampling` (`Sample.check`) is an error before the first step, also at temperature 0. The prompt and the new
## ids must fit `max_position_embeddings`.
def generate(
  g: Generator,
  ids: List[Int],
  max_new: Int,
  stop: List[Int] = [],
  sampling: Optional[Sample.Sampling] = none,
) -> Result[List[Int]]
  return error("generate needs max_new of 0 or more, got #{max_new}") if max_new < 0
  case sampling
    some(s) ->
      checked_sampling = Sample.check(s)
      return error(checked_sampling.message) if not checked_sampling.ok?

    none -> nil
  end

  limit = g.model.config.max_position_embeddings
  if max_new > 0 and ids.size + max_new - 1 > limit
    return error("generate needs at most #{limit} positions, a prompt of #{ids.size} and #{max_new} new ids need more")
  end

  cache = start()
  out: List[Int] = []
  feed = ids
  state = 0
  case sampling
    some(s) -> state = Sample.start(s)
    none -> nil
  end

  while out.size < max_new
    s = chosen_next(g, cache, feed, sampling, state)
    return error(s.message) if not s.ok?

    picked = s.unwrap!
    cache = picked.cache
    state = picked.state
    break if stop.include?(picked.id)
    out = out.push(picked.id)
    feed = [picked.id]
  end

  finish(cache)
  ok(out)
end

## The id after `feed` and the cache, by the greedy step or, with `sampling`, by a draw among the largest logits (`state`
## is the generator of the draws, which the answer carries on).
def chosen_next(
  g: Generator,
  cache: Cache,
  feed: List[Int],
  sampling: Optional[Sample.Sampling],
  state: Int,
) -> Result[Drawn]
  case sampling
    some(s) ->
      if s.temperature > 0.0
        top = try step_top(g, cache, feed, Sample.window(s, g.model.config.vocab_size))
        drawn = Sample.pick_top(top.ids, top.logits, s, state)
        if not drawn.ok?
          finish(top.cache)
          return error(drawn.message)
        end

        d = drawn.unwrap!
        return ok(Drawn(id: d.id, cache: top.cache, state: d.state))
      end

    none -> nil
  end

  p = try step(g, cache, feed)
  ok(Drawn(id: p.id, cache: p.cache, state: state))
end

## The first id of the maximum of `scores` (`[n]`) as `[1]` floats (exact in F32 below 2^24): `v` minus the
## largest of `v - id` over the ids that hold the maximum. A graph cannot answer an `I64` (what `argmax`
## makes) on ggml.
def picked(scores: Tensor.Tensor) -> Result[Tensor.Tensor]
  v = Tensor.shape(scores)[0]
  dtype = Tensor.dtype(scores)
  backend = Tensor.backend(scores)
  top = try Tensor.broadcast_to(try Tensor.max(scores, some([0]), true), [v])
  hit = try Tensor.to_dtype(try Tensor.eq(scores, top), dtype)
  ids = try Tensor.to_dtype(try Tensor.arange(0, v, 1, Tensor.Dtype.I32, backend), dtype)
  count = try Tensor.full([v], Tensor.Values.Float(v.to_f), dtype, backend)
  best = try Tensor.max(try Tensor.mul(hit, try Tensor.sub(count, ids)), some([0]), true)
  Tensor.sub(try Tensor.full([1], Tensor.Values.Float(v.to_f), dtype, backend), best)
end

private

## How long a call waits for the plans, which are held for a few VM steps.
PLANS_WAIT_MS = 50

## What the graph of a step answers about the next token: the logits `[vocab_size]`, the greedy pick as one float, or
## the `k` largest logits and their ids (`2k` floats: the logits, then the ids).
enum Head
  Logits
  Pick
  Top(Int)
end

## What one step gives: the last row of the head (the logits, the pick as one float, or the candidates) and the cache
## after it.
struct Advanced
  head: List[Tensor.Values]
  cache: Cache
end

struct Kv
  key: Tensor.Tensor
  value: Tensor.Tensor
end

struct Layer
  hidden: Tensor.Tensor
  key: Tensor.Tensor
  value: Tensor.Tensor
end

## The `k` largest of `logits` as `[2k]` floats: their values, then their ids (exact in F32 below 2^24).
def top_candidates(logits: Tensor.Tensor, k: Int) -> Result[Tensor.Tensor]
  if Tensor.dtype(logits) != Tensor.Dtype.F32
    return error("the logits are #{Tensor.dtype(logits)}: step_top needs F32, or the ids would round")
  end

  ids = try Tensor.top_ids(logits, k)
  values = try Tensor.gather(logits, ids)
  Tensor.concat([values, try Tensor.to_dtype(ids, Tensor.dtype(logits))], 0)
end

def advance(g: Generator, cache: Cache, ids: List[Int], mode: Head) -> Result[Advanced]
  m = g.model
  bad = checked(m, cache, ids)
  if not bad.ok?
    finish(cache)
    return error(bad.message)
  end

  n = ids.size
  limit = m.config.max_position_embeddings
  room = Padded.capacity(cache.length + n, limit)
  # A prompt is padded to the capacity, a later step is one token.
  width = 1
  if cache.length == 0
    width = room
  end

  backend = Tensor.backend(m.tokens.table)
  made = inputs(ids, cache.length, width, room, backend)
  if not made.ok?
    finish(cache)
    return error(made.message)
  end

  fed = made.unwrap!
  held = cache.keys + cache.values
  in_place = written_in_place?(backend, cache, width, room)
  reach = room
  if in_place and not g.whole
    reach = Padded.reach(cache.length + n, room)
  end

  outs = case backend
    Tensor.Backend.Ggml(_) -> compiled_step(g, cache, fed, room, reach, mode, in_place)
    _ -> eager_step(m, cache, fed, room, mode)
  end

  Lift.close(fed)
  # A step in place keeps the tensors of `cache` as the ones of the new cache; any other step is done with them (a
  # tensor closed twice is fine, as when the step failed after it closed its outputs).
  if not in_place or not outs.ok?
    Lift.close(held)
  end

  s = try outs
  ok(Advanced(head: s.head, cache: s.cache.with(length: cache.length + n)))
end

## Whether a step writes its row into the tensors of `cache`: on `Ggml`, one token after a cache that already has the
## capacity of the step.
def written_in_place?(backend: Tensor.Backend, cache: Cache, width: Int, room: Int) -> Bool
  case backend
    Tensor.Backend.Ggml(_) -> width == 1 and not cache.keys.empty? and Tensor.shape(cache.keys[0])[2] == room
    _ -> false
  end
end

## The graph of one step as a compiled function of its inputs on `Ggml`; `in_place` writes the new row into the cache's own
## tensors, which come back as the keys and values of the new one. Its attention reads the first `reach` rows of the
## cache. Each reach is a graph of its own, and a capacity keeps one: a step takes the graph of a larger reach when
## there is one (its mask hides the rows after the length), and compiling the graph of a reach ends the graphs of the
## smaller ones.
def compiled_step(
  g: Generator,
  cache: Cache,
  fed: List[Tensor.Tensor],
  room: Int,
  reach: Int,
  mode: Head,
  in_place: Bool,
) -> Result[Advanced]
  m = g.model
  args = fed + cache.keys + cache.values
  rows = 0
  if not cache.keys.empty?
    rows = Tensor.shape(cache.keys[0])[2]
  end

  width = Tensor.shape(fed[0])[0]
  behind: List[String] = []
  near = Padded.reach(1, room)
  while near < reach
    behind = behind.push(plan_key(width, rows, room, near, mode))
    near = Padded.reach(near + 1, room)
  end

  ahead: List[String] = []
  far = reach
  while far < room
    far = Padded.reach(far + 1, room)
    ahead = ahead.push(plan_key(width, rows, room, far, mode))
  end

  plan = try planned(
    g,
    plan_key(width, rows, room, reach, mode),
    ahead,
    behind,
    args.map { |t| Trace.spec_of(t) },
    ->(xs: List[Tensor.Tensor]) { stepped(Qwen3.traced(m), xs, room, reach, mode, in_place, true) },
  )

  outs = try Trace.run(plan, args)
  layers = m.config.num_hidden_layers
  head = Tensor.to_list(outs[0])
  Lift.close([outs[0]])
  if not head.ok?
    Lift.close(outs.drop(1))
  end

  ok(Advanced(head: try head, cache: Cache(keys: outs.drop(1).take(layers), values: outs.drop(1 + layers), length: 0)))
end

## The key of the step of `width` tokens over a cache of `rows` rows that leaves `room` and reads `reach` of them.
def plan_key(width: Int, rows: Int, room: Int, reach: Int, mode: Head) -> String
  "#{width} #{rows} #{room} #{reach} #{head_key(mode)}"
end

## The compiled function `key` of `g` for `specs`, or the first of `ahead` that `g` has, which do the same. With none
## of them it is compiled with `f` and kept at once, and the functions `behind` end: `g` has no more use for them.
def planned(
  g: Generator,
  key: String,
  ahead: List[String],
  behind: List[String],
  specs: List[Trace.Spec],
  f: Fn[List[Tensor.Tensor], Result[List[Tensor.Tensor]]],
) -> Result[Trace.Compiled]
  case take_plans(g)
    some(p) ->
      _ = g.plans.send(p, 0)
      for same in [key] + ahead
        return ok(p.graphs[same]) if p.graphs.has_key?(same)
      end

    none -> nil
  end

  compiled = try Trace.compile(specs, f)
  case take_plans(g)
    some(p) ->
      graphs = p.graphs
      for old in behind
        if graphs.has_key?(old)
          Trace.close(graphs[old])
          graphs = graphs.delete(old)
        end
      end

      _ = g.plans.send(Plans(graphs: graphs.put(key, compiled), compiles: p.compiles + 1), 0)
    none -> nil
  end

  ok(compiled)
end

## The plans, which the caller must put back. When they cannot be taken in time, a
## new empty `Plans` is put in the channel in their place and the answer is `none`.
def take_plans(g: Generator) -> Optional[Plans]
  taken = g.plans.recv(PLANS_WAIT_MS)
  return some(taken.unwrap!) if taken.ok?

  _ = g.plans.send(Plans(graphs: {}, compiles: 0), 0)
  none
end

## The same graph, run now and in an arena that keeps the new cache alone.
def eager_step(m: Qwen3.Qwen3, cache: Cache, fed: List[Tensor.Tensor], room: Int, mode: Head) -> Result[Advanced]
  layers = m.config.num_hidden_layers
  ran: Result[Advanced] = Native.arena do
    outs = try stepped(m, fed + cache.keys + cache.values, room, room, mode)
    ok(
      Advanced(
        head: try Tensor.to_list(outs[0]),
        cache: Cache(keys: outs.drop(1).take(layers), values: outs.drop(1 + layers), length: 0),
      ),
    )
  end

  ran
end

## The step: `xs` are the ids `[width]`, their positions, the one-hot `[1, width]` of the row whose
## prediction is wanted, and per layer the keys then the values. The outputs are the head (the logits `[vocab_size]`,
## or the choice `[1]`), then the keys and the values of every layer. The cache it leaves has `room` rows, of which
## the attention reads the first `reach` (every row the new tokens may see is among them).
def stepped(
  m: Qwen3.Qwen3,
  xs: List[Tensor.Tensor],
  room: Int,
  reach: Int,
  mode: Head,
  in_place: Bool = false,
  half: Bool = false,
) -> Result[List[Tensor.Tensor]]
  layers = m.config.num_hidden_layers
  width = Tensor.shape(xs[0])[0]
  d = m.config.hidden_size
  slots = Padded.Slots(positions: xs[1], capacity: room)
  h = try Tensor.reshape(try Nn.embedding(m.tokens, xs[0]), [1, width, d])
  mask = try Padded.padded_mask(slots.with(capacity: reach), Tensor.dtype(h))
  spread: Optional[Tensor.Tensor] = none
  if not in_place
    spread = some(try Padded.spread_of(slots, Tensor.dtype(h)))
  end

  keys: List[Tensor.Tensor] = []
  values: List[Tensor.Tensor] = []
  for i in 0...layers
    old: Optional[Kv] = none
    if xs.size > 3
      old = some(Kv(key: xs[3 + i], value: xs[3 + layers + i]))
    end

    r = try layer(m.stack.blocks[i], h, slots.positions, mask, spread, old, in_place, reach)
    h = r.hidden
    keys = keys.push(try cached(r.key, half))
    values = values.push(try cached(r.value, half))
  end

  last = try Tensor.reshape(try Tensor.matmul(xs[2], try Tensor.reshape(h, [width, d])), [1, 1, d])
  case m.stack.final_norm
    some(n) -> last = try Nn.layer_norm(n, last)
    none -> nil
  end

  logits = try Tensor.reshape(try Nn.dense(m.head, last), [m.config.vocab_size])
  head = logits
  case mode
    Head.Pick -> head = try picked(logits)
    Head.Top(k) -> head = try top_candidates(logits, k)
    Head.Logits -> nil
  end

  ok([head] + keys + values)
end

## The cache tensor `t` as it is kept: `F16` on `Ggml` (`half`), when it is not already.
def cached(t: Tensor.Tensor, half: Bool) -> Result[Tensor.Tensor]
  return ok(t) if not half or Tensor.dtype(t) == Tensor.Dtype.F16
  Tensor.to_dtype(t, Tensor.Dtype.F16)
end

## One block over the new tokens `h` `[1, width, d]`, as `Nn.encoder_block` with
## `norm_first`, the keys and values of the cache in the place of the block's own.
def layer(
  b: Nn.EncoderBlock,
  h: Tensor.Tensor,
  positions: Tensor.Tensor,
  mask: Tensor.Tensor,
  spread: Optional[Tensor.Tensor],
  old: Optional[Kv],
  in_place: Bool,
  reach: Int,
) -> Result[Layer]
  x = try Nn.layer_norm(b.norm1, h)
  heads = try Nn.project(b.attention, x, x, some(positions))
  key = heads.key
  value = heads.value
  if in_place
    key = try Tensor.set_rows(old.unwrap!.key, heads.key, positions)
    value = try Tensor.set_rows(old.unwrap!.value, heads.value, positions)
  else
    key = try Padded.written(try widened(old.map { |kv| kv.key }), heads.key, spread.unwrap!, 2)
    value = try Padded.written(try widened(old.map { |kv| kv.value }), heads.value, spread.unwrap!, 2)
  end

  seen = try Nn.attend(
    b.attention,
    heads.query,
    try Padded.reached(key, reach),
    try Padded.reached(value, reach),
    some(mask),
  )

  mid = try Tensor.add(h, seen)
  out = try Tensor.add(mid, try Nn.mlp(b.mlp, try Nn.layer_norm(b.norm2, mid)))
  ok(Layer(hidden: out, key: key, value: value))
end

## The old cache tensor `t` in the dtype of the new rows.
def widened(t: Optional[Tensor.Tensor]) -> Result[Optional[Tensor.Tensor]]
  case t
    some(x) ->
      return ok(t) if Tensor.dtype(x) == Tensor.Dtype.F32
      ok(some(try Tensor.to_dtype(x, Tensor.Dtype.F32)))
    none -> ok(t)
  end
end

## The ids padded to `width`, the positions (the rows of the new tokens, `room` for the padding), and the
## one-hot of the last real token.
def inputs(ids: List[Int], length: Int, width: Int, room: Int, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  n = ids.size
  tokens: List[Tensor.Values] = []
  at: List[Tensor.Values] = []
  pick: List[Tensor.Values] = []
  for i in 0...width
    if i < n
      tokens = tokens.push(Tensor.Values.Int(ids[i]))
      at = at.push(Tensor.Values.Int(length + i))
    else
      tokens = tokens.push(Tensor.Values.Int(0))
      at = at.push(Tensor.Values.Int(room))
    end

    if i == n - 1
      pick = pick.push(Tensor.Values.Float(1.0))
    else
      pick = pick.push(Tensor.Values.Float(0.0))
    end
  end

  made: List[Tensor.Tensor] = []
  for t in [
    Tensor.from_list(tokens, [width], Tensor.Dtype.I32, backend),
    Tensor.from_list(at, [width], Tensor.Dtype.I32, backend),
    Tensor.from_list(pick, [1, width], Tensor.Dtype.F32, backend),
  ]
    case t
      ok(x) -> made = made.push(x)
      error(e) ->
        Lift.close(made)
        return error(e.message)
    end
  end

  ok(made)
end

## An error unless `ids` may follow `cache`.
def checked(m: Qwen3.Qwen3, cache: Cache, ids: List[Int]) -> Result[Void]
  n = ids.size
  limit = m.config.max_position_embeddings
  return error("step needs ids") if n == 0
  return error("step takes one id after the first call, got #{n}") if n > 1 and cache.length > 0
  return error("step needs at most #{limit} tokens, got #{cache.length + n}") if cache.length + n > limit
  for id in ids
    return error("token id #{id} is outside 0...#{m.config.vocab_size}") if id < 0 or id >= m.config.vocab_size
  end

  ok()
end

def head_key(mode: Head) -> String
  case mode
    Head.Logits -> "logits"
    Head.Pick -> "pick"
    Head.Top(k) -> "top #{k}"
  end
end

## The id the pick found, an error for none (a NaN in the logits: no score equals the maximum).
def chosen(head: List[Tensor.Values], vocab: Int) -> Result[Int]
  case head[0]
    Tensor.Values.Float(f) ->
      id = try f.to_i
      return error("the pick found no id in a vocabulary of #{vocab}: the logits hold a NaN") if id < 0 or id >= vocab
      ok(id)
    _ -> error("the pick is a float")
  end
end
