## Greedy generation with Qwen3 and a key-value cache (`transformers/DESIGN.md`),
## the design of `asr/`'s decoder (`asr/asr.agc`, `ml#117` tracks sharing it):
##
## - The cache of a layer is `[1, kv_heads, rows, head_dim]`, the keys already
##   normalized and rotated, padded with zero rows to a capacity of 8, 16, 32
##   and so on (`capacity`), so a graph changes shape only when the capacity
##   grows. A step writes the new rows with a product by a 0/1 matrix
##   (`written`), and a mask hides the later positions and the padding.
## - On `Ggml` each (tokens, rows, capacity) is one compiled graph, kept by the
##   `Generator` of the model: the prompt, padded to a power-of-2 capacity
##   (8, 16, ...), takes only its last hidden row to the head, and every later
##   step is one token. The choice of the next id runs in the graph, so one
##   number leaves the device, not the logits.
## - Elsewhere the same graph runs eagerly in an arena. On the reference
##   backend attention needs `q * k` of at most 1,048,576, which the padded
##   prompt counts (a prompt of 600 tokens is padded to 1024). libtorch is not
##   tested.
##
## Batch of one. A prompt is fed in one step (`step` at length 0), then one token
## at a time. A `Generator` belongs to one task at a time, and `close` ends it
## before `Qwen3.close_model` frees the weights its graphs read.

import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Lift from "lift"
import Qwen3 from "qwen3"

## A model and the compiled steps it has run (on `Ggml`), which every sequence
## of the model shares: the channel holds one `Plans`, which a call takes to read
## or change it and puts back at once, and a graph goes in as soon as it is made.
struct Generator
  model: Qwen3.Qwen3
  plans: Channel[Plans]
end

## The compiled steps by shape, and how many compiles made them.
struct Plans
  graphs: Map[String, Trace.Compiled]
  compiles: Int
end

## What `step` keeps between calls: per layer the keys and the values of the
## `length` tokens so far, and `spare`, the keys and values of the cache before
## (on `Ggml`), which the next step overwrites with its own instead of making new
## tensors.
struct Cache
  keys: List[Tensor.Tensor]
  values: List[Tensor.Tensor]
  length: Int
  spare: List[Tensor.Tensor]
end

## `step`: the id with the largest logit after the last token fed (the first on a
## tie), and the cache.
struct Picked
  id: Int
  cache: Cache
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
  ok(Generator(model: m, plans: cell))
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
def start() -> Cache = Cache(keys: [], values: [], length: 0, spare: [])

## Feeds `ids` after the tokens of `cache`: a prompt of any length when the
## cache is empty, one id after that. The cache that `step` takes is dead, even
## when the step fails: on ggml its tensors become the spare of the cache it gives
## (a later step overwrites them), elsewhere they are closed. Use the one it gives;
## never step a cache twice or read one after it was stepped.
def step(g: Generator, cache: Cache, ids: List[Int]) -> Result[Picked]
  s = try advance(g, cache, ids, true)
  found = chosen(s.head, g.model.config.vocab_size)
  if not found.ok?
    finish(s.cache)
  end

  ok(Picked(id: try found, cache: s.cache))
end

## `step` with the logits of the last token fed, `[vocab_size]`, in place of the choice.
def step_logits(g: Generator, cache: Cache, ids: List[Int]) -> Result[Scored]
  s = try advance(g, cache, ids, false)
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

## Ends a sequence: closes the tensors of `cache`.
def finish(cache: Cache) -> Void
  Lift.close(cache.keys + cache.values + cache.spare)
end

## Greedy decoding from the prompt `ids`: at most `max_new` ids, stopping before
## a stop id (`stop`, such as `<|im_end|>`). The prompt and the new ids must fit
## `max_position_embeddings`.
def generate(g: Generator, ids: List[Int], max_new: Int, stop: List[Int] = []) -> Result[List[Int]]
  return error("generate needs max_new of 0 or more, got #{max_new}") if max_new < 0
  limit = g.model.config.max_position_embeddings
  if max_new > 0 and ids.size + max_new - 1 > limit
    return error("generate needs at most #{limit} positions, a prompt of #{ids.size} and #{max_new} new ids need more")
  end

  cache = start()
  out: List[Int] = []
  feed = ids
  while out.size < max_new
    s = step(g, cache, feed)
    return error(s.message) if not s.ok?

    picked = s.unwrap!
    cache = picked.cache
    break if stop.include?(picked.id)
    out = out.push(picked.id)
    feed = [picked.id]
  end

  finish(cache)
  ok(out)
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

## What one step gives: the last row of the head (the logits, or the pick as one
## float) and the cache after it.
struct Advanced
  head: List[Tensor.Values]
  cache: Cache
end

## Where a step writes into a padded cache: `positions` (`I32`, `[n]`) are the
## new tokens' rows, `capacity` the rows after the step.
struct Slots
  positions: Tensor.Tensor
  capacity: Int
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

## The rows a padded cache has for `needed` tokens: 8, 16, 32 and so on, at
## most `limit`.
def capacity(needed: Int, limit: Int) -> Int
  rows = 8
  while rows < needed
    rows *= 2
  end

  [rows, limit].min.unwrap!
end

def advance(g: Generator, cache: Cache, ids: List[Int], picking: Bool) -> Result[Advanced]
  m = g.model
  bad = checked(m, cache, ids)
  if not bad.ok?
    finish(cache)
    return error(bad.message)
  end

  n = ids.size
  limit = m.config.max_position_embeddings
  room = capacity(cache.length + n, limit)
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
  outs = case backend
    Tensor.Backend.Ggml(_) -> compiled_step(g, cache, fed, room, picking)
    _ -> eager_step(m, cache, fed, room, picking)
  end

  Lift.close(fed)
  # A compiled step keeps the old keys and values as the spare of the new cache, and closes the spare of the old one.
  if not compiled?(backend) or not outs.ok?
    Lift.close(held)
  end

  s = try outs
  ok(Advanced(head: s.head, cache: s.cache.with(length: cache.length + n)))
end

def compiled?(backend: Tensor.Backend) -> Bool
  case backend
    Tensor.Backend.Ggml(_) -> true
    _ -> false
  end
end

## The graph of one step as a compiled function of its inputs on `Ggml`.
def compiled_step(g: Generator, cache: Cache, fed: List[Tensor.Tensor], room: Int, picking: Bool) -> Result[Advanced]
  m = g.model
  args = fed + cache.keys + cache.values
  rows = 0
  if not cache.keys.empty?
    rows = Tensor.shape(cache.keys[0])[2]
  end

  layers = m.config.num_hidden_layers
  into: List[Optional[Tensor.Tensor]] = []
  reuse = Tensor.shape(fed[0])[0] == 1 and rows == room and fits(cache)
  if reuse
    lead: List[Optional[Tensor.Tensor]] = [none]
    into = lead + cache.spare.map { |t| some(t) }
  else
    Lift.close(cache.spare)
  end

  key = "#{Tensor.shape(fed[0])[0]} #{rows} #{room} #{picking}"
  planned_step = planned(
    g,
    key,
    args.map { |t| Trace.spec_of(t) },
    ->(xs: List[Tensor.Tensor]) { stepped(Qwen3.traced(m), xs, room, picking) },
  )

  ran: Result[List[Tensor.Tensor]] = error("no step ran")
  case planned_step
    # The plan stays cached in the generator, so a failed run leaves it in place (asr closes it: its plans die with
    # the call).
    ok(plan) -> ran = Trace.run(plan, args, into)
    error(e) -> ran = error(e.message)
  end

  if not ran.ok?
    if reuse
      Lift.close(cache.spare)
    end

    return error(ran.message)
  end

  outs = ran.unwrap!
  head = Tensor.to_list(outs[0])
  Lift.close([outs[0]])
  if not head.ok?
    Lift.close(outs.drop(1))
  end

  fresh = Cache(
    keys: outs.drop(1).take(layers),
    values: outs.drop(1 + layers),
    length: 0,
    spare: cache.keys + cache.values,
  )

  ok(Advanced(head: try head, cache: fresh))
end

## Whether the spare tensors of `cache` have the shapes and dtype of its own keys and values (so a step of the same
## capacity can write its new ones there).
def fits(cache: Cache) -> Bool
  have = cache.keys + cache.values
  return false if cache.spare.size != have.size or have.empty?
  (0...have.size).all? { |i|
    Tensor.shape(cache.spare[i]) == Tensor.shape(have[i]) and Tensor.dtype(cache.spare[i]) == Tensor.dtype(have[i])
  }
end

## The compiled function `key` of `g` for `specs`, compiled with `f` on a miss and
## kept at once.
def planned(
  g: Generator,
  key: String,
  specs: List[Trace.Spec],
  f: Fn[List[Tensor.Tensor], Result[List[Tensor.Tensor]]],
) -> Result[Trace.Compiled]
  case take_plans(g)
    some(p) ->
      _ = g.plans.send(p, 0)
      return ok(p.graphs[key]) if p.graphs.has_key?(key)
    none -> nil
  end

  compiled = try Trace.compile(specs, f)
  case take_plans(g)
    some(p) -> _ = g.plans.send(Plans(graphs: p.graphs.put(key, compiled), compiles: p.compiles + 1), 0)
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
def eager_step(m: Qwen3.Qwen3, cache: Cache, fed: List[Tensor.Tensor], room: Int, picking: Bool) -> Result[Advanced]
  layers = m.config.num_hidden_layers
  ran: Result[Advanced] = Native.arena do
    outs = try stepped(m, fed + cache.keys + cache.values, room, picking)
    ok(
      Advanced(
        head: try Tensor.to_list(outs[0]),
        cache: Cache(keys: outs.drop(1).take(layers), values: outs.drop(1 + layers), length: 0, spare: []),
      ),
    )
  end

  ran
end

## The step: `xs` are the ids `[width]`, their positions, the one-hot `[1, width]` of the row whose
## prediction is wanted, and per layer the keys then the values. The outputs are the head (the logits `[vocab_size]`,
## or the choice `[1]`), then the keys and the values of every layer.
def stepped(m: Qwen3.Qwen3, xs: List[Tensor.Tensor], room: Int, picking: Bool) -> Result[List[Tensor.Tensor]]
  layers = m.config.num_hidden_layers
  width = Tensor.shape(xs[0])[0]
  d = m.config.hidden_size
  slots = Slots(positions: xs[1], capacity: room)
  h = try Tensor.reshape(try Nn.embedding(m.tokens, xs[0]), [1, width, d])
  mask = try padded_mask(slots, Tensor.dtype(h))
  spread = try spread_of(slots, Tensor.dtype(h))
  keys: List[Tensor.Tensor] = []
  values: List[Tensor.Tensor] = []
  for i in 0...layers
    old: Optional[Kv] = none
    if xs.size > 3
      old = some(Kv(key: xs[3 + i], value: xs[3 + layers + i]))
    end

    r = try layer(m.stack.blocks[i], h, slots.positions, mask, spread, old)
    h = r.hidden
    keys = keys.push(r.key)
    values = values.push(r.value)
  end

  last = try Tensor.reshape(try Tensor.matmul(xs[2], try Tensor.reshape(h, [width, d])), [1, 1, d])
  case m.stack.final_norm
    some(n) -> last = try Nn.layer_norm(n, last)
    none -> nil
  end

  logits = try Tensor.reshape(try Nn.dense(m.head, last), [m.config.vocab_size])
  head = logits
  if picking
    head = try picked(logits)
  end

  ok([head] + keys + values)
end

## One block over the new tokens `h` `[1, width, d]`, as `Nn.encoder_block` with
## `norm_first`, the keys and values of the cache in the place of the block's own.
def layer(
  b: Nn.EncoderBlock,
  h: Tensor.Tensor,
  positions: Tensor.Tensor,
  mask: Tensor.Tensor,
  spread: Tensor.Tensor,
  old: Optional[Kv],
) -> Result[Layer]
  x = try Nn.layer_norm(b.norm1, h)
  heads = try Nn.project(b.attention, x, x, some(positions))
  key = try written(old.map { |kv| kv.key }, heads.key, spread)
  value = try written(old.map { |kv| kv.value }, heads.value, spread)
  seen = try Nn.attend(b.attention, heads.query, key, value, some(mask))
  mid = try Tensor.add(h, seen)
  out = try Tensor.add(mid, try Nn.mlp(b.mlp, try Nn.layer_norm(b.norm2, mid)))
  ok(Layer(hidden: out, key: key, value: value))
end

## `[n, capacity]`: `MASKED` where the key's row is after the query's position
## (later tokens and the padding), 0 elsewhere.
def padded_mask(s: Slots, dtype: Tensor.Dtype) -> Result[Tensor.Tensor]
  n = Tensor.shape(s.positions)[0]
  backend = Tensor.backend(s.positions)
  grid = [n, s.capacity]
  queries = try Tensor.broadcast_to(try Tensor.reshape(s.positions, [n, 1]), grid)
  keys = try Tensor.broadcast_to(
    try Tensor.reshape(try Tensor.arange(0, s.capacity, 1, Tensor.Dtype.I32, backend), [1, s.capacity]),
    grid,
  )

  blocked = try Tensor.full(grid, Tensor.Values.Float(Nn.MASKED), dtype, backend)
  Tensor.where(try Tensor.lt(queries, keys), blocked, try Tensor.zeros(grid, dtype, backend))
end

## `[capacity, n]`: 1 at row `positions[j]` of column `j`, 0 elsewhere. A
## position of `capacity` (the padding of a prompt) names no row.
def spread_of(s: Slots, dtype: Tensor.Dtype) -> Result[Tensor.Tensor]
  n = Tensor.shape(s.positions)[0]
  grid = [s.capacity, n]
  rows = try Tensor.broadcast_to(
    try Tensor.reshape(
      try Tensor.arange(0, s.capacity, 1, Tensor.Dtype.I32, Tensor.backend(s.positions)),
      [s.capacity, 1],
    ),
    grid,
  )

  at = try Tensor.broadcast_to(try Tensor.reshape(s.positions, [1, n]), grid)
  Tensor.to_dtype(try Tensor.eq(rows, at), dtype)
end

## `fresh` (`[1, kv_heads, n, head_dim]`) written into `old` (`[1, kv_heads, rows, head_dim]`,
## zero past the cache's length; none when empty) at the rows `spread`
## (`[capacity, n]`) names, padded with zeros to `capacity` rows. A product with a 0/1 matrix
## and a sum with zeros, so exact in F32.
def written(old: Optional[Tensor.Tensor], fresh: Tensor.Tensor, spread: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(fresh)
  heads = shape[1]
  n = shape[2]
  width = shape[3]
  room = Tensor.shape(spread)[0]
  flat = try Tensor.reshape(try Tensor.permute(fresh, [0, 2, 1, 3]), [n, heads * width])
  placed = try Tensor.permute(
    try Tensor.reshape(try Tensor.matmul(spread, flat), [1, room, heads, width]),
    [0, 2, 1, 3],
  )

  case old
    none -> ok(placed)
    some(o) ->
      rows = Tensor.shape(o)[2]
      if rows < room
        gap = try Tensor.zeros([1, heads, room - rows, width], Tensor.dtype(o), Tensor.backend(o))
        o = try Tensor.concat([o, gap], 2)
      end

      Tensor.add(o, placed)
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
