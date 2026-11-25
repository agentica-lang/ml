## Greedy or sampled generation with Granite MoE (`transformers/DESIGN.md`), the cache of `generate.agc`: the keys
## (already rotated) and the values of a layer `[1, kv_heads, rows, head_dim]`, padded to a capacity of 8, 16, 32 and so
## on (`F16` on `Ggml`), the compiled steps by shape and the pick of the next id in the graph. A prompt is padded to its
## capacity and the padding rows go through the experts too, but no real token reads them (the mask hides them), and an
## expert's answer for a token does not depend on the other tokens. The block of a layer is `GraniteMoe`'s: the
## multipliers of the config, the attention scale, and `Moe.moe` in place of the MLP.
##
## Batch of one. A `Generator` belongs to one task at a time, and `close` ends it before `GraniteMoe.close_model` frees the
## weights its graphs read.

import Padded from "pkg/nn/kv"
import Moe from "pkg/nn/moe"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Lift from "lift"
import GraniteMoe from "granite_moe"
import Sample from "sample"
import Stepping from "stepping"

## A model and the compiled steps it has run (on `Ggml`), which every sequence
## of the model shares: the channel holds one `Plans`, which a call takes to read
## or change it and puts back at once, and a graph goes in as soon as it is made.
struct Generator
  model: GraniteMoe.GraniteMoe
  plans: Channel[Stepping.Plans]
  ## How many ids a step may answer: the ids from `usable` on get a logit of `Nn.MASKED`, for a caller whose tokenizer has
  ## fewer tokens than the embedding has rows (`tok.pieces.size`). `vocab_size` (the default of `open`) is no mask, as in
  ## Hugging Face.
  usable: Int
  ## Every step reads the whole capacity of the cache, not only the rows in use: `false` from `open`. The tests set it
  ## to have the logits of the other way of reading the same cache.
  whole: Bool
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

## `step_logits`: the logits after the last token fed, and the cache.
struct Scored
  logits: List[Float]
  cache: Cache
end

## The generator of `m`. `usable` is how many ids it may answer (see `Generator.usable`); the whole vocabulary when it
## is 0.
def open(m: GraniteMoe.GraniteMoe, usable: Int = 0) -> Result[Generator]
  ids = try Stepping.usable_ids(usable, m.config.vocab_size)
  cell: Channel[Stepping.Plans] = Channel.new(1)
  try cell.send(Stepping.Plans(graphs: {}, compiles: 0), 1000)
  ok(Generator(model: m, plans: cell, usable: ids, whole: false))
end

## How many steps `g` has compiled: a shape it has seen compiles no more.
def compiles(g: Generator) -> Int
  case Stepping.take_plans(g.plans)
    some(p) ->
      _ = g.plans.send(p, 0)
      p.compiles
    none -> 0
  end
end

## Ends the pins of the compiled steps of `g`, which a later call compiles again.
## Call it before `GraniteMoe.close_model(g.model)`.
def close(g: Generator) -> Void
  case Stepping.take_plans(g.plans)
    some(p) ->
      for key in p.graphs.keys
        Trace.close(p.graphs[key])
      end

      _ = g.plans.send(Stepping.Plans(graphs: {}, compiles: p.compiles), 0)
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
def step(g: Generator, cache: Cache, ids: List[Int]) -> Result[Stepping.Picked[Cache]]
  s = try advance(g, cache, ids, Stepping.Head.Pick)
  found = Stepping.chosen(s.head, g.usable)
  if not found.ok?
    finish(s.cache)
  end

  ok(Stepping.Picked(id: try found, cache: s.cache))
end

## `step` with the logits of the last token fed, `[vocab_size]`, in place of the choice.
def step_logits(g: Generator, cache: Cache, ids: List[Int]) -> Result[Scored]
  s = try advance(g, cache, ids, Stepping.Head.Logits)
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
## are checked to be finite). `k` is at most the usable ids.
def step_top(g: Generator, cache: Cache, ids: List[Int], k: Int) -> Result[Stepping.Candidates[Cache]]
  if k < 1 or k > g.usable
    finish(cache)
    return error("step_top needs k from 1 to #{g.usable}, got #{k}")
  end

  s = try advance(g, cache, ids, Stepping.Head.Top(k))
  found = Stepping.ranked(s.head, k, g.usable)
  if not found.ok?
    finish(s.cache)
  end

  r = try found
  ok(Stepping.Candidates(ids: r.ids, logits: r.logits, cache: s.cache))
end

## Ends a sequence: closes the tensors of `cache`.
def finish(cache: Cache) -> Void
  Lift.close(cache.keys + cache.values)
end

## Decoding from the prompt `ids`: at most `max_new` ids, stopping before a stop id (`stop`, such as `<|im_end|>`).
## Greedy (the choice runs in the graph) unless `sampling` says how to draw (`Stepping.generate`): then each step reads
## the largest logits and `Sample.pick_top` draws among them. The prompt and the new ids must fit
## `max_position_embeddings`.
def generate(
  g: Generator,
  ids: List[Int],
  max_new: Int,
  stop: List[Int] = [],
  sampling: Optional[Sample.Sampling] = none,
) -> Result[List[Int]]
  Stepping.generate(decoder(g), ids, max_new, stop, sampling)
end

private

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

## What `Stepping.generate` needs of `g`.
def decoder(g: Generator) -> Stepping.Decoder[Cache]
  Stepping.Decoder(
    usable: g.usable,
    limit: g.model.config.max_position_embeddings,
    empty: start(),
    pick: ->(c: Cache, ids: List[Int]) { step(g, c, ids) },
    top: ->(c: Cache, ids: List[Int], k: Int) { step_top(g, c, ids, k) },
    finish: ->(c: Cache) { finish(c) },
  )
end

def advance(g: Generator, cache: Cache, ids: List[Int], mode: Stepping.Head) -> Result[Advanced]
  m = g.model
  bad = Stepping.check_ids(m.config.max_position_embeddings, m.config.vocab_size, cache.length, ids)
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
  made = Stepping.inputs(ids, cache.length, width, room, backend)
  if not made.ok?
    finish(cache)
    return error(made.message)
  end

  fed = made.unwrap!
  held = cache.keys + cache.values
  in_place = Stepping.in_place?(backend, cache.keys, width, room)
  reach = room
  if in_place and not g.whole
    reach = Padded.reach(cache.length + n, room)
  end

  outs = case backend
    Tensor.Backend.Ggml(_) -> compiled_step(g, cache, fed, room, reach, mode, in_place)
    _ -> eager_step(m, g.usable, cache, fed, room, mode)
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
  mode: Stepping.Head,
  in_place: Bool,
) -> Result[Advanced]
  m = g.model
  args = fed + cache.keys + cache.values
  rows = 0
  if not cache.keys.empty?
    rows = Tensor.shape(cache.keys[0])[2]
  end

  width = Tensor.shape(fed[0])[0]
  keys = Stepping.reaches(width, rows, room, reach, mode)
  plan = try Stepping.planned(
    g.plans,
    keys.key,
    keys.ahead,
    keys.behind,
    args.map { |t| Trace.spec_of(t) },
    ->(xs: List[Tensor.Tensor]) { stepped(GraniteMoe.traced(m), g.usable, xs, room, reach, mode, in_place, true) },
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

## The same graph, run now and in an arena that keeps the new cache alone.
def eager_step(
  m: GraniteMoe.GraniteMoe,
  usable: Int,
  cache: Cache,
  fed: List[Tensor.Tensor],
  room: Int,
  mode: Stepping.Head,
) -> Result[Advanced]
  layers = m.config.num_hidden_layers
  ran: Result[Advanced] = Native.arena do
    outs = try stepped(m, usable, fed + cache.keys + cache.values, room, room, mode)
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
  m: GraniteMoe.GraniteMoe,
  usable: Int,
  xs: List[Tensor.Tensor],
  room: Int,
  reach: Int,
  mode: Stepping.Head,
  in_place: Bool = false,
  half: Bool = false,
) -> Result[List[Tensor.Tensor]]
  layers = m.config.num_hidden_layers
  width = Tensor.shape(xs[0])[0]
  d = m.config.hidden_size
  slots = Padded.Slots(positions: xs[1], capacity: room)
  h = try GraniteMoe.times(
    try Tensor.reshape(try Nn.embedding(m.tokens, xs[0]), [1, width, d]),
    m.config.embedding_multiplier,
  )

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

    r = try layer(m, m.layers[i], h, slots.positions, mask, spread, old, in_place, reach)
    h = r.hidden
    keys = keys.push(try Stepping.cached(r.key, half))
    values = values.push(try Stepping.cached(r.value, half))
  end

  last = try Tensor.reshape(try Tensor.matmul(xs[2], try Tensor.reshape(h, [width, d])), [1, 1, d])
  logits = try Stepping.masked(
    try Tensor.reshape(try GraniteMoe.scores(m, try Nn.layer_norm(m.final_norm, last)), [m.config.vocab_size]),
    usable,
  )

  head = logits
  case mode
    Stepping.Head.Pick -> head = try Stepping.picked(logits)
    Stepping.Head.Top(k) -> head = try Stepping.top_candidates(logits, k)
    Stepping.Head.Logits -> nil
  end

  ok([head] + keys + values)
end

## One layer over the new tokens `h` `[1, width, d]`, as `GraniteMoe`'s own, the keys and values of the cache in the place
## of the layer's own.
def layer(
  m: GraniteMoe.GraniteMoe,
  l: GraniteMoe.Layer,
  h: Tensor.Tensor,
  positions: Tensor.Tensor,
  mask: Tensor.Tensor,
  spread: Optional[Tensor.Tensor],
  old: Optional[Kv],
  in_place: Bool,
  reach: Int,
) -> Result[Layer]
  x = try Nn.layer_norm(l.norm1, h)
  heads = try Nn.project(l.attention, x, x, some(positions))
  key = heads.key
  value = heads.value
  if in_place
    key = try Tensor.set_rows(old.unwrap!.key, heads.key, positions)
    value = try Tensor.set_rows(old.unwrap!.value, heads.value, positions)
  else
    key = try Padded.written(try Stepping.widened(old.map { |kv| kv.key }), heads.key, spread.unwrap!, 2)
    value = try Padded.written(try Stepping.widened(old.map { |kv| kv.value }), heads.value, spread.unwrap!, 2)
  end

  seen = try Nn.attend(
    l.attention,
    heads.query,
    try Padded.reached(key, reach),
    try Padded.reached(value, reach),
    some(mask),
  )

  mid = try Tensor.add(h, try GraniteMoe.times(seen, m.config.residual_multiplier))
  routed = try Moe.moe(l.moe, try Nn.layer_norm(l.norm2, mid))
  out = try Tensor.add(mid, try GraniteMoe.times(routed, m.config.residual_multiplier))
  ok(Layer(hidden: out, key: key, value: value))
end
