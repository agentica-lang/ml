## The decoder that Qwen3 (`generate.agc`), Granite MoE (`generate_granite_moe.agc`), LFM3 (`generate_lfm3.agc`) and
## Qwen3.5 (`generate_qwen35.agc`) share, over a key-value cache
## (`transformers/DESIGN.md`), the design of `asr/`'s decoder (`asr/asr.agc`); the cache helpers are shared in
## `nn/kv.agc`. A model gives a `Stepper`: its sizes, its backend, `traced` and its `stepped` function, the graph of
## one step; everything else is here:
##
## - The cache of a layer is `[1, kv_heads, rows, head_dim]`, the keys already normalized and rotated, padded with zero
##   rows to a capacity of 8, 16, 32 and so on (`capacity`), so a graph changes shape only when the capacity grows. The
##   model's `stepped` decides the dtype of what it keeps. A step writes the new rows with a product by a 0/1 matrix
##   (`written`), and a mask hides the later positions and the padding.
## - On `Ggml` each (tokens, rows, capacity) is one compiled graph, kept by the generator of the model: the prompt,
##   padded to a power-of-2 capacity (8, 16, ...), takes only its last hidden row to the head, and every later step is
##   one token. The choice of the next id runs in the graph, so one number leaves the device, not the logits. A
##   one-token step reads the rows of the cache in use, rounded up (`Padded.reach`), not the padding after them; each
##   reach is a graph, and a capacity keeps the one of its largest.
## - Elsewhere the same graph runs eagerly in an arena. On the reference backend attention needs `q * k` of at most
##   1,048,576, which the padded prompt counts (a prompt of 600 tokens is padded to 1024). libtorch is not tested.
##
## A model with layers that keep a state (LFM3's short convolutions, Qwen3.5's linear attention) says so in its
## `Stepper` (`states`, `taps`): the states are tensors of a size that does not depend on the length, new at every step,
## and a prompt is given the `Delta.Padding` of its convolutions, so that the padding rows leave the state as the real
## ones did. A step that gives another number of tensors than the `Stepper` says is an error.
##
## A model that reads images (Qwen3.5, SmolVLM) says so in its `Stepper` (`sighted`): a prompt starts from
## `start(sight)` and the step splices the image rows into the embeddings (`spliced`). A model with a multimodal rope
## (Qwen3.5) turns the attention layers by the three axes of `Sight.layout` (`Request.rotary`), and the later tokens
## by their index plus the `Cache.delta` of the prompt; a model with one position per token (`Stepper.rotated` false,
## SmolVLM) turns them by the index and takes only a `plain_sight`. Each number of image tokens compiles its own graph
## for the prompt.
##
## Batch of one. A prompt is fed in one step (`step` at length 0), then one token at a time. A `Generator` belongs to
## one task at a time, and `close` ends it before the weights its graphs read are freed.

import Padded from "pkg/nn/kv"
import Delta from "pkg/nn/delta"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Lift from "lift"
import Mrope from "mrope"
import Sample from "sample"
import Stepping from "stepping"

## One step to run: `xs` are the ids `[width]`, their positions, the one-hot `[1, width]` of the row whose prediction is
## wanted, for a prompt of a model with `taps` the `valid` and `keep` of its `Delta.Padding`, and per layer the keys then
## the values, then the states. The cache the step leaves has `room` rows, of which the attention
## reads the first `reach`; `in_place` writes the new row into the tensors of the old cache, and `compiled` says the
## step is traced for ggml (a model may keep its cache in `F16` then). `usable` and `mode` are those of the generator
## and of the call (`Stepping.Head`).
struct Request
  usable: Int
  xs: List[Tensor.Tensor]
  room: Int
  reach: Int
  mode: Stepping.Head
  in_place: Bool
  compiled: Bool
  ## What the attention layers turn by when it is not `xs[1]`: `[3, width]`, the three axes of a prompt with images, or
  ## `[width]` for the token after one (its index plus the `Cache.delta`).
  rotary: Optional[Tensor.Tensor] = none
  ## The image rows of a prompt with images (`spliced`).
  splice: Optional[Splice] = none
end

## The rows of the images of a prompt: `at` (`[n]`, `I32`) the prompt rows that are image tokens, and the `n` rows
## `features` (`[n, width]`) that take their place.
struct Splice
  at: Tensor.Tensor
  features: Tensor.Tensor
end

## The images of a prompt: their `features` (`[n, width]`, the rows of the image tokens in the order of the prompt,
## which the caller keeps and closes), the id of the `image_token` of the prompt and where `Mrope.layout` put the
## tokens.
struct Sight
  features: Tensor.Tensor
  image_token: Int
  layout: Mrope.Layout
end

## The old keys and values of one layer, which a step reads and (in place) writes.
struct Kv
  key: Tensor.Tensor
  value: Tensor.Tensor
end

## What a layer of a step gives: its hidden rows and its new keys and values.
struct Layer
  hidden: Tensor.Tensor
  key: Tensor.Tensor
  value: Tensor.Tensor
end

## What the attention layers of one step read, built once per step (`reading`): the positions of the new rows, the mask
## over the cache (the first `Request.reach` rows) and, when the step is not in place, the `Padded.spread_of` the
## positions that writes the new rows. A model whose layers have more than one mask (LFM3's sliding window) gives each
## layer its own `mask` (`with`). A model with a sectioned rope gives the `axes` of its masks (`Nn.axis_masks`), built
## once for all its layers.
struct Reading
  positions: Tensor.Tensor
  mask: Tensor.Tensor
  spread: Optional[Tensor.Tensor]
  axes: List[Tensor.Tensor] = []
end

## What `attend_cached` gives: the answer of the attention (not yet added to the stream) and the keys and values of the
## cache after the step, in the dtype of the old ones; a model that keeps them in `F16` on ggml converts them
## (`Stepping.cached`).
struct Attended
  answer: Tensor.Tensor
  key: Tensor.Tensor
  value: Tensor.Tensor
end

## What a model gives the decoder, `M` being its type. `stepped(model, request)` is the graph of a step: its outputs are
## the head (the logits `[vocab]`, the choice `[1]` or the candidates, by `request.mode`), then the keys and the values
## of every layer with a cache, then its states. `traced` is the model with its tensors as `Trace` constants.
struct Stepper[M]
  ## The layers with a cache of keys and values.
  layers: Int
  ## The positions of the model (`max_position_embeddings`).
  limit: Int
  vocab: Int
  ## The backend of the model, which the ids and the inputs of a step are made on (the one its table is on, unless the
  ## table is on the CPU under a model on a Vulkan device).
  backend: Tensor.Backend
  ## How many state tensors a step takes and gives, after the keys and the values (0 for a model with none).
  states: Int = 0
  ## The taps of the convolution of a model with states: a prompt (more than one row) is given a `Delta.Padding` for
  ## them (its `valid` is for the recurrent state of Qwen3.5; LFM3 reads only `keep`). 0 for a model with no convolution.
  taps: Int = 0
  ## Whether `stepped` reads `Request.splice` (and `Request.rotary`, if `rotated`): only then may a cache start from
  ## a `Sight`.
  sighted: Bool = false
  ## Whether the rope of the model turns by more than the index of a token (Qwen3.5's three axes and `Cache.delta`). A
  ## model with one position per token (`false`) is given no `rotary`, and a `Sight` for it must be a plain one
  ## (`plain_sight`): any other layout is an error.
  rotated: Bool = true
  traced: Fn[M, M]
  stepped: Fn[M, Request, Result[List[Tensor.Tensor]]]
end

## What a generator of a model is for the decoder: the model, its `Stepper`, and the compiled steps it has run (on
## `Ggml`), which every sequence of the model shares: the channel holds one `Plans`, which a call takes to read or
## change it and puts back at once, and a graph goes in as soon as it is made. A model file's `Generator` has the same
## fields (without the `Stepper`) and is turned into a `Core` by its `core`.
struct Core[M]
  model: M
  stepper: Stepper[M]
  plans: Channel[Stepping.Plans]
  ## How many ids a step may answer: the ids from `usable` on get a logit of `Nn.MASKED`. The embedding of a model can
  ## have more rows than its tokenizer has tokens (Qwen3: 151,936 and 151,669); with the tokenizer's count here
  ## (`tok.pieces.size`) the sampler cannot give an id that `Bpe.decode` rejects. `vocab_size` (the default of `open`)
  ## is no mask, as in Hugging Face.
  usable: Int
  ## Every step reads the whole capacity of the cache, not only the rows in use. The tests set it to have the logits of
  ## the other way of reading the same cache.
  whole: Bool
end

## What `step` keeps between calls: per layer the keys and the values of the `length` tokens so far, padded to the
## capacity. On `Ggml` a one-token step of the same capacity writes its row into these very tensors (they are the
## tensors of the cache it returns), so a cache that was stepped is dead.
struct Cache
  keys: List[Tensor.Tensor]
  values: List[Tensor.Tensor]
  length: Int
  ## The states of a model that has them, new tensors at every step.
  states: List[Tensor.Tensor] = []
  ## How far the position of the next token is from its index, after a prompt with images (0 for a text prompt).
  delta: Int = 0
  ## The images of the prompt that the first step reads (`start`); none after it.
  sight: Optional[Sight] = none
end

## `step_logits`: the logits after the last token fed, and the cache.
struct Scored
  logits: List[Float]
  cache: Cache
end

## The `Core` of `model`. `usable` is how many ids it may answer (see `Core.usable`); the whole vocabulary when it is 0.
def open[M](model: M, stepper: Stepper[M], usable: Int = 0) -> Result[Core[M]]
  ids = try Stepping.usable_ids(usable, stepper.vocab)
  cell: Channel[Stepping.Plans] = Channel.new(1)
  try cell.send(Stepping.Plans(graphs: {}, compiles: 0), 1000)
  ok(Core(model: model, stepper: stepper, plans: cell, usable: ids, whole: false))
end

## How many steps `g` has compiled: a shape it has seen compiles no more.
def compiles[M](g: Core[M]) -> Int
  case Stepping.take_plans(g.plans)
    some(p) ->
      _ = g.plans.send(p, 0)
      p.compiles
    none -> 0
  end
end

## Ends the pins of the compiled steps of `g`, which a later call compiles again. Call it before the weights of the
## model are closed.
def close[M](g: Core[M]) -> Void
  case Stepping.take_plans(g.plans)
    some(p) ->
      for key in p.graphs.keys
        Trace.close(p.graphs[key])
      end

      _ = g.plans.send(Stepping.Plans(graphs: {}, compiles: p.compiles), 0)
    none -> nil
  end
end

## The `Sight` of a model with one position per token: the `features` (`[n, width]`) at the tokens of `ids` equal to
## `image_token`, in order, positioned by their index.
def plain_sight(features: Tensor.Tensor, image_token: Int, ids: List[Int]) -> Sight
  Sight(features: features, image_token: image_token, layout: Mrope.indexed(ids, image_token))
end

## The empty cache, or, with `sight`, the one of a prompt with images: the first `step` splices their `features` in and
## positions the tokens by their `layout`, and the cache it gives remembers the `delta` for the tokens after. A model
## that does not read images (`Stepper.sighted`) refuses the step.
def start(sight: Optional[Sight] = none) -> Cache = Cache(keys: [], values: [], length: 0, sight: sight)

## Feeds `ids` after the tokens of `cache`: a prompt of any length when the cache is empty (the images of a prompt come
## from `start(sight)`), one id after that. The cache
## that `step` takes is dead afterwards, even when the step fails: on ggml a step of the same capacity writes the new
## row into its tensors, which are the ones of the cache it gives, elsewhere they are closed. Use the one it gives;
## never step a cache twice or read one after it was stepped.
def step[M](g: Core[M], cache: Cache, ids: List[Int]) -> Result[Stepping.Picked[Cache]]
  s = try advance(g, cache, ids, Stepping.Head.Pick)
  found = Stepping.chosen(s.head, g.usable)
  if not found.ok?
    finish(s.cache)
  end

  ok(Stepping.Picked(id: try found, cache: s.cache))
end

## `step` with the logits of the last token fed, `[vocab]`, in place of the choice.
def step_logits[M](g: Core[M], cache: Cache, ids: List[Int]) -> Result[Scored]
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
## graph (`Tensor.top_ids`), so `k` values and ids cross to the host and not `vocab` logits. Ids with equal logits at
## the edge of the `k` are not specified. A `-inf` logit among the `k` makes the step an error on ggml (its outputs are
## checked to be finite). `k` is at most the usable ids.
def step_top[M](g: Core[M], cache: Cache, ids: List[Int], k: Int) -> Result[Stepping.Candidates[Cache]]
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
  Lift.close(cache.keys + cache.values + cache.states)
end

## The `Reading` of the step `req` under `mask` (of the dtype of the stream), whose positions are the second input.
def reading(req: Request, mask: Tensor.Tensor) -> Result[Reading]
  positions = req.xs[1]
  spread: Optional[Tensor.Tensor] = none
  if not req.in_place
    spread = some(try Padded.spread_of(Padded.Slots(positions: positions, capacity: req.room), Tensor.dtype(mask)))
  end

  ok(Reading(positions: positions, mask: mask, spread: spread))
end

## The embeddings `h` (`[1, width, d]`) with the rows of the images of `req.splice` in place of the rows `at` names: the
## other rows are kept as they are.
def spliced(h: Tensor.Tensor, req: Request) -> Result[Tensor.Tensor]
  case req.splice
    none -> ok(h)
    some(images) ->
      shape = Tensor.shape(h)
      width = shape[1]
      n = Tensor.shape(images.at)[0]
      backend = Tensor.backend(h)
      dtype = Tensor.dtype(h)
      grid = [width, n]
      rows = try Tensor.broadcast_to(
        try Tensor.reshape(try Tensor.arange(0, width, 1, Tensor.Dtype.I32, backend), [width, 1]),
        grid,
      )

      hit = try Tensor.to_dtype(
        try Tensor.eq(rows, try Tensor.broadcast_to(try Tensor.reshape(images.at, [1, n]), grid)),
        dtype,
      )

      covered = try Tensor.matmul(hit, try Tensor.ones([n, 1], dtype, backend))
      keep = try Tensor.sub(try Tensor.ones([width, 1], dtype, backend), covered)
      kept = try Tensor.mul(try Tensor.reshape(h, [width, shape[2]]), try Tensor.broadcast_to(keep, [width, shape[2]]))

      placed = try Tensor.add(kept, try Tensor.matmul(hit, images.features))
      Tensor.reshape(placed, [1, width, shape[2]])
  end
end

## The attention `a` of a layer over the new rows `x` `[1, width, d]` (its input, normed): the keys and values of the
## rows are written into the `old` ones of the cache (`Tensor.set_rows` in the tensors themselves when `req.in_place`,
## else `Padded.written`), then the query attends over the first `req.reach` rows under the mask of `reading`. `old` is
## none for a prompt, which has no cache.
def attend_cached(
  a: Nn.Attention,
  x: Tensor.Tensor,
  old: Optional[Kv],
  reading: Reading,
  req: Request,
) -> Result[Attended]
  heads = try Nn.project(a, x, x, some(req.rotary.unwrap_or(reading.positions)), axes: reading.axes)
  key = heads.key
  value = heads.value
  if req.in_place
    key = try Tensor.set_rows(old.unwrap!.key, heads.key, reading.positions)
    value = try Tensor.set_rows(old.unwrap!.value, heads.value, reading.positions)
  else
    key = try Padded.written(try Stepping.widened(old.map { |kv| kv.key }), heads.key, reading.spread.unwrap!, 2)
    value = try Padded.written(try Stepping.widened(old.map { |kv| kv.value }), heads.value, reading.spread.unwrap!, 2)
  end

  seen = try Nn.attend(
    a,
    heads.query,
    try Padded.reached(key, req.reach),
    try Padded.reached(value, req.reach),
    some(reading.mask),
    gate: heads.gate,
  )

  ok(Attended(answer: seen, key: key, value: value))
end

## Decoding from the prompt `ids`: at most `max_new` ids, stopping before a stop id (`stop`, such as `<|im_end|>`).
## Greedy (the choice runs in the graph) unless `sampling` says how to draw (`Stepping.generate`): then each step reads
## the largest logits and `Sample.pick_top` draws among them. The prompt and the new ids must fit
## `max_position_embeddings`. `sight` is the images of the prompt (`start`).
def generate[M](
  g: Core[M],
  ids: List[Int],
  max_new: Int,
  stop: List[Int] = [],
  sampling: Optional[Sample.Sampling] = none,
  sight: Optional[Sight] = none,
) -> Result[List[Int]]
  Stepping.generate(decoder(g, start(sight)), ids, max_new, stop, sampling)
end

private

## What one step gives: the last row of the head (the logits, the pick as one float, or the candidates) and the cache
## after it.
struct Advanced
  head: List[Tensor.Values]
  cache: Cache
end

## The inputs of a step beyond its ids: what the rope turns by when it is not the index (`rotary`), and the images to
## splice in. `own` are the tensors the step made (closed after it), `tag` tells the compiled steps of another shape of
## them apart.
struct Feed
  rotary: Optional[Tensor.Tensor]
  splice: Optional[Splice]
  own: List[Tensor.Tensor]
  tag: String
end

## The tensors of `feed` that a compiled step takes after the cache, in the order that `traced` reads them.
def feed_args(f: Feed) -> List[Tensor.Tensor]
  args: List[Tensor.Tensor] = []
  case f.rotary
    some(r) -> args = args.push(r)
    none -> nil
  end

  case f.splice
    some(sp) -> args = args + [sp.at, sp.features]
    none -> nil
  end

  args
end

## `f` over the traced tensors `tail` of a compiled step, which are its `args`.
def feed_traced(f: Feed, tail: List[Tensor.Tensor]) -> Feed
  rotary: Optional[Tensor.Tensor] = none
  at = 0
  if f.rotary.some?
    rotary = some(tail[0])
    at = 1
  end

  splice: Optional[Splice] = none
  if f.splice.some?
    splice = some(Splice(at: tail[at], features: tail[at + 1]))
  end

  f.with(rotary: rotary, splice: splice)
end

## What `Stepping.generate` needs of `g`.
def decoder[M](g: Core[M], first: Cache) -> Stepping.Decoder[Cache]
  Stepping.Decoder(
    usable: g.usable,
    limit: g.stepper.limit,
    empty: first,
    pick: ->(c: Cache, ids: List[Int]) { step(g, c, ids) },
    top: ->(c: Cache, ids: List[Int], k: Int) { step_top(g, c, ids, k) },
    finish: ->(c: Cache) { finish(c) },
  )
end

def advance[M](g: Core[M], cache: Cache, ids: List[Int], mode: Stepping.Head) -> Result[Advanced]
  s = g.stepper
  bad = Stepping.check_ids(s.limit, s.vocab, cache.length, ids)
  if not bad.ok?
    finish(cache)
    return error(bad.message)
  end

  n = ids.size
  room = Padded.capacity(cache.length + n, s.limit)
  # A prompt is padded to the capacity, a later step is one token.
  width = 1
  if cache.length == 0
    width = room
  end

  backend = s.backend
  made = fed_inputs(s, ids, cache.length, width, room, backend)
  if not made.ok?
    finish(cache)
    return error(made.message)
  end

  fed = made.unwrap!
  seen = rotation(s, cache, ids, width, backend)
  if not seen.ok?
    Lift.close(fed)
    finish(cache)
    return error(seen.message)
  end

  feed = seen.unwrap!
  held = cache.keys + cache.values
  in_place = Stepping.in_place?(backend, cache.keys, width, room)
  reach = room
  if in_place and not g.whole
    reach = Padded.reach(cache.length + n, room)
  end

  outs = case backend
    Tensor.Backend.Ggml(_) -> compiled_step(g, cache, fed, feed, room, reach, mode, in_place)
    _ -> eager_step(g, cache, fed, feed, room, mode)
  end

  Lift.close(fed)
  Lift.close(feed.own)
  Lift.close(cache.states)
  # A step in place keeps the tensors of `cache` as the ones of the new cache; any other step is done with them (a
  # tensor closed twice is fine, as when the step failed after it closed its outputs).
  if not in_place or not outs.ok?
    Lift.close(held)
  end

  r = try outs
  delta = cache.delta
  case cache.sight
    some(sight) -> delta = sight.layout.delta
    none -> nil
  end

  ok(Advanced(head: r.head, cache: r.cache.with(length: cache.length + n, delta: delta)))
end

## The graph of one step as a compiled function of its inputs on `Ggml`; `in_place` writes the new row into the cache's
## own tensors, which come back as the keys and values of the new one. Its attention reads the first `reach` rows of the
## cache. Each reach is a graph of its own, and a capacity keeps one: a step takes the graph of a larger reach when
## there is one (its mask hides the rows after the length), and compiling the graph of a reach ends the graphs of the
## smaller ones.
def compiled_step[M](
  g: Core[M],
  cache: Cache,
  fed: List[Tensor.Tensor],
  feed: Feed,
  room: Int,
  reach: Int,
  mode: Stepping.Head,
  in_place: Bool,
) -> Result[Advanced]
  s = g.stepper
  args = fed + cache.keys + cache.values + cache.states + feed_args(feed)
  rows = 0
  if not cache.keys.empty?
    rows = Tensor.shape(cache.keys[0])[2]
  end

  width = Tensor.shape(fed[0])[0]
  keys = Stepping.reaches(width, rows, room, reach, mode)
  tag = feed.tag
  plan = try Stepping.planned(
    g.plans,
    keys.key + tag,
    keys.ahead.map { |k| k + tag },
    keys.behind.map { |k| k + tag },
    args.map { |t| Trace.spec_of(t) },
    ->(xs: List[Tensor.Tensor]) {
      split = xs.size - feed_args(feed).size
      seen = feed_traced(feed, xs.drop(split))
      s.stepped.call(
        s.traced.call(g.model),
        Request(
          usable: g.usable,
          xs: xs.take(split),
          room: room,
          reach: reach,
          mode: mode,
          in_place: in_place,
          compiled: true,
          rotary: seen.rotary,
          splice: seen.splice,
        ),
      )
    },
  )

  outs = try Trace.run(plan, args)
  sized = counted(s, outs)
  if not sized.ok?
    Lift.close(outs)
    return error(sized.message)
  end

  head = Tensor.to_list(outs[0])
  Lift.close([outs[0]])
  if not head.ok?
    Lift.close(outs.drop(1))
  end

  ok(Advanced(head: try head, cache: cache_of(s, outs.drop(1))))
end

## An error when a step gave other tensors than its `Stepper` says: the head, the keys and the values of every layer
## with a cache, then the states.
def counted[M](s: Stepper[M], outs: List[Tensor.Tensor]) -> Result[Void]
  want = 1 + 2 * s.layers + s.states
  return ok() if outs.size == want
  error("a step gave #{outs.size} tensors, its Stepper says #{want} (layers #{s.layers}, states #{s.states})")
end

## The cache of the outputs of a step after its head: the keys, the values and the states.
def cache_of[M](s: Stepper[M], outs: List[Tensor.Tensor]) -> Cache
  Cache(
    keys: outs.take(s.layers),
    values: outs.drop(s.layers).take(s.layers),
    states: outs.drop(2 * s.layers),
    length: 0,
  )
end

## The inputs of a step (`Stepping.inputs`) and, for a prompt of a model with convolutions, the `Delta.Padding` of
## them.
def fed_inputs[M](
  s: Stepper[M],
  ids: List[Int],
  length: Int,
  width: Int,
  room: Int,
  backend: Tensor.Backend,
) -> Result[List[Tensor.Tensor]]
  base = try Stepping.inputs(ids, length, width, room, backend)
  return ok(base) if width == 1 or s.taps == 0
  padded = Delta.padding(ids.size, width, s.taps, backend)
  if not padded.ok?
    Lift.close(base)
    return error(padded.message)
  end

  pad = padded.unwrap!
  ok(base + [pad.valid, pad.keep])
end

## The `Feed` of the step of `ids` after `cache`: the three axes and the image rows of a prompt that starts from a
## `Sight`, the position of the token after one (its index plus the `delta`), or nothing for a text prompt.
def rotation[M](s: Stepper[M], cache: Cache, ids: List[Int], width: Int, backend: Tensor.Backend) -> Result[Feed]
  bare = Feed(rotary: none, splice: none, own: [], tag: "")
  if cache.sight.none?
    return ok(bare) if cache.delta == 0
    at: List[Tensor.Values] = []
    for i in 0...width
      at = at.push(Tensor.Values.Int(cache.length + i + cache.delta))
    end

    rotary = try Tensor.from_list(at, [width], Tensor.Dtype.I32, backend)
    return ok(Feed(rotary: some(rotary), splice: none, own: [rotary], tag: " delta"))
  end

  sight = cache.sight.unwrap!
  return error("this model reads no images (Stepper.sighted)") if not s.sighted
  return error("a Sight is for the prompt: the cache already has #{cache.length} tokens") if cache.length > 0
  seen = sight.layout
  try fits(sight, ids)
  if not s.rotated
    try check_plain(seen)
  end

  n = seen.at.size
  return ok(bare) if n == 0
  shape = Tensor.shape(sight.features)
  return error("the sight has #{n} image tokens and features of #{shape}") if shape.size != 2 or shape[0] != n

  rows: List[Tensor.Values] = []
  for axis in [seen.time, seen.down, seen.across]
    for i in 0...width
      if i < axis.size
        rows = rows.push(Tensor.Values.Int(axis[i]))
      else
        rows = rows.push(Tensor.Values.Int(0))
      end
    end
  end

  rotary: Optional[Tensor.Tensor] = none
  own: List[Tensor.Tensor] = []
  if s.rotated
    made = try Tensor.from_list(rows, [3, width], Tensor.Dtype.I32, backend)
    rotary = some(made)
    own = [made]
  end

  where = Tensor.from_list(seen.at.map { |i| Tensor.Values.Int(i) }, [n], Tensor.Dtype.I32, backend)
  if not where.ok?
    Lift.close(own)
    return error(where.message)
  end

  images = Splice(at: where.unwrap!, features: sight.features)
  ok(Feed(rotary: rotary, splice: some(images), own: own + [images.at], tag: " sight #{n}"))
end

## An error unless `seen` is the layout of `plain_sight`: the index of each token on every axis, and no `delta`.
def check_plain(seen: Mrope.Layout) -> Result[Void]
  if seen.delta != 0
    return error("this model has one position for each token, the layout has a delta of #{seen.delta}")
  end

  for axis in [seen.time, seen.down, seen.across]
    for i in 0...axis.size
      if axis[i] != i
        return error("this model has one position for each token, the layout puts token #{i} at #{axis[i]}")
      end
    end
  end

  ok()
end

## An error unless the layout of `sight` is the one of the prompt `ids`: a position on each axis for every token, and
## the image rows `at` the tokens that are the sight's `image_token`, in order and no other.
def fits(sight: Sight, ids: List[Int]) -> Result[Void]
  seen = sight.layout
  for axis in [seen.time, seen.down, seen.across]
    if axis.size != ids.size
      return error("the sight has positions for #{axis.size} tokens, the prompt has #{ids.size}")
    end
  end

  last = -1
  for i in seen.at
    return error("the sight puts an image row at #{i}, outside the #{ids.size} tokens") if i < 0 or i >= ids.size
    return error("the image rows of the sight are not in increasing order (#{i} after #{last})") if i <= last
    if ids[i] != sight.image_token
      return error(
        "the sight puts an image row at #{i}, whose id #{ids[i]} is not the image token #{sight.image_token}",
      )
    end

    last = i
  end

  tokens = ids.select { |id| id == sight.image_token }.size
  if tokens != seen.at.size
    return error("the prompt has #{tokens} image tokens, the sight has #{seen.at.size} image rows")
  end

  ok()
end

## The same graph, run now and in an arena that keeps the new cache alone.
def eager_step[M](
  g: Core[M],
  cache: Cache,
  fed: List[Tensor.Tensor],
  feed: Feed,
  room: Int,
  mode: Stepping.Head,
) -> Result[Advanced]
  s = g.stepper
  ran: Result[Advanced] = Native.arena do
    outs = try s.stepped.call(
      g.model,
      Request(
        usable: g.usable,
        xs: fed + cache.keys + cache.values + cache.states,
        room: room,
        reach: room,
        mode: mode,
        in_place: false,
        compiled: false,
        rotary: feed.rotary,
        splice: feed.splice,
      ),
    )

    try counted(s, outs)
    ok(Advanced(head: try Tensor.to_list(outs[0]), cache: cache_of(s, outs.drop(1))))
  end

  ran
end
