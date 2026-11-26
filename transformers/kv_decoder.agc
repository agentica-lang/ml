## The decoder that Qwen3 (`generate.agc`), Granite MoE (`generate_granite_moe.agc`) and LFM3 (`generate_lfm3.agc`) share, over a key-value cache
## (`transformers/DESIGN.md`), the design of `asr/`'s decoder (`asr/asr.agc`); the cache helpers are shared in
## `nn/kv.agc`. A model gives a `Stepper`: its sizes, its token table, `traced` and its `stepped` function, the graph of
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
## A model with layers that keep a state (LFM3's short convolutions) says so in its `Stepper` (`states`, `taps`): the
## states are tensors of a size that does not depend on the length, new at every step, and a prompt is given the
## `Delta.Padding` of its convolutions, so that the padding rows leave the state as the real ones did.
##
## Batch of one. A prompt is fed in one step (`step` at length 0), then one token at a time. A `Generator` belongs to
## one task at a time, and `close` ends it before the weights its graphs read are freed.

import Padded from "pkg/nn/kv"
import Delta from "pkg/nn/delta"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Lift from "lift"
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

## What a model gives the decoder, `M` being its type. `stepped(model, request)` is the graph of a step: its outputs are
## the head (the logits `[vocab]`, the choice `[1]` or the candidates, by `request.mode`), then the keys and the values
## of every layer with a cache, then its states. `traced` is the model with its tensors as `Trace` constants.
struct Stepper[M]
  ## The layers with a cache of keys and values.
  layers: Int
  ## The positions of the model (`max_position_embeddings`).
  limit: Int
  vocab: Int
  ## The token table, which tells the backend of the model.
  table: Tensor.Tensor
  ## How many state tensors a step takes and gives, after the keys and the values (0 for a model with none).
  states: Int = 0
  ## The taps of the convolution of a model with states: a prompt (more than one row) is given a `Delta.Padding` for
  ## them. 0 for a model with no convolution.
  taps: Int = 0
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
  ## The states of a model that has them, new tensors at every step.
  states: List[Tensor.Tensor] = []
  length: Int
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

## The empty cache.
def start() -> Cache = Cache(keys: [], values: [], length: 0)

## Feeds `ids` after the tokens of `cache`: a prompt of any length when the cache is empty, one id after that. The cache
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

## Decoding from the prompt `ids`: at most `max_new` ids, stopping before a stop id (`stop`, such as `<|im_end|>`).
## Greedy (the choice runs in the graph) unless `sampling` says how to draw (`Stepping.generate`): then each step reads
## the largest logits and `Sample.pick_top` draws among them. The prompt and the new ids must fit
## `max_position_embeddings`.
def generate[M](
  g: Core[M],
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

## What `Stepping.generate` needs of `g`.
def decoder[M](g: Core[M]) -> Stepping.Decoder[Cache]
  Stepping.Decoder(
    usable: g.usable,
    limit: g.stepper.limit,
    empty: start(),
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

  backend = Tensor.backend(s.table)
  made = fed_inputs(s, ids, cache.length, width, room, backend)
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
    _ -> eager_step(g, cache, fed, room, mode)
  end

  Lift.close(fed)
  Lift.close(cache.states)
  # A step in place keeps the tensors of `cache` as the ones of the new cache; any other step is done with them (a
  # tensor closed twice is fine, as when the step failed after it closed its outputs).
  if not in_place or not outs.ok?
    Lift.close(held)
  end

  r = try outs
  ok(Advanced(head: r.head, cache: r.cache.with(length: cache.length + n)))
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
  room: Int,
  reach: Int,
  mode: Stepping.Head,
  in_place: Bool,
) -> Result[Advanced]
  s = g.stepper
  args = fed + cache.keys + cache.values + cache.states
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
    ->(xs: List[Tensor.Tensor]) {
      s.stepped.call(
        s.traced.call(g.model),
        Request(usable: g.usable, xs: xs, room: room, reach: reach, mode: mode, in_place: in_place, compiled: true),
      )
    },
  )

  outs = try Trace.run(plan, args)
  head = Tensor.to_list(outs[0])
  Lift.close([outs[0]])
  if not head.ok?
    Lift.close(outs.drop(1))
  end

  ok(Advanced(head: try head, cache: cache_of(s, outs.drop(1))))
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

## The same graph, run now and in an arena that keeps the new cache alone.
def eager_step[M](
  g: Core[M],
  cache: Cache,
  fed: List[Tensor.Tensor],
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
      ),
    )

    ok(Advanced(head: try Tensor.to_list(outs[0]), cache: cache_of(s, outs.drop(1))))
  end

  ran
end
