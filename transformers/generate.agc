## Greedy or sampled generation with Qwen3 and a key-value cache (`transformers/DESIGN.md`): `kv_decoder.agc` has the
## decoder (the cache, the compiled steps, the pick in the graph) and this file the graph of one step of Qwen3 and
## Qwen2.5. On `Ggml` the cache is `F16`, half the memory and of the bytes a decode step reads (the logits drift by
## about 5e-3 on values of 10 to 30); on the other backends it is `F32`. A `Generator` belongs to one task at a time,
## and `close` ends it before `Qwen3.close_model` frees the weights its graphs read.

import Padded from "pkg/nn/kv"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import KvDecoder from "kv_decoder"
import Lift from "lift"
import Placement from "placement"
import Qwen3 from "qwen3"
import Sample from "sample"
import Stepping from "stepping"

## A model and the compiled steps it has run (on `Ggml`), which every sequence of the model shares: the fields of a
## `KvDecoder.Core` (see there), which `core` makes of it.
struct Generator
  model: Qwen3.Qwen3
  plans: Channel[Stepping.Plans]
  ## See `KvDecoder.Core`.
  usable: Int
  whole: Bool
  ## Where the cache is kept (`Placement.Kv`).
  kv: Placement.Kv
end

type Cache = KvDecoder.Cache

type Scored = KvDecoder.Scored

## One greedy decode step for two sequences in one graph: the id after each cache, and the cache after the step.
struct Paired
  a: Stepping.Picked[Cache]
  b: Stepping.Picked[Cache]
end

## The generator of `m`. `usable` is how many ids it may answer (see `Generator.usable`); the whole vocabulary when it
## is 0.
def open(m: Qwen3.Qwen3, usable: Int = 0, kv: Placement.Kv = Placement.Kv.Device) -> Result[Generator]
  c = try KvDecoder.open(m, stepper(m, kv), usable)
  ok(Generator(model: m, plans: c.plans, usable: c.usable, whole: false, kv: kv))
end

## How many steps `g` has compiled: a shape it has seen compiles no more.
def compiles(g: Generator) -> Int = KvDecoder.compiles(core(g))

## Ends the pins of the compiled steps of `g`, which a later call compiles again. Call it before
## `Qwen3.close_model(g.model)`.
def close(g: Generator) -> Void
  KvDecoder.close(core(g))
end

## The empty cache, or with `sight` the one of a prompt with image features (`KvDecoder.start`): the rows are spliced
## into the embeddings of the image tokens, and the positions stay the index of each token (`KvDecoder.plain_sight`).
def start(sight: Optional[KvDecoder.Sight] = none) -> Cache = KvDecoder.start(sight)

## Feeds `ids` after the tokens of `cache`: a prompt of any length when the cache is empty, one id after that. The cache
## that `step` takes is dead afterwards, even when the step fails (`KvDecoder.step`).
def step(g: Generator, cache: Cache, ids: List[Int]) -> Result[Stepping.Picked[Cache]]
  KvDecoder.step(core(g), cache, ids)
end

## `step` with the logits of the last token fed, `[vocab_size]`, in place of the choice.
def step_logits(g: Generator, cache: Cache, ids: List[Int]) -> Result[Scored]
  KvDecoder.step_logits(core(g), cache, ids)
end

## `step_logits` for the `k` largest logits only: their ids and values, largest first, and the cache
## (`KvDecoder.step_top`).
def step_top(g: Generator, cache: Cache, ids: List[Int], k: Int) -> Result[Stepping.Candidates[Cache]]
  KvDecoder.step_top(core(g), cache, ids, k)
end

## One greedy decode step for two sequences in one graph. Both caches hold a prompt already: a prompt runs alone, with `step`. Each tower of the graph is the `stepped` of one
## row, with its own inputs, positions, mask and cache, so one row never reads the keys of the other; the weights are
## shared, and each tower is the same computation a step of that row alone runs. The step runs one graph, compiled
## once per pair of shapes and kept in the plans of `g` like any other step. The caches the step takes are dead
## afterwards, even when the step fails, as with `step`.
def step_pair(g: Generator, a: Cache, a_id: Int, b: Cache, b_id: Int) -> Result[Paired]
  c = core(g)
  s = c.stepper
  if a.length == 0 or b.length == 0
    finish(a)
    finish(b)
    return error("step_pair steps two decodes: a cache is empty, prefill it with step first")
  end

  made_a = prepared(s, a, a_id)
  if not made_a.ok?
    finish(a)
    finish(b)
    return error(made_a.message)
  end

  row_a = made_a.unwrap!
  made_b = prepared(s, b, b_id)
  if not made_b.ok?
    Lift.close(row_a.fed)
    finish(a)
    finish(b)
    return error(made_b.message)
  end

  row_b = made_b.unwrap!
  outs = case s.backend
    Tensor.Backend.Ggml(_) -> compiled_pair(c, row_a, row_b)
    Tensor.Backend.Xla(_) -> compiled_pair(c, row_a, row_b)
    _ -> eager_pair(c, row_a, row_b)
  end

  Lift.close(row_a.fed + row_b.fed)
  # A paired step never writes in place: the caches it was given are dead, even when the step fails (a tensor closed
  # twice is fine, as when the step failed after it closed its outputs).
  Lift.close(a.keys + a.values + b.keys + b.values)
  found = try outs
  try counted_pair(s, found)
  n = 1 + 2 * s.layers
  fa = Tensor.to_list(found[0])
  fb = Tensor.to_list(found[n])
  if not fa.ok? or not fb.ok?
    Lift.close(found)
    return error("step_pair: the picks are floats")
  end

  Lift.close([found[0], found[n]])
  ka = hosted_pair(s, found.drop(1).take(n - 1))
  if not ka.ok?
    Lift.close(found.drop(n + 1))
    return error(ka.message)
  end

  kb = hosted_pair(s, found.drop(n + 1))
  if not kb.ok?
    Lift.close(ka.unwrap!)
    return error(kb.message)
  end

  cache_a = paired_cache(s, a, ka.unwrap!)
  cache_b = paired_cache(s, b, kb.unwrap!)
  ida = Stepping.chosen(fa.unwrap!, c.usable)
  if not ida.ok?
    finish(cache_a)
    finish(cache_b)
    return error(ida.message)
  end

  idb = Stepping.chosen(fb.unwrap!, c.usable)
  if not idb.ok?
    finish(cache_a)
    finish(cache_b)
    return error(idb.message)
  end

  ok(Paired(a: Stepping.Picked(id: ida.unwrap!, cache: cache_a), b: Stepping.Picked(id: idb.unwrap!, cache: cache_b)))
end

## Ends a sequence: closes the tensors of `cache`.
def finish(cache: Cache) -> Void
  KvDecoder.finish(cache)
end

## Decoding from the prompt `ids`: at most `max_new` ids, stopping before a stop id (`stop`, such as `<|im_end|>`),
## greedy or sampled (`Stepping.generate`). The prompt and the new ids must fit `max_position_embeddings`. `sight` is
## the image features of the prompt.
def generate(
  g: Generator,
  ids: List[Int],
  max_new: Int,
  stop: List[Int] = [],
  sampling: Optional[Sample.Sampling] = none,
  sight: Optional[KvDecoder.Sight] = none,
) -> Result[List[Int]]
  KvDecoder.generate(core(g), ids, max_new, stop, sampling, sight)
end

private

## One row of a paired step: its inputs and the room they leave, with the rows the cache already has (for the plan
## key). The cache is untouched: the caller finishes it when this is an error.
struct Row
  fed: List[Tensor.Tensor]
  keys: List[Tensor.Tensor]
  values: List[Tensor.Tensor]
  room: Int
  rows: Int
end

def core(g: Generator) -> KvDecoder.Core[Qwen3.Qwen3]
  KvDecoder.Core(model: g.model, stepper: stepper(g.model, g.kv), plans: g.plans, usable: g.usable, whole: g.whole)
end

## What the decoder needs of `m`.
def stepper(m: Qwen3.Qwen3, kv: Placement.Kv) -> KvDecoder.Stepper[Qwen3.Qwen3]
  KvDecoder.Stepper(
    layers: m.config.num_hidden_layers,
    limit: m.config.max_position_embeddings,
    vocab: m.config.vocab_size,
    backend: m.device,
    sighted: true,
    rotated: false,
    kv: kv,
    traced: ->(p: Qwen3.Qwen3) { Qwen3.traced(p) },
    stepped: ->(p: Qwen3.Qwen3, req: KvDecoder.Request) { stepped(p, req) },
  )
end

## The step: `xs` are the ids `[width]`, their positions, the one-hot `[1, width]` of the row whose
## prediction is wanted, and per layer the keys then the values. The outputs are the head (the logits `[vocab_size]`,
## or the choice `[1]`), then the keys and the values of every layer. The cache it leaves has `room` rows, of which
## the attention reads the first `reach` (every row the new tokens may see is among them).
def stepped(m: Qwen3.Qwen3, req: KvDecoder.Request) -> Result[List[Tensor.Tensor]]
  usable = req.usable
  xs = req.xs
  room = req.room
  reach = req.reach
  mode = req.mode
  compiled = req.compiled
  layers = m.config.num_hidden_layers
  width = Tensor.shape(xs[0])[0]
  d = m.config.hidden_size
  slots = Padded.Slots(positions: xs[1], capacity: room)
  h = try KvDecoder.spliced(try Tensor.reshape(try Nn.embedding(m.tokens, xs[0]), [1, width, d]), req)
  mask = try Padded.padded_mask(slots.with(capacity: reach), Tensor.dtype(h))
  reading = try KvDecoder.reading(req, mask)
  keys: List[Tensor.Tensor] = []
  values: List[Tensor.Tensor] = []
  for i in 0...layers
    old: Optional[KvDecoder.Kv] = none
    if xs.size > 3
      old = some(KvDecoder.Kv(key: xs[3 + i], value: xs[3 + layers + i]))
    end

    r = try layer(m.stack.blocks[i], h, reading, old, req)
    h = r.hidden
    keys = keys.push(try Stepping.cached(r.key, compiled))
    values = values.push(try Stepping.cached(r.value, compiled))
  end

  last = try Tensor.reshape(try Tensor.matmul(xs[2], try Tensor.reshape(h, [width, d])), [1, 1, d])
  case m.stack.final_norm
    some(n) -> last = try Nn.layer_norm(n, last)
    none -> nil
  end

  logits = try Stepping.masked(try Tensor.reshape(try Nn.dense(m.head, last), [m.config.vocab_size]), usable)
  head = try Stepping.headed(logits, mode)

  ok([head] + keys + values)
end

## One block over the new tokens `h` `[1, width, d]`, as `Nn.encoder_block` with
## `norm_first`, the keys and values of the cache in the place of the block's own.
def layer(
  b: Nn.EncoderBlock,
  h: Tensor.Tensor,
  reading: KvDecoder.Reading,
  old: Optional[KvDecoder.Kv],
  req: KvDecoder.Request,
) -> Result[KvDecoder.Layer]
  x = try Nn.layer_norm(b.norm1, h)
  kept = try KvDecoder.attend_cached(b.attention, x, old, reading, req)
  mid = try Tensor.add(h, kept.answer)
  out = try Tensor.add(mid, try Nn.mlp(b.mlp, try Nn.layer_norm(b.norm2, mid)))
  ok(KvDecoder.Layer(hidden: out, key: kept.key, value: kept.value))
end

def prepared(s: KvDecoder.Stepper[Qwen3.Qwen3], cache: Cache, id: Int) -> Result[Row]
  bad = Stepping.check_ids(s.limit, s.vocab, cache.length, [id])
  return error(bad.message) if not bad.ok?

  room = Padded.capacity(cache.length + 1, s.limit)
  fed = try Stepping.inputs([id], cache.length + cache.delta, 1, room, s.backend)
  rows = 0
  if not cache.keys.empty?
    rows = Tensor.shape(cache.keys[0])[2]
  end

  ok(Row(fed: fed, keys: cache.keys, values: cache.values, room: room, rows: rows))
end

## The cache of one tower of a paired step after it: the keys, then the values, one token longer, at the same delta.
def paired_cache(s: KvDecoder.Stepper[Qwen3.Qwen3], old: Cache, kept: List[Tensor.Tensor]) -> Cache
  KvDecoder.Cache(keys: kept.take(s.layers), values: kept.drop(s.layers), length: old.length + 1, delta: old.delta)
end

## An error when a paired step gave other tensors than two heads with the keys and the values of every layer each.
def counted_pair(s: KvDecoder.Stepper[Qwen3.Qwen3], outs: List[Tensor.Tensor]) -> Result[Void]
  want = 2 * (1 + 2 * s.layers)
  return ok() if outs.size == want
  error("a paired step gave #{outs.size} tensors, two heads with the keys and values of #{s.layers} layers each")
end

## The keys and values of one tower of a paired step in the memory `Stepper.kv` says: a tensor that is somewhere else
## is copied and the copy replaces it (the old one is closed), as `KvDecoder.hosted` keeps the one row of one step.
def hosted_pair(s: KvDecoder.Stepper[Qwen3.Qwen3], outs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  kept: List[Tensor.Tensor] = []
  for i in 0...outs.size
    t = outs[i]
    on = Tensor.backend(t)
    if Placement.cache_backend(s.kv, on) == on
      kept = kept.push(t)
      next
    end

    moved = Tensor.to_device(t, Tensor.Device.Cpu)
    if not moved.ok?
      Lift.close(kept + outs.drop(i))
      return error(moved.message)
    end

    Lift.close([t])
    kept = kept.push(moved.unwrap!)
  end

  ok(kept)
end

## One paired step on `Ggml` or `Xla`: the two towers in one compiled graph, kept in the plans of `c` under a key of
## its own (no single-row key has the `pair` prefix). Each tower reads the room of its row, as a step that is not in
## place does.
def compiled_pair(c: KvDecoder.Core[Qwen3.Qwen3], a: Row, b: Row) -> Result[List[Tensor.Tensor]]
  s = c.stepper
  mode = Stepping.Head.Pick
  key = (
    "pair " +
    Stepping.plan_key(1, a.rows, a.room, a.room, mode) +
    " " +
    Stepping.plan_key(1, b.rows, b.room, b.room, mode)
  )

  na = a.fed.size + a.keys.size + a.values.size
  args = a.fed + a.keys + a.values + b.fed + b.keys + b.values
  plan = try Stepping.planned(
    c.plans,
    key,
    [],
    [],
    args.map { |t| Trace.spec_of(t) },
    ->(xs: List[Tensor.Tensor]) { fused(s, c.model, xs, a, b, na, c.usable, Stepping.half_cache?(s.backend)) },
  )

  Trace.run(plan, args)
end

## The two towers over `xs`: the head, then the keys and the values of every layer, of row `a`, then of row `b`.
def fused(
  s: KvDecoder.Stepper[Qwen3.Qwen3],
  model: Qwen3.Qwen3,
  xs: List[Tensor.Tensor],
  a: Row,
  b: Row,
  na: Int,
  usable: Int,
  compiled: Bool,
) -> Result[List[Tensor.Tensor]]
  seen = s.traced.call(model)
  out_a = try stepped(
    seen,
    KvDecoder.Request(
      usable: usable,
      xs: xs.take(na),
      room: a.room,
      reach: a.room,
      mode: Stepping.Head.Pick,
      in_place: false,
      compiled: compiled,
    ),
  )

  out_b = try stepped(
    seen,
    KvDecoder.Request(
      usable: usable,
      xs: xs.drop(na),
      room: b.room,
      reach: b.room,
      mode: Stepping.Head.Pick,
      in_place: false,
      compiled: compiled,
    ),
  )

  ok([out_a[0]] + out_a.drop(1) + [out_b[0]] + out_b.drop(1))
end

## One paired step elsewhere: the two towers run now, in an arena that keeps the new caches alone.
def eager_pair(c: KvDecoder.Core[Qwen3.Qwen3], a: Row, b: Row) -> Result[List[Tensor.Tensor]]
  ran: Result[List[Tensor.Tensor]] = Native.arena do
    out_a = try stepped(
      c.model,
      KvDecoder.Request(
        usable: c.usable,
        xs: a.fed + a.keys + a.values,
        room: a.room,
        reach: a.room,
        mode: Stepping.Head.Pick,
        in_place: false,
        compiled: false,
      ),
    )

    out_b = try stepped(
      c.model,
      KvDecoder.Request(
        usable: c.usable,
        xs: b.fed + b.keys + b.values,
        room: b.room,
        reach: b.room,
        mode: Stepping.Head.Pick,
        in_place: false,
        compiled: false,
      ),
    )

    ok([out_a[0]] + out_a.drop(1) + [out_b[0]] + out_b.drop(1))
  end

  ran
end
