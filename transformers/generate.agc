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

## The widest batch `step_batch` runs: one compiled graph over at most this many rows.
MAX_BATCH = 10

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

## The two rows of the compatibility wrapper `step_pair`: the id after each cache, and the cache after the step.
struct Paired
  a: Stepping.Picked[Cache]
  b: Stepping.Picked[Cache]
end

## A row of a batch: a decode-ready cache and the id to feed after it, as `step` takes them.
struct BatchRow
  cache: Cache
  id: Int
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

## Whether `step_batch` runs on `backend`: the reference backend and every ggml device — the CPU engine, a card, and
## the software Vulkan device (Mesa's llvmpipe) a CI runner without one has. The device is the ggml device the model was
## opened on, and a batch runs on that one: software or card is a property of the device, named in its log, never a path
## this code takes. XLA and libtorch step their rows alone, as they did before batching existed, until one of them is
## measured and added here. `step_batch` asks this of its backend first, before it reads any cache: a backend left out
## here is refused with every cache still the caller's, so a direct caller keeps its rows and a scheduler asks this
## once per round, as `serving.agc` does.
def pairable(backend: Tensor.Backend) -> Bool
  case backend
    Tensor.Backend.Reference -> true
    Tensor.Backend.Ggml(_) -> true
    _ -> false
  end
end

## The widest batch `step_batch` runs on `backend`: `MAX_BATCH` on the reference backend and ggml's CPU engine, 2 on
## a ggml Vulkan device (the width of its CI smoke, where a wider graph is not measured), and 0 where the rows step
## alone. `step_batch` refuses a wider batch before it reads any cache, and `serving.agc` clamps its groups to this.
def batch_width(backend: Tensor.Backend) -> Int
  case backend
    Tensor.Backend.Reference -> MAX_BATCH
    Tensor.Backend.Ggml(Tensor.Device.Cpu) -> MAX_BATCH
    Tensor.Backend.Ggml(Tensor.Device.Vulkan(_)) -> 2
    _ -> 0
  end
end

## Whether the steps of `g` may be batched: `pairable` of its backend.
def pairs(g: Generator) -> Bool = pairable(core(g).stepper.backend)

## The widest batch the steps of `g` take: `batch_width` of its backend.
def batch_limit(g: Generator) -> Int = batch_width(core(g).stepper.backend)

## One greedy decode step for the decode-ready `rows` in one graph, on the backends `pairable` names. Every cache holds
## a prompt already: a prompt runs alone, with `step`. The step is one forward over the batch: the projections, the
## attention and the MLP of every layer run on the rows `[n, 1, hidden]` at once, with the rows' own positions,
## additive mask `[n, 1, 1, room]` and cache — each row writes its new key and value into its own slice of the batch's
## cache `[n, kv_heads, room, head_dim]` and its attention reads only its own rows, so one row never reads the keys of
## another; the weights are shared, and each row's answer is the same a step of that row alone gives. A row may have
## its own prompt length, positions, capacity and cache: the batch is as wide as the largest room it pads to, and the
## cache each row gets back is cut to that row's own rows.
##
## On a ggml device the step is one graph, compiled once per batch of shapes and kept in the plans of `g` like any
## other step, so one batch round is one execution of the backend; on the other backends the same forward runs eagerly
## in an arena. The caches the step takes are dead afterwards, even when the step fails, as with `step`, and each cache
## it gives is the tensors of that row alone. `rows` is at most `batch_width` of its backend (`MAX_BATCH` on the
## reference backend and ggml's CPU engine, 2 on a ggml Vulkan device); a wider batch, paged KV with physical slots,
## token budgets, chunked prefill and batching on another device are not here (ml#92), and a prompt, a
## row that decodes alone and the pick of one row stay in `step`. On a backend `pairable` leaves out the step is
## refused before any cache is read, so that refusal leaves every cache to its caller; every other error, of a batch
## the backend accepted, consumes every cache as a failed `step` does. A ggml device batches as the device it is: the
## CPU engine, a card and the software Vulkan device all run the same forward, and the tests pin each row's ids to that
## device's own singleton run.
def step_batch(g: Generator, rows: List[BatchRow]) -> Result[List[Stepping.Picked[Cache]]]
  c = core(g)
  s = c.stepper
  # The capability of the backend first, before any cache is checked, prepared or finished: no work is done for a
  # backend that cannot run the step, and a caller that handed over live caches still owns all of them.
  if not pairable(s.backend)
    return error(
      "step_batch batches on the reference backend and ggml's devices only: this #{s.backend} steps its rows alone",
    )
  end

  width = batch_width(s.backend)
  return error("step_batch batches at most #{width} rows on this backend, got #{rows.size}") if rows.size > width

  return error("step_batch steps at least one decode") if rows.empty?

  empty = false
  for r in rows
    if r.cache.length == 0
      empty = true
    end
  end

  if empty
    for r in rows
      finish(r.cache)
    end

    return error("step_batch steps decodes: a cache is empty, prefill it with step first")
  end

  built: List[Row] = []
  for r in rows
    made = prepared(s, r.cache, r.id)
    if not made.ok?
      for b in built
        Lift.close(b.fed)
      end

      for q in rows
        finish(q.cache)
      end

      return error(made.message)
    end

    built = built.push(made.unwrap!)
  end

  outs = case s.backend
    Tensor.Backend.Ggml(_) -> compiled_batch(c, built)
    _ -> eager_batch(c, built)
  end

  for b in built
    Lift.close(b.fed)
  end

  # A batch step never writes in place: the caches it was given are dead, even when the step fails (a tensor closed
  # twice is fine, as when the step failed after it closed its outputs).
  for r in rows
    Lift.close(r.cache.keys + r.cache.values)
  end

  found = try outs
  size = 1 + 2 * s.layers
  counted = counted_batch(s, found, rows.size)
  if not counted.ok?
    Lift.close(found)
    return error(counted.message)
  end

  ids: List[Int] = []
  heads: List[Tensor.Tensor] = []
  for r in 0...rows.size
    head = found[r * size]
    heads = heads.push(head)
    pick = Tensor.to_list(head)
    if not pick.ok?
      Lift.close(heads)
      Lift.close(found)
      return error("step_batch: the picks are floats")
    end

    id = Stepping.chosen(pick.unwrap!, c.usable)
    if not id.ok?
      Lift.close(heads)
      Lift.close(found)
      return error(id.message)
    end

    ids = ids.push(id.unwrap!)
  end

  Lift.close(heads)
  made: List[List[Tensor.Tensor]] = []
  for r in 0...rows.size
    hosted = hosted_batch(s, found.drop(r * size + 1).take(2 * s.layers))
    if not hosted.ok?
      for m in made
        Lift.close(m)
      end

      Lift.close(found)
      return error(hosted.message)
    end

    made = made.push(hosted.unwrap!)
  end

  out: List[Stepping.Picked[Cache]] = []
  for r in 0...rows.size
    out = out.push(Stepping.Picked(id: ids[r], cache: stepped_cache(s, rows[r].cache, made[r])))
  end

  ok(out)
end

## One greedy decode step for two sequences in one graph: `step_batch` of the two `BatchRow`s, kept for the callers of
## the first paired step. The refusal of a backend that does not batch leaves both caches with the caller, as it did.
def step_pair(g: Generator, a: Cache, a_id: Int, b: Cache, b_id: Int) -> Result[Paired]
  made = try step_batch(g, [BatchRow(cache: a, id: a_id), BatchRow(cache: b, id: b_id)])
  ok(Paired(a: made[0], b: made[1]))
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

## One row of a batch step: its inputs and the room they leave, with the rows the cache already has (for the plan
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

## The cache of one row of a batch step after it: the keys, then the values, one token longer, at the same delta.
def stepped_cache(s: KvDecoder.Stepper[Qwen3.Qwen3], old: Cache, kept: List[Tensor.Tensor]) -> Cache
  KvDecoder.Cache(keys: kept.take(s.layers), values: kept.drop(s.layers), length: old.length + 1, delta: old.delta)
end

## An error when a batch step gave other tensors than one head per row and the keys and values of every layer each.
def counted_batch(s: KvDecoder.Stepper[Qwen3.Qwen3], outs: List[Tensor.Tensor], rows: Int) -> Result[Void]
  want = rows * (1 + 2 * s.layers)
  return ok() if outs.size == want
  error("a batch step of #{rows} rows gave #{outs.size} tensors, expected #{want}")
end

## The keys and values of one row of a batch step in the memory `Stepper.kv` says: a tensor that is somewhere else
## is copied and the copy replaces it (the old one is closed), as `KvDecoder.hosted` keeps the one row of one step.
def hosted_batch(s: KvDecoder.Stepper[Qwen3.Qwen3], outs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
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

## One batch step on `Ggml`: one compiled graph over the rows, kept in the plans of `c` under a key of its own (no
## single-row key has the `batch` prefix, and the rows' shapes are in the key in order). The graph takes the rows as a
## step that is not in place does and makes the batch's tensors itself (`batch_of_rows`), so a round binds only the
## rows' own tensors: one execution of the backend, nothing eager around it.
def compiled_batch(c: KvDecoder.Core[Qwen3.Qwen3], rows: List[Row]) -> Result[List[Tensor.Tensor]]
  s = c.stepper
  mode = Stepping.Head.Pick
  key = "batch " + rows.map { |r| Stepping.plan_key(1, r.rows, r.room, r.room, mode) }.join(" ")
  args: List[Tensor.Tensor] = []
  for r in rows
    args = args + r.fed + r.keys + r.values
  end

  plan = try Stepping.planned(
    c.plans,
    key,
    [],
    [],
    args.map { |t| Trace.spec_of(t) },
    ->(xs: List[Tensor.Tensor]) {
      batch_over(s, c.model, xs, rows, s.layers, c.usable, Stepping.half_cache?(s.backend))
    },
  )

  Trace.run(plan, args)
end

## One batch step over `xs`, the rows one after the other (each row's fed, keys and values): the rows' ids, positions
## and picks become one batch and the caches one (`batch_of_rows`), one forward runs over the batch (`batched`), and
## its outputs are the rows apart (`batch_outputs`). On a device the graph fuses all of it; elsewhere the same ops run
## eagerly.
def batch_over(
  s: KvDecoder.Stepper[Qwen3.Qwen3],
  model: Qwen3.Qwen3,
  xs: List[Tensor.Tensor],
  rows: List[Row],
  layers: Int,
  usable: Int,
  compiled: Bool,
) -> Result[List[Tensor.Tensor]]
  room = rows.map { |r| r.room }.max.unwrap!
  fed = try batch_of_rows(xs, layers, room)
  outs = try batched(s.traced.call(model), fed, usable, compiled)
  batch_outputs(outs, layers, compiled, rows.map { |r| r.room })
end

## The rows of a batch step as one batch: the ids `[n]`, the positions `[n]` (in row order) and the picks `[n, 1]`,
## then per layer the keys and the values of the batch's cache `[n, kv_heads, room, head_dim]`, each row zero-padded to
## the `room` of the batch (the larger room of the rows) and stacked.
def batch_of_rows(xs: List[Tensor.Tensor], layers: Int, room: Int) -> Result[List[Tensor.Tensor]]
  stride = 3 + 2 * layers
  n = xs.size / stride
  ids: List[Tensor.Tensor] = []
  positions: List[Tensor.Tensor] = []
  picks: List[Tensor.Tensor] = []
  parts: List[List[Tensor.Tensor]] = []
  for r in 0...n
    base = r * stride
    ids = ids.push(xs[base])
    positions = positions.push(xs[base + 1])
    picks = picks.push(xs[base + 2])
    parts = parts.push(xs.drop(base + 3).take(2 * layers))
  end

  fed = [try Tensor.concat(ids, 0), try Tensor.concat(positions, 0), try Tensor.concat(picks, 0)]
  caches = try batch_caches(parts, room)
  ok(fed + caches)
end

## The keys and the values of one layer (`parts` holds each row's keys then values) as the batch's
## `[n, kv_heads, room, head_dim]`, each as F32 (the write and the attention read F32, as a step of one row reads its
## widened cache), zero-padded to the `room` of the batch and stacked.
def batch_caches(parts: List[List[Tensor.Tensor]], room: Int) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for i in 0...parts[0].size
    padded: List[Tensor.Tensor] = []
    for part in parts
      padded = padded.push(try padded_to(try as_f32(part[i]), room))
    end

    out = out.push(try Tensor.concat(padded, 0))
  end

  ok(out)
end

## The cache tensor `t` `[1, kv_heads, rows, head_dim]` with the rows past its `room` of zero: the rows it already has
## when it is not shorter.
def padded_to(t: Tensor.Tensor, room: Int) -> Result[Tensor.Tensor]
  shape = Tensor.shape(t)
  return ok(t) if shape[2] == room
  gap = try Tensor.zeros([1, shape[1], room - shape[2], shape[3]], Tensor.dtype(t), Tensor.backend(t))
  Tensor.concat([t, gap], 2)
end

## One forward over the batch. `xs` are the ids `[n]`, their positions `[n]` and the picks `[n, 1]`, then per layer
## the keys and the values of the batch's cache `[n, kv_heads, room, head_dim]` (each row its own tokens: the rows past
## its length are zero, and a row never reads the keys of another). The outputs are the picks `[1]` in row order, then
## per layer the keys and the values of the batch's cache.
def batched(m: Qwen3.Qwen3, xs: List[Tensor.Tensor], usable: Int, compiled: Bool) -> Result[List[Tensor.Tensor]]
  layers = m.config.num_hidden_layers
  d = m.config.hidden_size
  tokens = xs[0]
  positions = xs[1]
  pick = xs[2]
  n = Tensor.shape(tokens)[0]
  room = Tensor.shape(xs[3])[2]
  keys_in = xs.drop(3).take(layers)
  values_in = xs.drop(3 + layers).take(layers)
  h = try Tensor.reshape(try Nn.embedding(m.tokens, tokens), [n, 1, d])
  mask = try Tensor.reshape(
    try Padded.padded_mask(Padded.Slots(positions: positions, capacity: room), Tensor.dtype(h)),
    [n, 1, 1, room],
  )

  spread = try batch_spread(positions, room, Tensor.dtype(h))
  keys_out: List[Tensor.Tensor] = []
  values_out: List[Tensor.Tensor] = []
  for i in 0...layers
    r = try batch_layer(m.stack.blocks[i], h, keys_in[i], values_in[i], spread, mask, positions)
    h = r.hidden
    keys_out = keys_out.push(r.key)
    values_out = values_out.push(r.value)
  end

  last = try Tensor.batch_matmul(try Tensor.reshape(pick, [n, 1, 1]), h)
  case m.stack.final_norm
    some(norm) -> last = try Nn.layer_norm(norm, last)
    none -> nil
  end

  logits = try Tensor.reshape(try Nn.dense(m.head, last), [n, m.config.vocab_size])
  heads = try batch_picks(logits, usable)
  ok(heads + keys_out + values_out)
end

## One block of the batch over the new rows `h` `[n, 1, d]`, as `Nn.encoder_block` with `norm_first`, the batch's cache in
## the place of the block's own: the rows are projected per row, the keys and values of each row are written into its
## own slice of the cache (a product with the `spread` of its position, exact in F32 as `Padded.written` is), and each
## row's query attends over its own rows of the cache under the rows' additive `mask`.
def batch_layer(
  b: Nn.EncoderBlock,
  h: Tensor.Tensor,
  old_keys: Tensor.Tensor,
  old_values: Tensor.Tensor,
  spread: Tensor.Tensor,
  mask: Tensor.Tensor,
  positions: Tensor.Tensor,
) -> Result[KvDecoder.Layer]
  x = try Nn.layer_norm(b.norm1, h)
  a = b.attention
  q = try split_heads(try Nn.dense(a.query, x), a.heads, a.query_norm)
  kv_heads = a.kv_heads.unwrap_or(a.heads)
  k = try split_heads(try Nn.dense(a.key, x), kv_heads, a.key_norm)
  v = try split_heads(try Nn.dense(a.value, x), kv_heads, none)
  case a.rope
    some(r) ->
      q = try rotated_batch(q, r, positions)
      k = try rotated_batch(k, r, positions)
    none -> nil
  end

  shape = Tensor.shape(old_keys)
  wide = try Tensor.broadcast_to(spread, shape)
  key = try Tensor.add(try as_f32(old_keys), try Tensor.mul(wide, try Tensor.broadcast_to(k, shape)))
  value = try Tensor.add(try as_f32(old_values), try Tensor.mul(wide, try Tensor.broadcast_to(v, shape)))
  seen = try Nn.attend(a, q, key, value, some(mask))
  mid = try Tensor.add(h, seen)
  out = try Tensor.add(mid, try Nn.mlp(b.mlp, try Nn.layer_norm(b.norm2, mid)))
  ok(KvDecoder.Layer(hidden: out, key: key, value: value))
end

## `[batch, 1, heads * head_width]` to `[batch, heads, 1, head_width]`: the heads split by head, each head normed
## (`norm`) before the permute, as `Nn.split_heads` of a step of one row.
def split_heads(projected: Tensor.Tensor, heads: Int, norm: Optional[Nn.LayerNorm]) -> Result[Tensor.Tensor]
  shape = Tensor.shape(projected)
  split = try Tensor.reshape(projected, [shape[0], shape[1], heads, shape[2] / heads])
  case norm
    some(n) -> split = try Nn.layer_norm(n, split)
    none -> nil
  end

  Tensor.permute(split, [0, 2, 1, 3])
end

## The heads of the batch `x` `[n, heads, 1, head_width]` turned by the rows' positions: the rows are swapped with the
## seq, which makes `[1, heads, n, head_width]` whose seq is the batch with one position per row, and the same
## permutation puts the rows back. A rope of another kind (`Nn.Rope.zero_tail`, `sections`) is an error: the batch
## step is Qwen3's plain rope.
def rotated_batch(x: Tensor.Tensor, r: Nn.Rope, positions: Tensor.Tensor) -> Result[Tensor.Tensor]
  if r.zero_tail.some? or r.sections.some?
    return error("a batch step turns by a plain rope: zero_tail and sections batch no rows")
  end

  brought = try Tensor.permute(x, [2, 1, 0, 3])
  turned = try Tensor.rope(brought, positions, base: r.base, dims: r.dims, neox: r.neox)
  Tensor.permute(turned, [2, 1, 0, 3])
end

## The one-hot of the rows' positions over the `room` of the batch: `[n, 1, room, 1]`, 1 where the key's row is that
## row's new token, 0 elsewhere. A product with it writes each row's token into its own slice of the cache, exact in
## F32 as `Padded.written` is.
def batch_spread(positions: Tensor.Tensor, room: Int, dtype: Tensor.Dtype) -> Result[Tensor.Tensor]
  n = Tensor.shape(positions)[0]
  grid = [n, room]
  backend = Tensor.backend(positions)
  queries = try Tensor.broadcast_to(try Tensor.reshape(positions, [n, 1]), grid)
  rows = try Tensor.broadcast_to(
    try Tensor.reshape(try Tensor.arange(0, room, 1, Tensor.Dtype.I32, backend), [1, room]),
    grid,
  )

  Tensor.reshape(try Tensor.to_dtype(try Tensor.eq(queries, rows), dtype), [n, 1, room, 1])
end

## The cache tensor `t` as the F32 the write and the attention read (`Stepping.widened` of a cache of one row).
def as_f32(t: Tensor.Tensor) -> Result[Tensor.Tensor]
  return ok(t) if Tensor.dtype(t) == Tensor.Dtype.F32
  Tensor.to_dtype(t, Tensor.Dtype.F32)
end

## The pick of each row: the logits `[n, vocab]` with the ids from `usable` on given a logit of `Nn.MASKED`
## (`Stepping.masked` of every row), then the first id of the maximum of each row (exact in F32 below 2^24,
## `Stepping.picked` of every row), as `[1]` floats in row order.
def batch_picks(logits: Tensor.Tensor, usable: Int) -> Result[List[Tensor.Tensor]]
  shape = Tensor.shape(logits)
  n = shape[0]
  vocab = shape[1]
  scores = logits
  if usable < vocab
    backend = Tensor.backend(logits)
    ids = try Tensor.broadcast_to(try Tensor.arange(0, vocab, 1, Tensor.Dtype.I32, backend), shape)
    edge = try Tensor.full(shape, Tensor.Values.Int(usable), Tensor.Dtype.I32, backend)
    blocked = try Tensor.full(shape, Tensor.Values.Float(Nn.MASKED), Tensor.Dtype.F32, backend)
    bias = try Tensor.where(try Tensor.lt(ids, edge), try Tensor.zeros(shape, Tensor.Dtype.F32, backend), blocked)
    scores = try Tensor.add(logits, bias)
  end

  top = try Tensor.max(scores, some([1]), true)
  hit = try Tensor.to_dtype(try Tensor.eq(scores, try Tensor.broadcast_to(top, shape)), Tensor.Dtype.F32)
  ids = try Tensor.to_dtype(
    try Tensor.broadcast_to(try Tensor.arange(0, vocab, 1, Tensor.Dtype.I32, Tensor.backend(scores)), shape),
    Tensor.Dtype.F32,
  )

  count = try Tensor.full(shape, Tensor.Values.Float(vocab.to_f), Tensor.Dtype.F32, Tensor.backend(scores))
  best = try Tensor.max(try Tensor.mul(hit, try Tensor.sub(count, ids)), some([1]), true)
  picks = try Tensor.reshape(
    try Tensor.sub(
      try Tensor.full([n, 1], Tensor.Values.Float(vocab.to_f), Tensor.Dtype.F32, Tensor.backend(scores)),
      best,
    ),
    [n],
  )

  out: List[Tensor.Tensor] = []
  for r in 0...n
    out = out.push(try Tensor.slice(picks, [r], [r + 1]))
  end

  ok(out)
end

## The batched outputs of a batch step as the rows apart, the order the step's outputs answer: each row's pick `[1]`,
## then its keys and values — each cache `[1, kv_heads, own, head_dim]`. Each slice is a tensor of the row's own: it
## shares nothing with another row's, so one sequence may be finished (or kept) without the others, and it is kept as
## the cache of a step is (`Stepping.cached`, F16 on ggml). Each row's cache is cut to the `room` of that row (the
## batch's cache is as wide as the largest room), so the cache a step gives has the rows the next step of the row reads.
def batch_outputs(
  outs: List[Tensor.Tensor],
  layers: Int,
  compiled: Bool,
  rooms: List[Int],
) -> Result[List[Tensor.Tensor]]
  n = rooms.size
  keys = outs.drop(n).take(layers)
  values = outs.drop(n + layers).take(layers)
  out: List[Tensor.Tensor] = []
  for r in 0...n
    a = try row_halves(keys, r, compiled, rooms[r])
    b = try row_halves(values, r, compiled, rooms[r])
    out = out.push(outs[r])
    out = out + a + b
  end

  ok(out)
end

## Each tensor `t` `[n, kv_heads, room, head_width]` of a batch's step as the slot `which` names:
## `[1, kv_heads, own, head_width]`, the cache of that row alone over its `own` rows, kept as `Stepping.cached` keeps
## a cache.
def row_halves(kept: List[Tensor.Tensor], which: Int, compiled: Bool, own: Int) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for t in kept
    shape = Tensor.shape(t)
    row = try Tensor.slice(t, [which, 0, 0, 0], [which + 1, shape[1], own, shape[3]])
    out = out.push(try Stepping.cached(row, compiled))
  end

  ok(out)
end

## One batch step elsewhere: one forward over the batch runs now, in an arena that keeps the batch's tensors alone.
def eager_batch(c: KvDecoder.Core[Qwen3.Qwen3], rows: List[Row]) -> Result[List[Tensor.Tensor]]
  ran: Result[List[Tensor.Tensor]] = Native.arena do
    args: List[Tensor.Tensor] = []
    for r in rows
      args = args + r.fed + r.keys + r.values
    end

    room = rows.map { |r| r.room }.max.unwrap!
    fed = try batch_of_rows(args, c.stepper.layers, room)
    outs = try batched(c.model, fed, c.usable, false)
    batch_outputs(outs, c.stepper.layers, false, rows.map { |r| r.room })
  end

  ran
end
