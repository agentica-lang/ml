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

## Whether `step_pair` runs on `backend`: the reference backend and ggml's CPU, the two it is built and tested on. Every
## other backend steps its rows alone, as it did before pairing existed — a Vulkan device, XLA and libtorch included,
## until one of them is measured and added here. Callers check this before they hand a cache over: a failed
## `step_pair` consumes both caches, so a scheduler that pairs where it does not run loses both rows.
def pairable(backend: Tensor.Backend) -> Bool
  case backend
    Tensor.Backend.Reference -> true
    Tensor.Backend.Ggml(Tensor.Device.Cpu) -> true
    _ -> false
  end
end

## Whether the steps of `g` may be paired: `pairable` of its backend.
def pairs(g: Generator) -> Bool = pairable(core(g).stepper.backend)

## One greedy decode step for two sequences in one graph, on the backends `pairable` names. Both caches hold a prompt
## already: a prompt runs alone, with `step`. The step is one forward over the pair: the projections, the attention and
## the MLP of every layer run on the two rows `[2, 1, hidden]` at once, with the rows' own positions, additive mask
## `[2, 1, 1, room]` and cache — each
## row writes its new key and value into its own half of the pair's cache `[2, kv_heads, room, head_dim]` and its
## attention reads only its own rows, so one row never reads the keys of the other; the weights are shared, and each
## row's answer is the same a step of that row alone gives. On a ggml device the step is one graph, compiled once per
## pair of shapes and kept in the plans of `g` like any other step, so one paired round is one execution of the
## backend; on the other backends the same forward runs eagerly in an arena. The caches the step takes are dead
## afterwards, even when the step fails, as with `step`, and each cache it gives is the tensors of that row alone. Two
## rows are the whole of this step, on the backends `pairable` names: a wider batch, paged KV with physical slots, token
## budgets, chunked prefill and pairing on another device are not here (ml#92), and a prompt, a row that decodes alone
## and the pick of one row stay in `step`.
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
  # The backends `pairable` names, before either cache is worth anything: the step consumes them as a failure does.
  if not pairable(s.backend)
    Lift.close(row_a.fed + row_b.fed)
    finish(a)
    finish(b)
    return error(
      "step_pair pairs on the reference backend and ggml's CPU only: this #{s.backend} steps its rows alone",
    )
  end

  outs = case s.backend
    Tensor.Backend.Ggml(_) -> compiled_pair(c, row_a, row_b)
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

## The cache of one row of a paired step after it: the keys, then the values, one token longer, at the same delta.
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

## One paired step on `Ggml`: one compiled graph over the two rows, kept in the plans of `c` under a key of its own
## (no single-row key has the `pair` prefix). The graph takes the two rows as a step that is not in place does and
## makes the pair's tensors itself (`pair_of_rows`), so a round binds only the rows' own tensors: one execution of the
## backend, nothing eager around it.
def compiled_pair(c: KvDecoder.Core[Qwen3.Qwen3], a: Row, b: Row) -> Result[List[Tensor.Tensor]]
  s = c.stepper
  mode = Stepping.Head.Pick
  key = (
    "pair " +
    Stepping.plan_key(1, a.rows, a.room, a.room, mode) +
    " " +
    Stepping.plan_key(1, b.rows, b.room, b.room, mode)
  )

  args = a.fed + a.keys + a.values + b.fed + b.keys + b.values
  plan = try Stepping.planned(
    c.plans,
    key,
    [],
    [],
    args.map { |t| Trace.spec_of(t) },
    ->(xs: List[Tensor.Tensor]) {
      paired_over(s, c.model, xs, a, b, s.layers, c.usable, Stepping.half_cache?(s.backend))
    },
  )

  Trace.run(plan, args)
end

## One paired step over `xs`, the two rows one after the other (a's fed, keys and values, then b's): the rows' ids,
## positions and picks become one batch and the caches one (`pair_of_rows`), one forward runs over the pair
## (`paired`), and its outputs are the rows apart (`pair_outputs`). On a device the graph fuses all of it; elsewhere
## the same ops run eagerly.
def paired_over(
  s: KvDecoder.Stepper[Qwen3.Qwen3],
  model: Qwen3.Qwen3,
  xs: List[Tensor.Tensor],
  a: Row,
  b: Row,
  layers: Int,
  usable: Int,
  compiled: Bool,
) -> Result[List[Tensor.Tensor]]
  fed = try pair_of_rows(xs, layers, [a.room, b.room].max.unwrap!)
  outs = try paired(s.traced.call(model), fed, usable, compiled)
  pair_outputs(outs, layers, compiled, a.room, b.room)
end

## The two rows of a paired step as one batch: the ids `[2]`, the positions `[2]` (row a's, then row b's) and the picks
## `[2, 1]`, then per layer the keys and the values of the pair's cache `[2, kv_heads, room, head_dim]`, each row
## zero-padded to the `room` of the pair (the larger room of the rows) and stacked.
def pair_of_rows(xs: List[Tensor.Tensor], layers: Int, room: Int) -> Result[List[Tensor.Tensor]]
  at = 3 + 2 * layers
  fed = [
    try Tensor.concat([xs[0], xs[at]], 0),
    try Tensor.concat([xs[1], xs[at + 1]], 0),
    try Tensor.concat([xs[2], xs[at + 2]], 0),
  ]

  caches = try pair_caches(xs.drop(3).take(2 * layers), xs.drop(at + 3).take(2 * layers), room)
  ok(fed + caches)
end

## The keys and the values of one layer (`one` is a's, then `other` is b's) as the pair's `[2, kv_heads, room, head_dim]`,
## each as F32 (the write and the attention read F32, as a step of one row reads its widened cache), zero-padded to the
## `room` of the pair and stacked.
def pair_caches(one: List[Tensor.Tensor], other: List[Tensor.Tensor], room: Int) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for i in 0...one.size
    a = try padded_to(try as_f32(one[i]), room)
    b = try padded_to(try as_f32(other[i]), room)
    out = out.push(try Tensor.concat([a, b], 0))
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

## One forward over the pair. `xs` are the ids `[2]`, their positions `[2]` and the picks `[2, 1]`, then per layer the
## keys and the values of the pair's cache `[2, kv_heads, room, head_dim]` (each row its own tokens: the rows past its
## length are zero, and a row never reads the keys of the other). The outputs are the pick of row a `[1]`, row b's,
## then per layer the keys and the values of the pair's cache.
def paired(m: Qwen3.Qwen3, xs: List[Tensor.Tensor], usable: Int, compiled: Bool) -> Result[List[Tensor.Tensor]]
  layers = m.config.num_hidden_layers
  d = m.config.hidden_size
  tokens = xs[0]
  positions = xs[1]
  pick = xs[2]
  keys_in = xs.drop(3).take(layers)
  values_in = xs.drop(3 + layers).take(layers)
  h = try Tensor.reshape(try Nn.embedding(m.tokens, tokens), [2, 1, d])
  mask = try Tensor.reshape(
    try Padded.padded_mask(Padded.Slots(positions: positions, capacity: Tensor.shape(keys_in[0])[2]), Tensor.dtype(h)),
    [2, 1, 1, Tensor.shape(keys_in[0])[2]],
  )

  spread = try pair_spread(positions, Tensor.shape(keys_in[0])[2], Tensor.dtype(h))
  keys_out: List[Tensor.Tensor] = []
  values_out: List[Tensor.Tensor] = []
  for i in 0...layers
    r = try paired_layer(m.stack.blocks[i], h, keys_in[i], values_in[i], spread, mask, positions)
    h = r.hidden
    keys_out = keys_out.push(r.key)
    values_out = values_out.push(r.value)
  end

  last = try Tensor.batch_matmul(try Tensor.reshape(pick, [2, 1, 1]), h)
  case m.stack.final_norm
    some(n) -> last = try Nn.layer_norm(n, last)
    none -> nil
  end

  logits = try Tensor.reshape(try Nn.dense(m.head, last), [2, m.config.vocab_size])
  heads = try pair_heads(logits, usable)
  ok(heads + keys_out + values_out)
end

## One block of the pair over the new rows `h` `[2, 1, d]`, as `Nn.encoder_block` with `norm_first`, the pair's cache in
## the place of the block's own: the rows are projected per row, the keys and values of each row are written into its
## own half of the cache (a product with the `spread` of its position, exact in F32 as `Padded.written` is), and each
## row's query attends over its own rows of the cache under the rows' additive `mask`.
def paired_layer(
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
  q = try batch_heads(try Nn.dense(a.query, x), a.heads, a.query_norm)
  kv_heads = a.kv_heads.unwrap_or(a.heads)
  k = try batch_heads(try Nn.dense(a.key, x), kv_heads, a.key_norm)
  v = try batch_heads(try Nn.dense(a.value, x), kv_heads, none)
  case a.rope
    some(r) ->
      q = try rotated_pair(q, r, positions)
      k = try rotated_pair(k, r, positions)
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
def batch_heads(projected: Tensor.Tensor, heads: Int, norm: Optional[Nn.LayerNorm]) -> Result[Tensor.Tensor]
  shape = Tensor.shape(projected)
  split = try Tensor.reshape(projected, [shape[0], shape[1], heads, shape[2] / heads])
  case norm
    some(n) -> split = try Nn.layer_norm(n, split)
    none -> nil
  end

  Tensor.permute(split, [0, 2, 1, 3])
end

## The heads of the pair `x` `[2, heads, 1, head_width]` turned by the rows' positions: the rows are swapped with the
## seq, which makes `[1, heads, 2, head_width]` whose seq is the pair with one position per row, and the same
## permutation puts the rows back. A rope of another kind (`Nn.Rope.zero_tail`, `sections`) is an error: the paired
## step is Qwen3's plain rope.
def rotated_pair(x: Tensor.Tensor, r: Nn.Rope, positions: Tensor.Tensor) -> Result[Tensor.Tensor]
  if r.zero_tail.some? or r.sections.some?
    return error("a paired step turns by a plain rope: zero_tail and sections pair no rows")
  end

  brought = try Tensor.permute(x, [2, 1, 0, 3])
  turned = try Tensor.rope(brought, positions, base: r.base, dims: r.dims, neox: r.neox)
  Tensor.permute(turned, [2, 1, 0, 3])
end

## The one-hot of the rows' positions over the `room` of the pair: `[2, 1, room, 1]`, 1 where the key's row is that
## row's new token, 0 elsewhere. A product with it writes each row's token into its own half of the cache, exact in
## F32 as `Padded.written` is.
def pair_spread(positions: Tensor.Tensor, room: Int, dtype: Tensor.Dtype) -> Result[Tensor.Tensor]
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

## The pick of each row: the logits `[2, vocab]` with the ids from `usable` on given a logit of `Nn.MASKED`
## (`Stepping.masked` of both rows), then the first id of the maximum of each row (exact in F32 below 2^24,
## `Stepping.picked` of both rows), as `[1]` floats.
def pair_heads(logits: Tensor.Tensor, usable: Int) -> Result[List[Tensor.Tensor]]
  shape = Tensor.shape(logits)
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
      try Tensor.full([2, 1], Tensor.Values.Float(vocab.to_f), Tensor.Dtype.F32, Tensor.backend(scores)),
      best,
    ),
    [2],
  )

  a = try Tensor.slice(picks, [0], [1])
  b = try Tensor.slice(picks, [1], [2])
  ok([a, b])
end

## The batched outputs of a paired step as the rows apart, the order the step's outputs answer: the pick of row a
## `[1]`, then its keys and values, then row b's pick, keys and values — each cache `[1, kv_heads, own, head_dim]`.
## Each slice is a tensor of the row's own: it shares nothing with the other row's, so one sequence may be finished
## (or kept) without the other, and it is kept as the cache of a step is (`Stepping.cached`, F16 on ggml). Each row's
## cache is cut to the `room` of that row (the pair's cache is as wide as the larger room), so the cache a step gives
## has the rows the next step of the row reads.
def pair_outputs(
  outs: List[Tensor.Tensor],
  layers: Int,
  compiled: Bool,
  a_room: Int,
  b_room: Int,
) -> Result[List[Tensor.Tensor]]
  keys = outs.drop(2).take(layers)
  values = outs.drop(2 + layers).take(layers)
  a = try halves(keys, 0, compiled, a_room) + try halves(values, 0, compiled, a_room)
  b = try halves(keys, 1, compiled, b_room) + try halves(values, 1, compiled, b_room)
  ok([outs[0]] + a + [outs[1]] + b)
end

## Each tensor `t` `[2, kv_heads, room, head_width]` of a pair's step as the slot `which` names:
## `[1, kv_heads, own, head_width]`, the cache of that row alone over its `own` rows, kept as `Stepping.cached` keeps
## a cache.
def halves(kept: List[Tensor.Tensor], which: Int, compiled: Bool, own: Int) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for t in kept
    shape = Tensor.shape(t)
    row = try Tensor.slice(t, [which, 0, 0, 0], [which + 1, shape[1], own, shape[3]])
    out = out.push(try Stepping.cached(row, compiled))
  end

  ok(out)
end

## One paired step elsewhere: one forward over the pair runs now, in an arena that keeps the batch's tensors alone.
def eager_pair(c: KvDecoder.Core[Qwen3.Qwen3], a: Row, b: Row) -> Result[List[Tensor.Tensor]]
  ran: Result[List[Tensor.Tensor]] = Native.arena do
    fed = try pair_of_rows(
      a.fed + a.keys + a.values + b.fed + b.keys + b.values,
      c.stepper.layers,
      [a.room, b.room].max.unwrap!,
    )

    outs = try paired(c.model, fed, c.usable, false)
    pair_outputs(outs, c.stepper.layers, false, a.room, b.room)
  end

  ran
end
