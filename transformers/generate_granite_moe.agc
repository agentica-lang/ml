## Greedy or sampled generation with Granite MoE (`transformers/DESIGN.md`): `kv_decoder.agc` has the decoder (the
## cache, the compiled steps, the pick in the graph) and this file the graph of one step of Granite MoE. The cache is
## `F32` on every backend (see `transformers/DESIGN.md`). A prompt is padded to its capacity and the padding rows go
## through the experts too, but no real token reads them (the mask hides them), and an expert's answer for a token does
## not depend on the other tokens. The block of a layer is `GraniteMoe`'s: the multipliers of the config, the attention
## scale, and `Moe.moe` in place of the MLP. A `Generator` belongs to one task at a time, and `close` ends it before
## `GraniteMoe.close_model` frees the weights its graphs read.

import Padded from "pkg/nn/kv"
import Moe from "pkg/nn/moe"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import KvDecoder from "kv_decoder"
import GraniteMoe from "granite_moe"
import Sample from "sample"
import Stepping from "stepping"

## A model and the compiled steps it has run (on `Ggml`), which every sequence of the model shares (`KvDecoder.Core`).
struct Generator
  model: GraniteMoe.GraniteMoe
  plans: Channel[Stepping.Plans]
  ## How many ids a step may answer: the ids from `usable` on get a logit of `Nn.MASKED`, for a caller whose tokenizer
  ## has fewer tokens than the embedding has rows (`tok.pieces.size`). `vocab_size` (the default of `open`) is no mask,
  ## as in Hugging Face.
  usable: Int
  ## Every step reads the whole capacity of the cache, not only the rows in use: `false` from `open`. The tests set it
  ## to have the logits of the other way of reading the same cache.
  whole: Bool
end

type Cache = KvDecoder.Cache

type Scored = KvDecoder.Scored

## The generator of `m`. `usable` is how many ids it may answer (see `Generator.usable`); the whole vocabulary when it
## is 0.
def open(m: GraniteMoe.GraniteMoe, usable: Int = 0) -> Result[Generator]
  c = try KvDecoder.open(m, stepper(m), usable)
  ok(Generator(model: m, plans: c.plans, usable: c.usable, whole: false))
end

## How many steps `g` has compiled: a shape it has seen compiles no more.
def compiles(g: Generator) -> Int = KvDecoder.compiles(core(g))

## Ends the pins of the compiled steps of `g`, which a later call compiles again. Call it before
## `GraniteMoe.close_model(g.model)`.
def close(g: Generator) -> Void
  KvDecoder.close(core(g))
end

## The empty cache.
def start() -> Cache = KvDecoder.start()

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

## Ends a sequence: closes the tensors of `cache`.
def finish(cache: Cache) -> Void
  KvDecoder.finish(cache)
end

## Decoding from the prompt `ids`: at most `max_new` ids, stopping before a stop id (`stop`, such as `<|im_end|>`),
## greedy or sampled (`Stepping.generate`). The prompt and the new ids must fit `max_position_embeddings`.
def generate(
  g: Generator,
  ids: List[Int],
  max_new: Int,
  stop: List[Int] = [],
  sampling: Optional[Sample.Sampling] = none,
) -> Result[List[Int]]
  KvDecoder.generate(core(g), ids, max_new, stop, sampling)
end

private

struct Kv
  key: Tensor.Tensor
  value: Tensor.Tensor
end

struct Layer
  hidden: Tensor.Tensor
  key: Tensor.Tensor
  value: Tensor.Tensor
end

def core(g: Generator) -> KvDecoder.Core[GraniteMoe.GraniteMoe]
  KvDecoder.Core(model: g.model, stepper: stepper(g.model), plans: g.plans, usable: g.usable, whole: g.whole)
end

## What the decoder needs of `m`.
def stepper(m: GraniteMoe.GraniteMoe) -> KvDecoder.Stepper[GraniteMoe.GraniteMoe]
  KvDecoder.Stepper(
    layers: m.config.num_hidden_layers,
    limit: m.config.max_position_embeddings,
    vocab: m.config.vocab_size,
    table: m.tokens.table,
    traced: ->(p: GraniteMoe.GraniteMoe) { GraniteMoe.traced(p) },
    stepped: ->(p: GraniteMoe.GraniteMoe, req: KvDecoder.Request) { stepped(p, req) },
  )
end

## The step: `xs` are the ids `[width]`, their positions, the one-hot `[1, width]` of the row whose
## prediction is wanted, and per layer the keys then the values. The outputs are the head (the logits `[vocab_size]`,
## or the choice `[1]`), then the keys and the values of every layer. The cache it leaves has `room` rows, of which
## the attention reads the first `reach` (every row the new tokens may see is among them).
def stepped(m: GraniteMoe.GraniteMoe, req: KvDecoder.Request) -> Result[List[Tensor.Tensor]]
  usable = req.usable
  xs = req.xs
  room = req.room
  reach = req.reach
  mode = req.mode
  in_place = req.in_place
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
    keys = keys.push(r.key)
    values = values.push(r.value)
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

## One layer over the new tokens `h` `[1, width, d]`, as `GraniteMoe`'s own, the keys and values of the cache in the
## place of the layer's own.
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
    key = try Padded.written(old.map { |kv| kv.key }, heads.key, spread.unwrap!, 2)
    value = try Padded.written(old.map { |kv| kv.value }, heads.value, spread.unwrap!, 2)
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
