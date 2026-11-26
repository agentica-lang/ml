## Greedy or sampled generation with Qwen3 and a key-value cache (`transformers/DESIGN.md`): `kv_decoder.agc` has the
## decoder (the cache, the compiled steps, the pick in the graph) and this file the graph of one step of Qwen3 and
## Qwen2.5. On `Ggml` the cache is `F16`, half the memory and of the bytes a decode step reads (the logits drift by
## about 5e-3 on values of 10 to 30); on the other backends it is `F32`. A `Generator` belongs to one task at a time,
## and `close` ends it before `Qwen3.close_model` frees the weights its graphs read.

import Padded from "pkg/nn/kv"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import KvDecoder from "kv_decoder"
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
end

type Cache = KvDecoder.Cache

type Scored = KvDecoder.Scored

## The generator of `m`. `usable` is how many ids it may answer (see `Generator.usable`); the whole vocabulary when it
## is 0.
def open(m: Qwen3.Qwen3, usable: Int = 0) -> Result[Generator]
  c = try KvDecoder.open(m, stepper(m), usable)
  ok(Generator(model: m, plans: c.plans, usable: c.usable, whole: false))
end

## How many steps `g` has compiled: a shape it has seen compiles no more.
def compiles(g: Generator) -> Int = KvDecoder.compiles(core(g))

## Ends the pins of the compiled steps of `g`, which a later call compiles again. Call it before
## `Qwen3.close_model(g.model)`.
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

def core(g: Generator) -> KvDecoder.Core[Qwen3.Qwen3]
  KvDecoder.Core(model: g.model, stepper: stepper(g.model), plans: g.plans, usable: g.usable, whole: g.whole)
end

## What the decoder needs of `m`.
def stepper(m: Qwen3.Qwen3) -> KvDecoder.Stepper[Qwen3.Qwen3]
  KvDecoder.Stepper(
    layers: m.config.num_hidden_layers,
    limit: m.config.max_position_embeddings,
    vocab: m.config.vocab_size,
    table: m.tokens.table,
    device: some(m.device),
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
  h = try Tensor.reshape(try Nn.embedding(m.tokens, xs[0]), [1, width, d])
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
