## Greedy or sampled generation with LFM3 (`transformers/DESIGN.md`): `kv_decoder.agc` has the decoder (the cache, the
## compiled steps, the pick in the graph, the `Delta.Padding` of a prompt) and this file the graph of one step of LFM3.
## An attention layer, full or sliding, keeps what Qwen3's layers keep: the keys (already normalized and rotated) and
## the values `[1, kv_heads, rows, head_dim]`, padded to a capacity of 8, 16, 32 and so on (`F16` on `Ggml`). A sliding
## layer keeps every row too and its mask hides the keys a window or more before the position (`Padded.padded_mask`
## with a window). A convolution layer keeps the last `taps - 1` rows of its input, `[taps - 1, width]` in `F32`, which
## do not grow with the length. A `Generator` belongs to one task at a time, and `close` ends it before
## `Lfm3.close_model` frees the weights its graphs read.

import Padded from "pkg/nn/kv"
import Delta from "pkg/nn/delta"
import Nn from "pkg/nn/nn"
import ShortConv from "pkg/nn/short_conv"
import Tensor from "pkg/tensor/tensor"
import KvDecoder from "kv_decoder"
import Lfm3 from "lfm3"
import Sample from "sample"
import Stepping from "stepping"

## A model and the compiled steps it has run (on `Ggml`), which every sequence of the model shares: the fields of a
## `KvDecoder.Core` (see there), which `core` makes of it.
struct Generator
  model: Lfm3.Lfm3
  plans: Channel[Stepping.Plans]
  ## See `KvDecoder.Core`.
  usable: Int
  whole: Bool
end

## The keys and values of the attention layers (in the order of the layers) and the states of the convolution layers.
type Cache = KvDecoder.Cache

type Scored = KvDecoder.Scored

## The generator of `m`. `usable` is how many ids it may answer (see `Generator.usable`); the whole vocabulary when it
## is 0. The tokenizer of LFM3 has 124,893 pieces and 140 added tokens (125,033) and the embedding 128,000 rows, so the ids
## in between have no text: pass `tok.pieces.size` to keep the sampler to ids that `Bpe.decode` accepts.
def open(m: Lfm3.Lfm3, usable: Int = 0) -> Result[Generator]
  c = try KvDecoder.open(m, stepper(m), usable)
  ok(Generator(model: m, plans: c.plans, usable: c.usable, whole: false))
end

## How many steps `g` has compiled: a shape it has seen compiles no more.
def compiles(g: Generator) -> Int = KvDecoder.compiles(core(g))

## Ends the pins of the compiled steps of `g`, which a later call compiles again. Call it before
## `Lfm3.close_model(g.model)`.
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

## One layer's answer: the hidden rows and what it leaves in the cache.
struct Layer
  hidden: Tensor.Tensor
  key: Optional[Tensor.Tensor]
  value: Optional[Tensor.Tensor]
  state: Optional[Tensor.Tensor]
end

## The answer of an attention and its keys and values after the new rows.
struct Attended
  out: Tensor.Tensor
  key: Tensor.Tensor
  value: Tensor.Tensor
end

def core(g: Generator) -> KvDecoder.Core[Lfm3.Lfm3]
  KvDecoder.Core(model: g.model, stepper: stepper(g.model), plans: g.plans, usable: g.usable, whole: g.whole)
end

def conv?(l: Lfm3.Layer) -> Bool
  case l.mixer
    Lfm3.Mixer.Conv(_) -> true
    _ -> false
  end
end

## What the decoder needs of `m`.
def stepper(m: Lfm3.Lfm3) -> KvDecoder.Stepper[Lfm3.Lfm3]
  convs = m.layers.select { |l| conv?(l) }.size
  KvDecoder.Stepper(
    layers: m.layers.size - convs,
    limit: m.config.max_position_embeddings,
    vocab: m.config.vocab_size,
    table: m.tokens.table,
    states: convs,
    taps: m.config.conv_L_cache,
    traced: ->(p: Lfm3.Lfm3) { Lfm3.traced(p) },
    stepped: ->(p: Lfm3.Lfm3, req: KvDecoder.Request) { stepped(p, req) },
  )
end

## The step: `xs` are the ids `[width]`, their positions, the one-hot `[1, width]` of the row whose prediction is
## wanted, for a prompt (`width` above 1) the `valid` and `keep` of the padding of the convolutions, and the cache: the
## keys then the values of the attention layers, then the state of each convolution layer. The outputs are the head
## (the logits `[vocab_size]`, the choice `[1]` or the candidates), then the keys, the values and the states of the new
## cache, which has `room` rows of keys, of which the attention reads the first `reach`.
def stepped(m: Lfm3.Lfm3, req: KvDecoder.Request) -> Result[List[Tensor.Tensor]]
  xs = req.xs
  width = Tensor.shape(xs[0])[0]
  d = m.config.hidden_size
  attention_layers = m.layers.select { |l| not conv?(l) }.size
  slots = Padded.Slots(positions: xs[1], capacity: req.room)
  h = try Tensor.reshape(try Nn.embedding(m.tokens, xs[0]), [1, width, d])
  reading = slots.with(capacity: req.reach)
  full_mask = try Padded.padded_mask(reading, Tensor.dtype(h))
  sliding_mask = full_mask
  if m.layers.any? { |l| sliding?(l) }
    sliding_mask = try Padded.padded_mask(reading, Tensor.dtype(h), m.config.sliding_window)
  end

  # A prompt is the only step with padding, and the only one without a cache.
  padding: Optional[Delta.Padding] = none
  lead = 3
  if width > 1
    padding = some(Delta.Padding(valid: xs[3], keep: xs[4]))
    lead = 5
  end

  spread: Optional[Tensor.Tensor] = none
  if not req.in_place
    spread = some(try Padded.spread_of(slots, Tensor.dtype(h)))
  end

  keys: List[Tensor.Tensor] = []
  values: List[Tensor.Tensor] = []
  states: List[Tensor.Tensor] = []
  attention_at = 0
  conv_at = 0
  for l in m.layers
    old_key: Optional[Tensor.Tensor] = none
    old_value: Optional[Tensor.Tensor] = none
    old_state: Optional[Tensor.Tensor] = none
    if xs.size > lead
      if conv?(l)
        old_state = some(xs[lead + 2 * attention_layers + conv_at])
      else
        old_key = some(xs[lead + attention_at])
        old_value = some(xs[lead + attention_layers + attention_at])
      end
    end

    mask = full_mask
    if sliding?(l)
      mask = sliding_mask
    end

    r = try layer(l, h, slots.positions, mask, spread, old_key, old_value, old_state, padding, req)
    h = r.hidden
    case r.key
      some(k) -> keys = keys.push(try Stepping.cached(k, req.compiled))
      none -> nil
    end

    case r.value
      some(v) -> values = values.push(try Stepping.cached(v, req.compiled))
      none -> nil
    end

    case r.state
      some(s) -> states = states.push(s)
      none -> nil
    end

    if conv?(l)
      conv_at += 1
    else
      attention_at += 1
    end
  end

  last = try Tensor.reshape(try Tensor.matmul(xs[2], try Tensor.reshape(h, [width, d])), [1, 1, d])
  last = try Nn.layer_norm(m.final_norm, last)
  logits = try Stepping.masked(try Tensor.reshape(try Nn.dense(m.head, last), [m.config.vocab_size]), req.usable)
  head = try Stepping.headed(logits, req.mode)
  ok([head] + keys + values + states)
end

def sliding?(l: Lfm3.Layer) -> Bool
  case l.mixer
    Lfm3.Mixer.Sliding(_) -> true
    _ -> false
  end
end

## One decoder layer over the new tokens `h` `[1, width, d]`, as `Lfm3.through`'s: the mixer (the convolution over its
## state, or the attention over the cache) and the MLP, each normed before and after.
def layer(
  l: Lfm3.Layer,
  h: Tensor.Tensor,
  positions: Tensor.Tensor,
  mask: Tensor.Tensor,
  spread: Optional[Tensor.Tensor],
  old_key: Optional[Tensor.Tensor],
  old_value: Optional[Tensor.Tensor],
  old_state: Optional[Tensor.Tensor],
  padding: Optional[Delta.Padding],
  req: KvDecoder.Request,
) -> Result[Layer]
  shape = Tensor.shape(h)
  x = try Nn.layer_norm(l.operator_norm, h)
  mixed = x
  key: Optional[Tensor.Tensor] = none
  value: Optional[Tensor.Tensor] = none
  state: Optional[Tensor.Tensor] = none
  case l.mixer
    Lfm3.Mixer.Conv(p) ->
      start = old_state
      if start.none?
        start = some(try ShortConv.short_start(p, Tensor.backend(h)))
      end

      r = try ShortConv.short_conv(p, try Tensor.reshape(x, [shape[1], shape[2]]), start.unwrap!, padding)
      mixed = try Tensor.reshape(r.out, shape)
      state = some(r.state)
    Lfm3.Mixer.Full(a) ->
      kept = try cached_attention(a, x, positions, mask, spread, old_key, old_value, req)
      mixed = kept.out
      key = some(kept.key)
      value = some(kept.value)
    Lfm3.Mixer.Sliding(a) ->
      kept = try cached_attention(a, x, positions, mask, spread, old_key, old_value, req)
      mixed = kept.out
      key = some(kept.key)
      value = some(kept.value)
  end

  mid = try Tensor.add(h, try Nn.layer_norm(l.post_operator_norm, mixed))
  ffn = try Nn.layer_norm(l.post_ffn_norm, try Nn.mlp(l.mlp, try Nn.layer_norm(l.ffn_norm, mid)))
  ok(Layer(hidden: try Tensor.add(mid, ffn), key: key, value: value, state: state))
end

def cached_attention(
  a: Nn.Attention,
  x: Tensor.Tensor,
  positions: Tensor.Tensor,
  mask: Tensor.Tensor,
  spread: Optional[Tensor.Tensor],
  old_key: Optional[Tensor.Tensor],
  old_value: Optional[Tensor.Tensor],
  req: KvDecoder.Request,
) -> Result[Attended]
  heads = try Nn.project(a, x, x, some(positions))
  whole_key = heads.key
  whole_value = heads.value
  if req.in_place
    whole_key = try Tensor.set_rows(old_key.unwrap!, heads.key, positions)
    whole_value = try Tensor.set_rows(old_value.unwrap!, heads.value, positions)
  else
    whole_key = try Padded.written(try Stepping.widened(old_key), heads.key, spread.unwrap!, 2)
    whole_value = try Padded.written(try Stepping.widened(old_value), heads.value, spread.unwrap!, 2)
  end

  seen = try Nn.attend(
    a,
    heads.query,
    try Padded.reached(whole_key, req.reach),
    try Padded.reached(whole_value, req.reach),
    some(mask),
    gate: heads.gate,
  )

  ok(Attended(out: seen, key: whole_key, value: whole_value))
end
