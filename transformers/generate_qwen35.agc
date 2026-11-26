## Greedy or sampled generation with Qwen3.5 (`transformers/DESIGN.md`), the cache of `generate.agc` made hybrid:
## `kv_decoder.agc` has the decoder (the cache, the compiled steps, the pick in the graph, the `Delta.Padding` of a
## prompt) and this file the graph of one step of Qwen3.5. A full-attention layer keeps what Qwen3's layers keep: the
## keys (already normalized and rotated) and the values `[1, kv_heads, rows, head_dim]`, padded to a capacity of 8, 16,
## 32 and so on (`F16` on `Ggml`). A linear-attention layer keeps two tensors of a size that does not depend on the
## length: the last rows of the input of its convolution `[taps - 1, channels]` and its recurrent state
## `[value_heads, value_dim, key_dim]`, in `F32`.
##
## A compiled step has one shape for the prompts of one capacity, so a prompt is padded to the capacity and the padding
## must change nothing: a full layer masks it (positions of `room` name no row), and a linear layer is given a
## `Delta.Padding` (`g` and `beta` of 0 on the padding rows, the convolution's rows picked at the last real token).
## Every later step is one token, which has no padding and no `Padding`.
##
## A `Generator` belongs to one task at a time, and `close` ends it before `Qwen3_5.close_model` frees the weights its
## graphs read.

import Padded from "pkg/nn/kv"
import Delta from "pkg/nn/delta"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import KvDecoder from "kv_decoder"
import Qwen3_5 from "qwen3_5"
import Sample from "sample"
import Stepping from "stepping"

## A model and the compiled steps it has run (on `Ggml`), which every sequence of the model shares: the fields of a
## `KvDecoder.Core` (see there), which `core` makes of it.
struct Generator
  model: Qwen3_5.Qwen3_5
  plans: Channel[Stepping.Plans]
  ## How many ids a step may answer: the ids from `usable` to the end of the vocabulary get no chance (a logit of
  ## `Nn.MASKED`). The embedding of Qwen3.5 has 248,320 rows and its tokenizer 248,070 tokens: with the tokenizer's
  ## count here (`tok.pieces.size`) the sampler cannot give an id that `Bpe.decode` rejects. `vocab_size` (the default
  ## of `open`) is no mask, as in Hugging Face.
  usable: Int
  ## Every step reads the whole capacity of the cache, not only the rows in use: `false` from `open`.
  whole: Bool
end

## The keys and values of the full-attention layers (in the order of the layers) and the two states of each
## linear-attention layer (the convolution's rows, then the recurrent state).
type Cache = KvDecoder.Cache

type Scored = KvDecoder.Scored

## The generator of `m`. `usable` is how many ids it may answer (see `Generator.usable`); the whole vocabulary when
## it is 0.
def open(m: Qwen3_5.Qwen3_5, usable: Int = 0) -> Result[Generator]
  c = try KvDecoder.open(m, stepper(m), usable)
  ok(Generator(model: m, plans: c.plans, usable: c.usable, whole: false))
end

## How many steps `g` has compiled: a shape it has seen compiles no more.
def compiles(g: Generator) -> Int = KvDecoder.compiles(core(g))

## Ends the pins of the compiled steps of `g`, which a later call compiles again. Call it before
## `Qwen3_5.close_model(g.model)`.
def close(g: Generator) -> Void
  KvDecoder.close(core(g))
end

## The empty cache.
def start() -> Cache = KvDecoder.start()

## Feeds `ids` after the tokens of `cache`: a prompt of any length when the cache is empty, one id after that. The
## cache that `step` takes is dead afterwards, even when the step fails (`KvDecoder.step`).
def step(g: Generator, cache: Cache, ids: List[Int]) -> Result[Stepping.Picked[Cache]]
  KvDecoder.step(core(g), cache, ids)
end

## `step` with the logits of the last token fed, `[vocab_size]`, in place of the choice.
def step_logits(g: Generator, cache: Cache, ids: List[Int]) -> Result[Scored]
  KvDecoder.step_logits(core(g), cache, ids)
end

## `step_logits` for the `k` largest logits only: their ids and values, largest first, and the cache
## (`KvDecoder.step_top`). `k` is at most the usable ids.
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

## One decoder layer's answer: the hidden rows, the keys and values of a full layer, the states of a linear one.
struct Mixed
  hidden: Tensor.Tensor
  kv: Optional[KvDecoder.Kv]
  states: List[Tensor.Tensor]
end

def core(g: Generator) -> KvDecoder.Core[Qwen3_5.Qwen3_5]
  KvDecoder.Core(model: g.model, stepper: stepper(g.model), plans: g.plans, usable: g.usable, whole: g.whole)
end

## The kinds of layer of a model: how many are full attention and how many linear.
def full_layers(m: Qwen3_5.Qwen3_5) -> Int = m.layers.select { |l| full?(l) }.size

def full?(l: Qwen3_5.Layer) -> Bool
  case l.mixer
    Qwen3_5.Mixer.Full(_) -> true
    Qwen3_5.Mixer.Linear(_) -> false
  end
end

## What the decoder needs of `m`: a cache of keys and values for each full layer, and two states for each linear one.
def stepper(m: Qwen3_5.Qwen3_5) -> KvDecoder.Stepper[Qwen3_5.Qwen3_5]
  full = full_layers(m)
  KvDecoder.Stepper(
    layers: full,
    limit: m.config.max_position_embeddings,
    vocab: m.config.vocab_size,
    backend: Tensor.backend(m.tokens.table),
    states: 2 * (m.layers.size - full),
    taps: m.config.linear_conv_kernel_dim,
    traced: ->(p: Qwen3_5.Qwen3_5) { Qwen3_5.traced(p) },
    stepped: ->(p: Qwen3_5.Qwen3_5, req: KvDecoder.Request) { stepped(p, req) },
  )
end

## The step: `xs` are the ids `[width]`, their positions, the one-hot `[1, width]` of the row whose prediction is
## wanted, for a prompt (`width` above 1) the `valid` and `keep` of the padding, and the cache (the keys then the values
## of the full layers, then the two states of each linear layer). The outputs are the head (the logits `[vocab_size]`,
## the choice `[1]` or the candidates), then the keys, the values and the states of the new cache, which has `room` rows
## of keys, of which the attention reads the first `reach`.
def stepped(m: Qwen3_5.Qwen3_5, req: KvDecoder.Request) -> Result[List[Tensor.Tensor]]
  xs = req.xs
  width = Tensor.shape(xs[0])[0]
  d = m.config.hidden_size
  full = full_layers(m)
  slots = Padded.Slots(positions: xs[1], capacity: req.room)
  h = try Tensor.reshape(try Nn.embedding(m.tokens, xs[0]), [1, width, d])
  mask = try Padded.padded_mask(slots.with(capacity: req.reach), Tensor.dtype(h))
  # A prompt is the only step with padding, and the only one without a cache.
  padding: Optional[Delta.Padding] = none
  lead = 3
  if width > 1
    padding = some(Delta.Padding(valid: xs[3], keep: xs[4]))
    lead = 5
  end

  reading = try KvDecoder.reading(req, mask)
  keys: List[Tensor.Tensor] = []
  values: List[Tensor.Tensor] = []
  states: List[Tensor.Tensor] = []
  full_at = 0
  linear_at = 0
  for l in m.layers
    old: Optional[KvDecoder.Kv] = none
    old_state: Optional[Delta.DeltaState] = none
    if xs.size > lead
      if full?(l)
        old = some(KvDecoder.Kv(key: xs[lead + full_at], value: xs[lead + full + full_at]))
      else
        base = lead + 2 * full + 2 * linear_at
        old_state = some(Delta.DeltaState(conv: xs[base], recurrent: xs[base + 1]))
      end
    end

    r = try layer(l, h, reading, old, old_state, padding, req)
    h = r.hidden
    case r.kv
      some(kv) ->
        keys = keys.push(try Stepping.cached(kv.key, req.compiled))
        values = values.push(try Stepping.cached(kv.value, req.compiled))
      none -> nil
    end

    states = states + r.states
    if full?(l)
      full_at += 1
    else
      linear_at += 1
    end
  end

  last = try Tensor.reshape(try Tensor.matmul(xs[2], try Tensor.reshape(h, [width, d])), [1, 1, d])
  last = try Nn.layer_norm(m.final_norm, last)
  logits = try Stepping.masked(try Tensor.reshape(try Nn.dense(m.head, last), [m.config.vocab_size]), req.usable)
  head = try Stepping.headed(logits, req.mode)

  ok([head] + keys + values + states)
end

## One decoder layer over the new tokens `h` `[1, width, d]`: the mixer (linear attention over the state, or full
## attention over the cache) and the MLP.
def layer(
  l: Qwen3_5.Layer,
  h: Tensor.Tensor,
  reading: KvDecoder.Reading,
  old: Optional[KvDecoder.Kv],
  old_state: Optional[Delta.DeltaState],
  padding: Optional[Delta.Padding],
  req: KvDecoder.Request,
) -> Result[Mixed]
  x = try Nn.layer_norm(l.norm1, h)
  shape = Tensor.shape(h)
  mixed = x
  kv: Optional[KvDecoder.Kv] = none
  states: List[Tensor.Tensor] = []
  case l.mixer
    Qwen3_5.Mixer.Linear(p) ->
      state = old_state
      if state.none?
        state = some(try Delta.delta_start(p, Tensor.backend(h)))
      end

      r = try Delta.gated_delta_net(p, try Tensor.reshape(x, [shape[1], shape[2]]), state.unwrap!, padding)
      mixed = try Tensor.reshape(r.out, shape)
      states = [r.state.conv, r.state.recurrent]
    Qwen3_5.Mixer.Full(a) ->
      kept = try KvDecoder.attend_cached(a, x, old, reading, req)
      mixed = kept.answer
      kv = some(KvDecoder.Kv(key: kept.key, value: kept.value))
  end

  mid = try Tensor.add(h, mixed)
  out = try Tensor.add(mid, try Nn.mlp(l.mlp, try Nn.layer_norm(l.norm2, mid)))
  ok(Mixed(hidden: out, kv: kv, states: states))
end
