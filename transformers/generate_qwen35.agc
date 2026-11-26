## Greedy or sampled generation with Qwen3.5 (`transformers/DESIGN.md`), the cache of `generate.agc` made hybrid. A
## full-attention layer keeps what Qwen3's layers keep: the keys (already normalized and rotated) and the values
## `[1, kv_heads, rows, head_dim]`, padded to a capacity of 8, 16, 32 and so on (`F16` on `Ggml`). A linear-attention
## layer keeps two tensors of a size that does not depend on the length: the last rows of the input of its convolution
## `[taps - 1, channels]` and its recurrent state `[value_heads, value_dim, key_dim]`, in `F32`.
##
## A compiled step has one shape for the prompts of one capacity, so a prompt is padded to the capacity and the padding
## must change nothing: a full layer masks it (positions of `room` name no row), and a linear layer is given a
## `Delta.Padding` (`g` and `beta` of 0 on the padding rows, the convolution's rows picked at the last real token).
## Every later step is one token, which has no padding and no `Padding`. The rest is as `kv_decoder.agc` (the plans, the
## pick in the graph, the reach of a decode step).
##
## Batch of one. A `Generator` belongs to one task at a time, and `close` ends it before `Qwen3_5.close_model` frees the
## weights its graphs read.

import Padded from "pkg/nn/kv"
import Delta from "pkg/nn/delta"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Lift from "lift"
import Qwen3_5 from "qwen3_5"
import Sample from "sample"
import Stepping from "stepping"

## A model and the compiled steps it has run (on `Ggml`), which every sequence of the model shares.
struct Generator
  model: Qwen3_5.Qwen3_5
  plans: Channel[Stepping.Plans]
  ## How many ids a step may answer: the ids from `usable` to the end of the vocabulary get no chance (a logit of
  ## `Nn.MASKED`). The embedding of Qwen3.5 has 248,320 rows and its tokenizer 248,070 tokens: with the tokenizer's
  ## count here (`tok.pieces.size`) the sampler cannot give an id that `Bpe.decode` rejects. `vocab_size` (the default of `open`) is no
  ## mask, as in Hugging Face.
  usable: Int
  ## Every step reads the whole capacity of the cache, not only the rows in use: `false` from `open`.
  whole: Bool
end

## What `step` keeps between calls: per full-attention layer the keys and the values of the `length` tokens so far,
## padded to the capacity, and per linear-attention layer its two states (the convolution's rows, then the recurrent
## state). On `Ggml` a one-token step of the same capacity writes its row into the keys and values themselves (they are
## the tensors of the cache it returns), so a cache that was stepped is dead.
struct Cache
  keys: List[Tensor.Tensor]
  values: List[Tensor.Tensor]
  states: List[Tensor.Tensor]
  length: Int
end

## `step_logits`: the logits after the last token fed, and the cache.
struct Scored
  logits: List[Float]
  cache: Cache
end

## The generator of `m`. `usable` is how many ids it may answer (see `Generator.usable`); the whole vocabulary when
## it is 0.
def open(m: Qwen3_5.Qwen3_5, usable: Int = 0) -> Result[Generator]
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

## Ends the pins of the compiled steps of `g`, which a later call compiles again. Call it before
## `Qwen3_5.close_model(g.model)`.
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
def start() -> Cache = Cache(keys: [], values: [], states: [], length: 0)

## Feeds `ids` after the tokens of `cache`: a prompt of any length when the cache is empty, one id after that. The
## cache that `step` takes is dead afterwards, even when the step fails. Use the one it gives; never step a cache twice
## or read one after it was stepped.
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

## `step_logits` for the `k` largest logits only: their ids and values, largest first, and the cache (as
## `Generate.step_top`). `k` is at most the usable ids.
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
  Lift.close(cache.keys + cache.values + cache.states)
end

## Decoding from the prompt `ids`: at most `max_new` ids, stopping before a stop id (`stop`, such as `<|im_end|>`).
## Greedy (the choice runs in the graph) unless `sampling` says how to draw (`Stepping.generate`). The prompt and the
## new ids must fit `max_position_embeddings`.
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

## What one step gives: the last row of the head and the cache after it.
struct Advanced
  head: List[Tensor.Values]
  cache: Cache
end

## One layer's answer: the hidden rows and what it leaves in the cache.
struct Layer
  hidden: Tensor.Tensor
  key: Optional[Tensor.Tensor]
  value: Optional[Tensor.Tensor]
  states: List[Tensor.Tensor]
end

## The kinds of layer of a model: how many are full attention and how many linear.
def full_layers(m: Qwen3_5.Qwen3_5) -> Int = m.layers.select { |l| full?(l) }.size

def full?(l: Qwen3_5.Layer) -> Bool
  case l.mixer
    Qwen3_5.Mixer.Full(_) -> true
    Qwen3_5.Mixer.Linear(_) -> false
  end
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
  made = fed_inputs(m, ids, cache.length, width, room, backend)
  if not made.ok?
    finish(cache)
    return error(made.message)
  end

  fed = made.unwrap!
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
  # The states of a step are new tensors; the keys and values are the cache's own when the step wrote in place.
  Lift.close(cache.states)
  if not in_place or not outs.ok?
    Lift.close(cache.keys + cache.values)
  end

  s = try outs
  ok(Advanced(head: s.head, cache: s.cache.with(length: cache.length + n)))
end

## The inputs of a step: the ids, their positions, the one-hot of the last real token (`Stepping.inputs`) and, for a
## prompt (more than one row), the `Delta.Padding` of the linear layers.
def fed_inputs(
  m: Qwen3_5.Qwen3_5,
  ids: List[Int],
  length: Int,
  width: Int,
  room: Int,
  backend: Tensor.Backend,
) -> Result[List[Tensor.Tensor]]
  base = try Stepping.inputs(ids, length, width, room, backend)
  return ok(base) if width == 1
  padded = Delta.padding(ids.size, width, m.config.linear_conv_kernel_dim, backend)
  if not padded.ok?
    Lift.close(base)
    return error(padded.message)
  end

  pad = padded.unwrap!
  ok(base + [pad.valid, pad.keep])
end

## The graph of one step as a compiled function of its inputs on `Ggml` (see `KvDecoder.compiled_step`).
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
    ->(xs: List[Tensor.Tensor]) { stepped(Qwen3_5.traced(m), g.usable, xs, room, reach, mode, in_place, true) },
  )

  outs = try Trace.run(plan, args)
  read_outs(m, outs)
end

## The head and the cache of the outputs of a step: the head, the keys, the values and the states.
def read_outs(m: Qwen3_5.Qwen3_5, outs: List[Tensor.Tensor]) -> Result[Advanced]
  full = full_layers(m)
  head = Tensor.to_list(outs[0])
  Lift.close([outs[0]])
  if not head.ok?
    Lift.close(outs.drop(1))
  end

  ok(
    Advanced(
      head: try head,
      cache: Cache(
        keys: outs.drop(1).take(full),
        values: outs.drop(1 + full).take(full),
        states: outs.drop(1 + 2 * full),
        length: 0,
      ),
    ),
  )
end

## The same graph, run now and in an arena that keeps the new cache alone.
def eager_step(
  g: Generator,
  cache: Cache,
  fed: List[Tensor.Tensor],
  room: Int,
  mode: Stepping.Head,
) -> Result[Advanced]
  m = g.model
  ran: Result[Advanced] = Native.arena do
    outs = try stepped(m, g.usable, fed + cache.keys + cache.values + cache.states, room, room, mode)
    read_outs(m, outs)
  end

  ran
end

## The step: `xs` are the ids `[width]`, their positions, the one-hot `[1, width]` of the row whose prediction is
## wanted, for a prompt (`width` above 1) the `valid` and `keep` of the padding, and the cache (the keys then the values of the full layers, then the
## two states of each linear layer). The outputs are the head (the logits `[vocab_size]`, the choice `[1]` or the
## candidates), then the keys, the values and the states of the new cache, which has `room` rows of keys, of which the
## attention reads the first `reach`.
def stepped(
  m: Qwen3_5.Qwen3_5,
  usable: Int,
  xs: List[Tensor.Tensor],
  room: Int,
  reach: Int,
  mode: Stepping.Head,
  in_place: Bool = false,
  half: Bool = false,
) -> Result[List[Tensor.Tensor]]
  width = Tensor.shape(xs[0])[0]
  d = m.config.hidden_size
  full = full_layers(m)
  slots = Padded.Slots(positions: xs[1], capacity: room)
  h = try Tensor.reshape(try Nn.embedding(m.tokens, xs[0]), [1, width, d])
  mask = try Padded.padded_mask(slots.with(capacity: reach), Tensor.dtype(h))
  # A prompt is the only step with padding, and the only one without a cache.
  padding: Optional[Delta.Padding] = none
  lead = 3
  if width > 1
    padding = some(Delta.Padding(valid: xs[3], keep: xs[4]))
    lead = 5
  end

  spread: Optional[Tensor.Tensor] = none
  if not in_place
    spread = some(try Padded.spread_of(slots, Tensor.dtype(h)))
  end

  keys: List[Tensor.Tensor] = []
  values: List[Tensor.Tensor] = []
  states: List[Tensor.Tensor] = []
  full_at = 0
  linear_at = 0
  for l in m.layers
    old_key: Optional[Tensor.Tensor] = none
    old_value: Optional[Tensor.Tensor] = none
    old_state: Optional[Delta.DeltaState] = none
    if xs.size > lead
      if full?(l)
        old_key = some(xs[lead + full_at])
        old_value = some(xs[lead + full + full_at])
      else
        base = lead + 2 * full + 2 * linear_at
        old_state = some(Delta.DeltaState(conv: xs[base], recurrent: xs[base + 1]))
      end
    end

    r = try layer(l, h, slots.positions, mask, spread, old_key, old_value, old_state, padding, in_place, reach)
    h = r.hidden
    case r.key
      some(k) -> keys = keys.push(try Stepping.cached(k, half))
      none -> nil
    end

    case r.value
      some(v) -> values = values.push(try Stepping.cached(v, half))
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
  logits = try Stepping.masked(try Tensor.reshape(try Nn.dense(m.head, last), [m.config.vocab_size]), usable)
  head = try Stepping.headed(logits, mode)

  ok([head] + keys + values + states)
end

## One decoder layer over the new tokens `h` `[1, width, d]`: the mixer (linear attention over the state, or full
## attention over the cache) and the MLP.
def layer(
  l: Qwen3_5.Layer,
  h: Tensor.Tensor,
  positions: Tensor.Tensor,
  mask: Tensor.Tensor,
  spread: Optional[Tensor.Tensor],
  old_key: Optional[Tensor.Tensor],
  old_value: Optional[Tensor.Tensor],
  old_state: Optional[Delta.DeltaState],
  padding: Optional[Delta.Padding],
  in_place: Bool,
  reach: Int,
) -> Result[Layer]
  x = try Nn.layer_norm(l.norm1, h)
  shape = Tensor.shape(h)
  mixed = x
  key: Optional[Tensor.Tensor] = none
  value: Optional[Tensor.Tensor] = none
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
      heads = try Nn.project(a, x, x, some(positions))
      whole_key = heads.key
      whole_value = heads.value
      if in_place
        whole_key = try Tensor.set_rows(old_key.unwrap!, heads.key, positions)
        whole_value = try Tensor.set_rows(old_value.unwrap!, heads.value, positions)
      else
        whole_key = try Padded.written(try Stepping.widened(old_key), heads.key, spread.unwrap!, 2)
        whole_value = try Padded.written(try Stepping.widened(old_value), heads.value, spread.unwrap!, 2)
      end

      mixed = try Nn.attend(
        a,
        heads.query,
        try Padded.reached(whole_key, reach),
        try Padded.reached(whole_value, reach),
        some(mask),
        gate: heads.gate,
      )

      key = some(whole_key)
      value = some(whole_value)
  end

  mid = try Tensor.add(h, mixed)
  out = try Tensor.add(mid, try Nn.mlp(l.mlp, try Nn.layer_norm(l.norm2, mid)))
  ok(Layer(hidden: out, key: key, value: value, states: states))
end
