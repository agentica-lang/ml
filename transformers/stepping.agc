## What the decoding steps of the models that keep a cache share (`generate.agc`, `generate_qwen35.agc`), so that a model
## writes only its own step and cache (`transformers/DESIGN.md`, "Generation"):
##
## - the loop that decodes (`generate` over a `Decoder`, the record that a generator fills in with its `pick`, `top`
##   and `finish`), greedy or sampled, which stops at a stop id and never asks for an id outside `usable`;
## - the compiled steps by shape (`Plans`, `reaches`, `planned`) and what the graph of a step answers (`Head`, the pick
##   of the largest logit in the graph, the `k` largest and their reading, `ranked`);
## - the inputs of a step (`inputs`), the check of the ids (`check_ids`), the cache tensors (`widened`, `cached`,
##   `in_place?`) and the mask of the ids a model must not answer (`masked`).

import Padded from "pkg/nn/kv"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Lift from "lift"
import Sample from "sample"

## The compiled steps by shape, and how many compiles made them.
struct Plans
  graphs: Map[String, Trace.Compiled]
  compiles: Int
end

struct Candidate
  id: Int
  logit: Float
end

## `Decoder.pick`: the id with the largest logit after the last token fed (the first on a tie), and the cache.
struct Picked[C]
  id: Int
  cache: C
end

## `Decoder.top`: the ids of the largest logits and the logits, largest first, and the cache.
struct Candidates[C]
  ids: List[Int]
  logits: List[Float]
  cache: C
end

## What `generate` needs of a generator, `C` being its cache. `pick` and `top` feed `ids` after the tokens of the cache
## (a prompt when it is empty, one id after that), as `Generate.step` and `Generate.step_top`, and each makes the cache
## it was given dead (a failing `pick` or `top` has already closed the cache it was given: `generate` does not close it
## again); `finish` ends a cache; `empty` is the cache of no tokens. `usable` is how many ids a step may
## answer and `limit` how many positions the model has.
struct Decoder[C]
  usable: Int
  limit: Int
  empty: C
  pick: Fn[C, List[Int], Result[Picked[C]]]
  top: Fn[C, List[Int], Int, Result[Candidates[C]]]
  finish: Fn[C, Void]
end

## The `k` largest logits of a step in the order of `Head.Top(k)`: their ids and values, largest first.
struct Ranking
  ids: List[Int]
  logits: List[Float]
end

## The keys of the compiled steps of one step: its own, the ones of a larger reach that do the same (`ahead`) and the
## ones of the smaller reaches that nothing needs after it (`behind`).
struct Reaches
  key: String
  ahead: List[String]
  behind: List[String]
end

## What the graph of a step answers about the next token: the logits `[vocab_size]`, the greedy pick as one float, or
## the `k` largest logits and their ids (`2k` floats: the logits, then the ids).
enum Head
  Logits
  Pick
  Top(Int)
end

## The first id of the maximum of `scores` (`[n]`) as `[1]` floats (exact in F32 below 2^24): `v` minus the
## largest of `v - id` over the ids that hold the maximum. A graph cannot answer an `I64` (what `argmax`
## makes) on ggml.
def picked(scores: Tensor.Tensor) -> Result[Tensor.Tensor]
  v = Tensor.shape(scores)[0]
  dtype = Tensor.dtype(scores)
  backend = Tensor.backend(scores)
  top = try Tensor.broadcast_to(try Tensor.max(scores, some([0]), true), [v])
  hit = try Tensor.to_dtype(try Tensor.eq(scores, top), dtype)
  ids = try Tensor.to_dtype(try Tensor.arange(0, v, 1, Tensor.Dtype.I32, backend), dtype)
  count = try Tensor.full([v], Tensor.Values.Float(v.to_f), dtype, backend)
  best = try Tensor.max(try Tensor.mul(hit, try Tensor.sub(count, ids)), some([0]), true)
  Tensor.sub(try Tensor.full([1], Tensor.Values.Float(v.to_f), dtype, backend), best)
end

## The `k` largest of `logits` as `[2k]` floats: their values, then their ids (exact in F32 below 2^24).
def top_candidates(logits: Tensor.Tensor, k: Int) -> Result[Tensor.Tensor]
  if Tensor.dtype(logits) != Tensor.Dtype.F32
    return error("the logits are #{Tensor.dtype(logits)}: step_top needs F32, or the ids would round")
  end

  ids = try Tensor.top_ids(logits, k)
  values = try Tensor.gather(logits, ids)
  Tensor.concat([values, try Tensor.to_dtype(ids, Tensor.dtype(logits))], 0)
end

## The key of the step of `width` tokens over a cache of `rows` rows that leaves `room` and reads `reach` of them.
def plan_key(width: Int, rows: Int, room: Int, reach: Int, mode: Head) -> String
  "#{width} #{rows} #{room} #{reach} #{head_key(mode)}"
end

## The compiled function `key` for `specs`, or the first of `ahead` that `plans` has, which do the same. With none
## of them it is compiled with `f` and kept at once, and the functions `behind` end: nothing has more use for them.
def planned(
  plans: Channel[Plans],
  key: String,
  ahead: List[String],
  behind: List[String],
  specs: List[Trace.Spec],
  f: Fn[List[Tensor.Tensor], Result[List[Tensor.Tensor]]],
) -> Result[Trace.Compiled]
  case take_plans(plans)
    some(p) ->
      _ = plans.send(p, 0)
      for same in [key] + ahead
        return ok(p.graphs[same]) if p.graphs.has_key?(same)
      end

    none -> nil
  end

  compiled = try Trace.compile(specs, f)
  case take_plans(plans)
    some(p) ->
      graphs = p.graphs
      for old in behind
        if graphs.has_key?(old)
          Trace.close(graphs[old])
          graphs = graphs.delete(old)
        end
      end

      _ = plans.send(Plans(graphs: graphs.put(key, compiled), compiles: p.compiles + 1), 0)
    none -> nil
  end

  ok(compiled)
end

## The plans, which the caller must put back. When they cannot be taken in time, a
## new empty `Plans` is put in the channel in their place and the answer is `none`.
def take_plans(plans: Channel[Plans]) -> Optional[Plans]
  taken = plans.recv(PLANS_WAIT_MS)
  return some(taken.unwrap!) if taken.ok?

  _ = plans.send(Plans(graphs: {}, compiles: 0), 0)
  none
end

## The id the pick found, an error for none (a NaN in the logits: no score equals the maximum).
def chosen(head: List[Tensor.Values], vocab: Int) -> Result[Int]
  case head[0]
    Tensor.Values.Float(f) ->
      id = try f.to_i
      return error("the pick found no id in a vocabulary of #{vocab}: the logits hold a NaN") if id < 0 or id >= vocab
      ok(id)
    _ -> error("the pick is a float")
  end
end

## The ids padded to `width`, the positions (the rows of the new tokens, `room` for the padding), and the
## one-hot of the last real token.
def inputs(ids: List[Int], length: Int, width: Int, room: Int, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  n = ids.size
  tokens: List[Tensor.Values] = []
  at: List[Tensor.Values] = []
  pick: List[Tensor.Values] = []
  for i in 0...width
    if i < n
      tokens = tokens.push(Tensor.Values.Int(ids[i]))
      at = at.push(Tensor.Values.Int(length + i))
    else
      tokens = tokens.push(Tensor.Values.Int(0))
      at = at.push(Tensor.Values.Int(room))
    end

    if i == n - 1
      pick = pick.push(Tensor.Values.Float(1.0))
    else
      pick = pick.push(Tensor.Values.Float(0.0))
    end
  end

  made: List[Tensor.Tensor] = []
  for t in [
    Tensor.from_list(tokens, [width], Tensor.Dtype.I32, backend),
    Tensor.from_list(at, [width], Tensor.Dtype.I32, backend),
    Tensor.from_list(pick, [1, width], Tensor.Dtype.F32, backend),
  ]
    case t
      ok(x) -> made = made.push(x)
      error(e) ->
        Lift.close(made)
        return error(e.message)
    end
  end

  ok(made)
end

## The keys of the step of `width` tokens over a cache of `rows` rows that leaves `room` and reads `reach` of them
## (`Padded.reach` is the step of the reach): see `planned`.
def reaches(width: Int, rows: Int, room: Int, reach: Int, mode: Head) -> Reaches
  behind: List[String] = []
  near = Padded.reach(1, room)
  while near < reach
    behind = behind.push(plan_key(width, rows, room, near, mode))
    near = Padded.reach(near + 1, room)
  end

  ahead: List[String] = []
  far = reach
  while far < room
    far = Padded.reach(far + 1, room)
    ahead = ahead.push(plan_key(width, rows, room, far, mode))
  end

  Reaches(key: plan_key(width, rows, room, reach, mode), ahead: ahead, behind: behind)
end

## The `k` largest logits that `top_candidates` gave (`head` is its `2k` floats), or an error when they are not ids
## below `usable`.
def ranked(head: List[Tensor.Values], k: Int, usable: Int) -> Result[Ranking]
  found: List[Candidate] = []
  for i in 0...k
    case head[i]
      Tensor.Values.Float(logit) ->
        case head[k + i]
          Tensor.Values.Float(id) -> found = found.push(Candidate(id: id.to_i.unwrap_or(-1), logit: logit))
          _ -> nil
        end

      _ -> nil
    end
  end

  if found.size != k or found.any? { |c| c.id < 0 or c.id >= usable }
    return error("step_top: the graph gave no ids, or ids outside the usable ones")
  end

  ordered = found.sort_by { |c| 0.0 - c.logit }
  ok(Ranking(ids: ordered.map { |c| c.id }, logits: ordered.map { |c| c.logit }))
end

## An error unless `ids` may follow a cache of `length` tokens in a model of `limit` positions and `vocab` ids.
def check_ids(limit: Int, vocab: Int, length: Int, ids: List[Int]) -> Result[Void]
  n = ids.size
  return error("step needs ids") if n == 0
  return error("step takes one id after the first call, got #{n}") if n > 1 and length > 0
  return error("step needs at most #{limit} tokens, got #{length + n}") if length + n > limit
  for id in ids
    return error("token id #{id} is outside 0...#{vocab}") if id < 0 or id >= vocab
  end

  ok()
end

## Whether a step writes its row into the tensors of a cache (`keys`, its first tensor of `[1, heads, rows, width]`): on
## `Ggml`, one token after a cache that already has the `room` rows of the step.
def in_place?(backend: Tensor.Backend, keys: List[Tensor.Tensor], width: Int, room: Int) -> Bool
  case backend
    Tensor.Backend.Ggml(_) -> width == 1 and not keys.empty? and Tensor.shape(keys[0])[2] == room
    _ -> false
  end
end

## The cache tensor `t` as it is kept: `F16` on `Ggml` (`half`), when it is not already.
def cached(t: Tensor.Tensor, half: Bool) -> Result[Tensor.Tensor]
  return ok(t) if not half or Tensor.dtype(t) == Tensor.Dtype.F16
  Tensor.to_dtype(t, Tensor.Dtype.F16)
end

## The old cache tensor `t` in `F32`.
def widened(t: Optional[Tensor.Tensor]) -> Result[Optional[Tensor.Tensor]]
  case t
    some(x) ->
      return ok(t) if Tensor.dtype(x) == Tensor.Dtype.F32
      ok(some(try Tensor.to_dtype(x, Tensor.Dtype.F32)))
    none -> ok(t)
  end
end

## `logits` `[vocab]` with the ids from `usable` on given a logit of `Nn.MASKED`, so that no step answers them: the
## padded rows of an embedding, which its tokenizer cannot decode. `logits` are `F32` and are returned as they are when
## `usable` is the vocabulary; any other dtype is an error then.
def masked(logits: Tensor.Tensor, usable: Int) -> Result[Tensor.Tensor]
  vocab = Tensor.shape(logits)[0]
  return ok(logits) if usable >= vocab
  return error("masked needs F32 logits, got #{Tensor.dtype(logits)}") if Tensor.dtype(logits) != Tensor.Dtype.F32
  backend = Tensor.backend(logits)
  ids = try Tensor.arange(0, vocab, 1, Tensor.Dtype.I32, backend)
  edge = try Tensor.full([vocab], Tensor.Values.Int(usable), Tensor.Dtype.I32, backend)
  blocked = try Tensor.full([vocab], Tensor.Values.Float(Nn.MASKED), Tensor.Dtype.F32, backend)
  bias = try Tensor.where(try Tensor.lt(ids, edge), try Tensor.zeros([vocab], Tensor.Dtype.F32, backend), blocked)
  Tensor.add(logits, bias)
end

## How many ids a generator opened with `usable` may answer in a vocabulary of `vocab`: all of them for 0, an error
## beyond the vocabulary.
def usable_ids(usable: Int, vocab: Int) -> Result[Int]
  if usable < 0 or usable > vocab
    return error("usable is 0 (the whole vocabulary) or from 1 to #{vocab}, got #{usable}")
  end

  return ok(vocab) if usable == 0
  ok(usable)
end

## Decoding from the prompt `ids`: at most `max_new` ids, stopping before a stop id (`stop`, such as `<|im_end|>`).
## Greedy (the choice runs in the graph) unless `sampling` says how to draw: then each step reads the largest logits
## (cut in the graph) and `Sample.pick_top` draws among them, the same ids for the same seed (up to float noise). An
## unusable `sampling` (`Sample.check`) is an error before the first step, also at temperature 0. The prompt and the new
## ids must fit the positions of the model (`Decoder.limit`).
def generate[C](
  d: Decoder[C],
  ids: List[Int],
  max_new: Int,
  stop: List[Int] = [],
  sampling: Optional[Sample.Sampling] = none,
) -> Result[List[Int]]
  return error("generate needs max_new of 0 or more, got #{max_new}") if max_new < 0
  case sampling
    some(s) ->
      checked_sampling = Sample.check(s)
      return error(checked_sampling.message) if not checked_sampling.ok?

    none -> nil
  end

  if max_new > 0 and ids.size + max_new - 1 > d.limit
    return error(
      "generate needs at most #{d.limit} positions, a prompt of #{ids.size} and #{max_new} new ids need more",
    )
  end

  cache = d.empty
  out: List[Int] = []
  feed = ids
  state = 0
  case sampling
    some(s) -> state = Sample.start(s)
    none -> nil
  end

  while out.size < max_new
    s = chosen_next(d, cache, feed, sampling, state)
    return error(s.message) if not s.ok?

    picked = s.unwrap!
    cache = picked.cache
    state = picked.state
    break if stop.include?(picked.id)
    out = out.push(picked.id)
    feed = [picked.id]
  end

  d.finish.call(cache)
  ok(out)
end

private

## How long a call waits for the plans, which are held for a few VM steps.
PLANS_WAIT_MS = 50

## `chosen_next`'s choice: the id, the cache after it and the state of the draws.
struct Drawn[C]
  id: Int
  cache: C
  state: Int
end

def head_key(mode: Head) -> String
  case mode
    Head.Logits -> "logits"
    Head.Pick -> "pick"
    Head.Top(k) -> "top #{k}"
  end
end

## The id after `feed` and the cache, by the greedy step or, with `sampling`, by a draw among the largest logits (`state`
## is the generator of the draws, which the answer carries on).
def chosen_next[C](
  d: Decoder[C],
  cache: C,
  feed: List[Int],
  sampling: Optional[Sample.Sampling],
  state: Int,
) -> Result[Drawn[C]]
  case sampling
    some(s) ->
      if s.temperature > 0.0
        top = try d.top.call(cache, feed, Sample.window(s, d.usable))
        drawn = Sample.pick_top(top.ids, top.logits, s, state)
        if not drawn.ok?
          d.finish.call(top.cache)
          return error(drawn.message)
        end

        r = drawn.unwrap!
        return ok(Drawn(id: r.id, cache: top.cache, state: r.state))
      end

    none -> nil
  end

  p = try d.pick.call(cache, feed)
  ok(Drawn(id: p.id, cache: p.cache, state: state))
end
