## What the compiled decoding steps of the models that keep a cache share (`generate.agc`, `generate_qwen35.agc`): the
## compiled steps by shape (`Plans`), what the graph of a step answers about the next token (`Head`), the pick of the
## largest logit in the graph, the inputs of a step, and the reading of what it gives back.

import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Lift from "lift"

## How long a call waits for the plans, which are held for a few VM steps.
PLANS_WAIT_MS = 50

## The compiled steps by shape, and how many compiles made them.
struct Plans
  graphs: Map[String, Trace.Compiled]
  compiles: Int
end

struct Candidate
  id: Int
  logit: Float
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

def head_key(mode: Head) -> String
  case mode
    Head.Logits -> "logits"
    Head.Pick -> "pick"
    Head.Top(k) -> "top #{k}"
  end
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
