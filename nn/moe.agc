## The mixture-of-experts block (`nn/DESIGN.md`): a router picks `top_k` of the experts for each token, every picked
## expert is a gated MLP, and the answers are added with the router's gates as weights.

import Nn from "nn"
import Tensor from "pkg/tensor/tensor"

## `router` is the `Dense` that scores the experts, `[experts, width]` with `out_in`, no bias. The experts are two
## stacked weights, one `[rows, columns]` matrix per expert as a checkpoint stores it: `gate_up`
## `[experts, 2 * inner, width]` (the first `inner` rows are the gate, which `activation` turns, the last `inner` the
## value it multiplies) and `down` `[experts, width, inner]`.
## `normalize` makes the gates the softmax over the scores of the picked experts (Granite MoE); without it they are the
## picked entries of the softmax over all the experts, which do not add up to 1 (Qwen3 MoE with `norm_topk_prob` off).
struct Moe
  router: Nn.Dense
  gate_up: Tensor.Tensor
  down: Tensor.Tensor
  top_k: Int
  activation: Nn.Activation
  normalize: Bool = true
end

## What the router decided for `tokens` rows: the `ids` `I32 [tokens, top_k]` of their experts (in no order) and the
## `gates` `[tokens, top_k]` of the same places.
struct Routing
  ids: Tensor.Tensor
  gates: Tensor.Tensor
end

## The `Routing` of `x` `[tokens, width]`. The ids come from `Tensor.top_ids`: an expert that ties with the `top_k`th
## in score may or may not be picked, on any backend.
def route(p: Moe, x: Tensor.Tensor) -> Result[Routing]
  logits = try Nn.dense(p.router, x)
  tokens = Tensor.shape(logits)[0]
  experts = Tensor.shape(logits)[1]
  ids = try Tensor.top_ids(logits, p.top_k)
  backend = Tensor.backend(x)
  probs = try Tensor.softmax(logits, 1)
  # the entry of `probs` that each id names: a mask of the ids against the expert numbers, and a sum over the experts
  shape = [tokens, p.top_k, experts]
  named = try Tensor.eq(
    try Tensor.broadcast_to(try Tensor.reshape(ids, [tokens, p.top_k, 1]), shape),
    try Tensor.broadcast_to(
      try Tensor.reshape(try Tensor.arange(0, experts, 1, Tensor.Dtype.I32, backend), [1, 1, experts]),
      shape,
    ),
  )

  every = try Tensor.broadcast_to(try Tensor.reshape(probs, [tokens, 1, experts]), shape)
  none_of = try Tensor.zeros(shape, Tensor.dtype(probs), backend)
  picked = try Tensor.sum(try Tensor.where(named, every, none_of), some([2]), false)
  return ok(Routing(ids: ids, gates: picked)) if !p.normalize
  total = try Tensor.sum(picked, some([1]), true)
  ok(Routing(ids: ids, gates: try Tensor.div(picked, try Tensor.broadcast_to(total, Tensor.shape(picked)))))
end

## The block over `x` `[..., width]` (any number of leading axes, as `Nn.dense` takes them), the same shape back.
def moe(p: Moe, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  try check(p)
  shape = Tensor.shape(x)
  return error("moe needs rank >= 2, got #{shape}") if shape.size < 2
  width = shape[shape.size - 1]
  if width != Tensor.shape(p.gate_up)[2]
    return error("moe needs a width of #{Tensor.shape(p.gate_up)[2]}, got #{width}")
  end

  rows = shape.reduce(1) do |acc, d| acc * d end / width
  flat = try Tensor.reshape(x, [rows, width])
  routing = try route(p, flat)
  inner = Tensor.shape(p.down)[2]
  both = try Tensor.expert_matmul(p.gate_up, try Tensor.reshape(flat, [rows, 1, width]), routing.ids)
  gate = try Tensor.slice(both, [0, 0, 0], [rows, p.top_k, inner])
  value = try Tensor.slice(both, [0, 0, inner], [rows, p.top_k, 2 * inner])
  hidden = try Tensor.mul(try Nn.activate(p.activation, gate), value)
  answers = try Tensor.expert_matmul(p.down, hidden, routing.ids)
  weighted = try Tensor.batch_matmul(try Tensor.reshape(routing.gates, [rows, 1, p.top_k]), answers)
  Tensor.reshape(weighted, shape)
end

private

## An error unless the router and the two stacked weights are about the same experts.
def check(p: Moe) -> Result[Void]
  gate_up = Tensor.shape(p.gate_up)
  down = Tensor.shape(p.down)
  router = Tensor.shape(p.router.w)
  return error("moe needs a gate_up of [experts, 2 * inner, width], got #{gate_up}") if gate_up.size != 3
  return error("moe needs a down of [experts, width, inner], got #{down}") if down.size != 3
  if down != [gate_up[0], gate_up[2], gate_up[1] / 2] or gate_up[1] % 2 != 0
    return error(
      "moe needs a down of #{[gate_up[0], gate_up[2], gate_up[1] / 2]} for a gate_up of #{gate_up}, got #{down}",
    )
  end

  if router != [gate_up[0], gate_up[2]] or !p.router.out_in
    return error("moe needs a router of [#{gate_up[0]}, #{gate_up[2]}] with out_in, got #{router}")
  end

  ok()
end
