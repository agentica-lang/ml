## The linear attention layer of Qwen3.5 (Gated DeltaNet), over `[tokens, width]` of one sequence, with the two
## states it carries from one call to the next (`nn/DESIGN.md`): the last rows of the convolution's input and the
## `[value_heads, value_dim, key_dim]` state of `Tensor.gated_delta_rule`. Both have a size that does not depend on
## the length of the sequence.

import Nn from "nn"
import Tensor from "pkg/tensor/tensor"

## `qkv`, `z`, `b` and `a` are the projections of the input, stored `[outputs, inputs]` (`out_in`), with no bias.
## `conv` is the depthwise convolution over the `qkv` channels, `[channels, taps]` (a checkpoint has `[channels, 1,
## taps]`). `dt_bias` and `a_log` are per value head, `norm` is the weight of the gated RMS norm over a value head.
## Each key head serves `value_heads / key_heads` value heads.
## The `eps` of the L2 normalization of the queries and the keys (Hugging Face's `l2norm`).
L2_EPS = 0.000001

struct GatedDeltaNet
  qkv: Nn.Dense
  z: Nn.Dense
  b: Nn.Dense
  a: Nn.Dense
  conv: Tensor.Tensor
  dt_bias: Tensor.Tensor
  a_log: Tensor.Tensor
  norm: Tensor.Tensor
  output: Nn.Dense
  key_heads: Int
  value_heads: Int
  key_dim: Int
  value_dim: Int
  eps: Float = 0.000001
end

## What a sequence leaves for the next call: the last `taps - 1` rows of the convolution's input `[taps - 1,
## channels]` and the recurrent state `[value_heads, value_dim, key_dim]`.
struct DeltaState
  conv: Tensor.Tensor
  recurrent: Tensor.Tensor
end

## The answers `[tokens, width]` of `gated_delta_net` and the state after the last token.
struct DeltaOut
  out: Tensor.Tensor
  state: DeltaState
end

## The two results of `causal_conv1d`.
struct Convolved
  out: Tensor.Tensor
  state: Tensor.Tensor
end

## A call over padded tokens: only the first `real` of the `tokens` are tokens, the rest is padding that must leave the
## state as the real ones left it (a compiled step has one shape for prompts of every length). `valid` is `[tokens]`,
## 1.0 for a real token and 0.0 for padding: the padding has `g` and `beta` of 0, so the recurrent state passes through it
## unchanged. `keep` is `[taps - 1, taps - 1 + tokens]`, the one-hot rows that select the `taps - 1` rows of the
## convolution's input that end at the last real token.
struct Padding
  valid: Tensor.Tensor
  keep: Tensor.Tensor
end

## The `Padding` of `tokens` rows of which the first `real` are tokens, for a convolution of `taps` taps.
def padding(real: Int, tokens: Int, taps: Int, backend: Tensor.Backend) -> Result[Padding]
  if real < 1 or real > tokens or taps < 2
    return error("padding needs 1 <= real <= tokens and taps >= 2, got #{real}, #{tokens} and #{taps}")
  end

  valid: List[Tensor.Values] = []
  for i in 0...tokens
    valid = valid.push(Tensor.Values.Float(if_real(i < real)))
  end

  keep: List[Tensor.Values] = []
  for j in 0...(taps - 1)
    for c in 0...(taps - 1 + tokens)
      keep = keep.push(Tensor.Values.Float(if_real(c == real + j)))
    end
  end

  ok(
    Padding(
      valid: try Tensor.from_list(valid, [tokens], Tensor.Dtype.F32, backend),
      keep: try Tensor.from_list(keep, [taps - 1, taps - 1 + tokens], Tensor.Dtype.F32, backend),
    ),
  )
end

## An error unless the heads and widths of `p` are what the layer runs: `Tensor.gated_delta_rule` takes one width for the
## keys and the values.
def check_layer(p: GatedDeltaNet) -> Result[Void]
  return error("a layer needs key heads, got #{p.key_heads}") if p.key_heads < 1
  return error("a layer needs value heads, got #{p.value_heads}") if p.value_heads < 1
  if p.value_heads % p.key_heads != 0
    return error("#{p.value_heads} value heads are not a multiple of #{p.key_heads} key heads")
  end

  if p.key_dim < 1 or p.key_dim != p.value_dim
    return error("a layer needs one head width for keys and values, got #{p.key_dim} and #{p.value_dim}")
  end

  return error("conv needs [channels, taps], got #{Tensor.shape(p.conv)}") if Tensor.shape(p.conv).size != 2
  ok()
end

## The empty state of `p`: zeros, for the first call.
def delta_start(p: GatedDeltaNet, backend: Tensor.Backend) -> Result[DeltaState]
  try check_layer(p)
  shape = Tensor.shape(p.conv)
  ok(
    DeltaState(
      conv: try Tensor.zeros([shape[1] - 1, shape[0]], Tensor.Dtype.F32, backend),
      recurrent: try Tensor.zeros([p.value_heads, p.value_dim, p.key_dim], Tensor.Dtype.F32, backend),
    ),
  )
end

## A depthwise convolution over time that sees the past only: `x` is `[tokens, channels]`, `weight` `[channels, taps]`
## and `state` the `taps - 1` rows before `x` (zeros at the start), so `out[t] = sum_j weight[:, j] * seen[t + j]`
## over `seen = state ++ x`. The answer is `[tokens, channels]` and the new state, the last `taps - 1` rows of `seen`, or
## with `keep` (`Padding.keep`) the rows it selects.
def causal_conv1d(
  x: Tensor.Tensor,
  weight: Tensor.Tensor,
  state: Tensor.Tensor,
  keep: Optional[Tensor.Tensor] = none,
) -> Result[Convolved]
  xs = Tensor.shape(x)
  ws = Tensor.shape(weight)
  return error("causal_conv1d needs x [tokens, channels], got #{xs}") if xs.size != 2
  if ws.size != 2 or ws[0] != xs[1] or ws[1] < 2
    return error("causal_conv1d needs a weight [#{xs[1]}, taps] of 2 taps or more, got #{ws}")
  end

  if Tensor.shape(state) != [ws[1] - 1, xs[1]]
    return error("causal_conv1d needs a state of [#{ws[1] - 1}, #{xs[1]}], got #{Tensor.shape(state)}")
  end

  tokens = xs[0]
  channels = xs[1]
  taps = ws[1]
  seen = try Tensor.concat([state, x], 0)
  total = try Tensor.zeros([tokens, channels], Tensor.Dtype.F32, Tensor.backend(x))
  for j in 0...taps
    rows = try Tensor.slice(seen, [j, 0], [j + tokens, channels])
    tap = try Tensor.reshape(try Tensor.slice(weight, [0, j], [channels, j + 1]), [1, channels])
    total = try Tensor.add(total, try Tensor.mul(rows, try Tensor.broadcast_to(tap, [tokens, channels])))
  end

  kept = case keep
    some(k) -> Tensor.matmul(k, seen)
    none -> Tensor.slice(seen, [tokens, 0], [tokens + taps - 1, channels])
  end

  ok(Convolved(out: total, state: try kept))
end

## `weight * rms_norm(x) * silu(z)` over the last axis, both `[rows, width]`: the norm of the output of the layer, which
## gates by `z` after the norm (the weight is a plain scale, not `1 + weight`).
def gated_rms_norm(x: Tensor.Tensor, z: Tensor.Tensor, weight: Tensor.Tensor, eps: Float) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  normed = try Tensor.mul(try Tensor.rms_norm(x, eps), try Tensor.broadcast_to(weight, shape))
  Tensor.mul(normed, try Tensor.silu(z))
end

## The layer over `x` `[tokens, width]`, after the `state` of the tokens before it. With `padding` the rows of `x` after
## its `real` ones are padding: their answers are not meaningful and the state is the one after the real rows.
def gated_delta_net(
  p: GatedDeltaNet,
  x: Tensor.Tensor,
  state: DeltaState,
  padding: Optional[Padding] = none,
) -> Result[DeltaOut]
  try check_layer(p)
  return error("gated_delta_net needs x [tokens, width], got #{Tensor.shape(x)}") if Tensor.shape(x).size != 2
  tokens = Tensor.shape(x)[0]
  keys = p.key_heads * p.key_dim
  values = p.value_heads * p.value_dim
  mixed = try causal_conv1d(try Nn.dense(p.qkv, x), p.conv, state.conv, padding.map { |pad| pad.keep })
  active = try Tensor.silu(mixed.out)
  q = try Tensor.reshape(try Tensor.slice(active, [0, 0], [tokens, keys]), [tokens, p.key_heads, p.key_dim])
  k = try Tensor.reshape(try Tensor.slice(active, [0, keys], [tokens, 2 * keys]), [tokens, p.key_heads, p.key_dim])
  v = try Tensor.reshape(
    try Tensor.slice(active, [0, 2 * keys], [tokens, 2 * keys + values]),
    [tokens, p.value_heads, p.value_dim],
  )

  q = try repeated(try Tensor.l2_norm(q, L2_EPS), p.value_heads / p.key_heads)
  k = try repeated(try Tensor.l2_norm(k, L2_EPS), p.value_heads / p.key_heads)
  per_head = [tokens, p.value_heads]
  beta = try Tensor.sigmoid(try Nn.dense(p.b, x))
  stretched = try Tensor.softplus(try Tensor.add(try Nn.dense(p.a, x), try Tensor.broadcast_to(p.dt_bias, per_head)))

  g = try Tensor.neg(try Tensor.mul(try Tensor.broadcast_to(try Tensor.exp(p.a_log), per_head), stretched))
  case padding
    some(pad) ->
      live = try Tensor.broadcast_to(try Tensor.reshape(pad.valid, [tokens, 1]), per_head)
      beta = try Tensor.mul(beta, live)
      g = try Tensor.mul(g, live)
    none -> nil
  end

  r = try Tensor.gated_delta_rule(q, k, v, g, beta, state.recurrent)
  rows = [tokens * p.value_heads, p.value_dim]
  gated = try gated_rms_norm(
    try Tensor.reshape(r.out, rows),
    try Tensor.reshape(try Nn.dense(p.z, x), rows),
    p.norm,
    p.eps,
  )

  out = try Nn.dense(p.output, try Tensor.reshape(gated, [tokens, values]))
  ok(DeltaOut(out: out, state: DeltaState(conv: mixed.state, recurrent: r.state)))
end

private

def if_real(live: Bool) -> Float
  return 1.0 if live
  0.0
end

## `x` `[tokens, heads, width]` with each head repeated `times` times in a row (`repeat_interleave` on the head axis),
## which is how Hugging Face gives the value heads their key heads (ggml's own broadcast is tiled instead).
def repeated(x: Tensor.Tensor, times: Int) -> Result[Tensor.Tensor]
  return ok(x) if times == 1
  s = Tensor.shape(x)
  spread = try Tensor.broadcast_to(try Tensor.reshape(x, [s[0], s[1], 1, s[2]]), [s[0], s[1], times, s[2]])
  Tensor.reshape(spread, [s[0], s[1] * times, s[2]])
end
