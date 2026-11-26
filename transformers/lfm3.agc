## LFM3 for inference, from a Hugging Face checkpoint (`transformers/DESIGN.md`): the embedding table, decoder layers
## that are a gated short convolution (`ShortConv`), a sliding-window attention or a global one (`Nn.Attention` with a
## `head_gate` and norms on the heads), each followed by a SwiGLU MLP, with a norm before and after each of the two
## branches (the sandwich), the final norm (`embedding_norm`) and the output head, tied to the embedding table. This is
## the forward over whole sequences, with no cache.
##
## `LiquidAI/LFM3-1.7B` is `bfloat16` on disk; the weights are `F32` in memory, or `F16` with `weights: F16` on ggml.

import Hub from "pkg/hub/hub"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import ShortConv from "pkg/nn/short_conv"
import Tensor from "pkg/tensor/tensor"
import Lift from "lift"
import Qwen3 from "qwen3"
import Source from "tokenizer/source"

## The rotary part of one kind of attention layer.
struct Rotary
  rope_theta: Float
  partial_rotary_factor: Float = 1.0
  rope_type: String = "default"
end

## The `rope_parameters` of `config.json`: one entry per kind of attention layer.
struct Rotaries
  full_attention: Optional[Rotary]
  sliding_attention: Optional[Rotary]
end

## The fields of `config.json` that the model reads.
struct Config
  model_type: String
  hidden_size: Int
  num_hidden_layers: Int
  num_attention_heads: Int
  num_key_value_heads: Int
  ## 0 when the config has none: `hidden_size / num_attention_heads`.
  head_dim: Int = 0
  intermediate_size: Int
  vocab_size: Int
  max_position_embeddings: Int
  norm_eps: Float
  conv_L_cache: Int = 7
  conv_bias: Bool = false
  layer_types: List[String]
  sliding_window: Optional[Int]
  use_sandwich_norm: Bool = true
  rope_parameters: Rotaries
  tie_word_embeddings: Bool = true
end

## What a layer does with the tokens: the short convolution, or an attention over every earlier token (`Full`) or over
## the last `Config.sliding_window` of them (`Sliding`).
enum Mixer
  Conv(ShortConv.ShortConv)
  Full(Nn.Attention)
  Sliding(Nn.Attention)
end

struct Layer
  operator_norm: Nn.LayerNorm
  mixer: Mixer
  post_operator_norm: Nn.LayerNorm
  ffn_norm: Nn.LayerNorm
  mlp: Nn.Mlp
  post_ffn_norm: Nn.LayerNorm
end

struct Lfm3
  config: Config
  tokens: Nn.Embedding
  layers: List[Layer]
  ## The norm after the last layer (`embedding_norm` in the checkpoint).
  final_norm: Nn.LayerNorm
  ## `[vocab_size, hidden_size]` (`out_in`): the embedding table itself when tied.
  head: Nn.Dense
end

## The model in `dir`, which holds `config.json` and `model.safetensors` (or the `model.safetensors.index.json` of a
## sharded one), as the official `LiquidAI/LFM3-*` repositories have it.
def open(dir: String, backend: Tensor.Backend, weights: Tensor.Dtype = Tensor.Dtype.F32) -> Result[Lfm3]
  load(try Qwen3.checkpoint(dir), try read_config(dir), backend, weights)
end

## The config of the model folder `dir`, with its `head_dim` filled in when the file leaves it out.
def read_config(dir: String) -> Result[Config]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  config = try Source.within("#{dir}/config.json", JSON.decode(text, Config))
  return ok(config) if config.head_dim != 0
  if config.num_attention_heads < 1 or config.hidden_size % config.num_attention_heads != 0
    return error("hidden_size #{config.hidden_size} is not a multiple of #{config.num_attention_heads} heads")
  end

  ok(config.with(head_dim: config.hidden_size / config.num_attention_heads))
end

## An error when `config` is a model this file does not run.
def check_config(config: Config) -> Result[Void]
  return error("model_type is #{config.model_type}, not lfm3") if config.model_type != "lfm3"
  return error("conv_bias is not supported") if config.conv_bias
  return error("use_sandwich_norm: false is not supported, only the sandwich of norms") if not config.use_sandwich_norm
  if config.layer_types.size != config.num_hidden_layers
    return error("#{config.layer_types.size} layer_types for #{config.num_hidden_layers} layers")
  end

  for kind in config.layer_types
    if kind != "conv" and kind != "full_attention" and kind != "sliding_attention"
      return error("layer type #{kind} is not supported, only conv, full_attention and sliding_attention")
    end
  end

  return error("conv_L_cache #{config.conv_L_cache} needs 2 or more") if config.conv_L_cache < 2

  if config.num_key_value_heads < 1 or config.num_attention_heads % config.num_key_value_heads != 0
    return error("#{config.num_attention_heads} heads are not a multiple of #{config.num_key_value_heads} key heads")
  end

  if config.head_dim < 2 or config.head_dim % 2 != 0
    return error("head_dim #{config.head_dim} is not an even number of 2 or more, which rope needs")
  end

  if config.layer_types.include?("sliding_attention")
    window = config.sliding_window.unwrap_or(0)
    return error("sliding_attention layers need a sliding_window of 1 or more, got #{window}") if window < 1
  end

  for kind in ["full_attention", "sliding_attention"]
    if config.layer_types.include?(kind)
      try rope_of(config, kind)
    end
  end

  ok()
end

## The model of an open checkpoint, its shapes checked against `config`. `weights` is `F32` or, on ggml, `F16`.
def load(
  c: Hub.Checkpoint,
  config: Config,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[Lfm3]
  return error("weights is F32 or F16, not #{weights}") if weights != Tensor.Dtype.F32 and weights != Tensor.Dtype.F16

  try check_config(config)
  width = config.hidden_size
  tokens = try Load.embedding(c, "model.embed_tokens.weight", backend, weights)
  try expect_shape("model.embed_tokens.weight", tokens.table, [config.vocab_size, width])
  layers: List[Layer] = []
  for i in 0...config.num_hidden_layers
    layers = layers.push(try layer(c, "model.layers.#{i}.", config.layer_types[i], config, backend, weights))
  end

  head = Nn.Dense(w: tokens.table, b: none, out_in: true)
  if not config.tie_word_embeddings
    head = try Load.dense(c, "lm_head", backend, weights)
    try expect_shape("lm_head.weight", head.w, [config.vocab_size, width])
  end

  ok(
    Lfm3(
      config: config,
      tokens: tokens,
      layers: layers,
      final_norm: try rms(c, "model.embedding_norm", config.norm_eps, backend),
      head: head,
    ),
  )
end

## The batch `rows` of token ids (all of one length), through the model.
def run(m: Lfm3, rows: List[List[Int]], last: Bool = false) -> Result[Tensor.Tensor]
  return error("run needs at least one row") if rows.empty?
  seq = rows[0].size
  return error("run needs at least one token") if seq == 0
  ids: List[Tensor.Values] = []
  for row in rows
    return error("run needs rows of one length, got #{row.size} and #{seq}") if row.size != seq
    for id in row
      return error("token id #{id} is outside 0...#{m.config.vocab_size}") if id < 0 or id >= m.config.vocab_size
      ids = ids.push(Tensor.Values.Int(id))
    end
  end

  tensor = try Tensor.from_list(ids, [rows.size, seq], Tensor.Dtype.I32, Tensor.backend(m.tokens.table))
  out = logits(m, tensor, last)
  Lift.close([tensor])
  out
end

## The logits `[batch, seq, vocab_size]` of the token ids `ids` (`I32`, `[batch, seq]`), each position reading the ones
## before it and itself; with `last`, only the last position's, `[batch, 1, vocab_size]`. `seq` may not pass
## `max_position_embeddings`.
def logits(m: Lfm3, ids: Tensor.Tensor, last: Bool = false) -> Result[Tensor.Tensor]
  shape = Tensor.shape(ids)
  return error("logits needs ids of [batch, seq], got #{shape}") if shape.size != 2
  if shape[1] > m.config.max_position_embeddings
    return error("#{shape[1]} tokens pass max_position_embeddings #{m.config.max_position_embeddings}")
  end

  out = try Lift.staged(
    m,
    ->(p: Lfm3) { traced(p) },
    [ids],
    ->(p: Lfm3, xs: List[Tensor.Tensor]) { ok([try graph(p, xs[0], last)]) },
  )

  ok(out[0])
end

## Closes the weights of `m` now. Nothing may use `m` afterwards. The logits that `run` gave belong to the caller.
def close_model(m: Lfm3) -> Void
  tensors = [m.tokens.table] + Lift.dense_parts(m.head) + Lift.norm_parts(m.final_norm)
  for l in m.layers
    tensors = tensors + Lift.norm_parts(l.operator_norm) + Lift.norm_parts(l.post_operator_norm)
    tensors = tensors + Lift.norm_parts(l.ffn_norm) + Lift.norm_parts(l.post_ffn_norm) + Lift.mlp_parts(l.mlp)
    case l.mixer
      Mixer.Conv(p) -> tensors = tensors + Lift.short_conv_parts(p)
      Mixer.Full(a) -> tensors = tensors + Lift.attention_parts(a)
      Mixer.Sliding(a) -> tensors = tensors + Lift.attention_parts(a)
    end
  end

  Lift.close(tensors)
end

## The tensors of `m` as `Trace` constants (`Lift`).
def traced(m: Lfm3) -> Lfm3
  m.with(
    tokens: Lift.embedding(m.tokens),
    layers: m.layers.map { |l| traced_layer(l) },
    final_norm: Lift.norm(m.final_norm),
    head: Lift.dense(m.head),
  )
end

private

def traced_layer(l: Layer) -> Layer
  mixer = case l.mixer
    Mixer.Conv(p) -> Mixer.Conv(Lift.short_conv(p))
    Mixer.Full(a) -> Mixer.Full(Lift.attention(a))
    Mixer.Sliding(a) -> Mixer.Sliding(Lift.attention(a))
  end

  l.with(
    operator_norm: Lift.norm(l.operator_norm),
    mixer: mixer,
    post_operator_norm: Lift.norm(l.post_operator_norm),
    ffn_norm: Lift.norm(l.ffn_norm),
    mlp: Lift.mlp(l.mlp),
    post_ffn_norm: Lift.norm(l.post_ffn_norm),
  )
end

def expect_shape(name: String, t: Tensor.Tensor, want: List[Int]) -> Result[Void]
  return error("#{name}: needs #{want}, got #{Tensor.shape(t)}") if Tensor.shape(t) != want
  ok()
end

## The values of a float tensor as a list.
def values_of(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end

## An RMS norm of `{prefix}.weight` (a plain scale): `beta` is unused, so zeros stand in.
def rms(c: Hub.Checkpoint, prefix: String, eps: Float, backend: Tensor.Backend) -> Result[Nn.LayerNorm]
  gamma = try Load.weight(c, "#{prefix}.weight", 1, backend)
  ok(
    Nn.LayerNorm(
      gamma: gamma,
      beta: try Tensor.zeros(Tensor.shape(gamma), Tensor.Dtype.F32, backend),
      eps: eps,
      rms: true,
    ),
  )
end

## The rotary of the attention layers of `kind`: `rope_theta`, and `zero_tail` for a `partial_rotary_factor` below 1.
def rope_of(config: Config, kind: String) -> Result[Nn.Rope]
  found = config.rope_parameters.sliding_attention
  if kind == "full_attention"
    found = config.rope_parameters.full_attention
  end

  r = try found.to_result("rope_parameters has no #{kind} entry")
  return error("rope_type #{r.rope_type} is not supported, only default") if r.rope_type != "default"
  return error("rope_theta #{r.rope_theta} needs a value above 0") if r.rope_theta <= 0.0
  rotated = try (config.head_dim.to_f * r.partial_rotary_factor).to_i
  if rotated < 2 or rotated > config.head_dim or rotated % 2 != 0
    return error("the rotary width #{rotated} of #{kind} is not an even number in 2 to head_dim #{config.head_dim}")
  end

  return ok(Nn.Rope(base: r.rope_theta)) if rotated == config.head_dim
  ok(Nn.Rope(base: r.rope_theta, zero_tail: some(rotated)))
end

## One layer under `p` (`model.layers.N.`).
def layer(
  c: Hub.Checkpoint,
  p: String,
  kind: String,
  config: Config,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[Layer]
  width = config.hidden_size
  eps = config.norm_eps
  f = "#{p}feed_forward."
  mlp = Nn.Mlp(
    up: try Load.dense(c, "#{f}w3", backend, weights),
    down: try Load.dense(c, "#{f}w2", backend, weights),
    activation: Nn.Activation.Silu,
    gate: some(try Load.dense(c, "#{f}w1", backend, weights)),
  )

  try expect_shape("#{f}w3.weight", mlp.up.w, [config.intermediate_size, width])
  try expect_shape("#{f}w1.weight", mlp.gate.unwrap!.w, [config.intermediate_size, width])
  try expect_shape("#{f}w2.weight", mlp.down.w, [width, config.intermediate_size])
  ok(
    Layer(
      operator_norm: try rms(c, "#{p}operator_norm", eps, backend),
      mixer: try mixer_of(c, p, kind, config, backend, weights),
      post_operator_norm: try rms(c, "#{p}post_operator_norm", eps, backend),
      ffn_norm: try rms(c, "#{p}ffn_norm", eps, backend),
      mlp: mlp,
      post_ffn_norm: try rms(c, "#{p}post_ffn_norm", eps, backend),
    ),
  )
end

## The mixer of a layer of `kind` under `p`.
def mixer_of(
  c: Hub.Checkpoint,
  p: String,
  kind: String,
  config: Config,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[Mixer]
  return ok(Mixer.Conv(try short_conv(c, "#{p}conv.", config, backend, weights))) if kind == "conv"
  attention = try attention_of(c, "#{p}self_attn.", kind, config, backend, weights)
  return ok(Mixer.Full(attention)) if kind == "full_attention"
  ok(Mixer.Sliding(attention))
end

## The short convolution under `a`: the convolution is stored `[width, 1, taps]`.
def short_conv(
  c: Hub.Checkpoint,
  a: String,
  config: Config,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[ShortConv.ShortConv]
  width = config.hidden_size
  taps = config.conv_L_cache
  stored = try Load.weight(c, "#{a}conv.weight", 3, Tensor.Backend.Reference)
  try expect_shape("#{a}conv.weight", stored, [width, 1, taps])
  p = ShortConv.ShortConv(
    input: try Load.dense(c, "#{a}in_proj", backend, weights),
    conv: try Tensor.floats(try values_of(stored), [width, taps], backend),
    output: try Load.dense(c, "#{a}out_proj", backend, weights),
  )

  try expect_shape("#{a}in_proj.weight", p.input.w, [3 * width, width])
  try expect_shape("#{a}out_proj.weight", p.output.w, [width, width])
  ok(p)
end

## The attention of `kind` under `a`, with the gate of one number per head.
def attention_of(
  c: Hub.Checkpoint,
  a: String,
  kind: String,
  config: Config,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[Nn.Attention]
  width = config.hidden_size
  heads = config.num_attention_heads
  q_width = heads * config.head_dim
  kv_width = config.num_key_value_heads * config.head_dim
  attention = Nn.Attention(
    query: try Load.dense(c, "#{a}q_proj", backend, weights),
    key: try Load.dense(c, "#{a}k_proj", backend, weights),
    value: try Load.dense(c, "#{a}v_proj", backend, weights),
    output: try Load.dense(c, "#{a}o_proj", backend, weights),
    heads: heads,
    kv_heads: some(config.num_key_value_heads),
    rope: some(try rope_of(config, kind)),
    query_norm: some(try rms(c, "#{a}q_norm", config.norm_eps, backend)),
    key_norm: some(try rms(c, "#{a}k_norm", config.norm_eps, backend)),
    head_gate: some(try Load.dense(c, "#{a}gate_proj", backend, weights)),
  )

  try expect_shape("#{a}q_proj.weight", attention.query.w, [q_width, width])
  try expect_shape("#{a}k_proj.weight", attention.key.w, [kv_width, width])
  try expect_shape("#{a}v_proj.weight", attention.value.w, [kv_width, width])
  try expect_shape("#{a}o_proj.weight", attention.output.w, [width, q_width])
  try expect_shape("#{a}gate_proj.weight", attention.head_gate.unwrap!.w, [heads, width])
  try expect_shape("#{a}q_norm.weight", attention.query_norm.unwrap!.gamma, [config.head_dim])
  try expect_shape("#{a}k_norm.weight", attention.key_norm.unwrap!.gamma, [config.head_dim])
  ok(attention)
end

## One sequence `x` `[seq, width]` through a layer: each branch is normed before and after it, then added.
def through(m: Lfm3, l: Layer, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  normed = try Nn.layer_norm(l.operator_norm, x)
  mixed = case l.mixer
    Mixer.Conv(p) -> (try ShortConv.short_conv(p, normed, try ShortConv.short_start(p, Tensor.backend(x)))).out
    Mixer.Full(a) -> try attended(a, normed, none)
    Mixer.Sliding(a) -> try attended(a, normed, try window(m, shape[0], Tensor.backend(x)))
  end

  h = try Tensor.add(x, try Nn.layer_norm(l.post_operator_norm, mixed))
  ffn = try Nn.layer_norm(l.post_ffn_norm, try Nn.mlp(l.mlp, try Nn.layer_norm(l.ffn_norm, h)))
  Tensor.add(h, ffn)
end

## The causal mask of a sliding layer over `seq` tokens, none when the window covers them all.
def window(m: Lfm3, seq: Int, backend: Tensor.Backend) -> Result[Optional[Tensor.Tensor]]
  w = m.config.sliding_window.unwrap_or(seq)
  return ok(none) if w >= seq
  ok(some(try Nn.causal_mask(seq, Tensor.Dtype.F32, backend, some(w))))
end

## `x` `[seq, width]` through the attention `a`, causal, under `mask` when there is one.
def attended(a: Nn.Attention, x: Tensor.Tensor, mask: Optional[Tensor.Tensor]) -> Result[Tensor.Tensor]
  shape = Tensor.shape(x)
  seen = try Nn.self_attention(a, try Tensor.reshape(x, [1, shape[0], shape[1]]), mask, mask.none?)
  Tensor.reshape(seen, shape)
end

## The embeddings, the layers, the final norm and the head, one sequence of the batch at a time.
def graph(m: Lfm3, ids: Tensor.Tensor, last: Bool) -> Result[Tensor.Tensor]
  shape = Tensor.shape(ids)
  batch = shape[0]
  seq = shape[1]
  width = m.config.hidden_size
  words = try Nn.embedding(m.tokens, try Tensor.reshape(ids, [batch * seq]))
  rows: List[Tensor.Tensor] = []
  for b in 0...batch
    h = try Tensor.slice(words, [b * seq, 0], [(b + 1) * seq, width])
    for l in m.layers
      h = try through(m, l, h)
    end

    h = try Nn.layer_norm(m.final_norm, h)
    if last
      h = try Tensor.slice(h, [seq - 1, 0], [seq, width])
    end

    rows = rows.push(try Nn.dense(m.head, h))
  end

  out = try Tensor.concat(rows, 0)
  Tensor.reshape(out, [batch, Tensor.shape(out)[0] / batch, m.config.vocab_size])
end
