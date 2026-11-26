## Qwen3.5 text models for inference, from a Hugging Face checkpoint (`transformers/DESIGN.md`): the embedding table,
## decoder layers that are either linear attention (Gated DeltaNet, `Delta.gated_delta_net`) or full attention with
## an output gate and rotary positions on part of each head (`Nn.Attention` with `gate`), each followed by a SwiGLU
## MLP, the final norm and the output head, tied to the embedding table when the config says so. This is the forward
## over whole sequences, with no cache (the rotary positions of an image prompt are in `generate_qwen35.agc`). The
## vision tower and the multi-token-prediction head of the checkpoint are not read. The norms of Qwen3.5 scale by `1
## + weight`: the loader adds the 1.
##
## `Qwen/Qwen3.5-0.8B` is `bfloat16` on disk, its text weights under `model.language_model.`; they are `F32` in memory.

import Hub from "pkg/hub/hub"
import Load from "pkg/hub/load"
import Delta from "pkg/nn/delta"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Lift from "lift"
import Qwen3 from "qwen3"
import Source from "tokenizer/source"

## The prefix of the text weights.
PREFIX = "model.language_model."

## The model types of a `config.json`, read first to refuse the mixture-of-experts model with its own message.
struct Kind
  model_type: String
  text_config: Optional[InnerKind]
end

struct InnerKind
  model_type: String
end

## The id that stands for an image in the prompt, from the top level of `config.json`.
struct ImageIds
  image_token_id: Int
end

## `config.json` of a checkpoint: the text model is in `text_config`.
struct Document
  model_type: String
  text_config: Text
  tie_word_embeddings: Optional[Bool]
end

## The rotary part of the text config. `mrope_section` is how many pairs of the rotated columns each axis (time, height,
## width) of an image prompt owns, interleaved (`Nn.Rope.sections`); `[11, 11, 10]` as in Hugging Face when the file has
## none (`sections_of`).
struct Rotary
  rope_theta: Float
  partial_rotary_factor: Float = 1.0
  rope_type: String = "default"
  mrope_section: Optional[List[Int]] = none
  mrope_interleaved: Optional[Bool] = none
end

## The fields of `text_config` that the model reads.
struct Text
  model_type: String
  hidden_size: Int
  num_hidden_layers: Int
  num_attention_heads: Int
  num_key_value_heads: Int
  head_dim: Int
  intermediate_size: Int
  vocab_size: Int
  max_position_embeddings: Int
  rms_norm_eps: Float
  hidden_act: String
  layer_types: List[String]
  linear_conv_kernel_dim: Int
  linear_key_head_dim: Int
  linear_num_key_heads: Int
  linear_num_value_heads: Int
  linear_value_head_dim: Int
  rope_parameters: Rotary
  attn_output_gate: Bool = true
  attention_bias: Bool = false
  tie_word_embeddings: Optional[Bool]
end

## What a layer does with the tokens: linear attention, or full attention.
enum Mixer
  Linear(Delta.GatedDeltaNet)
  Full(Nn.Attention)
end

struct Layer
  norm1: Nn.LayerNorm
  mixer: Mixer
  norm2: Nn.LayerNorm
  mlp: Nn.Mlp
end

struct Qwen3_5
  config: Text
  ## Whether the head is the embedding table (no `lm_head`).
  tied: Bool
  tokens: Nn.Embedding
  layers: List[Layer]
  final_norm: Nn.LayerNorm
  ## `[vocab_size, hidden_size]` (`out_in`): the embedding table itself when tied.
  head: Nn.Dense
end

## The model in `dir`, which holds `config.json` and `model.safetensors` (or the `model.safetensors.index.json` of a
## sharded one, as the official `Qwen/Qwen3.5-*` repositories have it).
def open(dir: String, backend: Tensor.Backend, weights: Tensor.Dtype = Tensor.Dtype.F32) -> Result[Qwen3_5]
  load(try Qwen3.checkpoint(dir), try read_config(dir), backend, weights)
end

## The text config of the model folder `dir`, with `tie_word_embeddings` of the file (the top level first) when it has
## one; `load` decides otherwise from the weights.
def read_config(dir: String) -> Result[Text]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  kind = try Source.within("#{dir}/config.json", JSON.decode(text, Kind))
  inner = kind.text_config.map { |t| t.model_type }.unwrap_or("")
  if kind.model_type == "qwen3_5_moe" or inner == "qwen3_5_moe_text"
    return error("qwen3_5_moe (a mixture of experts) is not supported, only the dense qwen3_5")
  end

  file = try Source.within("#{dir}/config.json", JSON.decode(text, Document))
  return error("model_type is #{file.model_type}, not qwen3_5") if file.model_type != "qwen3_5"

  tie = file.tie_word_embeddings
  case tie
    some(_) -> nil
    none -> tie = file.text_config.tie_word_embeddings
  end

  ok(file.text_config.with(tie_word_embeddings: tie))
end

## The id of the image token (`<|image_pad|>`) of the checkpoint in `dir`: the one that `Mrope.expand` repeats for
## each image and that a `KvDecoder.Sight` names.
def image_token(dir: String) -> Result[Int]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  ok((try Source.within("#{dir}/config.json", JSON.decode(text, ImageIds))).image_token_id)
end

## An error when `config` is a model this file does not run.
def check_config(config: Text) -> Result[Void]
  return error("text model_type is #{config.model_type}, not qwen3_5_text") if config.model_type != "qwen3_5_text"
  return error("hidden_act #{config.hidden_act} is not supported, only silu") if config.hidden_act != "silu"
  return error("attention_bias is not supported") if config.attention_bias
  return error("only an attention with an output gate (attn_output_gate) is supported") if not config.attn_output_gate
  if config.rope_parameters.rope_type != "default"
    return error("rope_type #{config.rope_parameters.rope_type} is not supported, only default")
  end

  if config.layer_types.size != config.num_hidden_layers
    return error("#{config.layer_types.size} layer_types for #{config.num_hidden_layers} layers")
  end

  for kind in config.layer_types
    if kind != "linear_attention" and kind != "full_attention"
      return error("layer type #{kind} is not supported, only linear_attention and full_attention")
    end
  end

  return error("head_dim #{config.head_dim} needs 1 or more") if config.head_dim < 1

  if config.num_key_value_heads < 1 or config.num_attention_heads % config.num_key_value_heads != 0
    return error("#{config.num_attention_heads} heads are not a multiple of #{config.num_key_value_heads} key heads")
  end

  if config.linear_num_key_heads < 1 or config.linear_num_value_heads % config.linear_num_key_heads != 0
    return error(
      "#{config.linear_num_value_heads} linear value heads are not a multiple of #{config.linear_num_key_heads} key heads",
    )
  end

  if config.linear_key_head_dim != config.linear_value_head_dim
    return error(
      "linear_key_head_dim #{config.linear_key_head_dim} and linear_value_head_dim #{config.linear_value_head_dim} differ, the delta rule needs one width",
    )
  end

  if config.linear_conv_kernel_dim < 2
    return error("linear_conv_kernel_dim #{config.linear_conv_kernel_dim} needs 2 or more")
  end

  sections = sections_of(config)
  if sections.size != 3 or sections[0] < 0 or sections[1] < 1 or sections[2] < 1
    return error(
      "mrope_section needs the numbers of time, height and width (1 or more for the last two), got #{sections}",
    )
  end

  if config.rope_parameters.mrope_interleaved == some(false)
    return error("mrope_interleaved false is not supported, only the interleaved M-RoPE")
  end

  dims = try rotary_dims(config)
  if dims < 1 or dims > config.head_dim or dims % 2 != 0
    return error("the rotary width #{dims} is not an even number in 2 to head_dim #{config.head_dim}")
  end

  ok()
end

## The `mrope_section` of the config.
def sections_of(config: Text) -> List[Int] = config.rope_parameters.mrope_section.unwrap_or([11, 11, 10])

## How many columns of each head rope turns: `head_dim * partial_rotary_factor`.
def rotary_dims(config: Text) -> Result[Int]
  (config.head_dim.to_f * config.rope_parameters.partial_rotary_factor).to_i
end

## The model of an open checkpoint, its shapes checked against `config`.
def load(
  c: Hub.Checkpoint,
  config: Text,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[Qwen3_5]
  return error("only weights: F32 is supported, not #{weights}") if weights != Tensor.Dtype.F32
  try check_config(config)
  tied = config.tie_word_embeddings.unwrap_or(not Hub.has?(c, "lm_head.weight"))
  tokens = try Load.embedding(c, "#{PREFIX}embed_tokens.weight", backend, weights)
  try expect_shape("#{PREFIX}embed_tokens.weight", tokens.table, [config.vocab_size, config.hidden_size])
  layers: List[Layer] = []
  for i in 0...config.num_hidden_layers
    layers = layers.push(try layer(c, "#{PREFIX}layers.#{i}.", config.layer_types[i], config, backend, weights))
  end

  head = Nn.Dense(w: tokens.table, b: none, out_in: true)
  if not tied
    head = try Load.dense(c, "lm_head", backend, weights)
    try expect_shape("lm_head.weight", head.w, [config.vocab_size, config.hidden_size])
  end

  ok(
    Qwen3_5(
      config: config,
      tied: tied,
      tokens: tokens,
      layers: layers,
      final_norm: try centered(c, "#{PREFIX}norm", config.rms_norm_eps, backend),
      head: head,
    ),
  )
end

## The batch `rows` of token ids (all of one length), through the model.
def run(m: Qwen3_5, rows: List[List[Int]], last: Bool = false) -> Result[Tensor.Tensor]
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
def logits(m: Qwen3_5, ids: Tensor.Tensor, last: Bool = false) -> Result[Tensor.Tensor]
  shape = Tensor.shape(ids)
  return error("logits needs ids of [batch, seq], got #{shape}") if shape.size != 2
  if shape[1] > m.config.max_position_embeddings
    return error("#{shape[1]} tokens pass max_position_embeddings #{m.config.max_position_embeddings}")
  end

  out = try Lift.staged(
    m,
    ->(p: Qwen3_5) { traced(p) },
    [ids],
    ->(p: Qwen3_5, xs: List[Tensor.Tensor]) { ok([try graph(p, xs[0], last)]) },
  )

  ok(out[0])
end

## Closes the weights of `m` now. Nothing may use `m` afterwards. The logits that `run` gave belong to the caller.
def close_model(m: Qwen3_5) -> Void
  tensors = [m.tokens.table] + Lift.dense_parts(m.head) + Lift.norm_parts(m.final_norm)
  for l in m.layers
    tensors = tensors + Lift.norm_parts(l.norm1) + Lift.norm_parts(l.norm2) + Lift.mlp_parts(l.mlp)
    case l.mixer
      Mixer.Linear(p) -> tensors = tensors + Lift.delta_parts(p)
      Mixer.Full(a) -> tensors = tensors + Lift.attention_parts(a)
    end
  end

  Lift.close(tensors)
end

## The tensors of `m` as `Trace` constants (`Lift`).
def traced(m: Qwen3_5) -> Qwen3_5
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
    Mixer.Linear(p) -> Mixer.Linear(Lift.delta(p))
    Mixer.Full(a) -> Mixer.Full(Lift.attention(a))
  end

  l.with(norm1: Lift.norm(l.norm1), mixer: mixer, norm2: Lift.norm(l.norm2), mlp: Lift.mlp(l.mlp))
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

## An RMS norm of `{prefix}.weight`, which Qwen3.5 applies as `1 + weight`: the gamma is that sum, made on the host,
## since a weight on ggml cannot be added to eagerly. `beta` is unused, so zeros stand in.
def centered(c: Hub.Checkpoint, prefix: String, eps: Float, backend: Tensor.Backend) -> Result[Nn.LayerNorm]
  stored = try Load.weight(c, "#{prefix}.weight", 1, Tensor.Backend.Reference)
  shape = Tensor.shape(stored)
  gamma = try Tensor.floats((try values_of(stored)).map { |w| 1.0 + w }, shape, backend)
  ok(Nn.LayerNorm(gamma: gamma, beta: try Tensor.zeros(shape, Tensor.Dtype.F32, backend), eps: eps, rms: true))
end

## A vector `{name}` as it is stored, on `backend`, of `[size]`.
def vector(c: Hub.Checkpoint, name: String, size: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  t = try Load.weight(c, name, 1, backend)
  try expect_shape(name, t, [size])
  ok(t)
end

## One layer under `p` (`model.language_model.layers.N.`).
def layer(
  c: Hub.Checkpoint,
  p: String,
  kind: String,
  config: Text,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[Layer]
  width = config.hidden_size
  eps = config.rms_norm_eps
  m = "#{p}mlp."
  mlp = Nn.Mlp(
    up: try Load.dense(c, "#{m}up_proj", backend, weights),
    down: try Load.dense(c, "#{m}down_proj", backend, weights),
    activation: Nn.Activation.Silu,
    gate: some(try Load.dense(c, "#{m}gate_proj", backend, weights)),
  )

  try expect_shape("#{m}up_proj.weight", mlp.up.w, [config.intermediate_size, width])
  try expect_shape("#{m}gate_proj.weight", mlp.gate.unwrap!.w, [config.intermediate_size, width])
  try expect_shape("#{m}down_proj.weight", mlp.down.w, [width, config.intermediate_size])
  mixer = try mixer_of(c, p, kind, config, backend, weights)

  ok(
    Layer(
      norm1: try centered(c, "#{p}input_layernorm", eps, backend),
      mixer: mixer,
      norm2: try centered(c, "#{p}post_attention_layernorm", eps, backend),
      mlp: mlp,
    ),
  )
end

## The mixer of a layer of `kind` under `p`.
def mixer_of(
  c: Hub.Checkpoint,
  p: String,
  kind: String,
  config: Text,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[Mixer]
  return ok(Mixer.Full(try full(c, "#{p}self_attn.", config, backend, weights))) if kind == "full_attention"

  ok(Mixer.Linear(try linear(c, "#{p}linear_attn.", config, backend, weights)))
end

## The Gated DeltaNet under `a`: the convolution is stored `[channels, 1, taps]`.
def linear(
  c: Hub.Checkpoint,
  a: String,
  config: Text,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[Delta.GatedDeltaNet]
  width = config.hidden_size
  keys = config.linear_num_key_heads * config.linear_key_head_dim
  values = config.linear_num_value_heads * config.linear_value_head_dim
  channels = 2 * keys + values
  taps = config.linear_conv_kernel_dim
  stored = try Load.weight(c, "#{a}conv1d.weight", 3, Tensor.Backend.Reference)
  try expect_shape("#{a}conv1d.weight", stored, [channels, 1, taps])
  p = Delta.GatedDeltaNet(
    qkv: try Load.dense(c, "#{a}in_proj_qkv", backend, weights),
    z: try Load.dense(c, "#{a}in_proj_z", backend, weights),
    b: try Load.dense(c, "#{a}in_proj_b", backend, weights),
    a: try Load.dense(c, "#{a}in_proj_a", backend, weights),
    conv: try Tensor.floats(try values_of(stored), [channels, taps], backend),
    dt_bias: try vector(c, "#{a}dt_bias", config.linear_num_value_heads, backend),
    a_log: try vector(c, "#{a}A_log", config.linear_num_value_heads, backend),
    norm: try vector(c, "#{a}norm.weight", config.linear_value_head_dim, backend),
    output: try Load.dense(c, "#{a}out_proj", backend, weights),
    key_heads: config.linear_num_key_heads,
    value_heads: config.linear_num_value_heads,
    key_dim: config.linear_key_head_dim,
    value_dim: config.linear_value_head_dim,
    eps: config.rms_norm_eps,
  )

  try expect_shape("#{a}in_proj_qkv.weight", p.qkv.w, [channels, width])
  try expect_shape("#{a}in_proj_z.weight", p.z.w, [values, width])
  try expect_shape("#{a}in_proj_b.weight", p.b.w, [config.linear_num_value_heads, width])
  try expect_shape("#{a}in_proj_a.weight", p.a.w, [config.linear_num_value_heads, width])
  try expect_shape("#{a}out_proj.weight", p.output.w, [width, values])
  ok(p)
end

## The gated full attention under `a`: `q_proj` holds a query and a gate for each head.
def full(
  c: Hub.Checkpoint,
  a: String,
  config: Text,
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
    rope: some(
      Nn.Rope(
        base: config.rope_parameters.rope_theta,
        dims: some(try rotary_dims(config)),
        sections: some(sections_of(config)),
      ),
    ),
    query_norm: some(try centered(c, "#{a}q_norm", config.rms_norm_eps, backend)),
    key_norm: some(try centered(c, "#{a}k_norm", config.rms_norm_eps, backend)),
    gate: true,
  )

  try expect_shape("#{a}q_proj.weight", attention.query.w, [2 * q_width, width])
  try expect_shape("#{a}k_proj.weight", attention.key.w, [kv_width, width])
  try expect_shape("#{a}v_proj.weight", attention.value.w, [kv_width, width])
  try expect_shape("#{a}o_proj.weight", attention.output.w, [width, q_width])
  try expect_shape("#{a}q_norm.weight", attention.query_norm.unwrap!.gamma, [config.head_dim])
  try expect_shape("#{a}k_norm.weight", attention.key_norm.unwrap!.gamma, [config.head_dim])
  ok(attention)
end

## One sequence `x` `[seq, width]` through a layer.
def through(l: Layer, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  normed = try Nn.layer_norm(l.norm1, x)
  mixed = case l.mixer
    Mixer.Linear(p) -> (try Delta.gated_delta_net(p, normed, try Delta.delta_start(p, Tensor.backend(x)))).out
    Mixer.Full(a) ->
      shape = Tensor.shape(x)
      seen = try Nn.self_attention(a, try Tensor.reshape(normed, [1, shape[0], shape[1]]), none, true)
      try Tensor.reshape(seen, shape)
  end

  h = try Tensor.add(x, mixed)
  Tensor.add(h, try Nn.mlp(l.mlp, try Nn.layer_norm(l.norm2, h)))
end

## The embeddings, the layers, the final norm and the head, one sequence of the batch at a time.
def graph(m: Qwen3_5, ids: Tensor.Tensor, last: Bool) -> Result[Tensor.Tensor]
  shape = Tensor.shape(ids)
  batch = shape[0]
  seq = shape[1]
  width = m.config.hidden_size
  words = try Nn.embedding(m.tokens, try Tensor.reshape(ids, [batch * seq]))
  rows: List[Tensor.Tensor] = []
  for b in 0...batch
    h = try Tensor.slice(words, [b * seq, 0], [(b + 1) * seq, width])
    for l in m.layers
      h = try through(l, h)
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
