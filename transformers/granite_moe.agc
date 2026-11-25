## IBM Granite MoE (`granitemoe`: `ibm-granite/granite-3.1-1b-a400m-instruct`) for inference, from a Hugging Face
## checkpoint (`transformers/DESIGN.md`): the embedding table, pre-norm decoder blocks with grouped-query attention (rotary
## positions with `rope_theta`, the scores scaled by `attention_multiplier`) whose MLP is a mixture of experts
## (`Moe.moe`), RMS norms, and the output head, tied to the embedding table when the config says so. Four multipliers of
## the config scale the embeddings, the two residual branches of each block and the logits. This is the forward over
## whole sequences, with no cache.
##
## The checkpoint is `bfloat16` on disk; the weights are `F32` in memory (5.3 GB for the 1B-A400M model).

import Hub from "pkg/hub/hub"
import Load from "pkg/hub/load"
import Moe from "pkg/nn/moe"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Lift from "lift"
import Qwen3 from "qwen3"
import Source from "tokenizer/source"

## The fields of `config.json` that the model reads. The four multipliers default to 1.0 and `tie_word_embeddings` to false, as
## in Hugging Face's `GraniteMoeConfig`; the other fields are required.
struct Config
  model_type: String
  hidden_size: Int
  num_hidden_layers: Int
  num_attention_heads: Int
  num_key_value_heads: Int
  intermediate_size: Int
  vocab_size: Int
  max_position_embeddings: Int
  rms_norm_eps: Float
  rope_theta: Float
  hidden_act: String
  num_local_experts: Int
  num_experts_per_tok: Int
  embedding_multiplier: Float = 1.0
  attention_multiplier: Float = 1.0
  residual_multiplier: Float = 1.0
  logits_scaling: Float = 1.0
  tie_word_embeddings: Bool = false
  attention_bias: Bool = false
  rope_scaling: Optional[Scaling]
end

## Any `rope_scaling` object: the model has none.
struct Scaling
  rope_type: Optional[String]
end

struct Layer
  norm1: Nn.LayerNorm
  attention: Nn.Attention
  norm2: Nn.LayerNorm
  moe: Moe.Moe
end

struct GraniteMoe
  config: Config
  tokens: Nn.Embedding
  layers: List[Layer]
  final_norm: Nn.LayerNorm
  ## `[vocab_size, hidden_size]` (`out_in`): the embedding table itself when tied.
  head: Nn.Dense
end

## The model in `dir`, which holds `config.json` and `model.safetensors` (or the `model.safetensors.index.json` of a
## sharded one).
def open(dir: String, backend: Tensor.Backend, weights: Tensor.Dtype = Tensor.Dtype.F32) -> Result[GraniteMoe]
  load(try Qwen3.checkpoint(dir), try read_config(dir), backend, weights)
end

## The `config.json` of the model folder `dir`.
def read_config(dir: String) -> Result[Config]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  Source.within("#{dir}/config.json", JSON.decode(text, Config))
end

## An error when `config` is a model this file does not run: another type, activation, attention bias or rope scaling
## (each would be read and ignored), a number of experts per token outside 1 to the number of experts, or head counts that do
## not divide.
def check_config(config: Config) -> Result[Void]
  return error("model_type is #{config.model_type}, not granitemoe") if config.model_type != "granitemoe"
  return error("hidden_act #{config.hidden_act} is not supported, only silu") if config.hidden_act != "silu"
  return error("attention_bias is not supported") if config.attention_bias
  return error("rope_scaling is not supported") if config.rope_scaling.some?
  if config.num_experts_per_tok < 1 or config.num_experts_per_tok > config.num_local_experts
    return error(
      "num_experts_per_tok #{config.num_experts_per_tok} is not in 1 to #{config.num_local_experts} experts",
    )
  end

  if config.num_key_value_heads < 1 or config.num_attention_heads % config.num_key_value_heads != 0
    return error("#{config.num_attention_heads} heads are not a multiple of #{config.num_key_value_heads} key heads")
  end

  if config.hidden_size % config.num_attention_heads != 0 or (config.hidden_size / config.num_attention_heads) % 2 != 0
    return error(
      "hidden_size #{config.hidden_size} does not give an even head width for #{config.num_attention_heads} heads",
    )
  end

  ok()
end

## The model of an open checkpoint, its shapes checked against `config`.
def load(
  c: Hub.Checkpoint,
  config: Config,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[GraniteMoe]
  return error("only weights: F32 is supported, not #{weights}") if weights != Tensor.Dtype.F32
  try check_config(config)
  tokens = try Load.embedding(c, "model.embed_tokens.weight", backend, weights)
  try expect_shape("model.embed_tokens.weight", tokens.table, [config.vocab_size, config.hidden_size])
  layers: List[Layer] = []
  for i in 0...config.num_hidden_layers
    layers = layers.push(try layer(c, "model.layers.#{i}.", config, backend, weights))
  end

  head = Nn.Dense(w: tokens.table, b: none, out_in: true)
  if not config.tie_word_embeddings
    head = try Load.dense(c, "lm_head", backend, weights)
    try expect_shape("lm_head.weight", head.w, [config.vocab_size, config.hidden_size])
  end

  ok(
    GraniteMoe(
      config: config,
      tokens: tokens,
      layers: layers,
      final_norm: try rms(c, "model.norm", config.rms_norm_eps, backend),
      head: head,
    ),
  )
end

## The batch `rows` of token ids (all of one length), through the model.
def run(m: GraniteMoe, rows: List[List[Int]], last: Bool = false) -> Result[Tensor.Tensor]
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
def logits(m: GraniteMoe, ids: Tensor.Tensor, last: Bool = false) -> Result[Tensor.Tensor]
  shape = Tensor.shape(ids)
  return error("logits needs ids of [batch, seq], got #{shape}") if shape.size != 2
  if shape[1] > m.config.max_position_embeddings
    return error("#{shape[1]} tokens pass max_position_embeddings #{m.config.max_position_embeddings}")
  end

  out = try Lift.staged(
    m,
    ->(p: GraniteMoe) { traced(p) },
    [ids],
    ->(p: GraniteMoe, xs: List[Tensor.Tensor]) { ok([try graph(p, xs[0], last)]) },
  )

  ok(out[0])
end

## Closes the weights of `m` now. Nothing may use `m` afterwards. The logits that `run` gave belong to the caller.
def close_model(m: GraniteMoe) -> Void
  tensors = [m.tokens.table] + Lift.dense_parts(m.head) + Lift.norm_parts(m.final_norm)
  for l in m.layers
    tensors = tensors + Lift.attention_parts(l.attention) + Lift.norm_parts(l.norm1) + Lift.norm_parts(l.norm2)
    tensors = tensors + Lift.moe_parts(l.moe)
  end

  Lift.close(tensors)
end

## The tensors of `m` as `Trace` constants (`Lift`).
def traced(m: GraniteMoe) -> GraniteMoe
  m.with(
    tokens: Lift.embedding(m.tokens),
    layers: m.layers.map { |l|
      l.with(
        norm1: Lift.norm(l.norm1),
        attention: Lift.attention(l.attention),
        norm2: Lift.norm(l.norm2),
        moe: Lift.moe(l.moe),
      )
    },
    final_norm: Lift.norm(m.final_norm),
    head: Lift.dense(m.head),
  )
end

## `x` times the constant `factor`. Internal: the steps of `Generate` use it.
def times(x: Tensor.Tensor, factor: Float) -> Result[Tensor.Tensor]
  Tensor.mul(x, try Tensor.full(Tensor.shape(x), Tensor.Values.Float(factor), Tensor.dtype(x), Tensor.backend(x)))
end

## The logits of the final-normed hidden rows `h`: the head, divided by `logits_scaling`. Internal, as `times`.
def scores(m: GraniteMoe, h: Tensor.Tensor) -> Result[Tensor.Tensor]
  out = try Nn.dense(m.head, h)
  Tensor.div(
    out,
    try Tensor.full(
      Tensor.shape(out),
      Tensor.Values.Float(m.config.logits_scaling),
      Tensor.dtype(out),
      Tensor.backend(out),
    ),
  )
end

private

def expect_shape(name: String, t: Tensor.Tensor, want: List[Int]) -> Result[Void]
  return error("#{name}: needs #{want}, got #{Tensor.shape(t)}") if Tensor.shape(t) != want
  ok()
end

## An RMS norm of `{prefix}.weight`: `beta` is unused, so zeros stand in.
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

## A weight of `{name}` of `shape`, as the checkpoint stores it.
def stacked(c: Hub.Checkpoint, name: String, shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  t = try Load.weight(c, name, shape.size, backend)
  try expect_shape(name, t, shape)
  ok(t)
end

## One decoder layer under `p` (`model.layers.N.`).
def layer(
  c: Hub.Checkpoint,
  p: String,
  config: Config,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[Layer]
  width = config.hidden_size
  head_width = width / config.num_attention_heads
  kv_width = config.num_key_value_heads * head_width
  eps = config.rms_norm_eps
  a = "#{p}self_attn."
  attention = Nn.Attention(
    query: try Load.dense(c, "#{a}q_proj", backend, weights),
    key: try Load.dense(c, "#{a}k_proj", backend, weights),
    value: try Load.dense(c, "#{a}v_proj", backend, weights),
    output: try Load.dense(c, "#{a}o_proj", backend, weights),
    heads: config.num_attention_heads,
    kv_heads: some(config.num_key_value_heads),
    rope: some(Nn.Rope(base: config.rope_theta)),
    scale: some(config.attention_multiplier),
  )

  try expect_shape("#{a}q_proj.weight", attention.query.w, [width, width])
  try expect_shape("#{a}k_proj.weight", attention.key.w, [kv_width, width])
  try expect_shape("#{a}v_proj.weight", attention.value.w, [kv_width, width])
  try expect_shape("#{a}o_proj.weight", attention.output.w, [width, width])
  m = "#{p}block_sparse_moe."
  experts = config.num_local_experts
  moe = Moe.Moe(
    router: Nn.Dense(w: try stacked(c, "#{m}router.layer.weight", [experts, width], backend), b: none, out_in: true),
    gate_up: try stacked(c, "#{m}input_linear.weight", [experts, 2 * config.intermediate_size, width], backend),
    down: try stacked(c, "#{m}output_linear.weight", [experts, width, config.intermediate_size], backend),
    top_k: config.num_experts_per_tok,
    activation: Nn.Activation.Silu,
  )

  ok(
    Layer(
      norm1: try rms(c, "#{p}input_layernorm", eps, backend),
      attention: attention,
      norm2: try rms(c, "#{p}post_attention_layernorm", eps, backend),
      moe: moe,
    ),
  )
end

## `x` `[batch, seq, width]` through a layer: each branch is scaled by `residual_multiplier` before its residual add.
def through(m: GraniteMoe, l: Layer, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  seen = try Nn.self_attention(l.attention, try Nn.layer_norm(l.norm1, x), none, true)
  h = try Tensor.add(x, try times(seen, m.config.residual_multiplier))
  routed = try Moe.moe(l.moe, try Nn.layer_norm(l.norm2, h))
  Tensor.add(h, try times(routed, m.config.residual_multiplier))
end

## The embeddings (times `embedding_multiplier`), the layers, the final norm and the head (divided by `logits_scaling`).
def graph(m: GraniteMoe, ids: Tensor.Tensor, last: Bool) -> Result[Tensor.Tensor]
  shape = Tensor.shape(ids)
  batch = shape[0]
  seq = shape[1]
  width = m.config.hidden_size
  words = try Nn.embedding(m.tokens, try Tensor.reshape(ids, [batch * seq]))
  h = try times(try Tensor.reshape(words, [batch, seq, width]), m.config.embedding_multiplier)
  for l in m.layers
    h = try through(m, l, h)
  end

  h = try Nn.layer_norm(m.final_norm, h)
  if last
    h = try Tensor.slice(h, [0, seq - 1, 0], [batch, seq, width])
  end

  scores(m, h)
end
