## Qwen3 for inference, from a Hugging Face checkpoint (`transformers/DESIGN.md`):
## the embedding table, pre-norm decoder blocks with grouped-query attention
## (query and key norms per head, rotary positions with `rope_theta`) and a
## SwiGLU MLP, RMS norms, and the output head, tied to the embedding table when
## the config says so. This is the forward over a whole sequence, with no cache.
##
## `Qwen/Qwen3-0.6B` is `bfloat16` on disk; the weights are `F32` in memory.

import Hub from "pkg/hub/hub"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Lift from "lift"
import Source from "tokenizer/source"

## The fields of `config.json` that the model reads.
struct Config
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
  rope_theta: Float
  hidden_act: String
  tie_word_embeddings: Bool = true
  attention_bias: Bool = false
  use_sliding_window: Bool = false
  rope_scaling: Optional[Scaling]
  sliding_window: Optional[Int]
end

## Any `rope_scaling` object: the model has none.
struct Scaling
  rope_type: Optional[String]
end

struct Qwen3
  config: Config
  tokens: Nn.Embedding
  ## The blocks and the final RMS norm.
  stack: Nn.EncoderStack
  ## `[vocab_size, hidden_size]` (`out_in`): the embedding table itself when tied.
  head: Nn.Dense
end

## The model in `dir`, which holds `config.json` and `model.safetensors` (or
## the `model.safetensors.index.json` of a sharded one).
def open(dir: String, backend: Tensor.Backend, weights: Tensor.Dtype = Tensor.Dtype.F32) -> Result[Qwen3]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  config = try Source.within("#{dir}/config.json", JSON.decode(text, Config))
  file = "#{dir}/model.safetensors"
  if not Path.real(file).ok?
    file = "#{dir}/model.safetensors.index.json"
  end

  load(try Hub.open(file), config, backend, weights)
end

## The model of an open checkpoint, its shapes checked against `config`.
def load(
  c: Hub.Checkpoint,
  config: Config,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[Qwen3]
  try check_config(config)
  width = config.hidden_size
  tokens = try Load.embedding(c, "model.embed_tokens.weight", backend, weights)
  try expect_shape("model.embed_tokens.weight", tokens.table, [config.vocab_size, width])
  blocks: List[Nn.EncoderBlock] = []
  for i in 0...config.num_hidden_layers
    blocks = blocks.push(try block(c, "model.layers.#{i}.", config, backend, weights))
  end

  head = try head_of(c, config, tokens, backend, weights)
  ok(
    Qwen3(
      config: config,
      tokens: tokens,
      stack: Nn.EncoderStack(blocks: blocks, final_norm: some(try rms(c, "model.norm", config.rms_norm_eps, backend))),
      head: head,
    ),
  )
end

## The batch `rows` of token ids (all of one length), through the model.
def run(m: Qwen3, rows: List[List[Int]], last: Bool = false) -> Result[Tensor.Tensor]
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

## The logits `[batch, seq, vocab_size]` of the token ids `ids` (`I32`,
## `[batch, seq]`), each position reading the ones before it and itself; with
## `last`, only the last position's, `[batch, 1, vocab_size]` (the one that
## picks the next token). `seq` may not pass `max_position_embeddings`.
def logits(m: Qwen3, ids: Tensor.Tensor, last: Bool = false) -> Result[Tensor.Tensor]
  shape = Tensor.shape(ids)
  return error("logits needs ids of [batch, seq], got #{shape}") if shape.size != 2
  if shape[1] > m.config.max_position_embeddings
    return error("#{shape[1]} tokens pass max_position_embeddings #{m.config.max_position_embeddings}")
  end

  out = try Lift.staged(
    m,
    ->(p: Qwen3) { traced(p) },
    [ids],
    ->(p: Qwen3, xs: List[Tensor.Tensor]) { ok([try graph(p, xs[0], last)]) },
  )

  ok(out[0])
end

## Closes the weights of `m` now. Nothing may use `m` afterwards: a call that
## does is an error, never a read of freed memory. The logits that `run` gave
## belong to the caller (`Lift.close`).
def close_model(m: Qwen3) -> Void
  tensors = [m.tokens.table] + Lift.dense_parts(m.head)
  for b in m.stack.blocks
    tensors = tensors + Lift.block_parts(b)
  end

  case m.stack.final_norm
    some(n) -> tensors = tensors + Lift.norm_parts(n)
    none -> nil
  end

  Lift.close(tensors)
end

## The tensors of `m` as `Trace` constants (`Lift`). Internal: the graphs of `Generate` use it.
def traced(m: Qwen3) -> Qwen3
  m.with(
    tokens: Lift.embedding(m.tokens),
    stack: Nn.EncoderStack(
      blocks: m.stack.blocks.map { |b| Lift.block(b) },
      final_norm: m.stack.final_norm.map { |n| Lift.norm(n) },
    ),
    head: Lift.dense(m.head),
  )
end

private

def check_config(config: Config) -> Result[Void]
  return error("model_type is #{config.model_type}, not qwen3") if config.model_type != "qwen3"
  return error("hidden_act #{config.hidden_act} is not supported, only silu") if config.hidden_act != "silu"
  return error("attention_bias is not supported") if config.attention_bias
  return error("rope_scaling is not supported") if config.rope_scaling.some?
  return error("sliding window attention is not supported") if config.use_sliding_window
  if config.num_key_value_heads < 1 or config.num_attention_heads % config.num_key_value_heads != 0
    return error("#{config.num_attention_heads} heads are not a multiple of #{config.num_key_value_heads} key heads")
  end

  return error("head_dim #{config.head_dim} is not even, which rope needs") if config.head_dim % 2 != 0
  ok()
end

def expect_shape(name: String, t: Tensor.Tensor, want: List[Int]) -> Result[Void]
  return error("#{name}: needs #{want}, got #{Tensor.shape(t)}") if Tensor.shape(t) != want
  ok()
end

## The output head: the embedding table when the config ties them
## (the file's `lm_head.weight`, if any, is then ignored), else `lm_head`.
def head_of(
  c: Hub.Checkpoint,
  config: Config,
  tokens: Nn.Embedding,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[Nn.Dense]
  return ok(Nn.Dense(w: tokens.table, b: none, out_in: true)) if config.tie_word_embeddings

  head = try Load.dense(c, "lm_head", backend, weights)
  try expect_shape("lm_head.weight", head.w, [config.vocab_size, config.hidden_size])
  ok(head)
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

## One decoder block under `p` (`model.layers.N.`).
def block(
  c: Hub.Checkpoint,
  p: String,
  config: Config,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[Nn.EncoderBlock]
  width = config.hidden_size
  q_width = config.num_attention_heads * config.head_dim
  kv_width = config.num_key_value_heads * config.head_dim
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
    query_norm: some(try rms(c, "#{a}q_norm", eps, backend)),
    key_norm: some(try rms(c, "#{a}k_norm", eps, backend)),
  )

  try expect_shape("#{a}q_proj.weight", attention.query.w, [q_width, width])
  try expect_shape("#{a}k_proj.weight", attention.key.w, [kv_width, width])
  try expect_shape("#{a}v_proj.weight", attention.value.w, [kv_width, width])
  try expect_shape("#{a}o_proj.weight", attention.output.w, [width, q_width])
  try expect_shape("#{a}q_norm.weight", attention.query_norm.unwrap!.gamma, [config.head_dim])
  try expect_shape("#{a}k_norm.weight", attention.key_norm.unwrap!.gamma, [config.head_dim])
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
  ok(
    Nn.EncoderBlock(
      norm1: try rms(c, "#{p}input_layernorm", eps, backend),
      attention: attention,
      norm2: try rms(c, "#{p}post_attention_layernorm", eps, backend),
      mlp: mlp,
    ),
  )
end

## The embeddings, the causal blocks, the final norm and the head.
def graph(m: Qwen3, ids: Tensor.Tensor, last: Bool) -> Result[Tensor.Tensor]
  shape = Tensor.shape(ids)
  batch = shape[0]
  seq = shape[1]
  width = m.config.hidden_size
  words = try Nn.embedding(m.tokens, try Tensor.reshape(ids, [batch * seq]))
  h = try Nn.encoder_stack(m.stack, try Tensor.reshape(words, [batch, seq, width]), none, true)
  if last
    h = try Tensor.slice(h, [0, seq - 1, 0], [batch, seq, width])
  end

  Nn.dense(m.head, h)
end
