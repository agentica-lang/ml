## Qwen3 and Qwen2.5 for inference, from a Hugging Face checkpoint (`transformers/DESIGN.md`):
## the embedding table, pre-norm decoder blocks with grouped-query attention
## (query and key norms per head, rotary positions with `rope_theta`) and a
## SwiGLU MLP, RMS norms, and the output head, tied to the embedding table when
## the config says so. This is the forward over a whole sequence, with no cache.
## `model_type` `qwen2` (Qwen2.5) has no query and key norms, a bias on the query, key and
## value projections, and a `head_dim` of `hidden_size / num_attention_heads`.
##
## `Qwen/Qwen3-0.6B` is `bfloat16` on disk; the weights are `F32` in memory.

import Gguf from "pkg/hub/gguf"
import Hub from "pkg/hub/hub"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Lift from "lift"
import Placement from "placement"
import Source from "tokenizer/source"

## The fields of `config.json` that the model reads.
struct Config
  model_type: String
  hidden_size: Int
  num_hidden_layers: Int
  num_attention_heads: Int
  num_key_value_heads: Int
  ## 0 when the config has none (Qwen2): `hidden_size / num_attention_heads`.
  head_dim: Int = 0
  intermediate_size: Int
  vocab_size: Int
  max_position_embeddings: Int
  rms_norm_eps: Float
  rope_theta: Float
  hidden_act: String
  tie_word_embeddings: Bool = true
  attention_bias: Bool = false
  mlp_bias: Bool = false
  use_sliding_window: Bool = false
  rope_scaling: Optional[Scaling]
  sliding_window: Optional[Int]
end

## The `image_token_id` of a `config.json`.
struct ImageToken
  image_token_id: Int
end

## `config.json` of a SmolVLM folder: the text model is in `text_config`.
struct Idefics3Document
  model_type: String
  text_config: Config
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
  ## `[vocab_size, hidden_size]` (`out_in`): the embedding table itself when tied, or a copy of it when the head is on
  ## another backend than the table (`layers_on_device`).
  head: Nn.Dense
  ## The backend the model was opened on: where the token ids and the first blocks are, whatever side the table is on
  ## (`layers_on_device`).
  device: Tensor.Backend
end

## An open checkpoint and its config, as `open` reads them.
struct Opened
  checkpoint: Hub.Checkpoint
  config: Config
end

## The model in `dir`, which holds `config.json` and `model.safetensors` (or
## the `model.safetensors.index.json` of a sharded one), or in the `.gguf` file
## `dir` names, whose key-values give the config.
## `layers_on_device: n` loads the weights of the blocks from the `n`th on on the CPU (`Ggml(Cpu)`), and a step runs
## those blocks there, with their KV cache in host memory (`kv_decoder.agc`); the first `n` blocks stay on the Vulkan
## device. With `n` below the number of blocks the embedding table is on the CPU, as in llama.cpp's `-ngl`, and the head
## and the final norm are on the device (a tied head is a second copy of the table there) so that the CPU does not do
## the product with the vocabulary and the card is idle less; with `n` of 0 they are on the CPU too. `none` keeps every
## block, the table and the head on the device, and on any other backend, or with an `n` of the number of blocks or
## more, `n` changes nothing.
def open(
  dir: String,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
  layers_on_device: Optional[Int] = none,
) -> Result[Qwen3]
  found = try opened(dir)
  load(found.checkpoint, found.config, backend, weights, layers_on_device)
end

## The `layers_on_device` for `open` on a Vulkan device that fills the card: the most blocks whose layout, with the
## KV cache of `context` tokens (`F32`) for each, fits the free memory of the device (`Tensor.device_memory`) less
## `reserve` bytes for the compute buffers, and not more than `max_device_bytes` when it is `some`. `none` on any other
## backend, where `layers_on_device` changes nothing. With `kv: Placement.Kv.Host` the cache is in host memory and counts
## for nothing (the step of the prompt still makes its cache on the card first, so leave room in `reserve` for it). The
## count is a reading of the moment: it does not hold the memory.
def layers_that_fit(
  dir: String,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
  max_device_bytes: Optional[Int] = none,
  context: Int = 2048,
  reserve: Int = 256000000,
  kv: Placement.Kv = Placement.Kv.Device,
) -> Result[Optional[Int]]
  case backend
    Tensor.Backend.Ggml(Tensor.Device.Vulkan(_)) -> nil
    _ -> return ok(none)
  end

  found = try opened(dir)
  free = (try Tensor.device_memory(backend)).free
  budget = Placement.budget(free, reserve, max_device_bytes)
  ok(some(try layers_within(found.checkpoint, found.config, weights, budget, context, kv)))
end

## The most blocks of the model of `c` that `budget` bytes of the card hold with `layers_on_device` (`Placement.fit`
## over the footprint of the checkpoint: the KV cache of `context` tokens in `F32` counted for each block, unless `kv` is
## `Host`).
def layers_within(
  c: Hub.Checkpoint,
  config: Config,
  weights: Tensor.Dtype,
  budget: Int,
  context: Int,
  kv: Placement.Kv = Placement.Kv.Device,
) -> Result[Int]
  config = try with_head_dim(config)
  cache = 0
  if kv == Placement.Kv.Device
    cache = 2 * config.num_key_value_heads * config.head_dim * context * 4
  end

  footprint = try Placement.footprint(
    c,
    "model.layers.",
    config.num_hidden_layers,
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.norm.weight",
    weights,
    cache,
    some(config.tie_word_embeddings),
  )

  ok(Placement.fit(footprint, budget))
end

## The `config.json` of the model folder `dir`.
def read_config(dir: String) -> Result[Config]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  Source.within("#{dir}/config.json", JSON.decode(text, Config))
end

## The Llama text model of the folder `dir` of a SmolVLM (`Idefics3ForConditionalGeneration`, `model_type` `idefics3`):
## the `text_config` of its `config.json`, its tensors under `model.text_model.` and its head `lm_head.weight`. The
## tower and the connector in the same file are not read, and `weights(m)` still names the tensors of a plain model
## (`model.layers...`).
def open_idefics3(
  dir: String,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
  layers_on_device: Optional[Int] = none,
) -> Result[Qwen3]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  file = try Source.within("#{dir}/config.json", JSON.decode(text, Idefics3Document))
  return error("model_type is #{file.model_type}, not idefics3") if file.model_type != "idefics3"
  load(try checkpoint(dir), file.text_config, backend, weights, layers_on_device, "model.text_model")
end

## The id of the image token of the SmolVLM folder `dir` (`image_token_id` of its `config.json`), the one that a
## `KvDecoder.Sight` names.
def image_token(dir: String) -> Result[Int]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  ok((try Source.within("#{dir}/config.json", JSON.decode(text, ImageToken))).image_token_id)
end

## The weights of the model folder `dir`: its `model.safetensors`, or the `model.safetensors.index.json` of a sharded one.
def checkpoint(dir: String) -> Result[Hub.Checkpoint]
  file = "#{dir}/model.safetensors"
  if not Path.real(file).ok?
    file = "#{dir}/model.safetensors.index.json"
  end

  Hub.open(file)
end

## The config of a `qwen3` GGUF, from the key-values that llama.cpp's converter writes. The output head is the file's
## `output.weight` when there is one (llama.cpp writes it even for a tied model), else the embedding table.
def gguf_config(c: Hub.Checkpoint) -> Result[Config]
  h = try c.gguf.to_result("the checkpoint is not a GGUF file")
  arch = try Gguf.string(h, "general.architecture")
  keys = try Gguf.int(h, "qwen3.attention.key_length")
  if keys != try Gguf.int(h, "qwen3.attention.value_length")
    return error("the key length #{keys} and the value length differ")
  end

  scaling: Optional[Scaling] = none
  if Gguf.has?(h, "qwen3.rope.scaling.type")
    scaling = some(Scaling(rope_type: some("in the GGUF")))
  end

  ok(
    Config(
      model_type: arch,
      hidden_size: try Gguf.int(h, "qwen3.embedding_length"),
      num_hidden_layers: try Gguf.int(h, "qwen3.block_count"),
      num_attention_heads: try Gguf.int(h, "qwen3.attention.head_count"),
      num_key_value_heads: try Gguf.int(h, "qwen3.attention.head_count_kv"),
      head_dim: keys,
      intermediate_size: try Gguf.int(h, "qwen3.feed_forward_length"),
      vocab_size: (try Hub.shape(c, "model.embed_tokens.weight"))[0],
      max_position_embeddings: try Gguf.int(h, "qwen3.context_length"),
      rms_norm_eps: try Gguf.float(h, "qwen3.attention.layer_norm_rms_epsilon"),
      rope_theta: try Gguf.float(h, "qwen3.rope.freq_base"),
      hidden_act: "silu",
      tie_word_embeddings: not Hub.has?(c, "lm_head.weight"),
      rope_scaling: scaling,
      sliding_window: none,
    ),
  )
end

## `config` with its `head_dim` filled in when `config.json` leaves it out: the hidden size over the heads.
def with_head_dim(config: Config) -> Result[Config]
  return ok(config) if config.head_dim != 0
  if config.num_attention_heads < 1 or config.hidden_size % config.num_attention_heads != 0
    return error("hidden_size #{config.hidden_size} is not a multiple of #{config.num_attention_heads} heads")
  end

  ok(config.with(head_dim: config.hidden_size / config.num_attention_heads))
end

## An error when `config` is a model this file does not run: another type, activation, attention bias, rope scaling or
## sliding window, or head counts that do not divide.
def check_config(config: Config) -> Result[Void]
  if config.model_type != "qwen3" and config.model_type != "qwen2" and config.model_type != "llama"
    return error("model_type is #{config.model_type}, not qwen3, qwen2 or llama")
  end

  return error("hidden_act #{config.hidden_act} is not supported, only silu") if config.hidden_act != "silu"
  return error("attention_bias is not supported") if config.attention_bias
  return error("mlp_bias is not supported") if config.mlp_bias
  return error("rope_scaling is not supported") if config.rope_scaling.some?
  return error("sliding window attention is not supported") if config.use_sliding_window
  if config.num_key_value_heads < 1 or config.num_attention_heads % config.num_key_value_heads != 0
    return error("#{config.num_attention_heads} heads are not a multiple of #{config.num_key_value_heads} key heads")
  end

  return error("head_dim #{config.head_dim} is not even, which rope needs") if config.head_dim % 2 != 0
  ok()
end

## The model of an open checkpoint, its shapes checked against `config`. The tensors of the embedding table, the blocks
## and the final norm are under `root` (`model.`), and the head is `lm_head` at the top.
def load(
  c: Hub.Checkpoint,
  config: Config,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
  layers_on_device: Optional[Int] = none,
  root: String = "model",
) -> Result[Qwen3]
  config = try with_head_dim(config)
  try check_config(config)
  if layers_on_device.some? and layers_on_device.unwrap! < 0
    return error("layers_on_device is 0 or more, got #{layers_on_device.unwrap!}")
  end

  width = config.hidden_size
  side = Placement.placement(backend, config.num_hidden_layers, layers_on_device)
  table = "#{root}.embed_tokens.weight"
  tokens = try Load.embedding(c, table, side.table, weights)
  try expect_shape(table, tokens.table, [config.vocab_size, width])
  blocks: List[Nn.EncoderBlock] = []
  for i in 0...config.num_hidden_layers
    blocks = blocks.push(
      try block(c, "#{root}.layers.#{i}.", config, Placement.layer_backend(backend, i, layers_on_device), weights),
    )
  end

  head = try head_of(c, config, tokens, side.head, weights, table)
  ok(
    Qwen3(
      config: config,
      tokens: tokens,
      stack: Nn.EncoderStack(
        blocks: blocks,
        final_norm: some(try rms(c, "#{root}.norm", config.rms_norm_eps, side.head)),
      ),
      head: head,
      device: backend,
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

  tensor = try Tensor.from_list(ids, [rows.size, seq], Tensor.Dtype.I32, m.device)
  out = logits(m, tensor, last)
  Lift.close([tensor])
  out
end

## The logits `[batch, seq, vocab_size]` of the token ids `ids` (`I32`,
## `[batch, seq]`), each position reading the ones before it and itself; with
## `last`, only the last position's, `[batch, 1, vocab_size]` (the one that
## picks the next token). `seq` may not pass `max_position_embeddings`. `ids` is on `m.device`, the backend the model
## was opened on (its table may be on the CPU with `layers_on_device`).
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

## The names of the embedding tables among `weights(m)`: what `Lora.save` of `train` must not write as the adapter
## of a linear layer.
def embeddings(_m: Qwen3) -> List[String] = ["model.embed_tokens.weight"]

## The weights of `m` by their Hugging Face names (`model.layers.0.self_attn.q_proj.weight`), for training part of the
## model (`Params.split` of `train`). A tied head is the embedding table and has no name of its own.
def weights(m: Qwen3) -> Map[String, Tensor.Tensor]
  out = {"model.embed_tokens.weight": m.tokens.table}
  if not m.config.tie_word_embeddings
    out = dense_weights(out, "lm_head", m.head)
  end

  for i in 0...m.stack.blocks.size
    b = m.stack.blocks[i]
    p = "model.layers.#{i}."
    out = dense_weights(out, "#{p}self_attn.q_proj", b.attention.query)
    out = dense_weights(out, "#{p}self_attn.k_proj", b.attention.key)
    out = dense_weights(out, "#{p}self_attn.v_proj", b.attention.value)
    out = dense_weights(out, "#{p}self_attn.o_proj", b.attention.output)
    out = norm_weights(out, "#{p}self_attn.q_norm", b.attention.query_norm)
    out = norm_weights(out, "#{p}self_attn.k_norm", b.attention.key_norm)
    out = norm_weights(out, "#{p}input_layernorm", some(b.norm1))
    out = norm_weights(out, "#{p}post_attention_layernorm", some(b.norm2))
    out = dense_weights(out, "#{p}mlp.up_proj", b.mlp.up)
    out = dense_weights(out, "#{p}mlp.down_proj", b.mlp.down)
    case b.mlp.gate
      some(g) -> out = dense_weights(out, "#{p}mlp.gate_proj", g)
      none -> nil
    end
  end

  norm_weights(out, "model.norm", m.stack.final_norm)
end

## `m` with the tensors of `weights` in place of its own: the names must be those of `weights(m)`, no more and no
## fewer, each with the shape the model holds. A tied head is rebuilt on the side of the table, which on a model with
## `layers_on_device` below its blocks is the CPU. Inside a loss the tensors are traced and `logits` of the result is part
## of the trace.
def with_weights(m: Qwen3, weights: Map[String, Tensor.Tensor]) -> Result[Qwen3]
  try fitting(m, weights)
  tokens = Nn.Embedding(table: try named(weights, "model.embed_tokens.weight"))
  head = Nn.Dense(w: tokens.table, b: none, out_in: true)
  if not m.config.tie_word_embeddings
    head = try with_dense(weights, "lm_head", m.head)
  end

  blocks: List[Nn.EncoderBlock] = []
  for i in 0...m.stack.blocks.size
    b = m.stack.blocks[i]
    p = "model.layers.#{i}."
    gate: Optional[Nn.Dense] = none
    case b.mlp.gate
      some(g) -> gate = some(try with_dense(weights, "#{p}mlp.gate_proj", g))
      none -> nil
    end

    attention = b.attention.with(
      query: try with_dense(weights, "#{p}self_attn.q_proj", b.attention.query),
      key: try with_dense(weights, "#{p}self_attn.k_proj", b.attention.key),
      value: try with_dense(weights, "#{p}self_attn.v_proj", b.attention.value),
      output: try with_dense(weights, "#{p}self_attn.o_proj", b.attention.output),
      query_norm: try with_norm(weights, "#{p}self_attn.q_norm", b.attention.query_norm),
      key_norm: try with_norm(weights, "#{p}self_attn.k_norm", b.attention.key_norm),
    )

    mlp = b.mlp.with(
      up: try with_dense(weights, "#{p}mlp.up_proj", b.mlp.up),
      down: try with_dense(weights, "#{p}mlp.down_proj", b.mlp.down),
      gate: gate,
    )

    blocks = blocks.push(
      b.with(
        norm1: (try with_norm(weights, "#{p}input_layernorm", some(b.norm1))).unwrap!,
        attention: attention,
        norm2: (try with_norm(weights, "#{p}post_attention_layernorm", some(b.norm2))).unwrap!,
        mlp: mlp,
      ),
    )
  end

  final_norm = try with_norm(weights, "model.norm", m.stack.final_norm)
  ok(m.with(tokens: tokens, stack: Nn.EncoderStack(blocks: blocks, final_norm: final_norm), head: head))
end

private

## The checkpoint and the config of `dir`: a GGUF file, or a folder with `config.json` and the safetensors.
def opened(dir: String) -> Result[Opened]
  if dir.end_with?(".gguf")
    c = try Hub.open(dir)
    return ok(Opened(checkpoint: c, config: try gguf_config(c)))
  end

  ok(Opened(checkpoint: try checkpoint(dir), config: try read_config(dir)))
end

## An error when `given` has a name `m` lacks or a tensor of another shape than the model's.
def fitting(m: Qwen3, given: Map[String, Tensor.Tensor]) -> Result[Void]
  own = weights(m)
  for name in given.keys.sort
    old = try own.get(name).to_result("the model has no weight #{name}")
    if Tensor.shape(given[name]) != Tensor.shape(old)
      return error("#{name}: the model holds #{Tensor.shape(old)}, got #{Tensor.shape(given[name])}")
    end
  end

  ok()
end

def dense_weights(out: Map[String, Tensor.Tensor], prefix: String, d: Nn.Dense) -> Map[String, Tensor.Tensor]
  named = out.put("#{prefix}.weight", d.w)
  case d.b
    some(b) -> named.put("#{prefix}.bias", b)
    none -> named
  end
end

## An RMS norm has one weight; its `beta` is not used.
def norm_weights(
  out: Map[String, Tensor.Tensor],
  prefix: String,
  n: Optional[Nn.LayerNorm],
) -> Map[String, Tensor.Tensor]
  case n
    some(p) -> out.put("#{prefix}.weight", p.gamma)
    none -> out
  end
end

def named(weights: Map[String, Tensor.Tensor], name: String) -> Result[Tensor.Tensor]
  weights.get(name).to_result("the weights have no #{name}")
end

def with_dense(weights: Map[String, Tensor.Tensor], prefix: String, d: Nn.Dense) -> Result[Nn.Dense]
  w = try named(weights, "#{prefix}.weight")
  case d.b
    some(_) -> ok(d.with(w: w, b: some(try named(weights, "#{prefix}.bias"))))
    none -> ok(d.with(w: w))
  end
end

def with_norm(
  weights: Map[String, Tensor.Tensor],
  prefix: String,
  n: Optional[Nn.LayerNorm],
) -> Result[Optional[Nn.LayerNorm]]
  case n
    some(p) -> ok(some(p.with(gamma: try named(weights, "#{prefix}.weight"))))
    none -> ok(none)
  end
end

def expect_shape(name: String, t: Tensor.Tensor, want: List[Int]) -> Result[Void]
  return error("#{name}: needs #{want}, got #{Tensor.shape(t)}") if Tensor.shape(t) != want
  ok()
end

## The output head on `backend`: the embedding table when the config ties them and it is there (else a copy of it on
## `backend`; the file's `lm_head.weight`, if any, is then ignored), else `lm_head`.
def head_of(
  c: Hub.Checkpoint,
  config: Config,
  tokens: Nn.Embedding,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
  table: String,
) -> Result[Nn.Dense]
  if config.tie_word_embeddings
    return ok(Nn.Dense(w: tokens.table, b: none, out_in: true)) if Tensor.backend(tokens.table) == backend

    copy = try Load.embedding(c, table, backend, weights)
    return ok(Nn.Dense(w: copy.table, b: none, out_in: true))
  end

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
  query_norm: Optional[Nn.LayerNorm] = none
  key_norm: Optional[Nn.LayerNorm] = none
  if config.model_type == "qwen3"
    query_norm = some(try rms(c, "#{a}q_norm", eps, backend))
    key_norm = some(try rms(c, "#{a}k_norm", eps, backend))
    try expect_shape("#{a}q_norm.weight", query_norm.unwrap!.gamma, [config.head_dim])
    try expect_shape("#{a}k_norm.weight", key_norm.unwrap!.gamma, [config.head_dim])
  end

  attention = Nn.Attention(
    query: try Load.dense(c, "#{a}q_proj", backend, weights),
    key: try Load.dense(c, "#{a}k_proj", backend, weights),
    value: try Load.dense(c, "#{a}v_proj", backend, weights),
    output: try Load.dense(c, "#{a}o_proj", backend, weights),
    heads: config.num_attention_heads,
    kv_heads: some(config.num_key_value_heads),
    rope: some(Nn.Rope(base: config.rope_theta)),
    query_norm: query_norm,
    key_norm: key_norm,
  )

  try expect_shape("#{a}q_proj.weight", attention.query.w, [q_width, width])
  try expect_shape("#{a}k_proj.weight", attention.key.w, [kv_width, width])
  try expect_shape("#{a}v_proj.weight", attention.value.w, [kv_width, width])
  try expect_shape("#{a}o_proj.weight", attention.output.w, [width, q_width])
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
