## The `asr/` package: Whisper's model forward over `tensor/`, `nn/` and
## `hub/`, loaded from a Hugging Face checkpoint (`asr/DESIGN.md`).
##
## `encode` turns a log-mel batch into the encoder output; `decode` turns
## token ids and that output into logits; `listen` and `step` decode one token
## at a time with a KV cache. Inference only; the log-mel frontend is
## `mel.agc`, the tokenizer `tokenizer.agc` and the rules of greedy decoding
## `greedy.agc`.

import Hub from "pkg/hub/hub"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"

## Whisper's layer norms use PyTorch's default.
EPS = 0.00001

## The prefix of every tensor `WhisperForConditionalGeneration` stores.
PREFIX = "model."

## The fields of Hugging Face's `config.json` the forward needs.
struct Config
  d_model: Int
  encoder_layers: Int
  decoder_layers: Int
  encoder_attention_heads: Int
  decoder_attention_heads: Int
  num_mel_bins: Int
  vocab_size: Int
  max_source_positions: Int
  max_target_positions: Int
end

struct Model
  config: Config
  conv1: Nn.Conv1d
  conv2: Nn.Conv1d
  encoder_positions: Nn.Embedding
  encoder: List[Nn.EncoderBlock]
  encoder_norm: Nn.LayerNorm
  tokens: Nn.Embedding
  decoder_positions: Nn.Embedding
  decoder: List[Nn.DecoderBlock]
  decoder_norm: Nn.LayerNorm
  ## The output projection, tied to `tokens`: its table transposed, no bias.
  logits: Nn.Dense
end

## One attention's keys and values, `[1, keys, d_model]` each.
struct Kv
  key: Tensor.Tensor
  value: Tensor.Tensor
end

## What `step` keeps between calls: per decoder block the projected encoder
## output, made once by `listen`, and the self-attention keys and values of
## the `length` tokens so far.
struct Cache
  cross: List[Heard]
  own: List[Kv]
  length: Int
end

## One block's cross-attention keys and values, split by head: `keys[h]` is
## `[head_width, frames]` (transposed), `values[h]` is `[frames, head_width]`.
struct Heard
  keys: List[Tensor.Tensor]
  values: List[Tensor.Tensor]
end

struct Stepped
  ## The logits of the last token fed, one per vocabulary id.
  logits: List[Float]
  cache: Cache
end

## Hugging Face's `config.json` at `path`.
def config(path: String) -> Result[Config] = within(path, JSON.decode(try File.read_text(path), Config))

## The checkpoint at `path` (a `.safetensors` file or the `.index.json` of a
## sharded one) with the `config.json` of its folder.
def open(path: String, backend: Tensor.Backend) -> Result[Model]
  parts = path.split("/")
  load(try Hub.open(path), try config(parts.take(parts.size - 1).push("config.json").join("/")), backend)
end

## The model from an open checkpoint, its shapes checked against `config`.
def load(c: Hub.Checkpoint, config: Config, backend: Tensor.Backend) -> Result[Model]
  e = "#{PREFIX}encoder."
  d = "#{PREFIX}decoder."
  width = config.d_model
  try expect_heads("encoder", width, config.encoder_attention_heads)
  try expect_heads("decoder", width, config.decoder_attention_heads)
  conv1 = try Load.conv1d(c, "#{e}conv1", 1, 1, backend)
  try expect_shape("#{e}conv1.weight", conv1.weight, [width, config.num_mel_bins, 3])
  conv2 = try Load.conv1d(c, "#{e}conv2", 2, 1, backend)
  try expect_shape("#{e}conv2.weight", conv2.weight, [width, width, 3])
  encoder_positions = try Load.embedding(c, "#{e}embed_positions.weight", backend)
  try expect_shape("#{e}embed_positions.weight", encoder_positions.table, [config.max_source_positions, width])
  tokens = try Load.embedding(c, "#{d}embed_tokens.weight", backend)
  try expect_shape("#{d}embed_tokens.weight", tokens.table, [config.vocab_size, width])
  decoder_positions = try Load.embedding(c, "#{d}embed_positions.weight", backend)
  try expect_shape("#{d}embed_positions.weight", decoder_positions.table, [config.max_target_positions, width])

  encoder: List[Nn.EncoderBlock] = []
  for i in 0...config.encoder_layers
    names = encoder_names("#{e}layers.#{i}.")
    encoder = encoder.push(
      try Load.encoder_block(c, names, config.encoder_attention_heads, Nn.Activation.Gelu, EPS, backend),
    )
  end

  decoder: List[Nn.DecoderBlock] = []
  for i in 0...config.decoder_layers
    names = decoder_names("#{d}layers.#{i}.")
    decoder = decoder.push(
      try Load.decoder_block(c, names, config.decoder_attention_heads, Nn.Activation.Gelu, EPS, backend),
    )
  end

  ok(
    Model(
      config: config,
      conv1: conv1,
      conv2: conv2,
      encoder_positions: encoder_positions,
      encoder: encoder,
      encoder_norm: try Load.layer_norm(c, "#{e}layer_norm", EPS, backend),
      tokens: tokens,
      decoder_positions: decoder_positions,
      decoder: decoder,
      decoder_norm: try Load.layer_norm(c, "#{d}layer_norm", EPS, backend),
      logits: Nn.Dense(w: try Tensor.transpose(tokens.table), b: none),
    ),
  )
end

## The encoder output `[batch, ceil(frames / 2), d_model]` of a log-mel batch
## `[batch, num_mel_bins, frames]`: two gelu convolutions (the second halves
## the frames), the positional table, the blocks and a final layer norm.
## `ceil(frames / 2)` may not pass `max_source_positions`.
def encode(m: Model, mel: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(mel)
  if shape.size != 3 or shape[1] != m.config.num_mel_bins
    return error("encode needs [batch, #{m.config.num_mel_bins}, frames], got #{shape}")
  end

  items: List[Tensor.Tensor] = []
  for b in 0...shape[0]
    one = try Tensor.reshape(try Tensor.slice(mel, [b, 0, 0], [b + 1, shape[1], shape[2]]), [shape[1], shape[2]])
    h = try Nn.activate(Nn.Activation.Gelu, try Nn.conv1d(m.conv1, one))
    h = try Nn.activate(Nn.Activation.Gelu, try Nn.conv1d(m.conv2, h))
    items = items.push(try Tensor.transpose(h))
  end

  h = try Nn.positional(m.encoder_positions, try Tensor.stack(items, 0))
  for block in m.encoder
    # On libtorch each block's intermediates are freed when it ends.
    out: Result[Tensor.Tensor] = Native.arena do Nn.encoder_block(block, h) end
    h = try out
  end

  Nn.layer_norm(m.encoder_norm, h)
end

## The logits `[batch, t, vocab_size]` of the integer token ids `[batch, t]`
## over `audio`, the output of `encode`. Every position is recomputed: no KV
## cache. `t` may not pass `max_target_positions`.
def decode(m: Model, tokens: Tensor.Tensor, audio: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(tokens)
  return error("decode needs token ids [batch, t], got #{shape}") if shape.size != 2
  heard = Tensor.shape(audio)
  if heard.size != 3 or heard[0] != shape[0] or heard[2] != m.config.d_model
    return error("decode needs audio [#{shape[0]}, frames, #{m.config.d_model}] from encode, got #{heard}")
  end

  rows = try Nn.embedding(m.tokens, try Tensor.reshape(tokens, [shape[0] * shape[1]]))
  h = try Nn.positional(m.decoder_positions, try Tensor.reshape(rows, [shape[0], shape[1], m.config.d_model]))
  for block in m.decoder
    h = try Nn.decoder_block(block, h, audio)
  end

  Nn.dense(m.logits, try Nn.layer_norm(m.decoder_norm, h))
end

## The empty cache over `audio`, the `[1, frames, d_model]` output of
## `encode`: the cross-attention keys and values of every decoder block.
def listen(m: Model, audio: Tensor.Tensor) -> Result[Cache]
  heard = Tensor.shape(audio)
  if heard.size != 3 or heard[0] != 1 or heard[2] != m.config.d_model
    return error("listen needs audio [1, frames, #{m.config.d_model}] from encode, got #{heard}")
  end

  frames = heard[1]
  d = m.config.d_model
  cross: List[Heard] = []
  for block in m.decoder
    a = block.cross_attention
    k = try Tensor.reshape(try Nn.dense(a.key, audio), [frames, d])
    v = try Tensor.reshape(try Nn.dense(a.value, audio), [frames, d])
    width = d / a.heads
    keys: List[Tensor.Tensor] = []
    values: List[Tensor.Tensor] = []
    for h in 0...a.heads
      keys = keys.push(try Tensor.transpose(try Tensor.slice(k, [0, h * width], [frames, (h + 1) * width])))
      values = values.push(try Tensor.slice(v, [0, h * width], [frames, (h + 1) * width]))
    end

    cross = cross.push(Heard(keys: keys, values: values))
  end

  ok(Cache(cross: cross, own: [], length: 0))
end

## `ids` fed after the tokens of `cache`: the logits of the last one, as
## `decode` gives them, and the cache that holds them all. Only an empty
## cache takes several ids at once (a prompt).
def step(m: Model, cache: Cache, ids: List[Int]) -> Result[Stepped]
  n = ids.size
  return error("step needs ids") if n == 0
  return error("step takes one id after the first call, got #{n}") if n > 1 and cache.length > 0
  if cache.length + n > m.config.max_target_positions
    return error("step needs at most #{m.config.max_target_positions} tokens, got #{cache.length + n}")
  end

  Native.arena do
    d = m.config.d_model
    backend = Tensor.backend(m.tokens.table)
    rows = try Nn.embedding(
      m.tokens,
      try Tensor.from_list(ids.map { |i| Tensor.Values.Int(i) }, [n], Tensor.Dtype.I64, backend),
    )

    positions = try Tensor.slice(m.decoder_positions.table, [cache.length, 0], [cache.length + n, d])
    h = try Tensor.reshape(try Tensor.add(rows, positions), [1, n, d])
    mask: Optional[Tensor.Tensor] = none
    if n > 1
      mask = some(try Nn.causal_mask(n, Tensor.dtype(h), backend))
    end

    own: List[Kv] = []
    for i in 0...m.decoder.size
      block = m.decoder[i]
      a = block.self_attention
      x = try Nn.layer_norm(block.norm1, h)
      k = try Nn.dense(a.key, x)
      v = try Nn.dense(a.value, x)
      if cache.length > 0
        k = try Tensor.concat([cache.own[i].key, k], 1)
        v = try Tensor.concat([cache.own[i].value, v], 1)
      end

      own = own.push(Kv(key: k, value: v))
      h = try Tensor.add(
        h,
        try Nn.dense(a.output, try Tensor.multi_head_attention(try Nn.dense(a.query, x), k, v, a.heads, mask)),
      )

      c = block.cross_attention
      x = try Nn.layer_norm(block.norm2, h)
      heard = try attend(try Tensor.reshape(try Nn.dense(c.query, x), [n, d]), cache.cross[i])
      h = try Tensor.add(h, try Nn.dense(c.output, try Tensor.reshape(heard, [1, n, d])))
      h = try Tensor.add(h, try Nn.mlp(block.mlp, try Nn.layer_norm(block.norm3, h)))
    end

    last = try Tensor.slice(h, [0, n - 1, 0], [1, n, d])
    logits = try Nn.dense(m.logits, try Nn.layer_norm(m.decoder_norm, last))
    ok(Stepped(logits: try floats(logits), cache: Cache(cross: cache.cross, own: own, length: cache.length + n)))
  end
end

private

## Attention of the queries `[n, d_model]` over one block's cached heads, as
## `Tensor.multi_head_attention` computes it, without slicing the keys and
## values again at every step.
def attend(queries: Tensor.Tensor, heard: Heard) -> Result[Tensor.Tensor]
  n = Tensor.shape(queries)[0]
  width = Tensor.shape(heard.values[0])[1]
  scale = try Tensor.full(
    [n, width],
    Tensor.Values.Float(1.0 / Math.sqrt(width.to_f)),
    Tensor.Dtype.F32,
    Tensor.backend(queries),
  )

  parts: List[Tensor.Tensor] = []
  for h in 0...heard.keys.size
    q = try Tensor.mul(try Tensor.slice(queries, [0, h * width], [n, (h + 1) * width]), scale)
    parts = parts.push(try Tensor.matmul(try Tensor.softmax(try Tensor.matmul(q, heard.keys[h]), 1), heard.values[h]))
  end

  Tensor.concat(parts, 1)
end

def floats(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("the logits hold Float values")
    end
  end

  ok(out)
end

## Whisper's names for encoder block `p` (`model.encoder.layers.N.`).
def encoder_names(p: String) -> Load.BlockNames
  Load.BlockNames(
    norm1: "#{p}self_attn_layer_norm",
    query: "#{p}self_attn.q_proj",
    key: "#{p}self_attn.k_proj",
    value: "#{p}self_attn.v_proj",
    output: "#{p}self_attn.out_proj",
    norm2: "#{p}final_layer_norm",
    up: "#{p}fc1",
    down: "#{p}fc2",
  )
end

## Whisper's names for decoder block `p` (`model.decoder.layers.N.`).
def decoder_names(p: String) -> Load.DecoderBlockNames
  Load.DecoderBlockNames(
    norm1: "#{p}self_attn_layer_norm",
    query: "#{p}self_attn.q_proj",
    key: "#{p}self_attn.k_proj",
    value: "#{p}self_attn.v_proj",
    output: "#{p}self_attn.out_proj",
    norm2: "#{p}encoder_attn_layer_norm",
    cross_query: "#{p}encoder_attn.q_proj",
    cross_key: "#{p}encoder_attn.k_proj",
    cross_value: "#{p}encoder_attn.v_proj",
    cross_output: "#{p}encoder_attn.out_proj",
    norm3: "#{p}final_layer_norm",
    up: "#{p}fc1",
    down: "#{p}fc2",
  )
end

def expect_heads(stack: String, width: Int, heads: Int) -> Result[Void]
  if heads < 1 or width % heads != 0
    return error("the #{stack} needs heads >= 1 dividing d_model #{width}, got #{heads}")
  end

  ok()
end

def expect_shape(name: String, t: Tensor.Tensor, want: List[Int]) -> Result[Void]
  return error("#{name}: the config needs #{want}, got #{Tensor.shape(t)}") if Tensor.shape(t) != want
  ok()
end

## `r`, with `prefix: ` before its error.
def within[T](prefix: String, r: Result[T]) -> Result[T]
  case r
    ok(v) -> ok(v)
    error(e) -> error("#{prefix}: #{e.message}")
  end
end
