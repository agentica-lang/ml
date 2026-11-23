## The `asr/` package: Whisper's model forward over `tensor/`, `nn/` and
## `hub/`, loaded from a Hugging Face checkpoint (`asr/DESIGN.md`).
##
## `encode` turns a log-mel batch into the encoder output; `decode` turns
## token ids and that output into logits. Inference only: no frontend, no
## tokenizer, no decoding loop.

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

## The encoder output `[batch, frames / 2, d_model]` of a log-mel batch
## `[batch, num_mel_bins, frames]`: two gelu convolutions (the second halves
## the frames), the positional table, the blocks and a final layer norm.
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
    h = try Nn.encoder_block(block, h)
  end

  Nn.layer_norm(m.encoder_norm, h)
end

## The logits `[batch, t, vocab_size]` of the integer token ids `[batch, t]`
## over `audio`, the output of `encode`. Every position is recomputed: no KV
## cache.
def decode(m: Model, tokens: Tensor.Tensor, audio: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(tokens)
  return error("decode needs token ids [batch, t], got #{shape}") if shape.size != 2
  rows: List[Tensor.Tensor] = []
  for b in 0...shape[0]
    ids = try Tensor.reshape(try Tensor.slice(tokens, [b, 0], [b + 1, shape[1]]), [shape[1]])
    rows = rows.push(try Nn.embedding(m.tokens, ids))
  end

  h = try Nn.positional(m.decoder_positions, try Tensor.stack(rows, 0))
  for block in m.decoder
    h = try Nn.decoder_block(block, h, audio)
  end

  Nn.dense(m.logits, try Nn.layer_norm(m.decoder_norm, h))
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

private

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
