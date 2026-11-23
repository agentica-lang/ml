# asr — design

Whisper's model forward in Agentica (Via B of `roadmap.md`), inference
only, over `tensor/`, `nn/` and `hub/`. Weights come from a Hugging Face
checkpoint with the names `WhisperForConditionalGeneration` stores
(`model.encoder.*`, `model.decoder.*`). `whisper/` is Via A, the whisper.cpp
binding; this package does not touch it.

## API

```text
struct Config   the fields of HF config.json: d_model, encoder_layers, decoder_layers,
                encoder_attention_heads, decoder_attention_heads, num_mel_bins,
                vocab_size, max_source_positions, max_target_positions
Asr.config(path) -> Result[Config]
Asr.open(path, backend) -> Result[Model]     a .safetensors or .index.json, config.json beside it
Asr.load(checkpoint, config, backend) -> Result[Model]
Asr.encode(m, mel [batch, num_mel_bins, frames]) -> [batch, ceil(frames / 2), d_model]
Asr.decode(m, tokens [batch, t] integer, audio) -> logits [batch, t, vocab_size]
```

Encoder: `conv1` (kernel 3, padding 1) → gelu → `conv2` (kernel 3, stride 2,
padding 1) → gelu → `+ embed_positions` → blocks → `layer_norm`. Decoder:
`embed_tokens + embed_positions` → blocks over the encoder output →
`layer_norm` → `hidden · embed_tokensᵀ` (the output projection is tied).

## Decisions

- **The config comes from `config.json`** beside the weights, decoded into
  a struct that keeps only the fields the forward needs; `load` takes one
  passed in. `load` checks the conv, position and token tables against it.
- **Conv padding is `tensor/`'s**: `Tensor.conv1d` takes `stride` and
  `padding`, so `Nn.Conv1d` adds only the bias.
- **GELU is the tanh form** (`Tensor.gelu`); Whisper uses the exact `erf`
  form. On the test model the drift is 1.7e-4 on the encoder output and
  3.8e-5 on the logits.
- **Positions are sliced**, not fixed: the encoder adds the first
  `ceil(frames / 2)` rows of its table (the decoder the first `t`), so a short input runs, where Hugging Face
  wants exactly `2 * max_source_positions` frames.
- **The conv stem loops over the batch**: `transpose` reverses every axis.
  The token lookup flattens the ids to `[batch * t]` and reshapes back.
- **What the model assumes**: the tensor names start with `model.`, the
  activation is gelu and `scale_embedding` is false (true of the released
  Whisper checkpoints; `config.json` is not checked for either). The tied
  output projection is a second, transposed copy of the vocab table.
- **Reference only**: libtorch has no `gelu_tanh_approx`, so the
  model loads there and `encode` stops at the first gelu (a test asserts it).

The tests use a tiny random Whisper (d_model 4, 2 heads, one block each,
3 mel bins, vocab 7, 6 frames) that `test/peer.py` (standard library only)
writes at test time with its config and its forward in doubles; run them
from `asr/`. The model is checked only against that peer, not yet against
a real checkpoint.
