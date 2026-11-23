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
Mel.log_mel(samples, mels, frames = 3000) -> Result[List[Float]]   row-major [mels, frames]
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

## The log-mel frontend (`mel.agc`)

16 kHz mono samples, padded with zeros or trimmed to `frames * 160` (30 s for
3000 frames), as Hugging Face's feature extractor does → a reflect-padded STFT
(Hann window of 400, hop 160, the last frame dropped) → the power spectrum →
the Slaney mel filterbank → `log10(max(x, 1e-10))` → a floor at the maximum
minus 8 → `(x + 4) / 4`. Fewer than 3000 frames are the spectrogram of that
shorter segment (its own end reflection and maximum), not a prefix of the 30 s
one. Samples are about -1..1: a non-finite one, or one past 1e150, is an error.

- **Agentica lists**, not tensor ops: the work is an FFT per frame, which the
  tensor API has no op for.
- **The FFT is mixed radix** (decimation in time): radix 2 while the length
  is even, then radix 5, since 400 = 2^4 · 5^2. whisper.cpp splits 400 the
  same way down to 25 and then runs a direct DFT; padding to 512 would change
  the bins. Two real frames go through one complex FFT, as the real and the
  imaginary part.
- **The mel filters are computed**, with librosa's formula (Slaney scale,
  Slaney area norm), which Whisper's `mel_filters.npz` and Hugging Face's
  extractor both use. That needs no file beside the checkpoint and gives 80 or
  128 bins.
- **`exp`, `log`, `log10`, `sin` and `cos` come from libm** through
  `mathx.agc`, a small required native module (`native/mathx.c`, built on
  install). Temporary: Agentica's `Math` has only `sqrt`, and the module goes
  once it has these (agentica-lang/agentica#1925). `test/mathx_test.agc`
  checks it against Python's `math`.
- **Silent frames are skipped**: a frame that starts after the last non-zero
  sample is `log10(1e-10)` (unless the end reflection reaches a sample), so a
  short clip costs only its own frames.
- 30 s of audio takes about 1–2 s.

`test/mel_test.agc` checks the output against `test/mel_peer.py`, the same
steps in Python doubles with a direct DFT (standard library only); the gap is
about 1e-13. It also checks 200 values of Hugging Face's
`WhisperFeatureExtractor` (its numpy path) for the first second of
`whisper/live/fixtures/es-hola.wav`, at 80 and 128 mels, within 1e-6:
`test/hf_golden.py` wrote `test/fixtures/hf_mel.json` once, by hand (it needs
`transformers`, so the tests do not run it). whisper.cpp keeps its mel
internal (its `debug_mode` dump is not wired), so it was not compared
directly; it runs the same steps in floats.

## Tests of the model

The tests use a tiny random Whisper (d_model 4, 2 heads, one block each,
3 mel bins, vocab 7, 6 frames) that `test/peer.py` (standard library only)
writes at test time with its config and its forward in doubles; run them
from `asr/`. The model is checked only against that peer, not yet against
a real checkpoint.
