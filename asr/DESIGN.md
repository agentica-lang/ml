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
Asr.listen(m, audio [1, frames, d_model]) -> Result[Cache]     the cross-attention keys and values
Asr.step(m, cache, ids) -> Result[Stepped]       the logits of the last id, and the cache
Asr.step_pick(m, cache, ids, mask) -> Result[Picked]   the id and log-probability of the largest masked logit, and the cache
Asr.transcribe(m, tok, pcm, options = Options()) -> Result[Transcript]
struct Options      language (none: detect), translate, timestamps, max_tokens
struct Transcript   text, language, language_probability, tokens, segments (start, stop, text)
Mel.log_mel(samples, mels, frames = 3000) -> Result[List[Float]]   row-major [mels, frames]
Tokenizer.open(path) -> Result[Tokenizer]       a tokenizer.json
Tokenizer.decode(tok, ids) -> Result[String]    text tokens only
Tokenizer.prompt(tok, language, job, timestamps) -> Result[List[Int]]
Tokenizer.seconds(tok, id) -> Optional[Float]   the time of a timestamp token
Greedy.rules(tok, suppress, begin_suppress, timestamps) -> Rules
Greedy.allowed(rules, logits, emitted) -> List[Float]   the forbidden ids at FORBIDDEN
Greedy.segments(tok, emitted, duration) -> Result[List[Segment]]
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
- **GELU follows `activation_function`**: `gelu` (every released Whisper
  checkpoint) is the exact `erf` form, `Tensor.gelu_erf`; `gelu_new` is the
  tanh form; anything else is an error. The test model checks both against
  the peer.
- **Positions are sliced**, not fixed: the encoder adds the first
  `ceil(frames / 2)` rows of its table (the decoder the first `t`), so a short input runs, where Hugging Face
  wants exactly `2 * max_source_positions` frames.
- **The conv stem loops over the batch**: `transpose` reverses every axis.
  The token lookup flattens the ids to `[batch * t]` and reshapes back.
- **What the model assumes**: the tensor names start with `model.` and
  `scale_embedding` is false (true of the released Whisper checkpoints;
  `config.json` is not checked for it). The tied
  output projection is a second, transposed copy of the vocab table.
- **Reference, libtorch and ggml**: the model runs on all three. The tiny
  test model matches the peer within 1e-5 on libtorch and on ggml, which
  compute in F32. On `Ggml` (the CPU or Vulkan) each encoder stage, each
  decoder block of `listen` (2 outputs per head, under ggml's 256) and each
  `step` is one compiled graph, with the model's tensors and the cache as
  trace constants. There the self-attention cache is padded with zero rows
  to a capacity of 8, 16, 32 and so on rows (at most `max_target_positions`):
  a step writes its rows at their positions with a 0/1 matrix product (exact
  in F32), and its mask hides the later positions and the padding. A step's
  graph changes shape only with the number of ids and the capacity, so a
  transcription compiles a few graphs and the next one none. The token ids
  and positions are I32, which ggml holds.

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
- **`exp`, `log`, `log10`, `sin` and `cos` come from Agentica's `Math`**
  (0.37.0). `test/math_test.agc` checks them against Python's `math`.
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

## The tokenizer (`tokenizer.agc`)

Decoding only: Whisper's decoder emits ids, and the prompt is special tokens,
so no text is ever encoded.

- **It reads `tokenizer.json`**, which Hugging Face writes beside every
  Whisper checkpoint: one file with the BPE vocabulary and the added tokens
  with their ids. The merges are not needed to decode.
- **Special tokens are found by id.** The added tokens give the ids of
  `<|endoftext|>`, `<|startoftranscript|>`, `<|translate|>`,
  `<|transcribe|>` and `<|notimestamps|>`, which must come in that order;
  every id from `<|endoftext|>` on is special and gives no text, and an id
  from `vocab_size` on is an error. The language tokens are the ids between
  `<|startoftranscript|>` and `<|translate|>` (99 in the first checkpoints,
  100 in large-v3), and the timestamps start right after `<|notimestamps|>`,
  50 a second.
- **English-only checkpoints** (`.en`) have `<|endoftext|>` at 50256 where
  the others have 50257: `multilingual` is false, `languages` is empty, and
  the prompt has no language or job token. As in Hugging Face, a language
  other than `en` or the translate job is an error there.
- **Decoding joins bytes first**: each character of a token's text stands for
  one byte (GPT-2's byte-level alphabet), the bytes of all the tokens are
  joined, and only then read as UTF-8, so a character split across tokens
  comes back whole; a broken one becomes U+FFFD, as Python's `"replace"`.
- `prompt` is `<|startoftranscript|>`, the language token, `<|transcribe|>`
  or `<|translate|>`, and `<|notimestamps|>` unless timestamps are wanted.
  With no language it is `<|startoftranscript|>` alone: the decoder's next
  token is then the language.

`test/tokenizer_test.agc` decodes a tiny vocabulary made for the tests
(`test/fixtures/tokenizer.json`) and checks it against
`test/tokenizer_peer.py` (standard library only); it also builds files with
the special ids of whisper-tiny, whisper-large-v3 and whisper-tiny.en. Checked
once against whisper-tiny's `tokenizer.json`: 206 id lists (5 encoded texts in
five scripts, 200 random ones, one with the prompt and timestamps) decode as
the peer and as Hugging Face's `WhisperTokenizerFast` do; it loads in about
0.1 s.

## Greedy decoding (`greedy.agc`) and the KV cache

The rules of Whisper's greedy decoding over one step's logits, as plain lists,
and the KV cache that feeds them one token at a time.

- **The rules** (`Greedy.allowed`), in OpenAI's order: the blank and `eot`
  are not emitted first, the config's `suppress_tokens` never (without them,
  only the specials: `sot` and those from `<|translate|>` to just before
  `<|notimestamps|>`, not OpenAI's non-speech symbols, which need the
  tokenizer's encoder; without `begin_suppress_tokens`: the blank `Ġ` and
  `eot`). With timestamps: `<|notimestamps|>` is out; timestamps come in pairs
  (after a pair, text; after text and one timestamp, a timestamp or `eot`);
  they never decrease; the first token is a timestamp of at most 1 s; and when
  the timestamps together (log-sum-exp) are likelier than the best text
  token, a timestamp must come. A forbidden id gets `FORBIDDEN` (-1e30), so
  the sums stay finite.
- **The pick** is the largest logit, the first one on a tie;
  `Greedy.probability` is a softmax in doubles.
- **Segments**: a timestamp opens a segment and the next one closes it; text
  with no closing timestamp starts at its opening one, as Hugging Face does
  (OpenAI starts it at 0), and ends at the clip's end; without timestamps the
  whole text is one segment from 0. A segment is `start`, `stop`, `text`
  (`end` is a reserved word in Agentica).
- **The KV cache** lives in `asr/`, over `nn/`'s layers: `listen` projects the
  encoder output to every block's cross-attention keys and values once, split
  by head (keys transposed); `step` keeps each block's self-attention keys and
  values and feeds one token at a time (a prompt at once, under the causal
  mask), and returns only the last token's logits.

`test/greedy_test.agc` checks the rules on hand-made logits and the segments
over `test/fixtures/tokenizer.json`; `test/model_test.agc` checks the steps
against the peer's full forward.

## Transcription (`transcribe`)

One window of 16 kHz samples (at most `2 * max_source_positions` frames: 30 s
for Whisper; no samples or more are an error) → `Mel.log_mel` of the whole
window, padded with zeros to 30 s as Hugging Face's feature extractor does
(not OpenAI's, which pads the log-mel) → `encode` → `listen` → greedy
decoding with `step` and `Greedy`. This is what Hugging Face's `generate`
does with `num_beams=1` and no sampling, and OpenAI's `transcribe` at
temperature 0 inside one window. A named language and the job are checked
against the tokenizer before the audio costs anything.

- **Greedy decoding on the device.** Without timestamps (`Options.timestamps`
  false), on ggml, `decide` calls `step_pick` with `Greedy.mask` (0 or
  `FORBIDDEN` per id: the begin-suppressed ids for the first token, the
  suppressed ids always), and only an id and a log-probability come back. With
  timestamps, the default, the rules need the logits of every step, so
  `step` and `Greedy.allowed` run as before. The step after
  `<|startoftranscript|>` always reads its logits.
- **The first step** feeds `<|startoftranscript|>` alone. `no_speech_prob` is
  the softmax of its logits at `<|nospeech|>` (`<|nocaptions|>` in the first
  checkpoints), as OpenAI's. Without `Options.language` it also detects the
  language: the language token with the largest logit wins, and
  `language_probability` is the softmax over the language tokens only. A
  named language gives 1.0; an English-only checkpoint is `en`. The prompt
  then starts again with the language, the job and `<|notimestamps|>` unless
  timestamps are wanted.
- **The suppressed ids** come from `config.json` (`suppress_tokens`,
  `begin_suppress_tokens`, optional in `Config`), else `Greedy.rules`'
  defaults.
- **Decoding stops** at `eot`, after `max_tokens` (at least 1), or at
  `max_target_positions` (448 for Whisper), Hugging Face's `max_length`;
  OpenAI's `sample_len` is 224. `tokens` are the ids after the prompt, without
  `eot`; the segments end at the clip's length.
- **`avg_logprob`** is OpenAI's: the log-softmax of each picked id under the
  rules, `eot` included, summed and divided by `tokens.size + 1`. Each segment
  carries it, and its ids with the timestamps around them, as OpenAI's do.
- **Out, on purpose**: beam search, temperature fallback (and the compression
  ratio it needs), initial prompts and word timestamps. Greedy at temperature
  0 is what the speech server asks for; each can come when something needs
  it.
- **Memory on libtorch**: a `Native.arena` frees what a block makes when it
  ends, and since Agentica 0.37.0 it keeps a returned tensor alone when the
  native ops that made it are `AN_FRESH` (`tensor/native/torch.c`: a handle
  owns its `torch::Tensor`, and a view holds its base by libtorch's refcount,
  not by pointing into another handle). So each encoder block runs in an arena
  that returns the block's output tensor, and the previous one is closed: the
  peak is one block. Each `step` returns the logits as a list and the new
  self-attention keys and values as tensors; `step_pick` returns the likeliest
  id and its log-probability instead, and on ggml the logits never leave the
  device (`Asr.picked`, in the step's graph); the loop closes the cache it
  replaced. A call runs in a task of its own, which closes what it opened when
  it ends (the encoder output and the cross-attention cache). Before 0.37.0
  an arena kept everything its result came from, so each block handed its
  output out as a list and built the tensor again outside: 1.5-2 s a call.

Checked against Hugging Face's `generate` (transformers 4.46.3, greedy) on
whisper-tiny: `tools/transcribe_check.agc` runs the five cases of
`test/fixtures/transcribe_golden.json` (`whisper/live/fixtures/es-hola.wav`
and whisper.cpp's `samples/jfk.wav`: detected and named language, transcribe
and translate, with and without timestamps). The token ids match exactly,
and so do the text and the segments. The language probability is within
1.3e-6 (es) and 1e-7 (en); with the tanh GELU it was 4.1e-4 and 1.2e-4.
`test/transcribe_golden.py` wrote the golden once, by hand (it needs `torch`
and `transformers`). The tests run neither: the checkpoint is not in git.

Speed on this machine (Ryzen 5 3550H, shared with other jobs, so the numbers
spread): with Agentica 0.37.0 libtorch takes 2.5 s for the 2.6 s of es-hola (a
real-time factor of 0.96) and 3.6–3.8 s for the 11 s of jfk (0.33–0.35); 1 s is
the encoder over the 30 s window and 0.06–0.08 s each decoder step. Before
0.37.0 (each block's output went out of its arena as a list) it was 5.4–6.7 s
(2.1–2.6) and 7.3–7.5 s (0.66–0.68), 3.2 s of it the encoder, and the run
peaked at 1.6 GB against 1.0 GB. The reference backend gave the same ids for
es-hola in 22 minutes (a real-time factor of about 500). Loading the checkpoint takes
0.3–0.5 s (it took 25–40 s before `Tensor.from_file`). On ggml the five cases
give the same ids, the text and the segments too,
and compile 12 graphs in all, 11 of them in the first case (104 when every
step compiled its own). In one run after another (a quiet machine), the
real-time factor was:

| backend | es-hola (2.6 s) | jfk (11 s) |
|---|---|---|
| libtorch, CPU | 0.96–0.98 | 0.33–0.35 |
| ggml, CPU | 0.95–1.07 | 0.36–0.41 |
| ggml, Vulkan (Vega iGPU) | 1.03–1.31 | 0.41–0.47 |

With another job on the CPU, ggml's CPU threads slow down most: 7–14 for
es-hola in one such run.

## Tests of the model

The tests use a tiny random Whisper (d_model 4, 2 heads, one block each,
3 mel bins, vocab 7, 6 frames) that `test/peer.py` (standard library only)
writes at test time with its config and its forward in doubles; run them
from `asr/`. The peer also writes `speech/`, the same shape with the 36 ids of
`test/fixtures/tokenizer.json`, on which `test/model_test.agc` runs
`transcribe` end to end and checks that its output keeps the rules.
