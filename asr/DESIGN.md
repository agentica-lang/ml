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
Asr.close_model(m)                          frees the weights and the log-mel plan; later use is an error
Asr.load(checkpoint, config, backend) -> Result[Model]
Asr.encode(m, mel [batch, num_mel_bins, frames]) -> [batch, ceil(frames / 2), d_model]
Asr.decode(m, tokens [batch, t] integer, audio) -> logits [batch, t, vocab_size]
Asr.listen(m, audio [1, frames, d_model]) -> Result[Cache]     the cross-attention keys and values
Asr.step(m, cache, ids) -> Result[Stepped]       the logits of the last id, and the cache
Asr.step_pick(m, cache, ids, mask) -> Result[Picked]   the id and log-probability of the largest masked logit, and the cache
Asr.transcribe(m, tok, pcm, options = Options()) -> Result[Transcript]
struct Options      language (none: detect), translate, timestamps, max_tokens
struct Transcript   text, language, language_probability, tokens, segments (start, stop, text)
Mel.features(samples, mels, frames, backend) -> Result[Tensor]     [1, mels, frames] on the backend
Mel.log_mel(samples, mels, frames = 3000) -> Result[List[Float]]   row-major [mels, frames], in doubles
Asr.features(model, samples, frames) -> Result[Tensor]             Mel.features with the model's plan
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
  output projection is the vocab table itself (`Dense.out_in`).
- **Reference, libtorch and ggml**: the model runs on all three. The tiny
  test model matches the peer within 1e-5 on libtorch and on ggml, which
  compute in F32. On `Ggml` (the CPU or Vulkan) each encoder stage, each
  decoder block of `listen` and each
  `step` is one compiled graph, with the model's tensors and the cache as
  trace constants. There the self-attention cache is padded with zero rows
  to a capacity of 8, 16, 32 and so on rows (at most `max_target_positions`):
  a step writes its rows at their positions with a 0/1 matrix product (exact
  in F32), and its mask hides the later positions and the padding. A step's
  graph changes shape only with the number of ids and the capacity, so a
  transcription compiles a few graphs and the next one none. The compiled
  steps and the encoder block are kept by the model (`Model.plans`, a channel
  of one value that holds the map, which a task takes and puts back at once),
  named by their kind and the shapes and dtypes of their inputs, so audio of
  another length gets its own. Every transcription of a model, from any task,
  finds them: `planned` looks the name up, and compiles and stores on a miss.
  Two tasks that miss the same name both compile, and one copy is kept. The
  kept copy has no pin (that stays with the task that compiled), so a run of
  it builds the graph again if the cache lost it. The map has no bound: a
  caller that sends many shapes through `encode` adds an entry for each. A task
  cancelled while it holds the map loses it; the next caller whose take times
  out (50 ms; a slow holder takes the same path) puts a new, empty map in the
  channel, which costs one round of compiles. A channel that was closed makes
  every call compile, as before. The token ids and positions are I32, which
  ggml holds.

## The log-mel frontend (`mel.agc`)

16 kHz mono samples, padded with zeros or trimmed to `frames * 160` (30 s for
3000 frames; the up to 200 samples after those still end the signal, for its
last frame), as Hugging Face's feature extractor does → a reflect-padded STFT
(Hann window of 400, hop 160, the last frame dropped) → the power spectrum →
the Slaney mel filterbank → `log10(max(x, 1e-10))` → a floor at the maximum
minus 8 → `(x + 4) / 4`. Fewer than 3000 frames are the spectrogram of that
shorter segment (its own end reflection and maximum), not a prefix of the 30 s
one. Samples are about -1..1: a non-finite one is an error, and so is one past
1e150 for `log_mel` (doubles) or past about 1.8e19 for the graph (`F32`, whose
power overflows there).

- **One graph of tensor ops** on every backend but `Reference`
  (`Mel.spectrogram`): the signal (padded by reflection) is read as blocks of
  one hop, a frame is two blocks and the first 80 samples of a third, and the
  windowed DFT is a matmul with a `[400, 402]` basis (the Hann window times the
  cosines, then the sines, of the 201 bins); the power, a second matmul with
  the mel filters, the log, the floor and the scale follow. It stays on the
  device and feeds `encode` with no list in between. It computes in `F32`: the
  gap to the exact value is about 1e-5 (the tokens and the language
  probabilities of the golden clips do not move). That holds for an `F32`
  matmul; with fp16 on Vulkan (the default, `Tensor.vulkan_fp16`) the DFT
  would run in fp16, which is not measured here and is not excepted. A sample that is not finite (or so
  large that the power overflows) makes the sum the graph also answers not
  finite, and that is an error (`relu` would hide a `NaN` in the features).
- **On `Ggml` the graph is compiled by `load`** into `Model.mel`
  (`Asr.features`), so the DFT basis (about 35 ms of Agentica to build, plus
  its upload and the filters', paid by `load`) is made once. A transcription
  runs in a task that closes what it opens, so constants made there would not
  last; the plan is a field and not an entry of `Model.plans`, which a timed
  out reader empties, because no transcription can build it again. Without the
  plan, the graph is traced and run once, uncached (`keep: false`, so it does
  not evict the graphs of the decoder). `Asr.features` closes its inputs and
  its check; the features are the caller's to close.
- **On `Reference`, the FFT in Agentica lists** (`Mel.log_mel`): the matmuls of
  the graph would run in the interpreter. It is also the exact model the graph
  is tested against, so it stays. The FFT is mixed radix (decimation in time):
  radix 2 while the length is even, then radix 5, since 400 = 2^4 · 5^2.
  whisper.cpp splits 400 the same way down to 25 and then runs a direct DFT;
  padding to 512 would change the bins. Two real frames go through one complex
  FFT, as the real and the imaginary part.
- **The mel filters are computed**, with librosa's formula (Slaney scale,
  Slaney area norm), which Whisper's `mel_filters.npz` and Hugging Face's
  extractor both use. That needs no file beside the checkpoint and gives 80 or
  128 bins.
- **`exp`, `log`, `log10`, `sin` and `cos` come from Agentica's `Math`**
  (0.37.0). `test/math_test.agc` checks them against Python's `math`.
- **`log_mel` skips silent frames**: a frame that starts after the last
  non-zero sample is `log10(1e-10)` (unless the end reflection reaches a
  sample), so a short clip costs only its own frames. The graph computes all
  3000 and takes about 35 ms on the ggml CPU backend for any clip.

`test/mel_test.agc` checks `log_mel` against `test/mel_peer.py`, the same
steps in Python doubles with a direct DFT (standard library only); the gap is
about 1e-13. It checks the graph against `log_mel` on `Reference` (small
sizes, 5e-5), and `Mel.features` on every backend the machine has (libtorch,
ggml, Vulkan) against 200 values of Hugging Face's `WhisperFeatureExtractor`
(its numpy path), within 5e-5. Those values are the first second of
`whisper/live/fixtures/es-hola.wav`, at 80 and 128 mels; `log_mel` is checked
against them within 1e-6. `test/hf_golden.py` wrote `test/fixtures/hf_mel.json`
once, by hand (it needs `transformers`, so the tests do not run it). An `Inf`,
a `NaN` and a 1e30 sample are errors on each backend. `test/model_test.agc`
checks that a ggml model's compiled plan and the traced graph give the same
features. whisper.cpp keeps its mel internal (its `debug_mode` dump is not
wired), so it was not compared directly; it runs the same steps in floats.

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
  encoder output to every block's cross-attention keys and values once, as
  `[1, heads, frames, head_width]` for `Tensor.attention`; `step` keeps each
  block's self-attention keys and values and feeds one token at a time (a
  prompt at once, under the causal mask), and returns only the last token's
  logits.

`test/greedy_test.agc` checks the rules on hand-made logits and the segments
over `test/fixtures/tokenizer.json`; `test/model_test.agc` checks the steps
against the peer's full forward.

## Transcription (`transcribe`)

One window of 16 kHz samples (up to `2 * max_source_positions` frames: 30 s
for Whisper; no samples are an error) → `Asr.features` of the whole window,
padded with zeros to 30 s as Hugging Face's feature extractor does (not
OpenAI's, which pads the log-mel) → `encode` → `listen` → greedy decoding with
`step` and `Greedy`. This is what Hugging Face's `generate` does with
`num_beams=1` and no sampling, and OpenAI's `transcribe` at temperature 0
inside one window. A named language and the job are checked against the
tokenizer before the audio costs anything.

- **Long audio (`sought`).** More than one window (3001 frames or more) is read
  as Hugging Face's long-form `generate` and OpenAI's `transcribe` read it,
  with their defaults: 30 s windows, greedy, no previous text in the prompt
  (`condition_on_prev_tokens` is false in Hugging Face, and no fallback or
  `no_speech_threshold` is set), the language detected once in the first
  window and kept. The log-mel of the whole clip is one graph (`Mel.features`
  with the clip's frames: the maximum that floors it is the clip's, and a
  window has the real samples around it, as Hugging Face's features have),
  and a window is a cut of it (`cut`: a graph of its own, not kept, since it
  differs for every `seek`), padded with zeros in the features, not in the
  audio, when fewer than 3000 frames remain, as Hugging Face's `generate`
  does. `Greedy.window` ports `_retrieve_segment`: two timestamps in a row end
  a segment; when the window's ids end with a lone timestamp it is done,
  otherwise the ids after the last pair are dropped and the next window starts
  at the first timestamp of that pair (a seek in 10 ms frames, twice the
  timestamp step); with no pair the ids are one segment up to the last
  timestamp. Segment times are the window's plus its start. `tokens` are the
  ids of the segments of all the windows, each with its own window's
  timestamps; `avg_logprob` is over all the windows' decoded ids and
  `no_speech_prob` and `language_probability` are the first window's. It needs
  timestamps, as Hugging Face's does (the seek reads them). The audio is a
  `List[Float]` and the whole log-mel one tensor (80 * 100 floats a second).
  The Hugging Face call matched is the feature extractor with
  `truncation=False, padding="longest"` (OpenAI's `log_mel_spectrogram` over
  the unpadded audio), not the default `processor(audio)`, which pads or cuts
  the audio to 30 s: so a clip is long when `samples / 160 > 3000` (480160
  samples), and 480000 to 480159 samples are one window, the samples past 30 s
  read as the end of the signal for its last frame. `Greedy.window` drops a
  segment with no text, and a seek of zero (a pair of timestamps at the start)
  finishes the window instead, so each window advances. The budget is
  `Options.max_windows`, by default `2 * ceil(frames / window) + 2`: when it
  runs out the transcript holds what was decoded and `truncated` is true,
  which a model that never advances reaches and normal audio does not. Only
  the first window reads the step after `<|startoftranscript|>` alone (the
  language and `no_speech_prob`).

- **Greedy decoding on the device.** On ggml, `decide` calls `step_pick` with
  `Greedy.mask` (0 or `FORBIDDEN` per id: the begin-suppressed ids for the
  first token, the suppressed ids always, `<|notimestamps|>` with timestamps),
  and only an id and a log-probability come back. The timestamp rules depend
  on the tokens so far, so `Greedy.state` turns them into four numbers per
  step: the first timestamp that may still come; 1 when no timestamp may come
  (the last two tokens are timestamps, or the only one is); 1 when no text
  token below `eot` may come (a timestamp opened a pair; `eot` may); 1 for
  the first token. `Asr.scored` reads them in the graph, with the rule that
  no text token may come when the timestamps together are likelier than the
  best text token (a log-sum-exp over the timestamp range against the
  maximum over the text). `Greedy.allowed` stays the reference. The tests
  check the graph against it on hand-made logits, one case per rule, where
  each rule changes the pick from the largest logit, and `step_pick` over ten
  histories of the tiny model. The step after `<|startoftranscript|>` always
  reads its logits, for `no_speech_prob` and the language.
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
  ratio it needs), conditioning a window on the text before it, initial
  prompts and word timestamps. Greedy at temperature
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

Checked against Hugging Face's `generate` (transformers 4.57.6, greedy) on
whisper-tiny: `tools/transcribe_check.agc` runs the seven cases of
`test/fixtures/transcribe_golden.json` (`whisper/live/fixtures/es-hola.wav` and
whisper.cpp's `samples/jfk.wav`: detected and named language, transcribe and
translate, with and without timestamps; and two clips of 73.5 s and 75.5 s
made of them, jfk six times with gaps of silence and es-hola 24 times, read in
three windows each). The token ids match exactly, and so do the text and the
segments (their times within 1e-9). The language probability is within
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
