# whisper

Speech-to-text in Agentica (Via A of the [roadmap](../roadmap.md)): a thin binding over the
pinned whisper.cpp engine, file input plus microphone capture and long-audio
chunking.

- `whisper.agc` — `extern module "whisper"`: `load_model` reads a ggml/gguf
  file into a `Model` resource, `transcribe_with` decodes 16 kHz mono PCM
  floats into timed `Segment`s as an `Options` asks (language, translation,
  beam search), `transcribe_detailed` adds each segment's tokens and
  probabilities and the detected language, and `transcribe` is `join_text` of
  the segments with the default options. They run in the compute pool, on the
  GPU when the engine was built with Vulkan, else on the CPU. One decode
  accepts up to five minutes of audio.
- `mic.agc` — microphone capture: 16 kHz mono s16le PCM from a command run
  through `Process` (pinned `arecord` invocation, configurable), plus the
  `Source` abstraction (`Samples` for synthetic PCM in tests, `Command` for
  a live capture). There is no native audio module: `Process` is the transport.
- `stream.agc` — long-audio transcription: a PCM stream cut into bounded
  windows with overlap, gated by pure-Agentica RMS energy VAD, each window
  decoded by `transcribe_with` (the long work stays in the compute pool)
  and merged back onto one timeline with absolute timestamps.
- `wav.agc` — a pure Agentica WAV reader (PCM, mono, 16 kHz, 16-bit).
- `native/whisper.c` — the thin C layer on the generated header
  `native/whisper_native.h`. It links the engine (never copied into `native/`)
  and owns one whisper context plus a mutex per model; inference on one
  context is serialized, teardown frees the context and destroys the mutex.
- `tools/build-engine.sh` — builds the pinned submodule at
  `third_party/whisper.cpp` for the CPU, or with `vulkan` for the CPU and
  Vulkan (static, PIC; needs cmake and a C/C++ compiler), and prints the
  `C_INCLUDE_PATH`/`LIBRARY_PATH` exports the module's build needs.
- `tools/fetch-model.sh` — downloads the two pinned multilingual models and
  checks their sha256 (see "Models" below). Weights never enter git
  (`models/` is ignored).
- `live/transcribe_jfk.agc` — the English end-to-end check as a program.
- `live/transcribe_detailed.agc` — the detailed decode as a program: language
  detection, timed tokens, `max_len` segments and beam search.
- `live/transcribe_spanish.agc` — the Spanish case: options, timed segments
  and translation, also as a program (see below).

## The calls

`load_model(path)` loads a ggml/gguf file once and returns a `Model`; the path
is copied, 4096 bytes at most.

`transcribe_with(model, pcm, options)` runs one decode of `pcm` (16 kHz mono
floats, 1 to 4 800 000 samples) and returns the `Segment`s the engine
emitted, in time order:

```agc
model = try Whisper.load_model("models/ggml-small.bin")
pcm = try Whisper.load("live/fixtures/es-hola.wav")
segments = try Whisper.transcribe_with(model, pcm, Whisper.Options(language: "es"))
for segment in segments
  puts("#{segment.start_ms} ms to #{segment.end_ms} ms: #{segment.text}")
end
```

- `Segment` is `start_ms: Int`, `end_ms: Int`, `text: String`: milliseconds
  from the first sample (the engine's hundredths of a second, times ten), and
  the text of that stretch trimmed of surrounding whitespace, which may leave
  it empty.
- `Options` is `language: String = ""` and `translate: Bool = false`, plus
  `beam_size`, `token_timestamps`, `max_len` and `language_probabilities` (below).
  `""` keeps the engine default (English), `"auto"` detects the language from
  the audio, and a code such as `"es"` pins it (whisper.cpp's own codes, 32
  bytes at most; an unknown code is an error). `translate: true` answers in
  English instead of the spoken language, whichever `language` is pinned.
  `beam_size` from 2 to 8 decodes with beam search; 0 (the default) or 1
  decodes greedily. An English-only model (`.en`) always decodes English and
  never translates, as whisper.cpp's server does.

`transcribe_detailed(model, pcm, options)` runs the same decode and returns
a `Transcript`, what an HTTP server such as whisper.cpp's `verbose_json` needs.
To get the segments and tokens that server gives, send its options:
`token_timestamps: true`, `max_len: 60` and `language_probabilities: true`
(below):

```agc
options = Whisper.Options(language: "auto", token_timestamps: true, max_len: 60, language_probabilities: true)
transcript = try Whisper.transcribe_detailed(model, pcm, options)
puts(Whisper.language_name(transcript.language))   # some("english")
```

- `Transcript` is `language` (the code decoded: the pinned one or the one
  detected), `segments` and `detection`.
- `SegmentDetail` is a `Segment` whose `text` is not trimmed, plus `tokens`,
  `avg_logprob` (the tokens' log probabilities over the segment's token
  count, special tokens included, as whisper.cpp's server computes it) and
  `no_speech_probability`.
- `Token` is `ids`, `text`, `probability` (of the first id),
  `log_probability` (of all of them), `start_ms` and `end_ms`. Special tokens
  are left out, and the tokens that together spell one UTF-8 character are
  one `Token`. The times are -10 unless `token_timestamps` is on.
- `max_len` (with `token_timestamps`) cuts the segments on token boundaries
  so that each holds at most that many characters (one longer token stays
  whole). A cut may fall inside a character: each maximal ill-formed UTF-8
  subpart becomes one U+FFFD, in tokens and segments alike, as the server's
  JSON writer does.
- `language_probabilities: true` runs the encoder once more on the first 30 s
  to fill `detection`: the most probable language code, its probability and
  the probability of every language. Without it, or with an English-only
  model, `detection` is none. On very short audio (well under a second) the
  detection still runs on 30 s of mostly padding, so its probabilities say
  little.

`language_name(code)` is the engine's English name of a code (`"es"` is
`"spanish"`), none for a code it does not know.

`transcribe(model, pcm)` is `transcribe_with` with `Options()` followed by
`join_text`:

```agc
text = try Whisper.transcribe(model, pcm)   # "¡Hola! Esto es una prueba de audio en español."
```

`join_text(segments)` is pure and public: each text trimmed, the empty ones
dropped, one space between them and none at either end, so no segments (or
nothing but whitespace) is `""`. The binding trims a segment on the native
side too, so `transcribe` no longer carries the single leading space the
tokenizer puts in front of a segment; the word sequence is what the checks
pin.

## Microphone capture

`Mic.capture(spec)` captures up to `spec.max_samples` samples (ten seconds
by default) of 16 kHz mono PCM from `spec.command`:

```agc
samples = try Mic.capture(Mic.default_capture())
```

- `Mic.default_capture()` is the pinned invocation, `arecord -q -f S16_LE
  -r 16000 -c 1 -t raw`, which writes raw samples to standard output.
  `Mic.Capture` keeps the command, its arguments, the per-read timeout and
  the sample cap, all tunable: `Mic.default_capture().with(max_samples:
  80_000)` captures five seconds, `.with(args: [...same flags..., "-D",
  "plughw:1,0"])` picks a card.
- Any command that writes the same raw format works: `sox -d -t raw
  -r 16000 -c 1 -b 16 -e signed-integer -` or `ffmpeg -f alsa -i default
  -ar 16000 -ac 1 -f s16le -` (both untested here: only `arecord` is
  pinned). Nothing here resamples: the command must output 16 kHz mono
  s16le.
- The capture reads standard output in ~1 s chunks until the cap or the end
  of the output, then closes the process on every path (a read timeout kills
  it first), and decodes with `Mic.decode_s16le`. The tool's standard error
  passes through to ours, so a missing device explains itself there; keep
  the invocation quiet (`-q`, `-hide_banner -nostats -loglevel error`) or
  route its stderr away (`sh -c '... 2>/dev/null'`), since a chatty child
  (an ffmpeg banner) fills that buffer and stalls the read until the
  timeout. Only `arecord` is pinned and tested.
- A command that dies on its own reports its status: a nonzero exit on the
  natural end of its output (ALSA xrun, unplug, exit 1) is
  `mic: the capture command failed (exit N; ...)`, and the partial audio is
  discarded, never returned — a truncated utterance cannot transcribe as if
  complete. Partial audio is returned only when the command exited 0 (for
  example `arecord -d 2` with a longer cap); on the cap path the kill is
  deliberate, so the status is meaningless and unchecked.
- `Mic.Source` is what the tests feed instead of a device:
  `Mic.Source.Samples(pcm)` for synthetic PCM, `Mic.Source.Command(spec)`
  for a live capture; `Mic.read` resolves either to samples.

## Long audio

`Stream.transcribe_stream(model, source, options)` transcribes a whole
`Source` — synthetic PCM or one microphone capture, of any length —
piecewise, so audio longer than the five-minute per-call bound is decoded in
bounded windows:

```agc
segments = try Stream.transcribe_stream(model, Mic.Source.Samples(pcm), Stream.default_options())
```

- `Stream.default_options()` is 30 s windows with 5 s of overlap, the
  30 ms / 0.02 energy gate and the engine's own default options.
  `Stream.check_options` validates a walk (window 1..300 s, overlap inside
  its window, frame 1..1000 ms, threshold 0.0..1.0) without audio or a model.
- The gate is RMS energy per frame: `Stream.frame_rms` over `frame_ms`
  (30 ms, 480 samples by default), `Stream.is_speech` when that RMS reaches
  `threshold` (0.02, about -34 dBFS: a quiet room sits below it,
  conversational speech sits above it). `Stream.has_speech` keeps a window
  when any frame is speech, so silent windows cost no engine call;
  `threshold: 0.0` keeps every window and turns the gate off. Very quiet
  speech below the threshold is skipped with the silence: tune the threshold
  down (or off) rather than up when words go missing.
- Each kept window is one `transcribe_with` call (one engine decode, in the
  compute pool), offset by its start (`Stream.offset_segments`) and merged
  (`Stream.merge_segments`): sorted by start, then each segment kept unless
  its text is empty or an already kept segment overlaps it in time with the
  same stripped text (case aside). Distinct hypotheses in the overlap are
  all kept: the merge never drops words, it only drops copies.
- Latency is the sum of the window decodes, and every returned segment is
  final (each window is decoded whole): there are no partial hypotheses. The
  merge only dedupes same-text overlaps: the same speech transcribed with
  different words in two windows is kept twice, by design (dropping words is
  worse than doubling them).
- The capture spawns one command with no process group: invoke the tool
  directly rather than through a `sh -c` wrapper with grandchildren, since
  closing kills the direct child only (see `Process` `pgroup`).

## Local setup

From this folder (order matters: installing already compiles the native
module, so the engine exports come first):

```sh
git submodule update --init ../third_party/whisper.cpp   # a fresh clone only
eval "$(sh tools/build-engine.sh | grep '^export ')"
agentica install
sh tools/fetch-model.sh
```

After the submodule moves to another commit, rebuild the engine and make
`agentica` link the module again: the native cache key covers the text of
`native/whisper.c` and the paths in `C_INCLUDE_PATH`/`LIBRARY_PATH`, not the
engine's contents (and a `touch` changes none of them), so `agentica install`
keeps the stale `libwhisper.so`. Remove its folder from
`~/.cache/agentica/pkg/native-build/` (the `built native whisper:` line names
it) and run `agentica install` again.

For the GPU, `sh tools/build-engine.sh vulkan` builds a second tree with
ggml's Vulkan backend (it needs the Vulkan headers and loader, `glslc` and
`SPIRV-Headers`; the script lists what is missing) and prints its exports
instead. Both trees give the module one library, `whisper_engine`, a GNU ld
linker script over the tree's archives, so one `agentica.toml` links either;
the package cache keeps one module per tree. A model then loads on the first
Vulkan device, and on the CPU when there is none. The script works on Linux
x86_64 only, for both trees (the GROUP linker script is GNU ld's, and the
flags pin AVX2).

One program links one ggml: `tensor/`'s ggml module or this engine's, never
both, since each carries its own copy of ggml's symbols and backends.

`tools/build-engine.sh` builds the pinned engine submodule (static; needs
cmake and a C/C++ compiler) and prints the
`C_INCLUDE_PATH`/`LIBRARY_PATH` exports; `agentica install` fetches
native-gen (the locked dependency that regenerates the header). Any
`agentica` invocation that builds the native module (`install`, `check` is
exempt, `test`, a run) needs those two exports set first — they are part of
the native build key, so changing them rebuilds correctly. Microphone
capture needs `arecord` (alsa-utils) plus a device, but only for
`Mic.capture` and `live/mic_live_test.agc`'s capture path: every other check
runs without either.

## Models

Both are multilingual, pinned by commit and sha256 in `tools/fetch-model.sh`,
and downloaded to `whisper/models/` (or the folder given first); git never
holds them:

```sh
sh tools/fetch-model.sh                # both models, into whisper/models
sh tools/fetch-model.sh models tiny    # one of them
sh tools/fetch-model.sh /tmp/w small   # another folder
```

- `ggml-tiny.bin` (74 MB) — every check that needs weights but not a
  particular language: the live test file and the English program.
- `ggml-small.bin` (466 MB) — the Spanish program, which a model this size
  hears well enough to time and word, and tiny does not.

## Checks

```sh
agentica check
agentica fmt --check
agentica deps/native-gen/cli.agc whisper.agc --output native/whisper_native.h --check
agentica test            # offline suite: test/ only, no model, no network
agentica test live/      # model-bound runner checks (needs models/ggml-tiny.bin)
agentica live/transcribe_jfk.agc       # English, real transcription (runs are not step-capped)
agentica live/transcribe_detailed.agc  # tokens, detection, max_len and beam search
agentica live/transcribe_spanish.agc   # Spanish, options and timed segments
agentica live/server_parity.agc path/to/ggml-medium.bin   # by hand: agreement with whisper.cpp's server
```

CI runs all of these but the last on the CPU tree. The Vulkan tree and
`live/server_parity.agc` are tested only locally (CI has no GPU and does not
fetch the medium model).

## What each layer covers

The layers differ in what they are allowed to need, so a failure points at
one of them:

- **Model-free and offline** (`agentica test`): the WAV reader on synthetic
  fixtures, the binding's pure half (`Options` defaults, `join_text`), the
  native error paths with no model, the capture layer with no device
  (hermetic shell commands: `printf`, `head`, `/dev/zero`), and the streaming
  layer with no model (the walk, the energy gate, window boundaries, offsets,
  the merge). It links the built engine, so CI builds it first.
- **Live, model-bound** (`agentica test live/`, needs the tiny model): real
  weights load through the FFI, real inference on 0.1 s of silence returns the
  empty string, empty audio is refused, and the streaming layer runs against
  the real engine.
- **Live programs** (`live/transcribe_*.agc`, need the models plus the engine;
  CI runs each one), because decoding a real sample costs far more steps than
  one test file may spend:
  - `live/transcribe_jfk.agc` (tiny): transcribes the 11 s English sample the
    whisper.cpp submodule carries and asserts the exact greedy transcript. A
    mismatch prints both texts and exits 1.
  - `live/transcribe_detailed.agc` (tiny): the same sample through
    `transcribe_detailed` with detection, token timestamps and `max_len: 60`,
    asserting the detected language, several segments that their tokens
    spell, ordered token times and probabilities in range, then one beam
    search decode.
  - `live/transcribe_spanish.agc` (small, `language: "es"`, then `translate:
    true`): transcribes a 2.6 s Spanish fixture and asserts a structural
    contract (ordered ranges that cover the audio, non-empty text) plus two
    words, not a transcript: the fixture is short and synthesised
    (`live/fixtures/es-hola.wav`, made with piper `es_ES-davefx-medium`, MIT),
    so which words survive is not worth betting on.
- **Agreement with whisper.cpp's server** (`live/server_parity.agc`, by hand,
  needs the medium model): `jfk.wav` and the Spanish fixture against the
  `verbose_json` answers that a whisper.cpp server with the same model gave
  (`live/fixtures/server-*-medium.json`): the languages, the detected
  probability and the top three languages, each segment's text, times, token
  ids, `avg_logprob` and `no_speech_prob`, and each token's text (non-ASCII
  included) and probability, within a tolerance. It passes on the CPU and
  the Vulkan tree. Token times are not compared: they move by up to 0.7 s
  between backends.
- **Live, microphone** (`live/mic_live_test.agc`, no model): where a capture
  command and a device exist it captures half a second and asserts the sample
  count; everywhere else it prints its SKIP lines and passes, so CI never
  depends on hardware. The test skips only on the device-less errors (a
  missing command, no audio, a command that died mid-capture, a hung read) and
  fails on anything else. A human with a mic runs it to prove the capture path.

## Bounds and errors

One `transcribe_with` call takes 1–4 800 000 samples (five minutes at 16 kHz);
more, none, a non-finite sample, or a finite one past the range of a `float`
(about 3.4e38) is an error, never a halt, and so is a
language code past 32 bytes, a `beam_size` outside 0–8, a `max_len` outside
0–100 000, an unknown language code, or more than a megabyte of text (the
segments' text, plus the tokens' for `transcribe_detailed`). `load_model`
copies paths of 4096 bytes at most. The library thread count and the
engine's other defaults stay. Every failure is an `error`; the native side
frees the audio and the text buffers on every path, and holds the per-model
mutex for the decode, the detection and the copy of the results, nothing
else.

One `Mic.capture` keeps 1–4 800 000 samples (five minutes at most) with a
timeout of at least 1 ms per read; a missing command, a timed-out read, no
audio at all, or a command that died mid-capture is an `error`, and the device
process is closed on every path. One `Stream.transcribe_stream` checks its walk
first, refuses empty input, then decodes each speech window within the
per-call bound and merges; window timestamps are absolute.

## Outside the repository checkout

A package installed through `agentica install` carries only `native/`: it
builds against the engine that `C_INCLUDE_PATH` and `LIBRARY_PATH` name, so run
`tools/build-engine.sh` from a checkout of this repository at the same commit
and export what it prints before the first run. `[native.whisper]` has no
`prebuilt` table: this package builds from source.
