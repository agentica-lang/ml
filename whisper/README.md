# whisper

Speech-to-text in Agentica (Via A of `roadmap.md`): a thin binding over the
pinned whisper.cpp engine. File input only — no microphone in this package.

- `whisper.agc` — `extern module "whisper"`: `load_model` reads a ggml/gguf
  file into a `Model` resource, `transcribe_with` decodes 16 kHz mono PCM
  floats into timed `Segment`s as an `Options` asks (language, translation),
  and `transcribe` is `join_text` of those segments with the default options.
  All three run in the compute pool; the engine is CPU-only.
- `wav.agc` — a pure Agentica WAV reader (PCM, mono, 16 kHz, 16-bit).
- `native/whisper.c` — the thin C layer on the generated header
  `native/whisper_native.h`. It links the engine (never copied into `native/`)
  and owns one whisper context plus a mutex per model; inference on one
  context is serialized, teardown frees the context and destroys the mutex.
- `tools/build-engine.sh` — builds the pinned submodule at
  `third_party/whisper.cpp` (CPU-only, static, PIC; needs cmake and a C/C++
  compiler) and prints the `C_INCLUDE_PATH`/`LIBRARY_PATH` exports the
  module's build needs.
- `tools/fetch-model.sh` — downloads the two pinned multilingual models and
  checks their sha256 (see "Models" below). Weights never enter git
  (`models/` is ignored).
- `live/transcribe_jfk.agc` — the English end-to-end check as a program.
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
- `Options` is `language: String = ""` and `translate: Bool = false`.
  `""` keeps the engine default (English), `"auto"` detects the language from
  the audio, and a code such as `"es"` pins it (whisper.cpp's own codes, 32
  bytes at most). `translate: true` answers in English instead of the spoken
  language, whichever `language` is pinned.

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

## Local setup

From this folder (order matters: installing already compiles the native
module, so the engine exports come first):

```sh
sh tools/build-engine.sh
eval "$(sh tools/build-engine.sh | grep '^export ')"
agentica install
sh tools/fetch-model.sh
```

`tools/build-engine.sh` builds the pinned engine submodule (CPU-only,
static; needs cmake and a C/C++ compiler) and prints the
`C_INCLUDE_PATH`/`LIBRARY_PATH` exports; `agentica install` fetches
native-gen (the locked dependency that regenerates the header). Any
`agentica` invocation that builds the native module (`install`, `check` is
exempt, `test`, a run) needs those two exports set first — they are part of
the native build key, so changing them rebuilds correctly.

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
agentica live/transcribe_spanish.agc   # Spanish, options and timed segments
```

## What each layer covers

The three layers differ in what they are allowed to need, so a failure points
at one of them:

- **Model-free and offline** (`agentica test`, 13 tests): the WAV reader on
  synthetic fixtures (formats, chunks, errors, file round-trip), the binding's
  pure half (`Options` defaults, `join_text` on hand-built `Segment`s:
  trimming, dropped empty segments, one space between, `""` for none), and
  the native error paths with no model (a missing file names itself in the
  error; a path past 4096 bytes is refused). It still links the built engine,
  so CI builds it first; no weights and no network are involved, which is why
  it runs before the download.
- **Live, model-bound** (`agentica test live/`, 3 tests, needs the downloaded
  tiny model): real weights load through the FFI, real inference on 0.1 s of
  silence returns the empty string from both `transcribe` and
  `transcribe_with` + `join_text`, and empty audio is refused with its
  message.
- **Live programs** (`live/transcribe_*.agc`, need the models plus the engine;
  CI runs each one), because decoding a real sample costs far more steps than
  one test file may spend (1 M × tests), while runs are not step-capped:
  - `live/transcribe_jfk.agc` (tiny, `language` default): transcribes the
    11 s English sample the whisper.cpp submodule carries and asserts the
    exact greedy transcript ("And so my fellow Americans ask not what your
    country can do for you, ask what you can do for your country."). Greedy
    decoding is the argmax at every step over pinned weights and pinned PCM,
    so the bytes are fixed; a mismatch prints both texts and exits 1.
  - `live/transcribe_spanish.agc` (small, `language: "es"`, then
    `translate: true`): transcribes a 2.6 s Spanish fixture and asserts the
    structural contract plus two words, not a transcript. The contract is:
    at least one segment; every segment a range with `start_ms < end_ms`;
    segments ordered, none starting before the one it follows; the first
    starting within 500 ms of the audio's start and the last ending within
    500 ms of its end, so the segments cover the audio it heard; and the
    joined text non-empty. The words are the two the fixture opens and closes
    on ("hola", "español", case aside). The translated pass asserts the other
    `Options` field is wired: non-empty English ("hello", "spanish"). An exact
    Spanish transcript is deliberately not pinned: the fixture is short and
    synthesised, so which words survive is the one thing not worth betting on.
- Deliberately untested, and why: non-finite samples (no NaN/Inf value is
  reachable from pure Agentica — division by zero is a runtime error and the
  WAV reader only yields finite floats), the five-minute input bound (covering
  it needs a 4.8 M-element list), the language-code bound and the megabyte
  text cap (each needs input or a decode nothing in the suite reaches). The
  guards stay as cheap defense at the FFI boundary.

### The Spanish fixture

`live/fixtures/es-hola.wav` is 84 KB: 2.647 s of mono 16-bit PCM at 16 kHz,
the sentence "Hola, esto es una prueba de audio en español." It was
synthesised for this test with piper (`piper-tts`, MIT) and its
`es_ES-davefx-medium` Spanish voice, then resampled to 16 kHz; no third-party
recording is redistributed, the words are this package's own, and it is small
enough to stay in git. A recording of a real speaker would validate more, but
it would need a licence to redistribute and would not fit the 100 KB budget;
synthesising at test time would need a TTS dependency in CI. Hence the
structural assertion above rather than a pinned transcript.

## Bounds and errors

One `transcribe_with` call takes 1–4 800 000 samples (five minutes at 16 kHz);
more, none, or a non-finite sample is an error, never a halt, and so is a
language code past 32 bytes or more than a megabyte of segment text.
`load_model` copies paths of 4096 bytes at most. Greedy decoding with the
library thread count: `Options` adds a language and a translation flag and
nothing else. Every failure is an `error`; the native side frees the audio,
the language copy and the segment list on every path, and holds the per-model
mutex only for the decode itself.

## Outside the repository checkout

A package installed through `agentica install` carries only `native/` (and
`live/` scripts read nothing outside the checkout), so building from the
submodule works for checkouts only. Downstream users need the prebuilt
libraries of `[native.whisper]` (one row per target, built from this source);
publishing them is the follow-up — this package builds from source.
