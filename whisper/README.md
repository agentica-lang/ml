# whisper

Speech-to-text in Agentica (Via A of `roadmap.md`): a thin binding over the
pinned whisper.cpp engine. File input only — no microphone in this package.

- `whisper.agc` — `extern module "whisper"`: `load_model` reads a ggml/gguf
  file into a `Model` resource, `transcribe` turns 16 kHz mono PCM floats into
  text. Both run in the compute pool; the engine is CPU-only.
- `wav.agc` — a pure Agentica WAV reader (PCM, mono, 16 kHz, 16-bit).
- `native/whisper.c` — the thin C layer on the generated header
  `native/whisper_native.h`. It links the engine (never copied into `native/`)
  and owns one whisper context plus a mutex per model; inference on one
  context is serialized, teardown frees the context and destroys the mutex.
- `tools/build-engine.sh` — builds the pinned submodule at
  `third_party/whisper.cpp` (CPU-only, static, PIC; needs cmake and a C/C++
  compiler) and prints the `C_INCLUDE_PATH`/`LIBRARY_PATH` exports the
  module's build needs.
- `tools/fetch-model.sh` — downloads the pinned tiny model and checks its
  sha256. Weights never enter git (`models/` is ignored).
- `live/transcribe_jfk.agc` — the end-to-end check as a program (see below).

## Local setup

From this folder:

```sh
agentica install
sh tools/build-engine.sh
eval "$(sh tools/build-engine.sh | grep ^export)"
sh tools/fetch-model.sh
```

`agentica install` fetches native-gen (the locked dependency that regenerates
the header); the engine build is separate because the engine is a pinned git
submodule, not an Agentica package. The two exports must be set for every
build below (they are part of the native build key, so changing them rebuilds
correctly).

## Checks

```sh
agentica check
agentica fmt --check
agentica deps/native-gen/cli.agc whisper.agc --output native/whisper_native.h --check
agentica test            # offline suite: test/ only, no model, no network
agentica test live/      # model-bound runner checks (needs models/ggml-tiny.bin)
agentica live/transcribe_jfk.agc   # real transcription (runs are not step-capped)
```

## What each layer covers

- `agentica test` (8 tests, offline): the WAV reader on synthetic fixtures
  (formats, chunks, errors, file round-trip) and the native error paths with
  no model (a missing file names itself in the error; a path past 4096 bytes
  is refused). It still links the built engine, so CI builds it first.
- `agentica test live/` (2 tests, needs the downloaded tiny model): real
  weights load through the FFI, real inference on 0.1 s of silence returns
  the empty string, and empty audio is refused with its message.
- `agentica live/transcribe_jfk.agc` (needs the model plus the submodule
  sample `third_party/whisper.cpp/samples/jfk.wav`): transcribes the 11 s
  English sample with the pinned tiny model and asserts the exact greedy
  transcript (" And so my fellow Americans ask not what your country can do
  for you, ask what you can do for your country."). Decoding 176000 samples
  takes ~14 M Agentica steps, past what one test file may spend (1 M × tests
  + 1, shared), so this runs as a program, where runs are not step-capped;
  a mismatch prints both texts and exits 1. CI runs all three layers.
- Deliberately untested, and why: non-finite samples (no NaN/Inf value is
  reachable from pure Agentica — division by zero is a runtime error and the
  WAV reader only yields finite floats), the five-minute input bound (covering
  it needs a 4.8 M-element list), and the megabyte text cap (the engine would
  have to emit it). The guards stay as cheap defense at the FFI boundary.

## Bounds and errors

One `transcribe` call takes 1–4 800 000 samples (five minutes at 16 kHz);
more, none, or a non-finite sample is an error, never a halt. `load_model`
copies paths of 4096 bytes at most. Greedy decoding with the library
defaults (language `"en"`): widening the options is a later change, not this
one.

## For fetched copies

A package installed through `agentica install` carries only `native/` (and
`live/` scripts read nothing outside the checkout), so building from the
submodule works for checkouts only. Downstream users need the prebuilt
libraries of `[native.whisper]` (one row per target, built from this source);
publishing them is the follow-up — this package builds from source.
