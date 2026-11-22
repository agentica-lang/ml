## Whisper speech-to-text (Via A): a thin binding over whisper.cpp.
##
## The `.agc` file is the one source of the native signatures: after each
## change to the `extern module` block, make the header again and check that
## it is current. native-gen comes from `agentica install` (see
## `agentica.lock`); from this folder:
##
## ```sh
## agentica install
## agentica deps/native-gen/cli.agc whisper.agc --output native/whisper_native.h
## agentica deps/native-gen/cli.agc whisper.agc --output native/whisper_native.h --check
## ```
##
## Audio enters as 16 kHz mono PCM floats in `Wav` (a pure Agentica WAV
## reader); a loaded model is a `Model` resource. `transcribe_with` runs one
## decode and returns the timed `Segment`s it emitted, and `transcribe` is
## `join_text` of that with the default `Options`, so the two never disagree.
## Both native calls run in the compute pool: loading reads and parses the
## whole weights file, and transcribing runs the full encoder/decoder. The
## engine is the pinned submodule at `third_party/whisper.cpp`, built CPU-only
## by `tools/build-engine.sh` (its libraries are linked, its sources are never
## copied); weights are downloaded by `tools/fetch-model.sh` and never enter
## git. What each test layer covers is in `README.md`. No microphone here:
## file input only (Fase W, Via A of `roadmap.md`).

import Wav from "wav"

export Wav.decode
export Wav.load

## How one transcription runs: the language it decodes and whether it
## translates. `Options()` is the engine's own default (English, no
## translation).
struct Options
  ## The language of the speech. `""` keeps the engine default (English),
  ## `"auto"` detects it from the audio, and a code such as `"es"` pins it.
  language: String = ""
  ## Translate to English instead of transcribing in the spoken language.
  translate: Bool = false
end

## One stretch of transcribed audio, timed by the engine: `start_ms` and
## `end_ms` are milliseconds from the first sample, and `text` is trimmed of
## surrounding whitespace, which may leave it empty.
struct Segment
  start_ms: Int
  end_ms: Int
  text: String
end

extern module "whisper"
  ## A loaded model: a resource whose state the native side owns (the
  ## whisper context plus the mutex that serializes inference on it).
  resource Model
  ## Loads the model at `path` (a ggml/gguf file); error when it cannot be read.
  compute def load_model(path: String) -> Result[Model]
  ## Transcribes 16 kHz mono PCM audio (at most five minutes: longer input is
  ## an error, so one call is bounded) as `options` asks, returning the
  ## segments the engine emitted, in time order; error when the model cannot
  ## run it.
  compute def transcribe_with(model: Model, pcm: List[Float], options: Options) -> Result[List[Segment]]
end

## Transcribes 16 kHz mono PCM audio with `Options()`: `join_text` of what
## `transcribe_with` returns, so plain text of a full decode.
def transcribe(model: Model, pcm: List[Float]) -> Result[String]
  segments = try transcribe_with(model, pcm, Options())
  ok(join_text(segments))
end

## The text of `segments` in the order given: each text trimmed, the empty
## ones dropped, a single space between them and none at either end. No
## segments (or nothing but whitespace) is the empty string.
def join_text(segments: List[Segment]) -> String
  parts = segments.map do |segment: Segment| segment.text.strip end
  kept = parts.select do |part: String| !part.empty? end
  kept.join(" ")
end
