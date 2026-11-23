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
## `transcribe_detailed` adds each segment's tokens and probabilities and the
## detected language. The loading and transcribing calls run in the compute
## pool: loading reads and parses the whole weights file, and transcribing
## runs the full encoder/decoder. The engine is the pinned submodule at
## `third_party/whisper.cpp`, built by `tools/build-engine.sh` for the CPU or
## with Vulkan, which it then uses (its libraries are linked, its sources are
## never copied); weights are downloaded by `tools/fetch-model.sh` and never enter
## git. What each test layer covers is in `README.md`. No microphone here:
## file input only (Fase W, Via A of `roadmap.md`).

import Wav from "wav"

export Wav.decode
export Wav.load

## How one transcription runs: the language it decodes, whether it
## translates, the search, the timing of tokens and language detection.
## `Options()` is the engine's own default (English, greedy, no translation).
struct Options
  ## The language of the speech. `""` keeps the engine default (English),
  ## `"auto"` detects it from the audio, and a code such as `"es"` pins it;
  ## a code the engine does not know is an error. An English-only model
  ## (`.en`) always decodes English and never translates.
  language: String = ""
  ## Translate to English instead of transcribing in the spoken language.
  translate: Bool = false
  ## Beam search with this many beams (2 to 8); 0 or 1 decodes greedily.
  beam_size: Int = 0
  ## Times each token (`Token.start_ms` and `end_ms` of `transcribe_detailed`).
  token_timestamps: Bool = false
  ## With `token_timestamps`, cuts the segments on token boundaries so that
  ## each holds at most this many characters (a single longer token stays
  ## whole); 0 is no limit. whisper.cpp's server uses 60 for `verbose_json`.
  max_len: Int = 0
  ## Also detects the language of the first 30 s of audio, whatever
  ## `language` pins, and returns every language's probability in
  ## `Transcript.detection`. It runs the encoder once more; an English-only
  ## model gives no detection.
  language_probabilities: Bool = false
end

## One stretch of transcribed audio, timed by the engine: `start_ms` and
## `end_ms` are milliseconds from the first sample, and `text` is trimmed of
## surrounding whitespace, which may leave it empty.
struct Segment
  start_ms: Int
  end_ms: Int
  text: String
end

## A token of a segment of `transcribe_detailed`, or the few tokens that
## together make whole UTF-8 text (the tokenizer may split one character).
## Special tokens (timestamps, end of text) are left out.
struct Token
  ## The ids of the tokens, in order.
  ids: List[Int]
  ## The text, as the tokenizer spells it (usually with its leading space).
  text: String
  ## The probability of the first token.
  probability: Float
  ## The sum of the log probabilities of the tokens.
  log_probability: Float
  ## When the token starts and ends, in ms from the first sample; -10 unless
  ## `Options.token_timestamps` is on.
  start_ms: Int
  end_ms: Int
end

## One segment of `transcribe_detailed`: a `Segment` plus what the engine
## knows of it.
struct SegmentDetail
  start_ms: Int
  end_ms: Int
  ## The text as the engine emitted it, not trimmed.
  text: String
  tokens: List[Token]
  ## The sum of the tokens' log probabilities over the number of tokens of
  ## the segment, special ones included (what whisper.cpp's server reports).
  avg_logprob: Float
  ## The probability that the segment holds no speech.
  no_speech_probability: Float
end

## The language detected on the first 30 s of audio.
struct Detection
  ## The most probable language, as a code (`"es"`).
  language: String
  ## Its probability, from 0 to 1.
  probability: Float
  ## The probability of every language, by code.
  probabilities: Map[String, Float]
end

## A whole decode of `transcribe_detailed`.
struct Transcript
  ## The language decoded, as a code: the pinned one, or the one detected.
  language: String
  segments: List[SegmentDetail]
  ## The detection, when `Options.language_probabilities` asked for it and the
  ## model is multilingual.
  detection: Optional[Detection]
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
  ## Like `transcribe_with`, with each segment's tokens and probabilities,
  ## the language decoded and, when `options` asks, the language detected.
  compute def transcribe_detailed(model: Model, pcm: List[Float], options: Options) -> Result[Transcript]
  ## The English name of a language code (`"es"` is `"spanish"`); none for a
  ## code the engine does not know.
  def language_name(code: String) -> Optional[String]
end

## Transcribes 16 kHz mono PCM audio with `Options()`: `join_text` of what
## `transcribe_with` returns, so plain text of a full decode. It delegates, so
## its errors name `transcribe_with`, not `transcribe`.
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
