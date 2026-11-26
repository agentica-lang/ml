---
title: whisper
description: A binding over the pinned whisper.cpp engine, with WAV reading, microphone capture and long-audio chunking.
weight: 6
---

`whisper` transcribes speech with whisper.cpp, a C and C++ engine for Whisper models. It reads 16 kHz WAV files and returns the text, with timed segments if you ask for them.

## Example

Load a model, read a 16 kHz mono WAV file, and print the text and the timed segments. The paths are placeholders:

```agentica
import Whisper from "pkg/whisper/whisper"

def main() -> Void
  model = Whisper.load_model("models/ggml-small.bin").unwrap!
  pcm = Whisper.load("clip.wav").unwrap!
  text = Whisper.transcribe(model, pcm).unwrap!
  puts(text)
  options = Whisper.Options(language: "es")
  segments = Whisper.transcribe_with(model, Whisper.widen(pcm), options).unwrap!
  for segment in segments
    puts("#{segment.start_ms} ms to #{segment.end_ms} ms: #{segment.text}")
  end
end
```

## Setup

`whisper` is a project of its own, not a member of the workspace of the other packages. It needs a C or C++ compiler and cmake (a build tool) to build the engine. The engine is whisper.cpp, a pinned git submodule.

It also needs a ggml model file. `tools/fetch-model.sh` downloads one.

## Main API

- `load_model(path)` loads a ggml or gguf model (the model file formats of whisper.cpp) into a `Model`. The path is at most 4096 bytes.
- `Whisper.load(path)` reads a WAV file as a `List[Float32]` of 16 kHz mono samples, 4 bytes each. Only PCM, mono, 16 kHz and 16-bit files are read.
- `Whisper.decode(data)` reads WAV bytes that are already in memory, by the same rules.
- `transcribe(model, pcm)` takes that `List[Float32]` and gives the text of one decode with the default options. It widens the whole clip for the length of the call (8 bytes a sample), so for a long clip use `Stream.transcribe_stream`, which widens one window at a time. A decode is one run of the model over the audio.
- `transcribe_with(model, pcm, options)` takes a `List[Float]`, because a native module cannot take a `Float32`: `Whisper.widen(pcm)` makes one from the samples of `load`, as a copy of 8 bytes a sample for the length of the call. It gives the timed `Segment`s of one decode, in time order. A `Segment` has `start_ms`, `end_ms` and `text`.
- `transcribe_detailed(model, pcm, options)` takes a `List[Float]` as `transcribe_with` does and gives a `Transcript`: the language, the segments with their tokens, and the detected language when asked for.
- `join_text(segments)` joins the texts of segments with single spaces.
- `language_name(code)` gives the English name of a language code, such as `"spanish"`, or none for a code the engine does not know.

`Whisper.Options` has these fields:

- `language`, default `""`, which means English. `"auto"` detects the language, and a code such as `"es"` pins it.
- `translate`, default `false`: answer in English.
- `beam_size`, default `0`. From 2 to 8 it decodes with beam search, which keeps several candidate outputs. `0` and `1` decode greedily, taking the most likely token at each step.
- `token_timestamps`, default `false`: times each token.
- `max_len`, default `0`: with `token_timestamps`, cuts the segments to at most this many characters.
- `language_probabilities`, default `false`: fills the detection with the probability of every language.

`Whisper.Transcript` has `language` (the code decoded), `segments` (a list of `SegmentDetail`) and `detection`, which is `some(Detection)` when `language_probabilities` asked for it on a multilingual model, and `none` otherwise.

- `SegmentDetail` has `start_ms`, `end_ms`, `text` (not trimmed), `tokens` (a list of `Token`), `avg_logprob` (the sum of the tokens' log probabilities over their number, special tokens included) and `no_speech_probability`.
- `Token` has `ids`, `text`, `probability` (of its first token), `log_probability` (the sum over its tokens), and `start_ms` and `end_ms`, which are -10 unless `token_timestamps` is on.
- `Detection` has `language`, its `probability`, and `probabilities`, a map from every code to its probability.

## Long audio and the microphone

The `stream` module (`pkg/whisper/stream`) transcribes audio of any length, in windows:

- `Stream.transcribe_stream(model, source, options)` reads a `Mic.Source`, checks the walk, decodes each window with speech and merges the segments onto one timeline, and gives a `List[Whisper.Segment]`. Its times count from the first sample. Audio with no speech gives no segments, and no samples at all is an error.
- `Stream.default_options()` gives a `StreamOptions` with `window_s` 30, `overlap_s` 5, the default `vad` and the default `Whisper.Options`. `window_s` is from 1 to 300 seconds, and `overlap_s` is from 0 to `window_s` minus 1. `Stream.check_options(options)` checks these, `frame_ms` (1..1000) and `threshold` (0.0..1.0), without audio or a model.
- `VadOptions` has `frame_ms` (30) and `threshold` (0.02). A window is kept when one frame's RMS reaches the threshold. `threshold: 0.0` keeps every window.
- `windows`, `offset_segments`, `merge_segments`, `has_speech` and `frame_rms` are the pure steps of the walk. Tests run them without a model.

The `mic` module (`pkg/whisper/mic`) captures audio:

- `Mic.Source` is `Samples(list)`, for PCM that is already there, or `Command(capture)`, for a live capture.
- `Mic.Capture` has `command` (`"arecord"`), `args`, `timeout_ms` (30 000) and `max_samples` (160 000, ten seconds). `Mic.default_capture()` gives the `arecord` capture of 16 kHz mono 16-bit PCM.
- `Mic.capture(spec)` runs the command and gives its samples as a `List[Float32]`. It is an error when the command cannot start, a read times out, the command dies mid-capture, or no audio comes.
- `Mic.read(source)` gives the samples of a `Mic.Source`, and `Mic.decode_s16le(data)` decodes raw 16-bit PCM bytes.

## Limits

- One decode takes 1 to 4 800 000 samples, which is five minutes of 16 kHz audio. Longer input, empty input and non-finite samples are errors.
- The language code is at most 32 bytes. `beam_size` is from 0 to 8, and `max_len` is from 0 to 100 000.
- The text of one call is at most one megabyte.
- A model runs one decode at a time. A mutex makes other calls wait.
- There are no partial results. Each decode returns its segments when it ends.
- The package reads WAV files only at the rate and format above. It does not resample (change the sample rate).
- Microphone capture needs `arecord`, the Linux recording tool, and a device. It reads 16 kHz mono audio. Nothing else in the package does.

## How it works

The decode runs in the compute pool, a set of background threads. It uses the GPU when the engine was built with Vulkan, and the CPU otherwise.
