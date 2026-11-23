---
title: whisper
description: A binding over the pinned whisper.cpp engine, with WAV reading, microphone capture and long-audio chunking.
weight: 6
---

`whisper` transcribes speech with whisper.cpp, which is a pinned git submodule. It is a project of its own, not a member of the workspace of the other packages. It needs a C/C++ compiler and cmake to build the engine, and a ggml model file, which `tools/fetch-model.sh` downloads.

The decode runs in the compute pool. It uses the GPU when the engine was built with Vulkan, and the CPU otherwise.

## Example

Load a model, read a 16 kHz mono WAV file, and print the text and the timed segments. The paths are placeholders.

```agentica
import Whisper from "pkg/whisper/whisper"

def main() -> Void
  model = Whisper.load_model("models/ggml-small.bin").unwrap!
  pcm = Whisper.load("clip.wav").unwrap!
  text = Whisper.transcribe(model, pcm).unwrap!
  puts(text)
  segments = Whisper.transcribe_with(model, pcm, Whisper.Options(language: "es")).unwrap!
  for segment in segments
    puts("#{segment.start_ms} ms to #{segment.end_ms} ms: #{segment.text}")
  end
end
```

## Main API

- `load_model(path)` loads a ggml or gguf model into a `Model`. The path is at most 4096 bytes.
- `Whisper.load(path)` reads a WAV file as a list of 16 kHz mono floats. Only PCM, mono, 16 kHz and 16-bit files are read.
- `transcribe(model, pcm)` gives the text of one decode with the default options.
- `transcribe_with(model, pcm, options)` gives the timed `Segment`s of one decode, in time order. A `Segment` has `start_ms`, `end_ms` and `text`.
- `transcribe_detailed(model, pcm, options)` gives a `Transcript`: the language, the segments with their tokens, and the detected language when asked for.
- `join_text(segments)` joins the texts of segments with single spaces.
- `language_name(code)` gives the English name of a language code, such as `"spanish"`.

`Whisper.Options` has these fields:

- `language`, default `""`, which means English. `"auto"` detects the language, and a code such as `"es"` pins it.
- `translate`, default `false`: answer in English.
- `beam_size`, default `0`. From 2 to 8 it decodes with beam search; `0` and `1` decode greedily.
- `token_timestamps`, default `false`: times each token.
- `max_len`, default `0`: with `token_timestamps`, cuts the segments to at most this many characters.
- `language_probabilities`, default `false`: fills the detection with the probability of every language.

Long audio and a live microphone are in two more modules. `stream.agc` cuts a PCM stream into 30 s windows with overlap, skips the silent ones with an energy gate, and merges the segments onto one timeline. `mic.agc` captures 16 kHz mono audio from a command, `arecord` by default.

## Limits

- One decode takes 1 to 4 800 000 samples, five minutes of 16 kHz audio. Longer input, empty input and non-finite samples are errors.
- The language code is at most 32 bytes, `beam_size` is from 0 to 8, and `max_len` is from 0 to 100 000.
- The text of one call is at most one megabyte.
- Inference on one model runs one decode at a time, because a mutex serializes it.
- There are no partial results: each decode returns its segments when it ends.
- The package reads WAV files only at the rate and format above. It does not resample.
- Microphone capture needs `arecord` and a device. Nothing else in the package does.
