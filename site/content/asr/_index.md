---
title: asr
description: Whisper in Agentica over tensor, nn and hub, from a Hugging Face checkpoint to a transcript of up to 30 s.
weight: 5
---

`asr` is Whisper, OpenAI's speech-to-text model, written in Agentica for inference. It turns up to 30 seconds of audio into text, with timed segments.

## Example

Transcribe one clip in Spanish. The folder `whisper-tiny` holds `model.safetensors`, `config.json` and `tokenizer.json`, from a Hugging Face Whisper checkpoint. The samples are 16 kHz mono floats: one channel, 16 000 samples per second. You can read them with `Whisper.load` from the `whisper` package, which uses `Wav.load` from its `wav` module.

```agentica
import Asr from "pkg/asr/asr"
import Tensor from "pkg/tensor/tensor"
import Tokenizer from "pkg/asr/tokenizer"

def transcribe_file(pcm: List[Float]) -> Result[Void]
  backend = Tensor.Backend.Reference
  model = try Asr.open("whisper-tiny/model.safetensors", backend)
  tok = try Tokenizer.open("whisper-tiny/tokenizer.json")
  options = Asr.Options(language: some("es"), timestamps: true)
  t = try Asr.transcribe(model, tok, pcm, options)
  puts(t.text)
  puts(t.language)
  for s in t.segments
    puts("#{s.start} #{s.stop} #{s.text}")
  end

  ok()
end

def main() -> Void
  transcribe_file([0.0]).unwrap!
end
```

## Main API

- `Asr.open(path, backend)` loads a `.safetensors` file, or the `.index.json` of a sharded one, with the `config.json` in its folder.
- `Asr.transcribe(model, tok, pcm, options)` runs the whole pipeline on one clip and gives a `Result` of `Transcript`.
- `Tokenizer.open(path)` reads a `tokenizer.json`.
- `Asr.encode(model, mel)` and `Asr.decode(model, tokens, audio)` are the model's forward passes (one run of the network), for callers who want the parts.
- `Asr.listen`, `Asr.step`, `Asr.step_pick` and the `Cache` they use decode one token at a time. The cache is a KV cache: it keeps the past keys and values, so earlier tokens are not computed again. `Asr.step` gives the logits of the last token; `Asr.step_pick(model, cache, ids, mask)` adds `mask` to them (0 for an id that may be emitted, a very negative number for one that may not) and gives the first largest id and its log-probability as a `Picked`.

`Asr.Options` has these fields:

- `language`, an `Optional[String]` such as `some("es")`. `none`, the default, detects the language.
- `translate`, default `false`: translate to English instead of transcribing.
- `timestamps`, default `true`: timestamp tokens, which give the segments. With `false`, on ggml, the next token is chosen on the device and only its id and log-probability come back; with `true` every step still reads the logits, so the faster path is not taken by default.
- `max_tokens`, an `Optional[Int]`. `none`, the default, decodes up to the model's `max_target_positions`.

`Asr.Transcript` has these fields:

- `text`, the text.
- `language` and `language_probability`, the language and its probability.
- `tokens`, the ids after the prompt, without the end-of-text token.
- `segments`, a list of `Greedy.Segment`, each with `start` and `stop` in seconds, `text`, `tokens` and `avg_logprob`.
- `avg_logprob` and `no_speech_prob`, as OpenAI's Whisper reports them.

## Limits

- Up to one window of audio: 30 s of 16 kHz audio for Whisper's checkpoints. A window is the chunk the model takes in one pass. Longer audio must be cut by the caller. The `whisper` package does that with its own engine.
- Decoding is greedy: at each step it takes the most likely token (temperature 0). It follows OpenAI's rules for suppressed tokens and timestamps. There is no beam search (a search that keeps several candidate outputs).
- The samples must be mono 16 kHz floats. The frontend, the part that turns audio into the model's input, does not resample (change the sample rate).
- `Tokenizer.vocab_size` must equal the model's `vocab_size`, or `transcribe` is an error.

## How it works

`asr` loads a Hugging Face checkpoint. Then it runs the log-mel frontend (it turns audio into log-mel features, the model's input), the encoder and greedy decoding. It depends on `tensor`, `nn` and `hub`.

`asr` runs on the reference backend, on libtorch and on ggml (CPU and Vulkan). See the [tensor page](../tensor/_index.md) for the backends.

A window is 2 * `max_source_positions` frames of 160 samples. For Whisper's checkpoints that is 30 seconds of 16 kHz audio.

`tools/transcribe_check.agc` compares the cases of `test/fixtures/transcribe_golden.json` with a real whisper-tiny checkpoint. It is run by hand, not by the tests. Against Hugging Face's `generate` on whisper-tiny, the token ids match.

## Speed

The [benchmarks folder](https://github.com/agentica-lang/ml/tree/main/benchmarks) times whisper-tiny against whisper.cpp on one machine, with the same weights, clips and threads. On the CPU the encoder is on par. The gap is in the decoder step, the log-mel and the load.
