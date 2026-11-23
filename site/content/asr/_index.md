---
title: asr
description: Whisper in Agentica over tensor, nn and hub, from a Hugging Face checkpoint to a transcript of up to 30 s.
weight: 5
---

`asr` is Whisper's model, written in Agentica, for inference. It loads a Hugging Face checkpoint, runs the log-mel frontend, the encoder and greedy decoding, and returns the text with its segments. It depends on `tensor`, `nn` and `hub`.

It runs on the reference backend and on libtorch. Its `mathx` module is a small native module over libm, so a C compiler is needed to build the package.

## Example

Transcribe one clip in Spanish with a Whisper checkpoint in a folder. The folder holds `model.safetensors`, `config.json` and `tokenizer.json`. The samples are 16 kHz mono floats, for example read with the `whisper` package's `Wav.load`.

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
- `Asr.encode(model, mel)` and `Asr.decode(model, tokens, audio)` are the model's forward passes, for callers who want the parts.
- `Asr.listen`, `Asr.step` and the `Cache` they use decode one token at a time with a KV cache.

`Asr.Options` has these fields:

- `language`, an `Optional[String]` such as `some("es")`. `none`, the default, detects the language.
- `translate`, default `false`: translate to English instead of transcribing.
- `timestamps`, default `true`: timestamp tokens, which give the segments.
- `max_tokens`, an `Optional[Int]`. `none`, the default, decodes up to the model's `max_target_positions`.

`Asr.Transcript` has these fields:

- `text`, the text.
- `language` and `language_probability`, the language and its probability.
- `tokens`, the ids after the prompt, without the end-of-text token.
- `segments`, a list of `Greedy.Segment`, each with `start` and `stop` in seconds, `text`, `tokens` and `avg_logprob`.
- `avg_logprob` and `no_speech_prob`, as OpenAI's Whisper reports them.

## Limits

- Up to one window of audio: 30 s of 16 kHz audio for Whisper's checkpoints, which is `2 * max_source_positions` frames of 160 samples. Longer audio must be cut by the caller; the `whisper` package does that with its own engine.
- Greedy decoding at temperature 0, with OpenAI's rules for the suppressed tokens and the timestamps. There is no beam search.
- The samples must be mono 16 kHz floats. The frontend does not resample.
- `Tokenizer.vocab_size` must equal the model's `vocab_size`, or `transcribe` is an error.
- `tools/transcribe_check.agc` compares the cases of `test/fixtures/transcribe_golden.json` with a real whisper-tiny checkpoint. It is run by hand, not by the tests. Against Hugging Face's `generate` on whisper-tiny, the token ids match.

## Speed

[`benchmarks/`](https://github.com/agentica-lang/ml/tree/main/benchmarks) times whisper-tiny against whisper.cpp on one machine, with the same weights, clips and threads. On the CPU the encoder is on par; the decoder step, the log-mel and the load are where the gap is.
