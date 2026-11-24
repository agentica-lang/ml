---
title: asr
description: Whisper in Agentica over tensor, nn and hub, from a Hugging Face checkpoint to a transcript of audio of any length.
weight: 5
---

`asr` is Whisper, OpenAI's speech-to-text model, written in Agentica for inference. It turns audio into text, with timed segments. Audio over 30 seconds is read in windows of 30 seconds.

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

- `Asr.open(path, backend, weights)` loads a `.safetensors` file, or the `.index.json` of a sharded one, with the `config.json` in its folder. With `weights: Dtype.F16` the big dense matrices and the token table (also the tied output head) are kept as half weights on ggml, which halves the weights' memory; whisper-medium transcribes 9% faster on Vulkan and no faster on the CPU, and whisper-tiny gains little (see the hub page for the numbers). `F32` is the default and the exact path, and Vulkan multiplies in fp16 by default (`Tensor.vulkan_fp16`), which breaks token-exactness against Hugging Face with `F32` weights too: `Tensor.vulkan_fp16(false)` before the first ggml call makes it exact, as the checks of this package do.
- `Asr.close_model(model)` frees the weights and the log-mel plan now, as a server does when it replaces a model. Using the model afterwards is an error, and a second call changes nothing.
- `Asr.config(path)` reads the `config.json` of a checkpoint as a `Config`: the sizes, head counts and position limits the forward needs, and the suppressed token lists. `Asr.load(checkpoint, config, backend, weights)` makes the model from a checkpoint from `Hub.open` and a `Config`. `open` does both.
- `Asr.transcribe(model, tok, pcm, options)` runs the whole pipeline on one clip and gives a `Result` of `Transcript`.
- `Asr.features(model, pcm, frames)` gives the log-mel of `pcm` as a `[1, num_mel_bins, frames]` `F32` tensor on the model's backend. The caller closes it.
- `Tokenizer.open(path)` reads a `tokenizer.json`. The tokenizer decodes ids to text; it does not encode text.
  - `Tokenizer.decode(tok, ids)` gives the text of the ids. Special and timestamp tokens give no text, and a character split across tokens joins back.
  - `Tokenizer.prompt(tok, language, job, timestamps)` gives the ids that start the decoder. `language` is `none` or a code, and `job` is `Tokenizer.Job.Transcribe` or `Tokenizer.Job.Translate`.
  - `Tokenizer.language_id(tok, code)` gives the id of a language, and `Tokenizer.seconds(tok, id)` the time of a timestamp token in seconds.
  - `Tokenizer.parse(text)` reads the text of a `tokenizer.json` instead of a path.
- `Asr.encode(model, mel)` and `Asr.decode(model, tokens, audio)` are the model's forward passes (one run of the network), for callers who want the parts.
- `Asr.listen`, `Asr.step`, `Asr.step_pick` and the `Cache` they use decode one token at a time. The cache is a KV cache: it keeps the past keys and values, so earlier tokens are not computed again, and its `length` counts the tokens so far. `Asr.step` gives a `Stepped`, with the logits of the last token and the new cache; `Asr.step_pick(model, cache, ids, mask)` adds `mask` to them (0 for an id that may be emitted, a very negative number for one that may not) and gives the first largest id and its log-probability as a `Picked`; with `stamps`, an `Asr.Stamps(rules, state)`, it applies Whisper's timestamp rules in the graph too: `rules` are the `Greedy.Rules`, and `state` is `Greedy.state(rules, emitted)` as an F32 tensor of 4 on the model's backend, made for each step and closed by the caller.

`Asr.Options` has these fields:

- `language`, an `Optional[String]` such as `some("es")`. `none`, the default, detects the language.
- `translate`, default `false`: translate to English instead of transcribing.
- `timestamps`, default `true`: timestamp tokens, which give the segments. On ggml the next token is chosen on the device, with or without the timestamp rules, and only its id and log-probability come back.
- `max_tokens`, an `Optional[Int]`. `none`, the default, decodes up to the model's `max_target_positions`. For audio over 30 s it applies to each window.
- `max_windows`, an `Optional[Int]`, for audio over 30 s: at most that many windows. `none`, the default, is `2 * ceil(frames / window) + 2`, which audio that seeks normally never reaches, so a model that hallucinates and never advances is stopped.

`Asr.Transcript` has these fields:

- `text`, the text.
- `language` and `language_probability`, the language and its probability.
- `tokens`, the ids after the prompt, without the end-of-text token.
- `segments`, a list of `Greedy.Segment`, each with `start` and `stop` in seconds, `text`, `tokens` and `avg_logprob`.
- `avg_logprob` and `no_speech_prob`, as OpenAI's Whisper reports them.
- `truncated`, `false` unless `max_windows` ran out before the end of the audio.

## Greedy rules

`pkg/asr/greedy` has the rules of greedy decoding over plain lists. Use it for a loop of your own over `step_pick`:

- `Greedy.rules(tok, suppress, begin_suppress, timestamps)` makes the `Rules` for a tokenizer. `suppress` and `begin_suppress` are the config's lists, or `none` for the defaults. `timestamps: false` turns the timestamp rules off.
- `Greedy.mask(rules, size, first)` gives the scores of a `step_pick` mask: 0 for an id that may come, `Greedy.FORBIDDEN` for one that may not. `Greedy.state(rules, emitted)` gives the four numbers that `Asr.Stamps` takes.
- `Greedy.allowed(rules, logits, emitted)` applies the rules to a list of logits, and `Greedy.pick(logits)` gives the first largest id.
- `Greedy.segments(tok, emitted, duration, avg_logprob)` makes the `Segment`s of emitted ids. `Greedy.window(tok, emitted, frames, avg_logprob)` also gives the ids to keep and how many 10 ms frames the next window advances.
- `Greedy.probability(logits, i)` and `Greedy.log_probability(logits, i)` score one id of a list of logits.

## Frontend

`pkg/asr/mel` is the log-mel frontend:

- `Mel.log_mel(samples, mels, frames)` gives the log-mel as a flat list, row-major `[mels, frames]`, computed with an exact FFT in doubles. `frames` defaults to 3000, which is 30 s. Non-finite samples, or samples far outside -1..1, are errors.
- `Mel.spectrogram(samples, mels, frames, backend)` is the same computation as one traced graph on any backend. In F32 it is about 1e-5 from `log_mel`.
- `Mel.features` picks between them: `Reference` gets `log_mel`, and every other backend the graph. `Asr.features` calls it.

## Limits

- Audio over one window (30 s of 16 kHz audio for Whisper's checkpoints; a window is the chunk the model takes in one pass) needs `timestamps: true`, the default. It is read as OpenAI's `transcribe` and Hugging Face's long-form `generate` (with `truncation=False` features) do: each window starts where the last timestamp of the one before said its segments end, with no previous text in the prompt, and the language is the first window's. For such audio:
  - the segments' times count from the start of the audio, and a segment with no text is left out;
  - `tokens` are the ids of the segments of every window, each window with its own timestamps;
  - `language_probability` and `no_speech_prob` are the first window's, and `avg_logprob` covers all the windows;
  - `max_tokens` applies to each window;
  - a window whose timestamps would seek by zero is done instead, so it always advances;
  - at most `max_windows` windows are decoded (see `Options`), and `truncated` says when that stopped the transcript short.
- Decoding is greedy: at each step it takes the most likely token (temperature 0). It follows OpenAI's rules for suppressed tokens and timestamps. There is no beam search (a search that keeps several candidate outputs).
- The samples must be mono 16 kHz floats. The frontend, the part that turns audio into the model's input, does not resample (change the sample rate).
- `Tokenizer.vocab_size` must equal the model's `vocab_size`, or `transcribe` is an error.

## How it works

`asr` loads a Hugging Face checkpoint. Then it runs the log-mel frontend (it turns audio into log-mel features, the model's input; on ggml and libtorch it is one graph of tensor ops), the encoder and greedy decoding. It depends on `tensor`, `nn` and `hub`.

`asr` runs on the reference backend, on libtorch and on ggml (CPU and Vulkan). See the [tensor page](../tensor/_index.md) for the backends.

A window is 2 * `max_source_positions` frames of 160 samples. For Whisper's checkpoints that is 30 seconds of 16 kHz audio.

`tools/transcribe_check.agc` compares the cases of `test/fixtures/transcribe_golden.json` with a real whisper-tiny checkpoint. It is run by hand, not by the tests. Against Hugging Face's `generate` on whisper-tiny, the token ids match.

## Speed

The [benchmarks folder](https://github.com/agentica-lang/ml/tree/main/benchmarks) times whisper-tiny against whisper.cpp on one machine, with the same weights, clips and threads. On the CPU the encoder is on par. The gap is in the decoder step and the load.
