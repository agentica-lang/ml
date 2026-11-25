# Speech to text with Whisper

Two packages transcribe:

- `asr`: Whisper written in Agentica over `tensor`, `nn` and `hub`. It loads a Hugging Face checkpoint and runs on
  ggml (CPU or Vulkan), libtorch or the reference backend. Its token ids equal Hugging Face's on the test clips. Use it
  by default.
- `whisper`: a binding over whisper.cpp. It is faster (`asr` takes about 2 to 3 times as long on the CPU, see
  `https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md`), but it needs the C++ engine built from a submodule and a ggml model file
  (`whisper/tools/fetch-model.sh` in a clone of the ml repository downloads one). Use it when speed matters more than setup.

## Prepare the audio

Whisper takes 16 kHz, mono, 16-bit PCM. The WAV reader refuses anything else and does not resample. Convert first:

```sh
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le clip.wav
```

## Transcribe a WAV with `asr`

Get `openai/whisper-tiny` (or `whisper-small`, `whisper-medium`) into a folder with `model.safetensors`,
`config.json` and `tokenizer.json` (`models.md`). `Wav.load` is the pure Agentica reader of the `whisper` package.

```agentica
import Asr from "pkg/asr/asr"
import Tensor from "pkg/tensor/tensor"
import Tokenizer from "pkg/asr/tokenizer"
import Wav from "pkg/whisper/wav"

def transcribe(dir: String, wav: String) -> Result[Asr.Transcript]
  backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  pcm = try Wav.load(wav)
  model = try Asr.open("#{dir}/model.safetensors", backend)
  tok = try Tokenizer.open("#{dir}/tokenizer.json")
  result = Asr.transcribe(model, tok, pcm, Asr.Options(language: none, timestamps: true))
  Asr.close_model(model)
  result
end

def main() -> Void
  t = transcribe("whisper-tiny", "clip.wav").unwrap!
  puts("#{t.language}: #{t.text}")
  for s in t.segments
    puts("#{s.start} #{s.stop} #{s.text}")
  end
end
```

- `language: none` detects the language; `some("es")` pins it. `translate: true` answers in English.
- Audio longer than 30 s is read in windows that seek by the timestamps, so it needs `timestamps: true` (the default).
  The segment times count from the start of the audio.
- `Options.max_tokens` and `max_windows` bound a run that hallucinates; `t.truncated` says when `max_windows` stopped it.
- Decoding is greedy. There is no beam search and no resampling.
- Whisper-tiny on ggml CPU runs at a real-time factor of about 0.24 (4 seconds for 17 s of audio). Vulkan fp16 can
  change tokens (`backends.md`, Settings).
- `Asr.open(path, backend, Tensor.Dtype.F16)` keeps half weights (`memory-and-errors.md`).

## whisper.cpp

```agentica
import Whisper from "pkg/whisper/whisper"

def main() -> Void
  model = Whisper.load_model("models/ggml-small.bin").unwrap!
  pcm = Whisper.load("clip.wav").unwrap!
  options = Whisper.Options(language: "es")
  for segment in Whisper.transcribe_with(model, pcm, options).unwrap!
    puts("#{segment.start_ms} ms to #{segment.end_ms} ms: #{segment.text}")
  end
end
```

`whisper` is a project of its own: it builds the pinned whisper.cpp with `whisper/tools/build-engine.sh` (clone the
ml repository at the tag, as `backends.md` does for ggml) and prints the exports to `eval`. One decode takes up to five
minutes of audio, `Whisper.Options.language` `"auto"` detects the language, and `Stream.transcribe_stream` cuts longer
audio into windows.

Pages: `https://www.agentica-lang.com/ml/asr/` and `https://www.agentica-lang.com/ml/whisper/`.
