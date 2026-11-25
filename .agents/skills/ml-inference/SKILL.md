---
name: ml-inference
description: >-
  Run machine-learning models from an Agentica app with the agentica-lang/ml
  packages (tensor, nn, hub, transformers, asr, whisper): install them by tag,
  pick a backend (reference, ggml on CPU or Vulkan, libtorch, XLA), load
  safetensors weights from a folder, generate text with Qwen3, embed sentences
  with DistilBERT or BERT, classify images with ResNet, and transcribe audio with Whisper. Use when an
  Agentica task runs a model or tensor code, or fixes an error of these
  packages. Not for training or fine-tuning (use ml-training), not for work on
  the ml repository itself, and not for writing plain Agentica (use
  agentica-programming).
---

# ml inference

The ml packages are tensors and models for Agentica. The API is functional: a
tensor never changes, and every op returns a new one. This skill runs a model
from an app. If the `agentica-programming` skill is available, read it first when the language is new to you.

## Install

Write the packages in the `agentica.toml` of the app, then lock and fetch them.
Use the tags of one release for every package: the packages ask for each other
by the tag of their own release, and the tags meet.

```toml
name = "app"
agentica = "0.38.0"

[dependencies]
tensor = { git = "https://github.com/agentica-lang/ml", tag = "tensor-0.4.0", subdir = "tensor" }
nn = { git = "https://github.com/agentica-lang/ml", tag = "nn-0.4.0", subdir = "nn" }
hub = { git = "https://github.com/agentica-lang/ml", tag = "hub-0.4.0", subdir = "hub" }
transformers = { git = "https://github.com/agentica-lang/ml", tag = "transformers-0.4.0", subdir = "transformers" }
asr = { git = "https://github.com/agentica-lang/ml", tag = "asr-0.4.0", subdir = "asr" }
# only to read WAV files with `pkg/whisper/wav` (it needs no C++ engine), or to use whisper.cpp:
whisper = { git = "https://github.com/agentica-lang/ml", tag = "whisper-0.4.0", subdir = "whisper" }
```

```sh
agentica update    # writes agentica.lock
agentica install   # writes deps/
```

Keep only the lines the app needs: `tensor` and `nn` for tensor code, add
`hub` for weights, `transformers` for Qwen3, BERT and ResNet models (the input of ResNet is made by the `vision` package: `Resize(256)`, `CenterCrop(224)`, `normalize`), `asr` for Whisper
(`whisper` only for the WAV reader or whisper.cpp). The tags
are released per package: list them with
`git ls-remote --tags https://github.com/agentica-lang/ml 'tensor-*'`.

## First program

A matrix product on the reference backend, which needs nothing native:

```agentica
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  a = Tensor.floats([1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap!
  b = Tensor.floats([5.0, 6.0, 7.0, 8.0], [2, 2]).unwrap!
  puts(Tensor.to_list(Tensor.matmul(a, b).unwrap!).unwrap!)
end
```

```output
[Values.Float(19.0), Values.Float(22.0), Values.Float(43.0), Values.Float(50.0)]
```

## Rules

- There is no operator overloading and no implicit broadcasting: `Tensor.add(a, b)`. Every op returns a
  `Result`: `try` it inside a function that returns `Result[T]`, `.unwrap!` only in `main`.
- Nothing is downloaded. A model is a local folder (`model.safetensors`, `config.json`, `tokenizer.json`):
  `references/models.md` says how to get one.
- `Backend.Reference` is slow and is for checks. A real model needs `Backend.Ggml(Device.Cpu)` or `Device.Vulkan(0)`,
  or libtorch. ggml and libtorch are native modules that you build once (`references/backends.md`); without
  them `Tensor.available(backend)` is `false` and an op answers `error("backend not available")`.
- On ggml an op other than create, read and copy is an error when called by itself, because ggml runs whole graphs.
  Use the model calls, or put your ops in `Trace.eval` (`references/backends.md`).
- A model holds native memory: close it (`Qwen3.close_model`, `Bert.close_model`, `Asr.close_model`), and close the
  tensors you are done with (`references/memory-and-errors.md`).
- Vulkan multiplies in fp16 by default, so greedy ids can differ from Hugging Face's (`references/backends.md`,
  Settings).

## References

Read only what the task needs.

| File in `references/` | Covers |
|---|---|
| `backends.md` | the five backends, building ggml, libtorch and the XLA plugin, picking one, `Trace.eval` and `Trace.compile` |
| `models.md` | getting a checkpoint, Qwen3 generation and sampling, the chat template, DistilBERT and BERT embeddings, `Hub` |
| `speech.md` | Whisper with `asr`: a WAV to text, timestamps, language, long audio; whisper.cpp |
| `memory-and-errors.md` | tensors on the device, closing, reading back, half weights, and the errors you will meet (a table of messages) |

The full API is on the site pages, one per package:
`https://www.agentica-lang.com/ml/tensor/`, `.../nn/`, `.../hub/`,
`.../transformers/`, `.../asr/`, `.../whisper/`. Training is the `ml-training`
skill.
