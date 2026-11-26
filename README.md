# ml

Machine learning for Agentica: tensors, training and inference over a
reference backend, libtorch, ggml (CPU and Vulkan) and XLA (CPU). The API is
functional, in the style of JAX and Nx: values are immutable, and a training
step returns new parameters.

Status: in progress. The API is not stable yet.

Works today: `tensor/` (the reference, libtorch, ggml and XLA backends, tracing
and gradients), `nn/` (layers, forward only), `train/` (losses, SGD and Adam, a
training loop, a data loader, LoRA with PEFT's files), `hub/` (safetensors and
GGUF checkpoints), `asr/` (Whisper on the reference, libtorch and ggml
backends, with Hugging Face's token ids on the test clips), `transformers/`
(BERT, DistilBERT, Qwen3, Qwen2.5, Qwen3.5, IBM Granite MoE, ResNet, ViT and
CLIP), `vision/` (images to tensors, on libvips) and `whisper/` (a binding over whisper.cpp).

Works, but slow: `asr/` on ggml. Its tokens match, but on the test clips it
runs about 1.5 times slower than whisper.cpp on CPU and about 1.7 times
slower on Vulkan (jfk). The numbers and the machine are in
[`benchmarks/README.md`](benchmarks/README.md).

Not yet: what the [roadmap](roadmap.md) lists, from vision-language models to CUDA,
serving and distributed training.

The tensor stack is designed in [`tensor/DESIGN.md`](tensor/DESIGN.md) and the
original design is in [issue #1](https://github.com/agentica-lang/ml/issues/1).

## Packages

One repository, one folder per package, each with its own `agentica.toml`, so
an app installs only what it needs (for example inference without training).
Each package is released on its own tag, `<package>-X.Y.Z`; the releases are
listed in [`CHANGELOG.md`](CHANGELOG.md). A project pins a member by its
tag. The API is not stable yet, so expect changes between versions.

| Folder | What | Needs |
|---|---|---|
| `tensor/` | the tensor API, the backend interface, a slow reference backend in Agentica, the libtorch (eager) backend behind a thin C++ layer, the ggml backend (CPU and Vulkan), which runs a traced function as one graph, and the XLA backend (CPU), which compiles it with the PJRT plugin | nothing native for the reference backend; libtorch 2.4.1+cpu, **optional**, for `Backend.Torch`; ggml (the `third_party/ggml` submodule, built by `tensor/tools/build-engine.sh`), **optional**, for `Backend.Ggml`; the XLA plugin (`agentica tools/install_xla.agc`), **optional**, for `Backend.Xla` |
| `nn/` | functional layers, forward only: dense, layer norm, embeddings, convolution, attention (grouped-query, rotary), MLP, transformer encoder and decoder blocks and stacks, seeded initializers, a mixture-of-experts block and the Gated DeltaNet layer of Qwen3.5 | `tensor` |
| `train/` | losses (`mse`, binary and softmax cross-entropy), optimizers (SGD, Adam, AdamW), a data loader (seeded shuffle, background prefetch), the training loop over `tensor/`'s gradients, training part of a model and LoRA adapters | `tensor`, `hub`, `nn` |
| `hub/` | `safetensors` checkpoints (one file or sharded with an index) and GGUF files, and `nn` layers loaded from them by name | `tensor`, `nn` |
| `asr/` | Whisper in Agentica: the model forward loaded from a Hugging Face checkpoint, its log-mel frontend, its tokenizer and `transcribe` (greedy decoding with a KV cache, language detection, timestamps, translation) for audio of any length, in windows of 30 s that seek; on the reference backend, libtorch and ggml (CPU and Vulkan) | `tensor`, `nn`, `hub` |
| `transformers/` | BERT, DistilBERT, Qwen3, Qwen2.5, Qwen3.5 text and IBM Granite MoE (the forward and generation with a KV cache) and ResNet-18/50, ViT-B/16 (image classification) and CLIP ViT-B/32 (image and text embeddings, zero-shot) from Hugging Face checkpoints, and the WordPiece and byte-level BPE tokenizers, which give Hugging Face's ids | `tensor`, `nn`, `hub`, `train` (for the fine-tune tool of `tools/`) |
| `vision/` | images for tensors: libvips decodes and transforms them (resize, crops, flip), their pixels reach a tensor as one buffer, `normalize` runs on the backend, and a folder per class makes batches for `train`'s loader | `tensor`, the `image` package (libvips with its development files) |
| `whisper/` | a binding over the pinned whisper.cpp engine, with WAV reading, microphone capture and long-audio chunking | whisper.cpp (a submodule), a C/C++ compiler |

An inference app depends on `nn` and one backend; a training app adds `train`.

Each package has a page on the site with its API and its limits:

- [tensor](https://www.agentica-lang.com/ml/tensor/)
- [nn](https://www.agentica-lang.com/ml/nn/)
- [train](https://www.agentica-lang.com/ml/train/), with its
  [data loader](https://www.agentica-lang.com/ml/train/data/)
- [hub](https://www.agentica-lang.com/ml/hub/)
- [asr](https://www.agentica-lang.com/ml/asr/)
- [transformers](https://www.agentica-lang.com/ml/transformers/)
- [vision](https://www.agentica-lang.com/ml/vision/)
- [whisper](https://www.agentica-lang.com/ml/whisper/)

Inside this repository `tensor`, `nn`, `train`, `hub`, `asr`, `transformers` and `vision` are members of
one Agentica workspace (the root `agentica.toml`), so they read each other from
their folders; `whisper` is a project of its own, not a member. Each package
pins its own Agentica version (`AGENTICA_VERSION`), and CI runs `check`,
`fmt --check` and `test` on each folder that has an `agentica.toml`, on that
version.

The members ask for each other by the tag of the same release
(`tag = "tensor-0.6.0"`). Inside the workspace the tag is ignored and the
folder is read. Outside it, an app that declares the packages it imports by
the tags of one release gets one source for each package, so it needs no
`override = true`:

```toml
[dependencies]
nn = { git = "https://github.com/agentica-lang/ml", tag = "nn-0.6.0", subdir = "nn" }
tensor = { git = "https://github.com/agentica-lang/ml", tag = "tensor-0.6.0", subdir = "tensor" }
```

The libtorch backend ships **inside `tensor/`**, not as a `torch/` package,
because a native handle is a `resource` that is nominal to its `extern module`
([`tensor/DESIGN.md`](tensor/DESIGN.md) §1, §7). libtorch is `optional = true` and never vendored or downloaded
by this repository: without it, everything still checks, formats and tests,
and `Backend.Torch` answers `error("backend not available")`.
[`tensor/README.md`](tensor/README.md) has the setup, the coverage table, the
tolerances and the measured cost of the dependency.

## Agent skills

Two skills in [`.agents/skills/`](.agents/skills/) guide an agent that uses the packages from an app: `ml-inference`
(install by tag, backends, loading weights, Qwen3, DistilBERT, Whisper, memory and errors) and `ml-training` (`fit`,
validation and checkpoints, LoRA and PEFT files, speed, checking against PyTorch). To give them to an agent, copy both
folders (`ml-inference` and `ml-training`; the second points to the first) into the `.agents/skills/` folder of your
project. Their code blocks are programs: `make check-skills` checks them, and runs the ones with an output on the
reference backend.

## Checking the native C

CI runs [cchecker](https://github.com/agentica-lang/cchecker) over the native C of `tensor/` (`ggml.c`, `ggml_graph.c`), `whisper/` and `asr/`: locks released on every path, and the ggml locks taken in order (device, cache, deferred), ownership, `gcc -fanalyzer` and clang-tidy. It fails on a finding that is not in `cchecker-baseline.json`. `torch.c` is left out: it is C++, and cchecker reads C.

```sh
git submodule update --init third_party/ggml third_party/whisper.cpp
make cchecker CCHECKER_FLAGS="--gcc gcc-14 --clang clang-19 --clang-tidy clang-tidy-19 --strict-baseline"   # what CI runs
```

It needs gcc 10+, clang, clang-tidy and the pinned `agentica` on PATH. No engine is built: the database (`build/cchecker/`, made by `tools/cchecker.sh`) only needs the headers of the submodules and of `agentica`. The version of cchecker is pinned in `tools/cchecker.sh`; the settings are `cchecker.toml`.

- Mark what cchecker cannot see with the macros of `tensor/native/cchecker.h` (a copy of cchecker's `include/cchecker.h`, empty in a normal build), for example `CC_ACQUIRES` on a function that returns with a lock held.
- A false positive of one line gets `// cchecker: ignore CODE: reason` next to it.
- To accept a finding, run `make cchecker-update-baseline`, review the diff of `cchecker-baseline.json`, and write the reason in `cchecker.toml` and in the PR.
