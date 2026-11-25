# ml

Machine learning for Agentica: tensors, training and inference over a
reference backend, libtorch and ggml (CPU and Vulkan). The API is functional, in
the style of JAX and Nx: values are immutable, and a training step returns new
parameters.

Status: in progress. The API is not stable yet.

Works today: `tensor/` (the reference, libtorch and ggml backends, tracing and
gradients), `nn/` (layers, forward only), `train/` (losses, SGD and Adam, a
training loop, a data loader), `hub/` (safetensors and GGUF checkpoints),
`asr/` (Whisper on the reference, libtorch and ggml backends, with Hugging
Face's token ids on the test clips), `transformers/` (BERT, Qwen3 and Qwen2.5)
and `whisper/` (a binding over whisper.cpp).

Works, but slow: `asr/` on ggml. Its tokens match, but on the test clips it
runs about 2 to 3 times slower than whisper.cpp on CPU and about 5 times
slower on Vulkan. The numbers and the machine are in
[`benchmarks/README.md`](benchmarks/README.md).

Not yet: what the [roadmap](roadmap.md) lists, from LoRA and inference
benchmarks to CUDA.

The tensor stack is designed in [`tensor/DESIGN.md`](tensor/DESIGN.md), with the
order of work in [`roadmap.md`](roadmap.md) and the original design in
[issue #1](https://github.com/agentica-lang/ml/issues/1).

## Packages

One repository, one folder per package, each with its own `agentica.toml`, so
an app installs only what it needs (for example inference without training).
Each package is released on its own tag, `<package>-X.Y.Z`; the first release
is 0.2.0 (see [`CHANGELOG.md`](CHANGELOG.md)). A project pins a member by its
tag. The API is not stable yet, so expect changes between versions.

| Folder | What | Needs |
|---|---|---|
| `tensor/` | the tensor API, the backend interface, a slow reference backend in Agentica, the libtorch (eager) backend behind a thin C++ layer, and the ggml backend (CPU and Vulkan), which runs a traced function as one graph | nothing native for the reference backend; libtorch 2.4.1+cpu, **optional**, for `Backend.Torch`; ggml (the `third_party/ggml` submodule, built by `tensor/tools/build-engine.sh`), **optional**, for `Backend.Ggml` |
| `nn/` | functional layers, forward only: dense, layer norm, embeddings, convolution, attention (grouped-query, rotary), MLP, transformer encoder and decoder blocks and stacks, seeded initializers | `tensor` |
| `train/` | losses (`mse`, binary and softmax cross-entropy), optimizers (SGD, Adam), a data loader (seeded shuffle, background prefetch), the training loop over `tensor/`'s gradients, training part of a model and LoRA adapters | `tensor`, `hub`, `nn` |
| `hub/` | `safetensors` checkpoints (one file or sharded with an index) read in Agentica, and `nn` layers loaded from them by name | `tensor`, `nn` |
| `asr/` | Whisper in Agentica: the model forward loaded from a Hugging Face checkpoint, its log-mel frontend, its tokenizer and `transcribe` (greedy decoding with a KV cache, language detection, timestamps, translation) for audio of any length, in windows of 30 s that seek; on the reference backend, libtorch and ggml (CPU and Vulkan) | `tensor`, `nn`, `hub` |
| `transformers/` | BERT and Qwen3 (the forward and greedy generation with a KV cache) from Hugging Face checkpoints, and the WordPiece and byte-level BPE tokenizers, which give Hugging Face's ids | `tensor`, `nn`, `hub`, `train` (for the fine-tune tool of `tools/`) |
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
- [whisper](https://www.agentica-lang.com/ml/whisper/)

Inside this repository `tensor`, `nn`, `train`, `hub`, `asr` and `transformers` are members of
one Agentica workspace (the root `agentica.toml`), so they read each other from
their folders; `whisper` is a project of its own, not a member. Each package
pins its own Agentica version (`AGENTICA_VERSION`), and CI runs `check`,
`fmt --check` and `test` on each folder that has an `agentica.toml`, on that
version.

Outside the workspace a member is fetched at the `ref` it writes for the
others.

```toml
[dependencies]
nn = { git = "https://github.com/agentica-lang/ml", tag = "nn-0.2.0", subdir = "nn" }
tensor = { git = "https://github.com/agentica-lang/ml", tag = "tensor-0.2.0", subdir = "tensor" }
```

The libtorch backend ships **inside `tensor/`**, not as a `torch/` package,
because a native handle is a `resource` that is nominal to its `extern module`
([`tensor/DESIGN.md`](tensor/DESIGN.md) §1, §7). libtorch is `optional = true` and never vendored or downloaded
by this repository: without it, everything still checks, formats and tests,
and `Backend.Torch` answers `error("backend not available")`.
[`tensor/README.md`](tensor/README.md) has the setup, the coverage table, the
tolerances and the measured cost of the dependency.

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
