# ml

Machine learning for Agentica: tensors, training and inference over a
reference backend and libtorch. The API is functional, in the style of JAX and
Nx: values are immutable, and a training step returns new parameters.

Status: in progress. `tensor/` (the reference and libtorch backends, tracing
and gradients), `nn/` (layers, forward only), `train/` (losses, SGD and Adam,
a training loop), `hub/` (safetensors checkpoints), `asr/` (Whisper's model
forward) and `whisper/` (a binding over whisper.cpp) are in. The tensor stack
is designed in [`tensor/DESIGN.md`](tensor/DESIGN.md), with the order of work
in [`roadmap.md`](roadmap.md) and the open decisions in
[issue #1](https://github.com/agentica-lang/ml/issues/1).

## Packages

One repository, one folder per package, each with its own `agentica.toml`, so
an app installs only what it needs (for example inference without training).
There are no releases yet: a project pins a member by a commit of `main`
(`ref`); later versions are prefixed tags, such as `tensor-0.1.0`.

| Folder | What | Needs |
|---|---|---|
| `tensor/` | the tensor API, the backend interface, a slow reference backend in Agentica, and the libtorch (eager) backend behind a thin C++ layer | nothing native for the reference backend; libtorch 2.4.1+cpu, **optional**, for `Backend.Torch` |
| `nn/` | functional layers, forward only: dense, layer norm, embeddings, convolution, attention, MLP, transformer encoder and decoder blocks, seeded initializers | `tensor` |
| `train/` | losses (`mse`, binary and softmax cross-entropy), optimizers (SGD, Adam), a data loader (seeded shuffle, background prefetch), the training loop over `tensor/`'s gradients | `tensor`, `hub` (and `nn` through it) |
| `hub/` | `safetensors` checkpoints (one file or sharded with an index) read in Agentica, and `nn` layers loaded from them by name | `tensor`, `nn` |
| `asr/` | Whisper in Agentica: the model forward loaded from a Hugging Face checkpoint, its log-mel frontend, its tokenizer and `transcribe` (greedy decoding with a KV cache, language detection, timestamps, translation) for up to 30 s of audio; on the reference backend and libtorch | `tensor`, `nn`, `hub`; a C compiler for `mathx`, a small libm module, temporary until Agentica's `Math` has `exp`, `log`, `log10`, `sin` and `cos` (agentica#1925) |
| `whisper/` | a binding over the pinned whisper.cpp engine, with WAV reading, microphone capture and long-audio chunking | whisper.cpp (a submodule), a C/C++ compiler |

An inference app depends on `nn` and one backend; a training app adds `train`.
Inside this repository `tensor`, `nn`, `train`, `hub` and `asr` are members of
one Agentica workspace (the root `agentica.toml`), so they read each other from
their folders; `whisper` is a project of its own, not a member. Each package
pins its own Agentica version (`AGENTICA_VERSION`), and CI runs `check`,
`fmt --check` and `test` on each folder that has an `agentica.toml`, on that
version.

Outside the workspace a member is fetched at the `ref` it writes for the
others.

```toml
[dependencies]
nn = { git = "https://github.com/agentica-lang/ml", ref = "COMMIT", subdir = "nn" }
tensor = { git = "https://github.com/agentica-lang/ml", ref = "COMMIT", subdir = "tensor" }
```

The libtorch backend ships **inside `tensor/`**, not as a `torch/` package,
because a native handle is a `resource` that is nominal to its `extern module`
([`tensor/DESIGN.md`](tensor/DESIGN.md) §1, §7;
[agentica-lang/agentica#1918](https://github.com/agentica-lang/agentica/issues/1918)
moves it out). libtorch is `optional = true` and never vendored or downloaded
by this repository: without it, everything still checks, formats and tests,
and `Backend.Torch` answers `error("backend not available")`.
[`tensor/README.md`](tensor/README.md) has the setup, the coverage table, the
tolerances and the measured cost of the dependency.

## Checking the native C

CI runs [cchecker](https://github.com/agentica-lang/cchecker) over the native C of `tensor/` (`ggml.c`, `ggml_graph.c`), `whisper/` and `asr/`: locks released on every path and taken in order (`tg_lock_device`, `cache_lock`, the model mutex), ownership, `gcc -fanalyzer` and clang-tidy. It fails on a finding that is not in `cchecker-baseline.json`. `torch.c` is left out: it is C++, and cchecker reads C.

```sh
git submodule update --init third_party/ggml third_party/whisper.cpp
make cchecker CCHECKER_FLAGS="--gcc gcc-14 --clang clang-19 --clang-tidy clang-tidy-19 --strict-baseline"   # what CI runs
```

It needs gcc 10+, clang, clang-tidy and the pinned `agentica` on PATH. No engine is built: the database (`build/cchecker/`, made by `tools/cchecker.sh`) only needs the headers of the submodules and of `agentica`. The version of cchecker is pinned in `tools/cchecker.sh`; the settings are `cchecker.toml`.

- Mark what cchecker cannot see with the macros of `tensor/native/cchecker.h` (a copy of cchecker's `include/cchecker.h`, empty in a normal build), for example `CC_ACQUIRES` on a function that returns with a lock held.
- A false positive of one line gets `// cchecker: ignore CODE: reason` next to it.
- To accept a finding, run `make cchecker-update-baseline`, review the diff of `cchecker-baseline.json`, and write the reason in `cchecker.toml` and in the PR.
