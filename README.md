# ml

Machine learning for Agentica: tensors, training and inference over adaptable
backends (libtorch, XLA, candle). The API is functional, in the style of JAX and
Nx: values are immutable, and a training step returns new parameters.

Status: in progress. `whisper/` is the first package; `tensor/` (reference
and libtorch backends, tracing and gradients), the first layers of `nn/`
([`nn/DESIGN.md`](nn/DESIGN.md)), the first of `train/`
([`train/DESIGN.md`](train/DESIGN.md)) and the safetensors reader of `hub/`
([`hub/DESIGN.md`](hub/DESIGN.md)) are in. The tensor stack is
designed in [`tensor/DESIGN.md`](tensor/DESIGN.md), with execution order in
[`roadmap.md`](roadmap.md) and the open decisions in
[issue #1](https://github.com/agentica-lang/ml/issues/1).

## Packages

One repository, one folder per package, each with its own `agentica.toml`, so
an app installs only what it needs (for example inference without training).
Versions are prefixed tags, such as `tensor-0.1.0`.

| Folder | What | Needs |
|---|---|---|
| `tensor/` | the tensor API, the backend interface, a slow reference backend in Agentica, and the libtorch (eager) backend behind a thin C++ layer | nothing native for the reference backend; libtorch 2.4.1+cpu, **optional**, for `Backend.Torch` |
| `xla/` | XLA backend (JIT, like JAX), through the PJRT C API | `tensor`, a PJRT plugin |
| `candle/` | candle backend (eager), through a thin C layer over Rust | `tensor`, candle |
| `nn/` | functional layers, forward only: dense, layer norm, embeddings, attention, MLP, a transformer encoder block, seeded initializers | `tensor` |
| `train/` | losses (`mse`, binary and softmax cross-entropy), optimizers (SGD, Adam), a data loader (seeded shuffle, background prefetch), the training loop over `tensor/`'s gradients | `tensor`, `hub` (and `nn` through it) |
| `hub/` | `safetensors` checkpoints (one file or sharded with an index) read in Agentica, and `nn` layers loaded from them by name; Hugging Face downloads later | `tensor`, `nn` |
| `asr/` | Whisper's model forward (conv stem, encoder, decoder, tied logits) loaded from a Hugging Face checkpoint; inference only, no frontend or decoding loop yet | `tensor`, `nn`, `hub` |

An inference app depends on `nn` and one backend; a training app adds `train`.
Inside this repository the packages are members of one Agentica workspace (the
root `agentica.toml`), so they read each other from their folders.

```toml
[dependencies]
nn = { git = "https://github.com/agentica-lang/ml", tag = "nn-0.1.0", subdir = "nn" }
tensor = { git = "https://github.com/agentica-lang/ml", tag = "tensor-0.1.0", subdir = "tensor" }
```

The libtorch backend ships **inside `tensor/`**, not as a `torch/` package: a
native handle is a `resource`, nominal to the `extern module` block that
declares it, so the arm that dispatches to libtorch and the type that carries
its handle have to be declared in the same package as the `Tensor`
([`tensor/DESIGN.md`](tensor/DESIGN.md) §1, §2, §7). The module is
`optional = true`, libtorch is never vendored or downloaded by this repository,
and
[agentica-lang/agentica#1918](https://github.com/agentica-lang/agentica/issues/1918)
(opaque resources that can cross modules) is the change that moves the backend
back out into its own package.

Installing libtorch is per machine:

```sh
unzip libtorch-cxx11-abi-shared-with-deps-2.4.1+cpu.zip -d "$HOME/opt"
export TORCH_DIR="$HOME/opt/libtorch"
cd tensor && eval "$(sh tools/build-torch.sh)"
```

Without it, everything still checks, formats and tests: the run warns that the
module is not built, `Tensor.available(Tensor.Backend.Torch(Tensor.Device.Cpu))`
is false, and the `Backend.Torch` arms answer
`error("backend not available")` like the backends that are not written yet.
[`tensor/README.md`](tensor/README.md) has the coverage table, the tolerances and
the measured cost of the dependency.
