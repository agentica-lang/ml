# ml

Machine learning for Agentica: tensors, training and inference over adaptable
backends (libtorch, XLA, candle). The API is functional, in the style of JAX and
Nx: values are immutable, and a training step returns new parameters.

Status: design only. The design and its open decisions are in [issue #1](https://github.com/agentica-lang/ml/issues/1);
no package exists yet.

## Packages

One repository, one folder per package, each with its own `agentica.toml`, so
an app installs only what it needs (for example inference without training).
Versions are prefixed tags, such as `tensor-0.1.0`.

| Folder | What | Needs |
|---|---|---|
| `tensor/` | the tensor API, the backend interface, tracing to a graph, a slow reference backend in Agentica | nothing native |
| `torch/` | libtorch backend (eager), through a thin C layer | `tensor`, libtorch |
| `xla/` | XLA backend (JIT, like JAX), through the PJRT C API | `tensor`, a PJRT plugin |
| `candle/` | candle backend (eager), through a thin C layer over Rust | `tensor`, candle |
| `nn/` | layers and models, forward only: what inference needs | `tensor` |
| `train/` | gradients (computed in Agentica over the graph), losses, optimizers, the training loop | `nn` |
| `hub/` | `safetensors` weights and Hugging Face models (later) | `nn` |

An inference app depends on `nn` and one backend; a training app adds `train`.

```toml
[dependencies]
nn = { git = "https://github.com/agentica-lang/ml", tag = "nn-0.1.0", subdir = "nn" }
torch = { git = "https://github.com/agentica-lang/ml", tag = "torch-0.1.0", subdir = "torch" }
```
