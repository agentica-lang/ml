---
title: Agentica ML
---

Agentica ML is machine learning for [Agentica](https://github.com/agentica-lang/agentica): tensors, training and inference. It is functional, like JAX (a Python library) and Nx (an Elixir one): a value never changes, and a training step returns new parameters.

The code is in [agentica-lang/ml](https://github.com/agentica-lang/ml), one folder per package.

## A first program

The gradient of `sum(w * w)` at `w = [1, 2, 3]`, on the reference backend. The function is traced once, so the value and the gradient come from one evaluation:

```agentica
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

def main() -> Void
  w = Tensor.floats([1.0, 2.0, 3.0], [3]).unwrap!
  both = Trace.value_and_grad([w]) do |ps|
    Tensor.sum(try Tensor.mul(ps[0], ps[0]))
  end
  result = both.unwrap!
  puts(Tensor.to_list(result.value).unwrap!)
  puts(Tensor.to_list(result.grads[0]).unwrap!)
end
```

```output
[Values.Float(14.0)]
[Values.Float(2.0), Values.Float(4.0), Values.Float(6.0)]
```

## Backends

A gradient is built in Agentica. A function runs once on tracing tensors, which record its ops as a graph, and the gradient is built from that graph. So each backend only has to compute the forward ops.

There are three backends:

- `Reference`: pure Agentica code. It is slow. It is the answer the tests trust, and the other backends are checked against it.
- `Torch`: libtorch (the C++ library under PyTorch), on the CPU.
- `Ggml`: ggml (a small tensor library in C), on the CPU or on Vulkan (a GPU API). It runs a traced function as one compiled graph.

The [tensor page](tensor/_index.md) has their limits.

## Status

In progress. No package has a release yet. The packages in the repository are `tensor`, `nn`, `train`, `hub`, `asr` and `whisper`. The [roadmap](../../roadmap.md) says what is done and what comes next.

Each package has its own `agentica.toml`, the project file that lists dependencies. So an app installs only what it needs: an inference app takes `nn` and a backend, and a training app adds `train`.

## Install a package

Add the package to the `agentica.toml` of your project. Give the repository, its folder as `subdir`, and a commit of `main` as `ref`:

```toml
[dependencies]
tensor = { git = "https://github.com/agentica-lang/ml", ref = "COMMIT", subdir = "tensor" }
```

Replace `COMMIT` with a commit ID from `main`.

A package that depends on another, such as `nn` on `tensor`, gets it at the `ref` that the package names.

Inside the repository, `tensor`, `nn`, `train`, `hub` and `asr` are members of one workspace. They read each other from their folders. `whisper` is a project of its own.
