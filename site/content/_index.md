---
title: Agentica ML
---

Machine learning for [Agentica](https://github.com/agentica-lang/agentica): tensors, training and inference in a functional style, in the manner of JAX and Nx. Values are immutable, an op returns a new tensor, and a training step returns new parameters. The code is in [agentica-lang/ml](https://github.com/agentica-lang/ml), one folder per package.

Every backend only computes forward. `grad` is written in Agentica: a function is traced into a graph of tensor ops, and the gradient graph is built from it. There are three backends: `Reference`, a pure Agentica backend that is the oracle for the tests; `Torch`, libtorch on the CPU; and `Ggml`, on the CPU or Vulkan, which runs a traced function as one compiled graph. The [tensor page](tensor/_index.md) has their limits.

## Status

In progress, and no package has a release yet. `tensor`, `nn`, `train`, `hub`, `asr` and `whisper` are in; the [roadmap](../../roadmap.md) says what is done and what comes next. Each package has its own `agentica.toml`, so an app installs only what it needs: an inference app depends on `nn` and a backend, and a training app adds `train`.

## Install a package

Add the package with the repository, its folder as `subdir`, and a commit of `main` as `ref`, in the `agentica.toml` of your project:

```toml
[dependencies]
tensor = { git = "https://github.com/agentica-lang/ml", ref = "COMMIT", subdir = "tensor" }
```

A package that depends on another, such as `nn` on `tensor`, pulls it in at the `ref` it writes. Inside the repository `tensor`, `nn`, `train`, `hub` and `asr` are members of one workspace and read each other from their folders; `whisper` is a project of its own.

## A first program

The gradient of `sum(w * w)` at `w = [1, 2, 3]`, on the reference backend. The function is traced once, so the value and the gradient come from one evaluation:

```agentica
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

def main() -> Void
  values = [1.0, 2.0, 3.0].map { |x| Tensor.Values.Float(x) }
  w = Tensor.from_list(values, [3], Tensor.Dtype.F32, Tensor.Backend.Reference).unwrap!
  both = Trace.value_and_grad([w]) do |ps| Tensor.sum(try Tensor.mul(ps[0], ps[0])) end
  result = both.unwrap!
  puts(Tensor.to_list(result.value).unwrap!)
  puts(Tensor.to_list(result.grads[0]).unwrap!)
end
```

```output
[Values.Float(14.0)]
[Values.Float(2.0), Values.Float(4.0), Values.Float(6.0)]
```
