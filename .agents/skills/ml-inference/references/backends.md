# Backends

A tensor carries its backend. An op over tensors of two backends is an error.

| Backend | Runs | Needs |
|---|---|---|
| `Backend.Reference` | Agentica code; the oracle the others are checked against; slow | nothing |
| `Backend.Ggml(Device.Cpu)` | ggml on the CPU, one compiled graph per traced function | the ggml module, built once |
| `Backend.Ggml(Device.Vulkan(0))` | ggml on the first Vulkan device that is not a CPU (an iGPU counts) | ggml built with Vulkan |
| `Backend.Torch(Device.Cpu)` | libtorch, one op at a time | libtorch 2.4.1 |
| `Backend.Xla(Device.Cpu)` | XLA on the CPU, one compiled function at a time | the XLA plugin, installed once |

Which to use:

- A model (Qwen3, BERT, DistilBERT, Whisper): ggml, on Vulkan when the machine has a GPU, else on the CPU. These
  models are tested on `Reference` and ggml; libtorch is not tested for them.
- A test or a check against a known answer: `Reference`.
- Training an MLP-sized model on the CPU: XLA (`ml-training`). XLA runs only what the StableHLO emitter covers (the ops
  of an MLP and its gradients), so it does not run the model packages.

`Tensor.available(backend)` says whether a backend works on this machine. A native module that is not built prints
`warning: native module ggml not built` when the program starts, and its ops answer `error("backend not available")`.
The same goes for `warning: native module torch not built` when libtorch is absent: it comes with about 20 lines of
compiler errors from `deps/tensor/native/torch.c` (and `ggml.c` for the ggml warning). They are expected and harmless
when you do not use that backend.

## Build ggml (Linux x86_64)

The ggml sources are a submodule of the ml repository, not part of the installed package. Clone the tag once, build
ggml with the script, and `eval` its exports in the shell that runs your app (the build cache of the native module
depends on them, so a shell without them builds nothing):

```sh
git clone --depth 1 --branch tensor-0.7.0 https://github.com/agentica-lang/ml ~/src/ml
git -C ~/src/ml submodule update --init --depth 1 third_party/ggml
eval "$(sh ~/src/ml/tensor/tools/build-engine.sh | grep '^export ')"                # CPU, about a minute
eval "$(JOBS=2 sh ~/src/ml/tensor/tools/build-engine.sh vulkan | grep '^export ')"  # CPU and Vulkan, long the first time
agentica app.agc
```

The Vulkan build needs the Vulkan headers and loader, `glslc` and SPIRV-Headers (Debian: `libvulkan-dev glslc
spirv-headers`); the script lists what is missing. The result is cached in `~/.cache/agentica-ml`, so a second run builds
nothing. A CPU build is enough to run on the CPU; the Vulkan build serves both.

## libtorch

```sh
export TORCH_DIR="$HOME/opt/libtorch"        # libtorch 2.4.1+cpu, shared library release; or Debian's libtorch-dev
eval "$(sh ~/src/ml/tensor/tools/build-torch.sh)"
```

If you built ggml too, run the torch line after it: both modules then build with `g++`.

## XLA

```sh
agentica deps/tensor/tools/install_xla.agc    # Linux x86_64 (AVX) or aarch64
```

It downloads the pinned plugin from our GitHub release of agentica-lang/ml, checks its sha256 and keeps it in
`~/.cache/agentica-ml/xla/` (`AGENTICA_ML_CACHE` moves the cache). Nothing of XLA is linked.

## Pick a backend

```agentica
import Tensor from "pkg/tensor/tensor"

def pick() -> Tensor.Backend
  candidates = [Tensor.Backend.Ggml(Tensor.Device.Vulkan(0)), Tensor.Backend.Ggml(Tensor.Device.Cpu)]

  for backend in candidates
    return backend if Tensor.available(backend)
  end

  Tensor.Backend.Reference
end

def main() -> Void
  puts(Tensor.available(pick()))
end
```

```output
true
```

## Run ops on ggml and XLA

These backends compute a traced function, not single ops. `Trace.eval` traces your function, compiles it as one graph
and runs it. The function gets traced tensors and answers a `Result`:

```agentica
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

def main() -> Void
  backend = Tensor.Backend.Reference
  x = Tensor.floats([1.0, -2.0, 3.0, 4.0], [2, 2], backend).unwrap!
  w = Tensor.floats([0.5, 1.0, 1.5, 0.0], [2, 2], backend).unwrap!
  y = Trace.eval([x, w]) do |ps| Tensor.relu(try Tensor.matmul(ps[0], ps[1])) end

  puts(Tensor.to_list(y.unwrap!).unwrap!)
end
```

```output
[Values.Float(0.0), Values.Float(1.0), Values.Float(7.5), Values.Float(3.0)]
```

Change `backend` to a ggml backend and the same code runs there. `Trace.eval` traces and plans on every call. For a
function you run many times on tensors of the same shapes, compile once with `Trace.compile` and call `Trace.run`:

```agentica
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

def main() -> Void
  backend = Tensor.Backend.Reference
  x = Tensor.floats([1.0, -2.0, 3.0, 4.0], [2, 2], backend).unwrap!
  w = Tensor.floats([0.5, 1.0, 1.5, 0.0], [2, 2], backend).unwrap!
  compiled = Trace.compile([Trace.spec_of(x), Trace.spec_of(w)]) do |ps|
    ok([try Tensor.relu(try Tensor.matmul(ps[0], ps[1]))])
  end

  for _round in 0...2
    out = Trace.run(compiled.unwrap!, [x, w]).unwrap!
    puts(Tensor.to_list(out[0]).unwrap!)
  end

  Trace.close(compiled.unwrap!)
end
```

```output
[Values.Float(0.0), Values.Float(1.0), Values.Float(7.5), Values.Float(3.0)]
[Values.Float(0.0), Values.Float(1.0), Values.Float(7.5), Values.Float(3.0)]
```

Whatever changes between runs must be an input of the compiled function. A tensor that the function brings in with
`Trace.constant` is fixed. `Trace.run` refuses a different shape: make one `Compiled` for each shape.

## Settings

- `Tensor.vulkan_fp16(false)` makes Vulkan products exact `F32`, for the whole process. Call it before the first ggml
  call; a later call is an error. The default (fp16) is about 1.7 times faster and can change greedy ids.
- `TENSOR_THREADS=N` sets the ggml CPU threads (default: one per physical core, at most 512).
- `TENSOR_LOAD=agentica` reads weights through Agentica instead of C: slower, for small models only.
- `Tensor.to_device(t, Device.Vulkan(0))` moves a ggml tensor between ggml devices.

Page: `https://www.agentica-lang.com/ml/tensor/`.
