# Speed: backend, batch size, compile, export

These are rules of thumb from the benchmarks of the ml repository
(`https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md`: the machine, the methods and the tables). They
tell what costs what; measure your own model.

## Pick a backend

| Training | Backend | Why |
|---|---|---|
| a check against a known answer, a test | `Reference` | exact, slow |
| an MLP-sized model on the CPU | `Backend.Xla(Device.Cpu)` | the CPU training path; on one machine its step took about 0.75 to 0.89 of PyTorch eager's time from batch 256 (those rows ran later, on a quieter machine than the others, so the gap is not confirmed) |
| a model with a GPU (an iGPU counts) | `Backend.Ggml(Device.Vulkan(0))` | beats PyTorch eager from about batch 256 (Adam) or 1024 (SGD) on the same machine |
| DistilBERT, Qwen3 or another model of `transformers` | `Backend.Ggml(Device.Cpu)` or Vulkan | XLA runs only the ops of an MLP and its gradients (no layer norm, attention or embedding yet) |

To train on XLA, install the plugin once (`ml-inference/references/backends.md`) and set `backend` to
`Tensor.Backend.Xla(Tensor.Device.Cpu)` when you make the dataset and the parameters.

Fine-tuning DistilBERT (66 M weights) on ggml CPU takes about 1.5 times PyTorch's time per step, and a LoRA run about
1.35 times PEFT's, with the same loss to 1e-7. A LoRA run needs about a third of the memory of a full fine-tune and each
step takes half the time. LoRA on XLA, and applying the adapter on the activations, are
[ml#236](https://github.com/agentica-lang/ml/issues/236).

## What costs what

- **Batch size.** A step has a fixed cost: a few ms of dispatches on a GPU. A batch of 64 is slower than PyTorch on
  ggml; raise it to 256 or more on Vulkan, to 1024 on the ggml CPU.
- **The first step of each shape compiles.** It takes a fraction of a second for an MLP. `fit` compiles once per batch
  shape, so a last partial batch is a second compile (use `drop_last: true`), and padded text batches should use few
  lengths. ggml keeps 32 graphs.
- **`fit` over a step you compile yourself.** On ggml and XLA the batch is gathered inside the step, so `fit` adds
  little over a compiled step. Compile your own only for a step `fit` cannot express.
- **Precision.** Vulkan fp16 changes gradients by up to 10% at MLP sizes: turn it off when you compare against PyTorch
  (`ml-inference/references/backends.md`, Settings).
- **Threads.** ggml CPU uses one thread per physical core (`TENSOR_THREADS=N` changes it; 8 against 4 made no
  difference here).
- **Prefetch.** `Loader.prefetch` hides the cost of building a batch on libtorch and the reference. On ggml and XLA the
  gather is inside the step and no prefetch task runs. A batch function that decodes images or tokenizes takes
  `fit_batches(..., prefetch: 2)`, on every backend.
- **Memory.** LoRA keeps gradients and moments only for `A` and `B`. Train in `F32`.

## Compile a step yourself

`Trace.compile(specs, f)` traces `f` once for inputs of the given shapes and `Trace.run` calls it again and again with
no tracing. `f` may compute `Trace.gradients` and the update, so a whole training step is one compiled function that
answers the loss and the new weights. Whatever changes between runs must be an input:

```agentica
import Loss from "pkg/train/loss"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

def step(ps: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  rows = Tensor.shape(ps[1])[0]
  loss = try Loss.mse(try Tensor.reshape(try Tensor.matmul(ps[1], ps[0]), [rows]), ps[2])
  grads = try Trace.gradients(loss, [ps[0]])
  rate = try Tensor.full(Tensor.shape(ps[0]), Tensor.Values.Float(0.05), Tensor.Dtype.F32, Tensor.backend(ps[0]))
  ok([loss, try Tensor.sub(ps[0], try Tensor.mul(grads[0], rate))])
end

def main() -> Void
  x = Tensor.floats([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0], [4, 2]).unwrap!
  y = Tensor.floats([1.0, 2.0, 3.0, 4.0], [4]).unwrap!
  w = Tensor.zeros([2, 1]).unwrap!
  specs = [Trace.spec_of(w), Trace.spec_of(x), Trace.spec_of(y)]
  compiled = Trace.compile(specs, ->(ps: List[Tensor.Tensor]) { step(ps) }).unwrap!
  first = 0.0
  last = 0.0
  for i in 0...20
    out = Trace.run(compiled, [w, x, y]).unwrap!
    case Tensor.to_list(out[0]).unwrap![0]
      Tensor.Values.Float(f) -> last = f
      _ -> nil
    end

    if i == 0
      first = last
    end

    w = out[1]
  end

  puts(last < first)
  Trace.close(compiled)
end
```

```output
true
```

On ggml and XLA a compiled step updates `adam_step`'s moments in place: the tensors you pass as moments are consumed, so
read the new ones that `Trace.run` returns. `Trace.run(compiled, args, into)` writes outputs into tensors that exist, so
a loop allocates nothing.

## Export to StableHLO

`Trace.stablehlo(specs, f)` gives the traced function as a StableHLO program that XLA (through PJRT) and IREE read: one
`@main` that takes the inputs, then the constants, with static shapes. `Train.stablehlo(loss, params, batch, optimizer)`
gives the step of `fit` as one program: the arguments are the parameters, the moments, the optimizer's numbers
(`Optim.step_inputs(state)`) and the batch; the results are the loss, the new parameters and the new moments.

```agentica
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

def main() -> Void
  x = Tensor.floats([1.0, 2.0], [2]).unwrap!
  y = Tensor.floats([3.0, 4.0], [2]).unwrap!
  specs = [Trace.spec_of(x), Trace.spec_of(y)]
  module = Trace.stablehlo(specs, ->(xs: List[Tensor.Tensor]) { ok([try Tensor.add(xs[0], xs[1])]) }).unwrap!
  puts(module.text.strip)
end
```

```output
module {
  func.func public @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tensor<2xf32>) {
    %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> (tensor<2xf32>)
    func.return %0 : tensor<2xf32>
  }
}
```

The emitter covers the ops of an MLP and its training step (`add`, `sub`, `mul`, `div`, `exp`, `log`, `tanh`, `relu`,
`gelu_tanh_approx`, `matmul`, `transpose`, `reshape`, `broadcast_to`, `sum`, `mean`, `max`, `where`, `slice`, `full`
and a few more); another op is an error that names it. The repository checks the text against snapshots and against
IREE's outputs.

Pages: `https://www.agentica-lang.com/ml/train/` and `https://www.agentica-lang.com/ml/tensor/`.
