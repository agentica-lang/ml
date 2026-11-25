# Speed: backend, batch size, compile, export

The numbers are from `benchmarks/README.md` of the ml repository (one machine: Ryzen 5 3550H, 4 cores and 8 threads, a
Vega 8 iGPU, Agentica 0.37 to 0.38). They tell what costs what; measure your own model.

## Pick a backend

| Training | Backend | Why |
|---|---|---|
| a check against a known answer, a test | `Reference` | exact, slow |
| an MLP-sized model on the CPU | `Backend.Xla(Device.Cpu)` | the CPU training path: 0.75 to 0.89 of PyTorch eager's time at batch 256 and up |
| a model with a GPU (an iGPU counts) | `Backend.Ggml(Device.Vulkan(0))` | from batch 256 (Adam) or 1024 (SGD) it beats PyTorch eager on this machine: 0.45 to 0.90 of its time |
| DistilBERT, Qwen3 or another model of `transformers` | `Backend.Ggml(Device.Cpu)` or Vulkan | XLA runs only the ops of an MLP and its gradients (no layer norm, attention or embedding yet) |

One Adam step of a 784-512-512-10 relu MLP with softmax cross-entropy, in ms (`Train.fit`, batch on the device):

| Batch | ggml CPU | ggml Vulkan | XLA CPU | PyTorch eager CPU |
|---|---|---|---|---|
| 64 | 9.1 | 9.0 | 4.1 | 5.5 |
| 256 | 14.1 | 10.0 | 9.0 | 11.1 |
| 1024 | 38.0 | 22.8 | 28.5 | 34.2 |
| 4096 | 192.5 | 76.6 | 128.4 | 155.6 |

Fine-tuning DistilBERT (66 M weights) on 1000 STS-B pairs, one epoch of 63 steps at batch 16 pairs, ggml CPU, against
PyTorch with the same rules: full fine-tune 2035 ms per step against 1363 (1.49 times; 2.7 GB of memory against
1.9 GB); LoRA (r 8 on `q_lin` and `v_lin`) 1022 ms against PEFT's 758 (1.35 times; 0.96 GB against 0.94 GB). The LoRA
step is 2 times shorter and 2.8 times smaller in memory than the full one, and reaches nearly the same Spearman
(0.8326 against 0.8340). The loss agrees with PyTorch's to 1.4e-7. LoRA on XLA and the activation-space adapter that
would close the gap are ml#236.

## What costs what

- **Batch size.** A step has a fixed cost: a few ms of dispatches on a GPU (about 2 ms on the iGPU). A batch of 64 is
  slower than PyTorch on ggml (1.4 to 2.5 times); raise it to 256 or more on Vulkan, to 1024 on the ggml CPU.
- **The first step of each shape compiles.** It takes 0.3 to 0.5 s on ggml and 0.24 to 0.56 s on XLA for an MLP, and
  about 0.45 s more than a warm step for DistilBERT. `fit` compiles once per batch shape, so a last partial batch is a
  second compile (use `drop_last: true`), and padded text batches should use few lengths. ggml keeps 32 graphs.
- **`fit` over a compiled step.** On ggml and XLA the batch is gathered inside the step, so `fit` adds 0.2 to 1.4 ms
  at batch 64 over a hand-compiled step, and 5 to 12 ms at batch 4096. Hand-compile only for a step `fit` cannot
  express.
- **Precision.** Vulkan multiplies in fp16 by default, which changes gradients by up to 10% at MLP sizes. Call
  `Tensor.vulkan_fp16(false)` first when you compare against PyTorch (it is slower).
- **Threads.** ggml CPU uses one thread per physical core. `TENSOR_THREADS=N` changes it; 8 against 4 made no
  difference here.
- **Prefetch.** `Loader.prefetch` hides the cost of building a batch on libtorch and the reference. On ggml and XLA the
  gather is inside the step and no prefetch task runs.
- **Memory.** LoRA keeps gradients and moments only for `A` and `B`, so it needs a third of the memory of a full
  fine-tune. Train in `F32`: half and quantized weights have no gradient.

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
    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }
}
```

The emitter covers the ops of an MLP and its training step (`add`, `sub`, `mul`, `div`, `exp`, `log`, `tanh`, `relu`,
`gelu_tanh_approx`, `matmul`, `transpose`, `reshape`, `broadcast_to`, `sum`, `mean`, `max`, `where`, `slice`, `full`
and a few more); another op is an error that names it. The repository checks the text against snapshots and against
IREE's outputs.

Pages: `https://www.agentica-lang.com/ml/train/` and `https://www.agentica-lang.com/ml/tensor/`.
