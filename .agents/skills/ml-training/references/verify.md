# Check a run against PyTorch

A training run is right when a peer that you trust follows it step by step. The recipe the ml repository uses
(`https://github.com/agentica-lang/ml/tree/main/benchmarks`): same start, same data, same optimizer, then compare the loss of every step.

## The rules of a fair comparison

- **Same weights at step 0.** Write the initial parameters to a safetensors file from Agentica and load that file in
  PyTorch. Do not rely on two seeded initializers agreeing.
- **Same batches in the same order.** Use `shuffle: false`, or make the batches yourself with `fit_batches` and give
  the peer the same rows.
- **Same optimizer constants.** `Optim.Adam` and `AdamW` take `lr`, `beta1`, `beta2`, `eps` (defaults 0.9, 0.999,
  1e-8) and `weight_decay` as PyTorch does. AdamW decays a weight by `lr * weight_decay` first, then steps: PyTorch's
  `AdamW`. Match the groups: no decay on biases and layer norms is two parameter groups in PyTorch. Schedules are those
  of Hugging Face's `Trainer`; there is no gradient clipping and no dropout in a step here.
- **Same precision.** Train in `F32`. Turn Vulkan fp16 off (`ml-inference/references/backends.md`, Settings). On the CPU, ggml and PyTorch add in
  different orders, so expect float32 rounding, not equality.
- **One step is one epoch of one batch** when you want a loss per step: `fit_batches(..., count: 1, epochs: steps)`
  gives a `history` entry for each step.

## The Agentica side

This program writes the starting weights, trains for 20 steps and prints one loss per line. It writes `init.safetensors`
and `losses.txt` into the current folder:

```agentica
import Data from "pkg/train/data"
import Loss from "pkg/train/loss"
import Nn from "pkg/nn/nn"
import Optim from "pkg/train/optim"
import SafetensorsWrite from "pkg/hub/safetensors_write"
import Tensor from "pkg/tensor/tensor"
import Train from "pkg/train/train"

def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  hidden = try Nn.dense(Nn.Dense(w: ps[0], b: some(ps[1])), b.inputs)
  Loss.cross_entropy(try Nn.dense(Nn.Dense(w: ps[2], b: some(ps[3])), try Tensor.relu(hidden)), b.targets)
end

def main() -> Void
  backend = Tensor.Backend.Reference
  up = Nn.init_dense(4, 8, 1, backend).unwrap!
  down = Nn.init_dense(8, 3, 2, backend).unwrap!
  params = [up.w, up.b.unwrap!, down.w, down.b.unwrap!]
  SafetensorsWrite.write(
    "init.safetensors",
    {"w1": params[0], "b1": params[1], "w2": params[2], "b2": params[3]},
  ).unwrap!

  xs = [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.1, 0.0, 0.0, 0.3, 0.9, 0.2]
  inputs = Tensor.floats(xs, [4, 4], backend).unwrap!
  labels = Tensor.from_list(
    [Tensor.Values.Int(0), Tensor.Values.Int(1), Tensor.Values.Int(2), Tensor.Values.Int(0)],
    [4],
    Tensor.Dtype.I32,
    backend,
  ).unwrap!

  loader = Data.Loader(dataset: Data.dataset(inputs, labels).unwrap!, batch_size: 4, shuffle: false)
  adam = Optim.Optimizer.Adam(Optim.Adam(lr: 0.01))
  fit = Train.fit(&loss, params, loader, adam, 20).unwrap!
  File.write_text("losses.txt", fit.history.map { |l: Float| "#{l}" }.join("\n")).unwrap!
  puts(fit.history.size)
end
```

Note that `Nn.dense` expects a `[in, out]` weight unless `out_in: true` is set: `Nn.init_dense` makes `[in, out]`. A
Hugging Face weight is `[out, in]`: load it with `Load.dense`, which sets `out_in`.

## The PyTorch peer

Pin the peer, as `run.sh` in `https://github.com/agentica-lang/ml/tree/main/benchmarks/stsb` does: `torch==2.6.0` (the CPU build),
`safetensors==0.6.2`, and for the Hugging Face and PEFT checks `transformers==4.57.6` and `peft==0.21.2`. The
tolerances below were measured with these versions on one machine: another version, or another CPU that adds in
another order, can move them.

```python
import torch, torch.nn.functional as F
from safetensors.torch import load_file

p = load_file("init.safetensors")
w1, b1, w2, b2 = [p[k].clone().requires_grad_() for k in ("w1", "b1", "w2", "b2")]
x = torch.tensor([[.1, .2, .3, .4], [.9, .8, .7, .6], [.5, .5, .1, 0.], [0., .3, .9, .2]])
y = torch.tensor([0, 1, 2, 0])
opt = torch.optim.Adam([w1, b1, w2, b2], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
ours = [float(v) for v in open("losses.txt").read().split()]
for step, want in enumerate(ours):
    loss = F.cross_entropy(F.relu(x @ w1 + b1) @ w2 + b2, y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(step, loss.item(), want, abs(loss.item() - want))
```

Read the gaps, not the losses. The repository's guards: an MLP step against PyTorch in float64 within 8e-7 with SGD and
7.9e-4 with Adam (which amplifies float32 rounding), and a DistilBERT fine-tune within 2.4e-7 per step. Compare the
movement of the loss (the loss at step `i` minus the loss at step 0) when the learning rate is small: a loss within 1e-3
passes for an optimizer that does nothing.

## Find the first difference

When two backends disagree on one function, `Trace.diff` runs the traced graph on both and names the first node whose
output is outside the tolerance while its inputs are within:

```agentica
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

def main() -> Void
  x = Tensor.floats([0.5, 1.0, 1.5, 2.0], [2, 2]).unwrap!
  w = Tensor.floats([1.0, -1.0, 0.5, 0.25], [2, 2]).unwrap!
  ggml = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  d = Trace.diff([x, w], Tensor.Backend.Reference, ggml, rtol: 0.001) do |xs|
    Tensor.tanh(try Tensor.matmul(xs[0], xs[1]))
  end

  puts(d.unwrap!.nodes)
end
```

`d.first` is a `NodeDiff` (`op`, `shape`, `max_abs`, `max_rel`) or `none`; `d.worst` lists the five nodes with the
largest gap. Run it on the loss of a layer, then of two layers, to find the layer that drifts.

## Check an adapter against PEFT

PEFT reads the files `Lora.save` writes, and `Lora.open` reads the files PEFT writes. Check both ways with the same
inputs, in `F32`, with `fp16` off:

```python
from transformers import AutoModel
from peft import PeftModel

base = AutoModel.from_pretrained("msmarco-distilbert-cos-v5")
model = PeftModel.from_pretrained(base, "adapter")   # the folder Lora.save wrote
```

- Compare the sentence embeddings of the same sentences from PEFT's model and from `DistilBert.embed` on the model with
  `Lora.merged` weights.
- For the other way, save an adapter with `model.save_pretrained("adapter")` in PEFT, then `Lora.open("adapter",
  backend)` and `Lora.merged`. An adapter that this package cannot compute (DoRA, rsLoRA, a regular-expression
  `target_modules`, `modules_to_save`, an embedding adapter) is an error that names the feature, never a silent change.
- A full run of this check is `run.sh ours-lora` and `peer-lora` in `https://github.com/agentica-lang/ml/tree/main/benchmarks/stsb`:
  both sides start from one adapter folder, and the 63 steps agree to 1.4e-7.

Page: `https://www.agentica-lang.com/ml/train/`.
