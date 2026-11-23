---
title: tensor
description: Tensors over a reference backend, libtorch on the CPU and ggml on the CPU and Vulkan, with tracing and gradients.
weight: 1
---

`tensor` is the tensor API that the other packages build on: creation, elementwise and reduction ops, matrix products, shape ops, 1-D convolution, and the ops built from them (softmax, layer norm, attention).

Values are immutable: an op returns a new `Tensor`. Every op returns a `Result`, so a shape, dtype or backend mismatch is an `error`, not a halt. There is no operator overloading and no implicit broadcasting: write `Tensor.add(a, b)`, and widen a size-1 axis with `Tensor.broadcast_to` first.

A tensor carries its backend, so tensors on different backends can live in one program. An op over tensors on two backends is an error.

## Example

The matrix product of two tensors on the reference backend, and the gradient of a sum of it. The `Trace` call runs the function once and gives the value and the gradient together.

```agentica
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

def main() -> Void
  a = Tensor.floats([1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap!
  b = Tensor.floats([5.0, 6.0, 7.0, 8.0], [2, 2]).unwrap!
  puts(Tensor.to_list(Tensor.matmul(a, b).unwrap!).unwrap!)
  both = Trace.value_and_grad([a]) do |ps|
    Tensor.sum(try Tensor.matmul(ps[0], Trace.constant(b)))
  end
  result = both.unwrap!
  puts(Tensor.to_list(result.value).unwrap!)
  puts(Tensor.to_list(result.grads[0]).unwrap!)
end
```

```output
[Values.Float(19.0), Values.Float(22.0), Values.Float(43.0), Values.Float(50.0)]
[Values.Float(134.0)]
[Values.Float(11.0), Values.Float(15.0), Values.Float(11.0), Values.Float(15.0)]
```

## Backends

| Backend | Where it computes | Needs |
|---|---|---|
| `Backend.Reference` | Agentica code over lists of floats; the oracle the other backends are checked against | nothing |
| `Backend.Torch(Device.Cpu)` | libtorch 2.4.1, one op at a time | libtorch (optional) |
| `Backend.Ggml(Device.Cpu)` | ggml on the CPU | the ggml module (optional) |
| `Backend.Ggml(Device.Vulkan(i))` | ggml on Vulkan device `i`, counted over the devices that are not a CPU | a ggml build with Vulkan |
| `Backend.Trace` | nothing: it records a graph of the ops, see below | nothing |

`Reference` and `Torch` serve `Device.Cpu` only.

`Tensor.available(backend)` says whether a backend works on this machine; without its native module the backend answers `error("backend not available")`.

`Torch` has 43 of the 45 primitives: `batch_matmul` and `to_device` answer an error that names the op.

On `Ggml`, eager calls only create, read and copy tensors; every other op is an error that points to `Trace`, because ggml is fast when a whole computation is one graph.

libtorch and ggml are never vendored: `tensor/README.md` has the setup.

## Creating and reading

- `zeros`, `ones`, `empty`, `full`, `like`, `arange` (integer dtypes only), `scalar_to_tensor`.
- `floats(values, shape, backend)` makes an `F32` tensor from a `List[Float]`.
- `from_list(values, shape, dtype, backend)` takes a `List[Values]` with one value per element of `shape`; `Values` is `Float`, `Int` or `Bool`.
- `dtype` defaults to `F32` and `backend` to `Reference` in `zeros`, `ones`, `full` and `from_list`.
- `to_list(t)` reads a tensor back as a `List[Values]`.
- `shape`, `dtype` and `backend` read the fields of a tensor, with no call to a backend.

## Ops

- Elementwise, floats only: `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `pow`, `relu`, `sigmoid`, `tanh`, `gelu_tanh_approx`, `gelu_erf`, `clamp`.
- Comparisons: `eq` and `lt` (floats or integers) give `1.0` or `0.0`; `where(cond, a, b)` selects with a `Bool` tensor.
- Reductions: `sum`, `mean`, `max`, `argmax`, each with `axes` (`none` for every axis, or a list) and `keep_dims`. `argmax` gives `I64`.
- Products: `matmul` (rank 2), `transpose` (reverses every axis), `batch_matmul` (Reference only).
- Shape: `reshape`, `squeeze`, `unsqueeze`, `slice`, `concat`, `stack`, `broadcast_to`, `gather` (rows at integer indices).
- Convolution: `conv1d` (stride and padding; no dilation or groups), `avg_pool1d`.
- Movement and dtype: `to_dtype`, `to_device`, `copy`, `contiguous`, `stop_grad`.
- Built from the above: `softmax`, `layer_norm`, `rms_norm`, `gelu` (the tanh form), `silu`, `dot`, `linear`, `embedding`, `multi_head_attention`, `greedy_step`.

## Tracing and gradients

`Trace` calls a function on `Trace` tensors, which record a graph instead of computing, and then runs the graph on a real backend.

- `Trace.eval(args, f)` gives the value of `f` on the backend of `args`.
- `Trace.grad(args, f)` gives the gradient of a scalar function for each argument.
- `Trace.value_and_grad(args, f)` gives `value` and `grads` from one evaluation.
- `Trace.eval_many(args, f)` gives several tensors from one evaluation: `f` answers a `List` of tensors, and on `Ggml` they are one compiled graph.
- `Trace.constant(t)` lifts a tensor from outside, such as data or labels, into the trace. The gradient is not taken for it.

Ops with no backward rule make the gradient an error that names the op: `pow`, `clamp`, `to_dtype`, `batch_matmul`, `conv1d`, `avg_pool1d`, and a `to_device` across devices.

`gelu_erf` is refused by `Trace`; use `gelu_tanh_approx` when you train.

On `Ggml`, `Trace.eval`, `eval_many`, `grad` and `value_and_grad` compile the whole graph, forward and backward, into one ggml graph. The graph is cached by its structure, so a retrace with new values compiles nothing. ggml lowers the ops of an MLP and of softmax cross-entropy with their gradients, plus `slice` and `gather`; `tensor/DESIGN.md` §7 has the list.

## Dtypes and fp16

- `Dtype` has `F32` (the compute dtype), `F16`, `F64`, `I8`, `I16`, `I32`, `I64`, `U8` and `Bool`.
- `Reference` computes `F16` and `F64` exactly, in doubles.
- ggml holds `F32` and `I32`. `Bool`, `I8`, `I16` and `U8` are held as `F32`, which is exact for them. `F16`, `F64` and `I64` are refused on ggml.
- `Tensor.vulkan_fp16(true)` is the fp16 opt-in: Vulkan products run in fp16 for the whole process. Call it before any other ggml call, `available` included; a later change is an error. Gradients change by up to about 10 % at MLP sizes.
- `Tensor.to_device(t, Device.Vulkan(0))` moves a ggml tensor to another ggml device.

## Limits

- Exact integers: `F32` holds integers exactly up to 2^24 (16 777 216). On ggml, an `I32` value beyond that, or an `arange` or fill beyond it, is an error, not a wrong number.
- Output cap: one ggml graph returns at most 256 outputs. `value_and_grad` returns the value and one gradient per argument, so ggml takes at most 255 arguments.
- Rank: ggml holds tensors of rank 4 at most.
- Gather: on ggml, the source of a `gather` holds at most 2^24 rows, because its integer indices are exact in `F32` only below that.
- Non-finite values: `exp`, `log`, `sqrt`, `pow`, `sigmoid`, `tanh` and `gelu_tanh_approx` refuse `NaN` and infinite inputs, and a ggml output that is not finite is an error.
- Integers: an integer overflow is an error, never a wrap-around. On libtorch, readback of `I64` past 2^53 loses precision.
- Speed: `Reference` is for checks. On one machine, a 64 x 64 matmul took about 6.6 ms there and 33 µs on libtorch (`tensor/README.md`).

## More

`tensor/README.md` has the setup, the coverage table and the tolerances. `tensor/DESIGN.md` is the contract.
