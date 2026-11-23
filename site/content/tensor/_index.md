---
title: tensor
description: Tensors over a reference backend, libtorch on the CPU and ggml on the CPU and Vulkan, with tracing and gradients.
weight: 1
---

`tensor` is the tensor library: arrays of numbers with a shape, and ops on them, such as `add`, `matmul` and `softmax`. Every op returns a new tensor, and no tensor changes in place.

## Example

Add a 2 by 2 tensor of numbers and a tensor of ones:

```agentica
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  a = Tensor.floats([1.0, 2.0, 3.0, 4.0], [2, 2]).unwrap!
  b = Tensor.ones([2, 2]).unwrap!
  c = Tensor.add(a, b).unwrap!
  puts(Tensor.to_list(c).unwrap!)
end
```

```output
[Values.Float(2.0), Values.Float(3.0), Values.Float(4.0), Values.Float(5.0)]
```

Every op returns a `Result`. A shape, dtype or backend mismatch is an `error`, not a halt.

There is no operator overloading and no implicit broadcasting. Write `Tensor.add(a, b)`, and widen a size-1 axis with `Tensor.broadcast_to` first.

## Create and read

A tensor has a shape, a dtype (its number type, such as `F32`, a 32-bit float) and a backend (where its values are computed).

- `zeros`, `ones`, `empty`, `full`, `like`, `arange` (integer dtypes only), `scalar_to_tensor`.
- `floats(values, shape, backend)` makes an `F32` tensor from a `List[Float]`.
- `from_list(values, shape, dtype, backend)` takes a `List[Values]` with one value per element of `shape`. `Values` is `Float`, `Int` or `Bool`.
- `dtype` defaults to `F32` and `backend` to `Reference` in `zeros`, `ones`, `full` and `from_list`.
- `to_list(t)` reads a tensor back as a `List[Values]`.
- `shape`, `dtype` and `backend` read the fields of a tensor, with no call to a backend.

## Ops

- Elementwise, one value at a time, floats only: `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `pow`, `relu`, `sigmoid`, `tanh`, `gelu_tanh_approx`, `gelu_erf`, `clamp`.
- Comparisons: `eq` and `lt` (floats or integers) give `1.0` or `0.0`. `where(cond, a, b)` selects with a `Bool` tensor.
- Reductions: `sum`, `mean`, `max`, `argmax`, each with `axes` (`none` for every axis, or a list) and `keep_dims`. `argmax` gives `I64`.
- Matrix products: `matmul` (rank 2), `transpose` (reverses every axis), `batch_matmul` (Reference only).
- Shape: `reshape`, `squeeze`, `unsqueeze`, `slice`, `concat`, `stack`, `broadcast_to`, `gather` (rows at integer indices).
- Convolution: `conv1d` (stride and padding; no dilation or groups), `avg_pool1d`.
- Movement and dtype: `to_dtype`, `to_device`, `copy`, `contiguous`, `stop_grad`.
- Built from the above: `softmax`, `layer_norm`, `rms_norm`, `gelu` (the tanh form), `silu`, `dot`, `linear`, `embedding`, `multi_head_attention`, `greedy_step`.

## Backends

`Torch` runs on libtorch, the C++ library under PyTorch. `Ggml` runs on ggml, a small tensor library in C.

| Backend | Where it computes | Needs |
|---|---|---|
| `Backend.Reference` | Agentica code over lists of floats; the answer the tests trust, and the other backends are checked against it | nothing |
| `Backend.Torch(Device.Cpu)` | libtorch 2.4.1, one op at a time | libtorch (optional) |
| `Backend.Ggml(Device.Cpu)` | ggml on the CPU | the ggml module (optional) |
| `Backend.Ggml(Device.Vulkan(i))` | ggml on Vulkan device `i`, counted over the devices that are not a CPU | a ggml build with Vulkan |
| `Backend.Trace` | nothing: it records a graph of the ops, see below | nothing |

A device is where the values are computed. `Reference` and `Torch` serve `Device.Cpu` only.

A tensor carries its backend, so tensors on different backends can live in one program. An op over tensors on two backends is an error.

`Tensor.available(backend)` says whether a backend works on this machine. A backend needs its native module (compiled C or C++ code). Without it, the backend answers `error("backend not available")`.

`Torch` has 43 of the 45 primitives (the basic ops). `batch_matmul` and `to_device` answer an error that names the op.

On `Ggml`, a direct call to an op is an eager call: it runs the op right away. Eager calls only create, read and copy tensors. Every other op is an error that points to `Trace`, because ggml is fast when a whole computation is one graph.

`Tensor.to_device(t, Device.Vulkan(0))` moves a ggml tensor to another ggml device.

## Tracing and gradients

The matrix product of two tensors on the reference backend, and the gradient of a sum of it. The `Trace` call runs the function once and gives the value and the gradient together:

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

`Trace` runs a function on `Trace` tensors. They record a graph instead of computing, and the graph then runs on a real backend.

- `Trace.eval(args, f)` gives the value of `f` on the backend of `args`.
- `Trace.grad(args, f)` gives the gradient of a scalar function for each argument.
- `Trace.value_and_grad(args, f)` gives `value` and `grads` from one evaluation.
- `Trace.eval_many(args, f)` gives several tensors from one evaluation: `f` answers a `List` of tensors, and on `Ggml` they are one compiled graph.
- `Trace.constant(t)` adds a tensor from outside, such as data or labels, to the trace. No gradient is taken for it.

A backward rule tells how to get the gradient of an op. These ops have no backward rule, so the gradient is an error that names the op: `pow`, `clamp`, `to_dtype`, `batch_matmul`, `conv1d`, `avg_pool1d`, and a `to_device` across devices.

`gelu_erf` is refused by `Trace`. Use `gelu_tanh_approx` when you train.

On `Ggml`, `Trace.eval`, `eval_many`, `grad` and `value_and_grad` compile the whole graph, forward and backward, into one ggml graph. See [How it works](#how-it-works).

## Dtypes and fp16

- `Dtype` has `F32` (the dtype that ops compute in), `F16`, `F64`, `I8`, `I16`, `I32`, `I64`, `U8` and `Bool`.
- `Reference` computes `F16` and `F64` exactly, in doubles (64-bit floats).
- ggml holds `F32` and `I32` as they are. `Bool`, `I8`, `I16` and `U8` are held as `F32`, which is exact for them. `F16`, `F64` and `I64` are refused on ggml.
- fp16 (16-bit floats) is opt-in. `Tensor.vulkan_fp16(true)` makes Vulkan products run in fp16 for the whole process. Call it before any other ggml call, `available` included. A later change is an error. Gradients change by up to about 10 % at MLP sizes.

## Limits

- Exact integers: `F32` holds integers exactly up to 2^24 (16 777 216). On ggml, an `I32` value beyond that, or an `arange` or fill beyond it, is an error, not a wrong number.
- Output cap: one ggml graph returns at most 256 outputs. `value_and_grad` returns the value and one gradient per argument, so ggml takes at most 255 arguments.
- Rank: ggml holds tensors of rank 4 at most.
- Gather: on ggml, the source of a `gather` holds at most 2^24 rows, because its integer indices are exact in `F32` only below that.
- Non-finite values: `exp`, `log`, `sqrt`, `pow`, `sigmoid`, `tanh` and `gelu_tanh_approx` refuse `NaN` and infinite inputs. A ggml output that is not finite is an error.
- Integers: an integer overflow is an error, never a wrap-around. On libtorch, readback of `I64` past 2^53 loses precision.
- Speed: `Reference` is for checks. On one machine, a 64 x 64 matmul took about 6.6 ms on `Reference` and 33 µs on libtorch (`tensor/README.md`).

## How it works

- libtorch and ggml are not in this repository, and they are never vendored. `tensor/README.md` has the setup.
- On `Ggml`, a traced function becomes one ggml graph, forward and backward. The graph is cached by its structure, so a retrace with new values compiles nothing.
- ggml maps the ops of an MLP and of softmax cross-entropy, with their gradients, to its own graph ops. It does the same for `slice` and `gather`. `tensor/DESIGN.md` §7 has the list.

## More

`tensor/README.md` has the setup, the coverage table and the tolerances. `tensor/DESIGN.md` is the contract.
