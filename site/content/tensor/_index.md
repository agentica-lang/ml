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
- `from_file(path, offset, shape, format, backend, dtype)` makes an `F32` tensor from the little-endian floats that start at byte `offset` of a file. `format` is `FloatFormat.F32` (the default), `F16`, `BF16` or `Q8_0`. A `Q8_0` range is ggml's blocks of 32 values (34 bytes: a binary16 scale and 32 signed bytes), and it loads as `F32` on every backend, each value the scale times the byte, as ggml dequantizes it (exact). libtorch and ggml read the file themselves, so the weights never become a list. `Reference` reads it in Agentica, which is slow on a large file. With the environment variable `TENSOR_LOAD=agentica` (an environment variable) libtorch and ggml take the bytes from an Agentica read instead (`File.seek` and `File.read_bytes`), which is slower and uses more memory, for small models. A short file, a bad offset or a missing file gives an `error`.
- `from_file(..., dtype: Dtype.F16)` makes a half weight: the ggml backend keeps the values in 16 bits, converted at load from `F32`, `F16` or `BF16`, which halves the memory and the bandwidth of a weight and speeds up the decoding that streams it. A value beyond 65504 is an `error` that names its element; a value too small for F16 (a BF16 one, for instance) becomes a subnormal or zero. Only ggml holds one (any other backend gives an `error` naming the option), and a trace may use it only as the weight of `matmul` and `linear` (use `linear(..., out_in: true)` with a checkpoint's `[out, in]` weight, or every run copies the transposed weight) and as the table of `gather` and `embedding`, whose rows stay `F16` in the trace and are widened with `to_dtype(rows, F32)`. Any other use is an `error` naming the op (an `F16` output is allowed only for the cache below). Widening an `F16` input with `to_dtype(x, F32)` runs, a half weight passed as an input included (the lowering cannot tell it from a cache): it makes a full `F32` copy on every run; a half weight that is a constant of the function is refused. It is for inference: `value_and_grad` refuses a half weight. `matmul` answers `F32` and the activations stay `F32` too (on ggml CPU and on Vulkan at its default, within 1e-7 of an exact sum); with `Tensor.vulkan_fp16(true)` the product runs in fp16 (about 2e-3 off on outputs of 0.4, and an activation beyond 65504 overflows), as it does for an `F32` weight, and a CPU kernel that rounds activations to F16 would be good to 2^-11 relative per product. `F32` stays the default and the exact path.
- `from_file(..., format: FloatFormat.Q8_0, dtype: Dtype.Q8_0)` makes a quantized weight: the ggml backend keeps the blocks of the Q8_0 file as they are, about a quarter of the memory of `F32`. It is a rank-2 weight whose rows hold whole blocks of 32 values, and any other backend or another `format` gives an `error`. A trace uses it as a half weight is used (the weight of `linear`, the table of `gather` and `embedding`, no gradient, never an output), and `linear` must be called with `out_in: true`: a `matmul` by the weight itself is an `error`, since its transpose would copy blocks. The rows of `embedding` come out exact. ggml rounds the activations of a product to Q8_0 blocks (llama.cpp does the same), so the answer is good to the half step of that rounding and of its scale's rounding to F16, `amax * (1 / 254 + 2^-11)` of a block of 32 activations per product, and not to `F32`'s 1e-7. `to_list` of one dequantizes it.
- `append_file(t, path)` appends the values of the `F32` tensor `t`, row-major and little-endian, to a file that exists: the bytes that `from_file` reads back with `FloatFormat.F32`, for writing a checkpoint. libtorch and ggml write from C, so no value becomes a list; `Reference` encodes them in Agentica (slow on a large tensor, and every NaN is written as the canonical one). A tensor of another dtype or a missing file gives an `error`.
- `convert_file(src, src_offset, src_format, count, dst, dst_offset, dst_format)` writes the `count` floats of `src_format` that start at byte `src_offset` of one file into the existing file `dst` as `dst_format` (`F32`, `F16` or `BF16`), for writing a checkpoint. C does it in chunks on several threads, so no tensor and no list exists. The same format is a copy of the bytes; a narrower one rounds to nearest even, and a value beyond 65504 is an `error` for `F16`. It needs the ggml module and works on files only, whatever the backend.
- `dtype` defaults to `F32` and `backend` to `Reference` in `zeros`, `ones`, `full` and `from_list`.
- `to_list(t)` reads a tensor back as a `List[Values]`.
- `shape`, `dtype` and `backend` read the fields of a tensor, with no call to a backend.

## Ops

- Elementwise, one value at a time, floats only: `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `pow`, `relu`, `sigmoid`, `tanh`, `gelu_tanh_approx`, `gelu_erf`, `clamp`.
- Comparisons: `eq` and `lt` (floats or integers) give `1.0` or `0.0`. `where(cond, a, b)` selects with a `Bool` tensor.
- Reductions: `sum`, `mean`, `max`, `argmax`, each with `axes` (`none` for every axis, or a list) and `keep_dims`. `argmax` gives `I64`.
- Matrix products: `matmul` (rank 2), `transpose` (reverses every axis), `batch_matmul` (not on `Torch`).
- Axes: `permute(t, axes)` puts input axis `axes[i]` at output axis `i`.
- Attention: `attention(q, k, v, mask: none, scale: none, causal: false)` is `softmax(scale * q k^T + mask) v` over `[batch, heads, q_len, head_dim]` tensors. `k` and `v` may have fewer heads, as long as they divide the heads of `q` (grouped-query attention). The mask is added to the scores and is `[q_len, k_len]` or `[batch, heads, q_len, k_len]`, with `1` allowed for `batch` and `heads`; it takes finite values and `-inf`, and it has a gradient. `causal: true` hides the keys after each query, with the last query on the last key, so it needs `q_len <= k_len`. A query with every key at `-inf`, or a `NaN` anywhere, is an error on every backend: to hide a padding position use a large finite negative number, as Hugging Face does. What a query whose every key is hidden by such a number returns is unspecified: the mean of `v` on `Reference` and in the fallback of ggml, the softmax of its scores on ggml's flash attention. On `Reference`, `q_len * k_len` of the causal mask stops at 1,048,576. The scale defaults to `1 / sqrt(head_dim)`.
- Rotary embedding: `rope(x, positions, base: 10000.0, dims: none, neox: true, inverse: false)` turns the first `dims` columns of each row of `x` (`[..., seq, width]`) in pairs by the angle `positions[row] * base^(-2i/dims)`. `neox` pairs column `i` with `i + dims/2`, and `neox: false` pairs `2i` with `2i + 1`. `positions` is an integer vector with one position per row, shared by every batch and head, and `inverse: true` turns the pairs back.
- Shape: `reshape`, `squeeze`, `unsqueeze`, `slice`, `concat`, `stack`, `broadcast_to`, `gather` (rows at integer indices), `top_ids(x, k)` (the ids of the `k` largest of a float vector, `I32` `[k]`: descending on the reference, in no order on ggml, where it is one device node and only `k` ids cross to the host; the derived `top_k` reads the whole vector), `set_rows(cache, rows, positions)` (a copy of `cache` `[..., rows, width]` with its rows `positions[i]` replaced, the positions distinct; see `Trace.run`), `scatter_add(indices, updates, rows)` (the inverse of `gather`: it adds the rows of `updates` into `rows` zero rows, and a repeated index sums its rows).
- Convolution: `conv1d` (stride and padding; no dilation or groups), `avg_pool1d`.
- Movement and dtype: `to_dtype`, `to_device`, `copy`, `contiguous`, `stop_grad`.
- Sorting and decoding: `top_k(x, k)` gives the `k` largest values of a rank-1 float tensor, descending, as a `TopK` with `values` and `indices` (`I64`). Ties keep an unspecified order, and a non-finite value is an `error`. `beam_search(logits, width)` reads `[steps, vocab]` logits and gives the `width` best token rows as `I64 [width, steps]`, best first. It ranks each path by the sum of its logits, with no softmax and no end token, and `width` is from 1 to `vocab`.
- Optimizer kernel: `adam_step(p, g, m, v, hyper)` is the update of Adam and AdamW, which `train/`'s `Optim` uses (you call `Optim`, not this). It gives the new `[p, m, v]` from a weight, its gradient, the two moments and `hyper`, a rank-1 tensor of 7 numbers (the learning rate, `beta1`, `beta2`, `eps`, the weight decay, `1 / (1 - beta1^t)` and `1 / (1 - beta2^t)`). In a trace of `F32` tensors of rank 3 or less, ggml runs it as one `ggml_opt_step_adamw` node, on copies of the arguments in `Trace.eval` and `Trace.eval_many`; elsewhere it is a composition of the ops above, which is also what `Trace.stablehlo` writes for the node. In a function compiled by `Trace.compile` it writes the moments `m` and `v` in place when they are inputs that no other node reads (once each, not outputs) and their two results are outputs: `Trace.run` then gives back those very tensors, with the new values, so the tensors given are consumed (even when the run fails), as for `set_rows`; `Trace.donated(compiled)` says which outputs are such inputs. The weight is never written. It has no gradient.
- Built from the above: `softmax`, `gelu` (the tanh form), `silu`, `dot`, `linear`, `embedding`, `multi_head_attention` (over `[batch, seq, width]`, it is `attention` between a split and a join of the heads). `layer_norm` and `rms_norm` are built from them too, but a trace records each as one node, so ggml runs one norm (ggml's Vulkan backend can fuse the product by a weight after an rms norm) and the gradient has a rule of its own. `greedy_step(logits, temperature: none)` takes the argmax of rank-1 logits; a `temperature` above 0 scales them first. `dropout(x, p)` is the identity at inference, and `p` must lie in `0..1`.

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

`Torch` has 47 of the 49 primitives (the basic ops). `batch_matmul` and `to_device` answer an error that names the op.

On `Ggml`, a direct call to an op is an eager call: it runs the op right away. Eager calls only create, read and copy tensors. Every other op is an error that points to `Trace`, because ggml is fast when a whole computation is one graph.

`Tensor.to_device(t, Device.Vulkan(0))` moves a ggml tensor to another ggml device.

The ggml CPU backend runs a graph on one thread per physical core of the machine, not per hardware thread: its threads wait for each other by spinning, so a second thread on a core adds stalls when the machine is busy. On Linux the cores are the ones this process may use (a cpuset limits them; a CPU quota does not). Set the environment variable `TENSOR_THREADS` to a positive integer, at most 512 (ggml's limit), to choose the number; any other value leaves the default.

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
- `Trace.gradients(out, wrt)` is the same step on traced tensors: it gives the gradient graph of the traced scalar `out` for each traced tensor of `wrt`, and zeros for a tensor `out` does not depend on.
- `Trace.value_and_grad(args, f)` gives `value` and `grads` from one evaluation.
- `Trace.eval_many(args, f)` gives several tensors from one evaluation: `f` answers a `List` of tensors, and on `Ggml` they are one compiled graph.
- `Trace.constant(t)` adds a tensor from outside, such as data or labels, to the trace. No gradient is taken for it.
- `Trace.node_count(outs)` gives the number of distinct nodes that `outs` reach, which is what one evaluation computes.
- `Trace.compile(specs, f)` traces `f` once and gives a `Compiled` function. `Trace.run(compiled, args)` runs it any number of times and gives what `eval_many` gives, without tracing `f` or planning the graph again.

`Trace.eval` and `eval_many` trace `f` and plan the graph on every call, then reuse the compiled ggml graph by its structure. `Trace.compile` does that work once:

- `specs` lists each input's shape, dtype and backend (`Trace.spec_of(t)` takes them from a tensor). `f` gets one traced input per spec and answers a `List` of tensors.
- Whatever changes between runs must be an input: the parameters of a training step, a KV cache. A tensor that `f` brings in with `Trace.constant` is fixed: every run reads the same tensor. So is the value of a `full` in `f`.
- A constant must stay open while any task may run the `Compiled` function. A run that starts after its owner closed a constant is an `error` that names the leaf. A run that already bound it keeps its memory until the run ends, so a close in the middle of a run does not break that run.
- `run` checks its arguments against the specs, one tensor per spec. A different shape, dtype, backend or count is an `error` naming the input. It never compiles another graph; make one `Compiled` for each shape.
- On ggml the graph lives in the process-wide cache of 32 graphs (least recently used out), and `compile` pins it there: the cache does not evict it while the pin lives. The pin belongs to the task that called `compile` and ends with that task or at `Trace.close(compiled)`; a function that no longer runs should be closed, or its pin takes a slot until its task ends. The `Compiled` still runs after the pin ends, and a run that finds its graph evicted builds it again. A run that finds its graph in use by another task sends the nodes again and compiles another copy, though it still does not trace `f` or plan the graph. When no slot is free (every graph is in use or pinned), `compile` pins nothing and the function runs as an ordinary graph, as `eval` and `eval_many` do. `Trace.close_constants(compiled)` also closes the constants the function read from outside its inputs (for a function that owns them, such as a log-mel graph's filters): a run after it is an error naming a closed constant.
- `Trace.run(compiled, args, into)` writes the outputs that `into` names into tensors that exist (one entry per output, `none` for a new tensor), so a loop that feeds an output back as the next input allocates nothing: the tensor has the output's shape, dtype and backend, is not an input or a constant of the run or another output (that is an `error`), is not closed, and is overwritten by the run.
- `set_rows` in a function run on ggml writes the cache in place. The cache must be an input of the function that no other node reads, the result of the `set_rows` must be an output, and an input is not also an output; a function that breaks one of these does not compile, and the error names the node. `Trace.run` then gives back the very tensor it was given as that output, with the rows written: the tensor given is consumed, whoever holds it sees the new rows (and nothing else may use it after). The cache is consumed even when the run fails: a position outside the cache is found after the write, and the rows are clamped into it. The cache must be `F32`, or `F16` (the key-value cache of `transformers`): the rows are `F32` and ggml rounds them to `F16` as it writes them, within 2^-11 relative (a value beyond 65504 becomes infinite, as in llama.cpp); `to_dtype(x, F16)` makes an `F16` tensor on ggml, `to_dtype(cache, F32)` widens an `F16` input, and `attention` reads an `F16` key and value (with an `F32` query) without a copy. `Trace.eval` and `eval_many` are functional: they write a copy and leave the cache alone. `into` cannot name a tensor for such an output, and the cache cannot be given for two inputs. On every other backend the result is a new tensor and the cache is not written. `set_rows` has no gradient.
- A cached ggml graph owns no weights and no data: it is the structure (ops, shapes, dtypes), and a run binds tensors to it. A `Compiled` function leaves its constants bound to its pinned graph between runs, so a run binds only its inputs; closing a constant, or the pin, unbinds them. A `Compiled` function holds only what `f` brought in with `Trace.constant`, so a function that takes its weights as inputs (the encoder blocks of `asr/` do) serves any checkpoint of the same shapes, and one that closes over its weights as constants belongs to those tensors.
- `f` may compute `Trace.gradients` and the update of the parameters, so a training step (forward, backward and update) is one `Compiled` function that answers the loss and the new parameters.
- Several tasks may run one `Compiled` function.
- On `Reference` and `Torch` the traced graph is kept and evaluated node by node on every run.

### Finding a parity bug

When two backends give different results, `Trace.diff(args, a, b, f)` finds the first node where they disagree. It traces `f` once, runs the graph on the backends `a` and `b` (`args` are copied to a backend only when they live elsewhere), and compares every node that the result of `f` reaches:

```agentica
d = try Trace.diff(args, Tensor.Backend.Reference, Tensor.Backend.Torch(Tensor.Device.Cpu)) do |xs|
  Tensor.exp(try Tensor.add(xs[0], xs[1]))
end
```

- `d.first` is the first node, in graph order, whose output is outside the tolerance while its inputs are within, or `none`. It is a `NodeDiff`: the `node` id, `op`, `shape`, `dtype`, the shapes of its arguments (`inputs`), `max_abs` and `max_rel` (the largest absolute and relative gap over its elements) and `within`. `d.nodes` counts the nodes compared, and `d.worst` lists the five with the largest `max_abs`.
- A float element agrees when `|a - b| <= atol + rtol * max(|a|, |b|)`; the defaults are `rtol: 0.0001` and `atol: 0.000001`, and `Trace.diff(args, a, b, rtol: 0.001) do ... end` changes them. Integers and `Bool`s must be equal. A non-finite value that the other side does not have is an infinite gap.
- On `Ggml` the nodes run 400 at a time, to stay under the 512 outputs of one graph, and each slice runs its ancestors again from the inputs. `Reference` and `Torch` run all the nodes at once.
- The comparison runs on the host, so it is slow on big activations.
- An op that overflows to infinity on `Reference` or `Torch` makes the next node an `error`, and `diff` returns it.
- Everything `f` needs from outside must be an argument. A `Trace.constant` lives on one backend, and running it on the other is an `error`.

A backward rule tells how to get the gradient of an op. These ops have no backward rule, so the gradient is an error that names the op: `pow`, `clamp`, `to_dtype`, `batch_matmul`, `conv1d`, `avg_pool1d`, and a `to_device` across devices.

On `Ggml`, `Trace.eval`, `eval_many`, `grad` and `value_and_grad` compile the whole graph, forward and backward, into one ggml graph. See [How it works](#how-it-works).

## StableHLO

`Trace.stablehlo(specs, f)` traces `f` once, over inputs of the shapes and dtypes of `specs`, and gives a `Module`: `text` is a StableHLO program that XLA (through PJRT) and IREE read, and `constants` are the tensors that `f` brought in with `Trace.constant`.

- The program is one `@main` that takes the inputs, then the constants, and returns the tensors `f` answered. Shapes are static.
- It covers the ops of an MLP forward and of a training step: `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `tanh`, `relu`, `gelu_tanh_approx`, `matmul`, `transpose`, `reshape`, `broadcast_to`, `sum`, `mean`, `max`, `eq`, `lt`, `where`, `arange`, `slice`, `full` and `stop_grad`. Another op is an `error` that names it.
- `Train.stablehlo(loss, params, batch, optimizer)` gives the step of `fit` as one program: the arguments are the parameters, the optimizer's moments, the inputs of the step (`Optim.step_inputs(state)`: SGD's learning rate as a rank-0 tensor, Adam's 7 numbers as a rank-1 tensor, AdamW's two such rows), then the batch; the results are the loss, the new parameters and the new moments. Feed the results back in, with the inputs of the next step, to train.
- The repository checks the text against snapshots, and the reference backend against the outputs IREE gave for the same text (`make iree-goldens` makes them; IREE is not needed to run the tests).

## Dtypes and fp16

- `Dtype` has `F32` (the dtype that ops compute in), `F16`, `F64`, `I8`, `I16`, `I32`, `I64`, `U8`, `Bool` and `Q8_0` (a quantized weight of ggml, from `from_file` only).
- `Reference` computes `F16` and `F64` exactly, in doubles (64-bit floats).
- ggml holds `F32` and `I32` as they are. `Bool`, `I8`, `I16` and `U8` are held as `F32`, which is exact for them. `F64` and `I64` are refused on ggml, and `F16` and `Q8_0` are held only as a half or a quantized weight from `from_file`, except an `F16` cache (`to_dtype(x, F16)`, written by `set_rows`, read by `attention`, whole or as a `slice` of it, and widened by `to_dtype`). A slice that starts at the first element is read by flash attention where it is, with no copy.
- Vulkan products run in fp16 (16-bit floats) by default, which is about 1.7 times faster for the products of a model. They are not exact against Hugging Face: the greedy ids of a model can differ. `Tensor.vulkan_fp16(false)` makes them run in exact `F32` for the whole process. Call it before any other ggml call (`available` of a Vulkan device included). A later change is an error. A `GGML_VK_DISABLE_F16` exported by the user, with no call, keeps `F32`. Gradients change by up to about 10 % at MLP sizes with fp16.

## Limits

- Exact integers: `F32` holds integers exactly up to 2^24 (16 777 216). On ggml, an `I32` value beyond that, or an `arange` or fill beyond it, is an error, not a wrong number.
- Output cap: one ggml graph returns at most 512 outputs. `value_and_grad` returns the value and one gradient per argument, so ggml takes at most 511 arguments.
- Rank: ggml holds tensors of rank 4 at most.
- Positions: on ggml, the `positions` of `rope` are any integer dtype but `I64`, which is refused. ggml turns by an angle it makes in `F32`, so at positions in the thousands it is a few 1e-4 off the reference.
- Gather: on ggml, the source of a `gather`, the answer of a `scatter_add` and the cache of a `set_rows` and the vector of a `top_ids` hold at most 2^24 rows, because its integer indices are exact in `F32` only below that.
- Non-finite values: `exp`, `log`, `sqrt`, `pow`, `sigmoid`, `tanh` and `gelu_tanh_approx` refuse `NaN` and infinite inputs, except that `exp(-inf)` is `0`. A ggml output that is not finite, or whose sum overflows `F32` (`exp(80)` over 10 000 values), is an error, except the cache that `set_rows` writes in place and the first moment that `adam_step` writes in place, which are not checked (the second moment is, and the new weight).
- Integers: an integer overflow is an error, never a wrap-around. On libtorch, readback of `I64` past 2^53 loses precision.
- Speed: `Reference` is for checks. On one machine, a 64 x 64 matmul took about 6.6 ms on `Reference` and 33 µs on libtorch (`tensor/README.md`).

## How it works

- libtorch and ggml are not in this repository, and they are never vendored. `tensor/README.md` has the setup.
- On `Ggml`, a traced function becomes one ggml graph, forward and backward. The graph is cached by its structure, so a retrace with new values compiles nothing.
- ggml maps the ops of an MLP and of softmax cross-entropy, with their gradients, to its own graph ops. It does the same for `slice`, `gather` and `scatter_add`, and for attention: `attention` is one flash attention kernel when the mask is absent or made of fills (0 and a large negative number, as the causal and the padded masks are), and two matrix products and one softmax otherwise; `rope` is ggml's rotary embedding. `tensor/DESIGN.md` §7 has the list.

## More

`tensor/README.md` has the setup, the coverage table and the tolerances. `tensor/DESIGN.md` is the contract.
