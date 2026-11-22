## The libtorch (CPU, eager) backend of the tensor stack: the §3.1 primitives
## libtorch can express, behind the thin C++ layer of `native/torch.c`
## (`tensor/DESIGN.md` §3.1, §8, §9).
##
## `tensor/` is imported as `Ref`: this package borrows its `Dtype`, `Device`
## and `Values` vocabulary and, in `test/`, compares against its reference
## backend, which is the oracle (`tensor/DESIGN.md` §11). It does not use
## `Ref.Tensor` itself: that type's handle names no native resource, so a torch
## tensor is this module's own `Tensor`, whose handle is
## `Handle.Native(TensorHandle)`. Nothing here changes `tensor/`'s public API;
## the PR body's "design delta needed (dispatch)" says what wiring the
## `Backend.Torch` arm would take, and it is not wired in M2.
##
## Rules this file keeps, inherited from `tensor/` and the repo's FFI rules:
## - Every fallible op returns `Result` and reports `error`, never a halt: a
##   missing native module, a wrong rank, a wrong dtype, shapes that cannot
##   pair up or reduce, a non-finite input to `exp`/`log`, a `log` of a
##   non-positive value, and a `to_dtype` result that does not fit its width
##   are all `error`s.
## - Shapes, dtypes and devices are checked here, before the native call
##   (`tensor/DESIGN.md` §9); `native/torch.c` re-checks cheaply and is never
##   the only check.
## - Device is explicit, never global (§4): every op takes it first. `Cpu` is
##   the only device served by libtorch's CPU release, so `Cuda` is an `error`.
## - Lifetimes (§8): one `Native.arena` per call. The result is the block's
##   value, so it leaves the arena; the arguments were opened by the caller
##   and are not this arena's to close. A caller that runs several ops per step
##   wraps them in one `Native.arena` of its own, so the intermediates die at
##   the end of that step, newest first (`test/lifetime_test.agc`).
## - Non-finite values never enter `exp`/`log`: the native layer refuses them
##   before touching its own numbers, because a wrapper cannot see stored
##   values without a full readback.
## - Numeric compute is `F32` (`tensor/DESIGN.md` §2): the other dtypes exist
##   for creation, readback and `to_dtype`, and the arithmetic ops take float
##   dtypes only, exactly as `tensor/`'s wrapper does.
##
## Not implemented in M2 (the README's coverage table says why for each):
## `sub`, `div`, `neg`, `sqrt`, `pow`, `sigmoid`, `tanh`, `gelu_tanh_approx`,
## `eq`, `lt`, `where`, `clamp`, `max`, `argmax`, `squeeze`, `unsqueeze`,
## `slice`, `concat`, `stack`, `broadcast_to`, `gather`, `batch_matmul`,
## `conv1d`, `avg_pool1d`, `to_device`, `copy`, `contiguous`. Each is
## expressible in libtorch; none is among the fourteen native calls M2 was cut
## to, and adding one is a native-side change of its own.

import Ref from "pkg/tensor/tensor"
import Dtypes from "src/dtypes"
import Raw from "src/native"
import Shapes from "src/shapes"

export Ref.Device
export Ref.Dtype
export Ref.Values

## How a torch tensor's values are held: the native handle owns one
## `torch::Tensor`, freed by `TORCH_TensorHandle_FREE` at `Native.close`, at
## the end of the arena that made it, or at the end of the task. The `Plain`
## arm of `Ref.Handle` has no spelling here: this package's tensors are native
## by construction, and `tensor/` cannot name this handle (design delta).
enum Handle
  Native(Raw.TensorHandle)
end

## One tensor of the torch backend: the shape and dtype this wrapper validated,
## and the native handle. Immutable like `Ref.Tensor`: every op returns a new
## one, so a training step returning new parameters falls out later.
struct Tensor
  shape: List[Int]
  dtype: Ref.Dtype
  handle: Handle
end

## True when the native module is loaded. `[native.torch] optional = true`
## in `agentica.toml` means a checkout without libtorch still checks and
## tests; every op below answers `error` instead of a `RUNTIME-NATIVE-LOAD`
## halt, and the conformance tests assert this state rather than passing
## silently.
def available() -> Bool = Native.load("torch").ok?

## The shape of `t`: a field read, no native call.
def shape(t: Tensor) -> List[Int] = t.shape

## The dtype of `t`: a field read, no native call.
def dtype(t: Tensor) -> Ref.Dtype = t.dtype

## True once `t`'s handle is closed. Tests use it to check that a result left
## its arena and that an arena did not close its arguments.
def closed(t: Tensor) -> Bool
  case t.handle
    Handle.Native(h) -> Native.closed?(h)
  end
end

## Creation: zeros of `shape` and `dtype` on `device`.
def zeros(device: Ref.Device, shape: List[Int], dtype: Ref.Dtype) -> Result[Tensor]
  _ = try cpu(device, "zeros")
  n = try Shapes.volume(shape)
  from_values("zeros", shape, dtype, List.repeat(0.0, n))
end

## Creation: ones of `shape` and `dtype` on `device`.
def ones(device: Ref.Device, shape: List[Int], dtype: Ref.Dtype) -> Result[Tensor]
  _ = try cpu(device, "ones")
  n = try Shapes.volume(shape)
  from_values("ones", shape, dtype, List.repeat(1.0, n))
end

## Creation: `empty` is deterministic zeros, as in the reference: there is no
## uninitialized memory to expose over a boundary that fills every element.
def empty(device: Ref.Device, shape: List[Int], dtype: Ref.Dtype) -> Result[Tensor]
  _ = try cpu(device, "empty")
  n = try Shapes.volume(shape)
  from_values("empty", shape, dtype, List.repeat(0.0, n))
end

## Creation: every element `value` (tagged for `dtype`).
def full(device: Ref.Device, shape: List[Int], value: Ref.Values, dtype: Ref.Dtype) -> Result[Tensor]
  _ = try cpu(device, "full")
  n = try Shapes.volume(shape)
  x = try Dtypes.value_to_float(value, dtype)
  from_values("full", shape, dtype, List.repeat(x, n))
end

## Creation: zeros of the shape and dtype of `t`; its values are not read.
def like(device: Ref.Device, t: Tensor) -> Result[Tensor]
  _ = try cpu(device, "like")
  n = try Shapes.volume(t.shape)
  from_values("like", t.shape, t.dtype, List.repeat(0.0, n))
end

## Creation: `start, start + step, ...` below `stop` (above for `step < 0`),
## integer dtypes only: positions are indices (`tensor/DESIGN.md` §5).
def arange(device: Ref.Device, start: Int, stop: Int, step: Int, dtype: Ref.Dtype) -> Result[Tensor]
  _ = try cpu(device, "arange")
  return error("arange needs an Int dtype, got #{Dtypes.name(dtype)}") if !Dtypes.is_int_dtype(dtype)
  return error("arange needs a nonzero step, got #{step}") if step == 0
  values: List[Float] = []
  v = start
  if step > 0
    while v < stop
      values = values.push(v.to_f)
      v += step
    end
  else
    while v > stop
      values = values.push(v.to_f)
      v += step
    end
  end

  from_values("arange", [values.size], dtype, values)
end

## Creation: the one `value` as a rank-0 tensor.
def scalar_to_tensor(device: Ref.Device, value: Ref.Values, dtype: Ref.Dtype) -> Result[Tensor]
  _ = try cpu(device, "scalar_to_tensor")
  x = try Dtypes.value_to_float(value, dtype)
  from_values("scalar_to_tensor", [], dtype, [x])
end

## Creation: row-major `data` of `shape`; the count must match.
def from_list(device: Ref.Device, data: List[Ref.Values], shape: List[Int], dtype: Ref.Dtype) -> Result[Tensor]
  _ = try cpu(device, "from_list")
  n = try Shapes.volume(shape)
  return error("from_list needs #{n} values for #{shape}, got #{data.size}") if data.size != n
  values: List[Float] = []
  for v in data
    values = values.push(try Dtypes.value_to_float(v, dtype))
  end

  from_values("from_list", shape, dtype, values)
end

## Host readback: the row-major values tagged with `t`'s dtype. This is
## `Ref.Tensor.to_list` for this backend: the boundary carries doubles, so the
## tagging happens here, as the reference tags its own storage.
def to_list(t: Tensor) -> Result[List[Ref.Values]]
  _ = try loaded("to_list")
  raw = try Raw.readback(handle_of(t))

  out: List[Ref.Values] = []
  for x in raw
    out = out.push(try Dtypes.float_to_value(x, t.dtype))
  end

  ok(out)
end

## Elementwise `a + b`, float only, exact shapes.
def add(device: Ref.Device, a: Tensor, b: Tensor) -> Result[Tensor]
  _ = try cpu(device, "add")
  _ = try loaded("add")
  _ = try pair(a, b, "add")
  one(a, a.dtype, a.shape) do Raw.add(handle_of(a), handle_of(b)) end
end

## Elementwise `a * b`, float only, exact shapes.
def mul(device: Ref.Device, a: Tensor, b: Tensor) -> Result[Tensor]
  _ = try cpu(device, "mul")
  _ = try loaded("mul")
  _ = try pair(a, b, "mul")
  one(a, a.dtype, a.shape) do Raw.mul(handle_of(a), handle_of(b)) end
end

## The rank-2 product `[rows, inner] . [inner, cols]`, float only.
def matmul(device: Ref.Device, a: Tensor, b: Tensor) -> Result[Tensor]
  _ = try cpu(device, "matmul")
  _ = try loaded("matmul")
  _ = try same_dtype(a, b, "matmul")
  _ = try float_only(a, "matmul")
  return error("matmul needs rank-2 tensors, got #{a.shape} and #{b.shape}") if a.shape.size != 2 or b.shape.size != 2

  return error("matmul needs #{a.shape[1]} in the inner dimension, got #{b.shape[0]}") if a.shape[1] != b.shape[0]
  one(a, a.dtype, [a.shape[0], b.shape[1]]) do Raw.matmul(handle_of(a), handle_of(b)) end
end

## `max(0, x)`, elementwise, float only.
def relu(device: Ref.Device, t: Tensor) -> Result[Tensor]
  _ = try cpu(device, "relu")
  _ = try loaded("relu")
  _ = try float_only(t, "relu")
  one(t, t.dtype, t.shape) do Raw.relu(handle_of(t)) end
end

## `exp(x)`, elementwise, float only; a non-finite input is an `error`.
def exp(device: Ref.Device, t: Tensor) -> Result[Tensor]
  _ = try cpu(device, "exp")
  _ = try loaded("exp")
  _ = try float_only(t, "exp")
  one(t, t.dtype, t.shape) do Raw.exp(handle_of(t)) end
end

## `log(x)`, elementwise, float only; a non-finite or non-positive input is
## an `error`.
def log(device: Ref.Device, t: Tensor) -> Result[Tensor]
  _ = try cpu(device, "log")
  _ = try loaded("log")
  _ = try float_only(t, "log")
  one(t, t.dtype, t.shape) do Raw.log(handle_of(t)) end
end

## Sums the named `axes`, or every axis for `none`, float only.
def sum(device: Ref.Device, t: Tensor, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Tensor]
  _ = try cpu(device, "sum")
  _ = try loaded("sum")
  _ = try float_only(t, "sum")
  axs = try Shapes.normalize_axes(axes, t.shape.size, "sum")
  one(t, t.dtype, Shapes.reduced_shape(t.shape, axs, keep_dims)) do Raw.sum(handle_of(t), axs, keep_dims) end
end

## `sum` divided by the reduced count, float only; reducing zero elements is
## an `error`, as in the reference.
def mean(device: Ref.Device, t: Tensor, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Tensor]
  _ = try cpu(device, "mean")
  _ = try loaded("mean")
  _ = try float_only(t, "mean")
  axs = try Shapes.normalize_axes(axes, t.shape.size, "mean")
  return error("mean of zero elements is an error") if Shapes.reduced_count(t.shape, axs) == 0
  one(t, t.dtype, Shapes.reduced_shape(t.shape, axs, keep_dims)) do Raw.mean(handle_of(t), axs, keep_dims) end
end

## The same values with `shape`; the element counts must agree.
def reshape(device: Ref.Device, t: Tensor, shape: List[Int]) -> Result[Tensor]
  _ = try cpu(device, "reshape")
  _ = try loaded("reshape")
  n = try Shapes.volume(shape)
  m = try Shapes.volume(t.shape)
  return error("reshape needs #{n} values for #{shape}, got #{m}") if m != n
  one(t, t.dtype, shape) do Raw.reshape(handle_of(t), shape) end
end

## Reverses every axis: rank 2 swaps rows and columns, a rank below 2 is
## unchanged. A true axis permutation is derived from `slice`/`stack` later.
def transpose(device: Ref.Device, t: Tensor) -> Result[Tensor]
  _ = try cpu(device, "transpose")
  _ = try loaded("transpose")
  one(t, t.dtype, t.shape.reverse) do Raw.transpose(handle_of(t)) end
end

## `t` in `dtype`: float to int truncates towards zero, anything nonzero
## becomes `true`, an int width that cannot hold a value is an `error`.
def to_dtype(device: Ref.Device, t: Tensor, dtype: Ref.Dtype) -> Result[Tensor]
  _ = try cpu(device, "to_dtype")
  _ = try loaded("to_dtype")
  one(t, dtype, t.shape) do Raw.to_dtype(handle_of(t), Dtypes.code(dtype)) end
end

private

## The native handle of `t`: the one arm `Handle` has.
def handle_of(t: Tensor) -> Raw.TensorHandle
  case t.handle
    Handle.Native(h) -> h
  end
end

## `device` is the one libtorch's CPU release serves.
def cpu(device: Ref.Device, op: String) -> Result[Bool]
  case device
    Ref.Device.Cpu -> ok(true)
    Ref.Device.Cuda(n) -> error("#{op} runs on Cpu only, got Cuda(#{n})")
  end
end

## The native module is loaded: a checkout without libtorch says so here,
## before any call would halt with `RUNTIME-NATIVE-LOAD`.
def loaded(op: String) -> Result[Bool]
  opened = Native.load("torch")
  return ok(true) if opened.ok?
  error(
    "#{op} needs the native torch module (#{opened.message}): run " +
    "`eval \"$(sh tools/build-torch.sh)\"` in the shell that runs agentica " +
    "(libtorch is a system dependency, never vendored)",
  )
end

## One `create` call: the values are already validated storage, and the result
## leaves the arena as the block's value.
def from_values(op: String, shape: List[Int], dtype: Ref.Dtype, values: List[Float]) -> Result[Tensor]
  _ = try loaded(op)
  Native.arena do
    h = try Raw.create(shape, values, Dtypes.code(dtype))
    ok(Tensor(shape: shape, dtype: dtype, handle: Handle.Native(h)))
  end
end

## One native call whose result is a tensor of `out_shape` and `out_dtype`,
## inside the arena of this op: the block returns `Result[Tensor]`, so the
## handle it names is what the arena keeps. `out_dtype` is `t`'s dtype for
## every op but `to_dtype`, and `t` stays the argument the op reads.
def one(t: Tensor, out_dtype: Ref.Dtype, out_shape: List[Int], call: Fn[Result[Raw.TensorHandle]]) -> Result[Tensor]
  Native.arena do
    case call.call()
      ok(h) -> ok(Tensor(shape: out_shape, dtype: out_dtype, handle: Handle.Native(h)))
      error(e) -> error(e)
    end
  end
end

## Two tensors of one shape and one float dtype: what `add` and `mul` take.
def pair(a: Tensor, b: Tensor, op: String) -> Result[Bool]
  return error("#{op} needs two tensors of one shape, got #{a.shape} and #{b.shape}") if a.shape != b.shape
  try same_dtype(a, b, op)
  try float_only(a, op)
  ok(true)
end

## Two tensors share their dtype.
def same_dtype(a: Tensor, b: Tensor, op: String) -> Result[Bool]
  return error("#{op} needs two #{Dtypes.name(a.dtype)} tensors, got #{Dtypes.name(b.dtype)}") if a.dtype != b.dtype

  ok(true)
end

## `t` computes: float dtypes only.
def float_only(t: Tensor, op: String) -> Result[Bool]
  return error("#{op} needs a float dtype, got #{Dtypes.name(t.dtype)}") if !Dtypes.is_float_dtype(t.dtype)
  ok(true)
end
