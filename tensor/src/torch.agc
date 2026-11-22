## The libtorch (CPU, eager) backend of `tensor/`: the §3.1 primitives libtorch
## can express, called from the `Backend.Torch(device)` arms of `../tensor.agc`
## (`tensor/DESIGN.md` §3.1, §7, §8, §9).
##
## This module lives inside `tensor/` rather than in a package of its own
## because a native handle is a resource, and a resource is nominal to the
## `extern module` block that declares it: `Tensor.handle` cannot name a
## handle declared elsewhere (`tensor/DESIGN.md` §2, §7). The module it drives
## is `src/native.agc`; agentica-lang/agentica#1918 would make the handle
## opaque and let this file move back out.
##
## Rules this file keeps, inherited from the public wrappers and the repo's FFI
## rules:
## - Every fallible op returns `Result` and reports `error`, never a halt: a
##   missing native module (`error("backend not available")`), a `Plain`
##   handle handed to this backend, a wrong rank, a wrong dtype, shapes that
##   cannot pair up or reduce, a non-finite input to `exp`/`log`, a `log` of a
##   non-positive value, and a `to_dtype` result that does not fit its width.
## - `../tensor.agc` has already validated shapes, dtypes and backends before
##   it calls here (`tensor/DESIGN.md` §9). The checks below are the re-check a
##   backend owes, never the only check, and `native/torch.c` re-checks the
##   ones libtorch would otherwise answer wrongly (broadcasting, wrapping).
## - Device is explicit, never global (§4): every function takes it first.
##   `Cpu` is the only device libtorch's CPU release serves, so `Cuda` is an
##   `error`.
## - Lifetimes (§8): one `Native.arena` per call. The result is the block's
##   value, so it leaves the arena; the arguments were opened by the caller and
##   are not this arena's to close. A caller that runs several ops per step
##   wraps them in one `Native.arena` of its own (`test/torch_lifetime_test.agc`).
## - Non-finite values never enter `exp`/`log`: the native layer refuses them
##   before touching its own numbers, because a wrapper cannot see stored
##   values without a full readback.
## - Numeric compute is `F32` (§2): the other dtypes exist for creation,
##   readback and `to_dtype`, and the arithmetic ops take float dtypes only,
##   exactly as `../tensor.agc`'s wrappers do.
##
## Not implemented, and answered with `unsupported` below so the dispatch arm
## says so in one place: `sub`, `div`, `neg`, `sqrt`, `pow`, `sigmoid`, `tanh`,
## `gelu_tanh_approx`, `eq`, `lt`, `where`, `clamp`, `max`, `argmax`,
## `batch_matmul`, `squeeze`, `unsqueeze`, `slice`, `concat`, `stack`,
## `broadcast_to`, `gather`, `conv1d`, `avg_pool1d`, `to_device`, `copy`,
## `contiguous`. Each is expressible in libtorch; none is among the thirteen
## native calls this backend started with, and adding one is a native-side
## change of its own (`README.md`, "What is implemented").

import Raw from "native"
import TorchDtypes from "torch_dtypes"
import TorchShapes from "torch_shapes"
import Types from "types"

## True when the native module is loaded. `[native.torch] optional = true` in
## `agentica.toml` means a checkout without libtorch still checks and tests;
## the dispatch answers `error("backend not available")` instead of halting
## with `RUNTIME-NATIVE-LOAD`, and the conformance tests assert that state.
def available() -> Bool = Native.load("torch").ok?

## The answer of an op this backend does not implement yet. One arm of
## `../tensor.agc` calls it per missing primitive, so the gap is a `Result`
## error naming the op instead of a halt or a wrong number.
def unsupported(op: String) -> Result[Types.Tensor]
  error("#{op} is not implemented on the torch backend yet (see tensor/README.md)")
end

## Creation: zeros of `shape` and `dtype` on `device`.
def zeros(device: Types.Device, shape: List[Int], dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, "zeros")
  n = try TorchShapes.volume(shape)
  from_values("zeros", device, shape, dtype, List.repeat(0.0, n))
end

## Creation: ones of `shape` and `dtype` on `device`.
def ones(device: Types.Device, shape: List[Int], dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, "ones")
  n = try TorchShapes.volume(shape)
  from_values("ones", device, shape, dtype, List.repeat(1.0, n))
end

## Creation: `empty` is deterministic zeros, as in the reference: there is no
## uninitialized memory to expose over a boundary that fills every element.
def empty(device: Types.Device, shape: List[Int], dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, "empty")
  n = try TorchShapes.volume(shape)
  from_values("empty", device, shape, dtype, List.repeat(0.0, n))
end

## Creation: every element `value` (tagged for `dtype`).
def full(device: Types.Device, shape: List[Int], value: Types.Values, dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, "full")
  n = try TorchShapes.volume(shape)
  x = try TorchDtypes.value_to_float(value, dtype)
  from_values("full", device, shape, dtype, List.repeat(x, n))
end

## Creation: zeros of the shape and dtype of `t`; its values are not read.
def like(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "like")
  n = try TorchShapes.volume(t.shape)
  from_values("like", device, t.shape, t.dtype, List.repeat(0.0, n))
end

## Creation: `start, start + step, ...` below `stop` (above for `step < 0`),
## integer dtypes only: positions are indices (`tensor/DESIGN.md` §5).
def arange(device: Types.Device, start: Int, stop: Int, step: Int, dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, "arange")
  return error("arange needs an Int dtype, got #{TorchDtypes.name(dtype)}") if !TorchDtypes.is_int_dtype(dtype)
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

  from_values("arange", device, [values.size], dtype, values)
end

## Creation: the one `value` as a rank-0 tensor.
def scalar_to_tensor(device: Types.Device, value: Types.Values, dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, "scalar_to_tensor")
  x = try TorchDtypes.value_to_float(value, dtype)
  from_values("scalar_to_tensor", device, [], dtype, [x])
end

## Creation: row-major `data` of `shape`; the count must match.
def from_list(
  device: Types.Device,
  data: List[Types.Values],
  shape: List[Int],
  dtype: Types.Dtype,
) -> Result[Types.Tensor]
  _ = try ready(device, "from_list")
  n = try TorchShapes.volume(shape)
  return error("from_list needs #{n} values for #{shape}, got #{data.size}") if data.size != n
  values: List[Float] = []
  for v in data
    values = values.push(try TorchDtypes.value_to_float(v, dtype))
  end

  from_values("from_list", device, shape, dtype, values)
end

## Host readback: the row-major values tagged with `t`'s dtype. This is the
## reference's `to_list` for this backend: the boundary carries doubles, so the
## tagging happens here, as the reference tags its own storage.
def to_list(t: Types.Tensor) -> Result[List[Types.Values]]
  _ = try usable("to_list")
  h = try handle_of(t)
  raw = try Raw.readback(h)

  out: List[Types.Values] = []
  for x in raw
    out = out.push(try Types.float_to_value(x, t.dtype))
  end

  ok(out)
end

## Elementwise `a + b`, float only, exact shapes.
def add(device: Types.Device, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "add")
  _ = try pair(a, b, "add")
  ha = try handle_of(a)
  hb = try handle_of(b)
  one(device, a.dtype, a.shape) do Raw.add(ha, hb) end
end

## Elementwise `a * b`, float only, exact shapes.
def mul(device: Types.Device, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "mul")
  _ = try pair(a, b, "mul")
  ha = try handle_of(a)
  hb = try handle_of(b)
  one(device, a.dtype, a.shape) do Raw.mul(ha, hb) end
end

## The rank-2 product `[rows, inner] . [inner, cols]`, float only.
def matmul(device: Types.Device, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "matmul")
  _ = try same_dtype(a, b, "matmul")
  _ = try float_only(a, "matmul")
  return error("matmul needs rank-2 tensors, got #{a.shape} and #{b.shape}") if a.shape.size != 2 or b.shape.size != 2

  return error("matmul needs #{a.shape[1]} in the inner dimension, got #{b.shape[0]}") if a.shape[1] != b.shape[0]

  ha = try handle_of(a)
  hb = try handle_of(b)
  one(device, a.dtype, [a.shape[0], b.shape[1]]) do Raw.matmul(ha, hb) end
end

## `max(0, x)`, elementwise, float only.
def relu(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "relu")
  _ = try float_only(t, "relu")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.relu(h) end
end

## `exp(x)`, elementwise, float only; a non-finite input is an `error`.
def exp(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "exp")
  _ = try float_only(t, "exp")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.exp(h) end
end

## `log(x)`, elementwise, float only; a non-finite or non-positive input is
## an `error`.
def log(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "log")
  _ = try float_only(t, "log")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.log(h) end
end

## Sums the named `axes`, or every axis for `none`, float only.
def sum(
  device: Types.Device,
  t: Types.Tensor,
  axes: Optional[List[Int]] = none,
  keep_dims: Bool = false,
) -> Result[Types.Tensor]
  _ = try ready(device, "sum")
  _ = try float_only(t, "sum")
  axs = try TorchShapes.normalize_axes(axes, t.shape.size, "sum")
  h = try handle_of(t)
  one(device, t.dtype, TorchShapes.reduced_shape(t.shape, axs, keep_dims)) do Raw.sum(h, axs, keep_dims) end
end

## `sum` divided by the reduced count, float only; reducing zero elements is
## an `error`, as in the reference.
def mean(
  device: Types.Device,
  t: Types.Tensor,
  axes: Optional[List[Int]] = none,
  keep_dims: Bool = false,
) -> Result[Types.Tensor]
  _ = try ready(device, "mean")
  _ = try float_only(t, "mean")
  axs = try TorchShapes.normalize_axes(axes, t.shape.size, "mean")
  return error("mean of zero elements is an error") if TorchShapes.reduced_count(t.shape, axs) == 0
  h = try handle_of(t)
  one(device, t.dtype, TorchShapes.reduced_shape(t.shape, axs, keep_dims)) do Raw.mean(h, axs, keep_dims) end
end

## The same values with `shape`; the element counts must agree.
def reshape(device: Types.Device, t: Types.Tensor, shape: List[Int]) -> Result[Types.Tensor]
  _ = try ready(device, "reshape")
  n = try TorchShapes.volume(shape)
  m = try TorchShapes.volume(t.shape)
  return error("reshape needs #{n} values for #{shape}, got #{m}") if m != n
  h = try handle_of(t)
  one(device, t.dtype, shape) do Raw.reshape(h, shape) end
end

## Reverses every axis: rank 2 swaps rows and columns, a rank below 2 is
## unchanged. A true axis permutation is derived from `slice`/`stack` later.
def transpose(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "transpose")
  h = try handle_of(t)
  one(device, t.dtype, t.shape.reverse) do Raw.transpose(h) end
end

## `t` in `dtype`: float to int truncates towards zero, anything nonzero
## becomes `true`, an int width that cannot hold a value is an `error`.
def to_dtype(device: Types.Device, t: Types.Tensor, dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, "to_dtype")
  h = try handle_of(t)
  one(device, dtype, t.shape) do Raw.to_dtype(h, TorchDtypes.code(dtype)) end
end

private

## The native handle of `t`: the `Native` arm. A `Plain` tensor has no native
## handle, and asking this backend to read one is a `Result` error, not a
## halt: it means a tensor of the reference backend was labelled `Torch`.
def handle_of(t: Types.Tensor) -> Result[Raw.TensorHandle]
  case t.handle
    Types.Handle.Native(h) -> ok(h)
    Types.Handle.Plain(_) -> error("a Plain tensor has no native handle (backend #{Types.backend_name(t.backend)})")
  end
end

## The one device libtorch's CPU release serves, and the module it needs.
def ready(device: Types.Device, op: String) -> Result[Bool]
  case device
    Types.Device.Cpu -> usable(op)
    Types.Device.Cuda(n) -> error("#{op} runs on Cpu only, got Cuda(#{n})")
  end
end

## The native module is loaded: a checkout without libtorch says so here,
## before any call would halt with `RUNTIME-NATIVE-LOAD`. The message is the
## one the unwritten backends answer with, plus where to look.
def usable(op: String) -> Result[Bool]
  opened = Native.load("torch")
  return ok(true) if opened.ok?
  error(
    "#{op}: backend not available (#{opened.message}): run " +
    "`eval \"$(sh tools/build-torch.sh)\"` in the shell that runs agentica " +
    "(libtorch is a system dependency, never vendored)",
  )
end

## One `create` call: the values are already validated storage, and the result
## leaves the arena as the block's value.
def from_values(
  op: String,
  device: Types.Device,
  shape: List[Int],
  dtype: Types.Dtype,
  values: List[Float],
) -> Result[Types.Tensor]
  _ = try usable(op)
  Native.arena do
    h = try Raw.create(shape, values, TorchDtypes.code(dtype))
    ok(tensor_of(device, shape, dtype, h))
  end
end

## One native call whose result is a tensor of `out_shape` and `out_dtype`,
## inside the arena of this op: the block returns `Result[Types.Tensor]`, so
## the handle it names is what the arena keeps. `out_dtype` is the input's
## dtype for every op but `to_dtype`.
def one(
  device: Types.Device,
  out_dtype: Types.Dtype,
  out_shape: List[Int],
  call: Fn[Result[Raw.TensorHandle]],
) -> Result[Types.Tensor]
  Native.arena do
    case call.call()
      ok(h) -> ok(tensor_of(device, out_shape, out_dtype, h))
      error(e) -> error(e)
    end
  end
end

## The `Tensor` a fresh handle makes.
def tensor_of(device: Types.Device, shape: List[Int], dtype: Types.Dtype, h: Raw.TensorHandle) -> Types.Tensor
  Types.Tensor(shape: shape, dtype: dtype, backend: Types.Backend.Torch(device), handle: Types.Handle.Native(h))
end

## Two tensors of one shape and one float dtype: what `add` and `mul` take.
def pair(a: Types.Tensor, b: Types.Tensor, op: String) -> Result[Bool]
  return error("#{op} needs two tensors of one shape, got #{a.shape} and #{b.shape}") if a.shape != b.shape
  try same_dtype(a, b, op)
  try float_only(a, op)
  ok(true)
end

## Two tensors share their dtype.
def same_dtype(a: Types.Tensor, b: Types.Tensor, op: String) -> Result[Bool]
  if a.dtype != b.dtype
    return error("#{op} needs two #{TorchDtypes.name(a.dtype)} tensors, got #{TorchDtypes.name(b.dtype)}")
  end

  ok(true)
end

## `t` computes: float dtypes only.
def float_only(t: Types.Tensor, op: String) -> Result[Bool]
  return error("#{op} needs a float dtype, got #{TorchDtypes.name(t.dtype)}") if !TorchDtypes.is_float_dtype(t.dtype)
  ok(true)
end
