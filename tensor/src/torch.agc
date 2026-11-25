## The libtorch (CPU, eager) backend of `tensor/`: the §3.1 primitives libtorch
## can express, called from the `Backend.Torch(device)` arms of `../tensor.agc`
## (`tensor/DESIGN.md` §3.1, §7, §8, §9).
##
## This module lives inside `tensor/` rather than in a package of its own
## because a native handle is a resource, and a resource is nominal to the
## `extern module` block that declares it: `Tensor.handle` cannot name a
## handle declared elsewhere (`tensor/DESIGN.md` §2, §7). The module it drives
## is `src/native.agc`.
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
##   `Cpu` is the only device libtorch's CPU release serves.
## - Lifetimes (§8): one `Native.arena` per call. The result is the block's
##   value, so it leaves the arena; the arguments were opened by the caller and
##   are not this arena's to close. A caller that runs several ops per step
##   wraps them in one `Native.arena` of its own (`test/torch_lifetime_test.agc`).
## - Non-finite values never enter a transcendental: `exp`, `log`, `sqrt`,
##   `pow`, `sigmoid` and `tanh` refuse them in the native layer before it
##   touches its own numbers, because a wrapper cannot see stored values
##   without a full readback. `div` refuses a zero divisor the same way, and
##   `log`/`sqrt`/`pow` refuse their out-of-domain inputs there too.
## - Numeric compute is `F32` (§2): the other dtypes exist for creation,
##   readback and `to_dtype`, and the arithmetic ops take float dtypes only,
##   exactly as `../tensor.agc`'s wrappers do.
##
## Not implemented, and answered with `unsupported` below so the dispatch arm
## says so in one place: `batch_matmul` and `to_device`.
## Each is expressible in libtorch, none is among the ops this backend was
## asked for, and nothing here claims it has one (`README.md`, "What is
## implemented").

import FileRange from "file_range"
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
  filled("zeros", device, shape, dtype, 0.0)
end

## Creation: ones of `shape` and `dtype` on `device`.
def ones(device: Types.Device, shape: List[Int], dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, "ones")
  filled("ones", device, shape, dtype, 1.0)
end

## Creation: `empty` is deterministic zeros, as in the reference: there is no
## uninitialized memory to expose over a boundary that fills every element.
def empty(device: Types.Device, shape: List[Int], dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, "empty")
  filled("empty", device, shape, dtype, 0.0)
end

## Creation: every element `value` (tagged for `dtype`).
def full(device: Types.Device, shape: List[Int], value: Types.Values, dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, "full")
  x = try TorchDtypes.value_to_float(value, dtype)
  filled("full", device, shape, dtype, x)
end

## Creation: zeros of the shape and dtype of `t`; its values are not read.
def like(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "like")
  filled("like", device, t.shape, t.dtype, 0.0)
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

## Creation: row-major `F32` `values` of `shape`, without the `Values` tags
## `from_list` takes; the wrapper checked the count.
def floats(device: Types.Device, values: List[Float], shape: List[Int]) -> Result[Types.Tensor]
  _ = try ready(device, "from_list")
  from_values("from_list", device, shape, Types.Dtype.F32, values)
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

## Creation: an `F32` tensor of `shape` from the floats of `format` at byte
## `offset` of the file `path`, read by C into the tensor's own memory; with
## `TENSOR_LOAD=agentica` Agentica reads the range and C widens the bytes.
def from_file(
  device: Types.Device,
  path: String,
  offset: Int,
  shape: List[Int],
  format: Types.FloatFormat,
) -> Result[Types.Tensor]
  _ = try ready(device, "from_file")
  if FileRange.agentica_read?()
    n = try Types.volume(shape)
    bytes = try FileRange.read_range(path, offset, try Types.file_bytes(format, n))
    return one(device, Types.Dtype.F32, shape) do Raw.from_bytes(bytes, shape, Types.file_code(format)) end
  end

  one(device, Types.Dtype.F32, shape) do Raw.load(path, offset, shape, Types.file_code(format)) end
end

## Appends the `F32` values of `t` to the file `path`, written by C.
def append_file(t: Types.Tensor, path: String) -> Result[Void]
  _ = try usable("append_file")
  Raw.append_file(try handle_of(t), path)
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

## Elementwise `a - b`, float only, exact shapes.
def sub(device: Types.Device, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "sub")
  _ = try pair(a, b, "sub")
  ha = try handle_of(a)
  hb = try handle_of(b)
  one(device, a.dtype, a.shape) do Raw.sub(ha, hb) end
end

## Elementwise `a * b`, float only, exact shapes.
def mul(device: Types.Device, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "mul")
  _ = try pair(a, b, "mul")
  ha = try handle_of(a)
  hb = try handle_of(b)
  one(device, a.dtype, a.shape) do Raw.mul(ha, hb) end
end

## Elementwise `a / b`, float only, exact shapes; a zero divisor is an
## `error`, as in the reference, never an infinity the next op computes on.
def div(device: Types.Device, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "div")
  _ = try pair(a, b, "div")
  ha = try handle_of(a)
  hb = try handle_of(b)
  one(device, a.dtype, a.shape) do Raw.div(ha, hb) end
end

## Elementwise `-a`, float only.
def neg(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "neg")
  _ = try float_only(t, "neg")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.neg(h) end
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

## `a . bᵀ` for `a` `[rows, inner]` and `b` `[cols, inner]`, float only: `b` is read as it is, where `matmul` of a
## transposed `b` would copy it.
def linear(device: Types.Device, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "linear")
  _ = try same_dtype(a, b, "linear")
  _ = try float_only(a, "linear")
  return error("linear needs rank-2 tensors, got #{a.shape} and #{b.shape}") if a.shape.size != 2 or b.shape.size != 2

  return error("linear needs #{a.shape[1]} in the inner dimension, got #{b.shape[1]}") if a.shape[1] != b.shape[1]

  ha = try handle_of(a)
  hb = try handle_of(b)
  one(device, a.dtype, [a.shape[0], b.shape[0]]) do Raw.linear(ha, hb) end
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

## `sqrt(x)`, elementwise, float only; a non-finite input or one below zero
## is an `error`.
def sqrt(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "sqrt")
  _ = try float_only(t, "sqrt")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.sqrt(h) end
end

## `base ** exponent`, elementwise, float only, exact shapes; a non-finite
## input or a base at or below zero is an `error`, as in the reference.
def pow(device: Types.Device, base: Types.Tensor, exponent: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "pow")
  _ = try pair(base, exponent, "pow")
  hb = try handle_of(base)
  he = try handle_of(exponent)
  one(device, base.dtype, base.shape) do Raw.pow(hb, he) end
end

## Sigmoid, elementwise, float only; a non-finite input is an `error`.
def sigmoid(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "sigmoid")
  _ = try float_only(t, "sigmoid")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.sigmoid(h) end
end

## Tanh, elementwise, float only; a non-finite input is an `error`.
def tanh(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "tanh")
  _ = try float_only(t, "tanh")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.tanh(h) end
end

## The tanh approximation of GELU, float only; a non-finite input is an
## `error`.
def gelu_tanh_approx(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "gelu_tanh_approx")
  _ = try float_only(t, "gelu_tanh_approx")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.gelu_tanh_approx(h) end
end

## The exact (erf) GELU, float only; a non-finite input is an `error`.
def gelu_erf(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "gelu_erf")
  _ = try float_only(t, "gelu_erf")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.gelu_erf(h) end
end

## `a == b` elementwise as `Bool`, float or int dtypes, exact shapes.
def eq(device: Types.Device, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "eq")
  _ = try comparable(a, b, "eq")
  ha = try handle_of(a)
  hb = try handle_of(b)
  one(device, Types.Dtype.Bool, a.shape) do Raw.eq(ha, hb) end
end

## `a < b` elementwise as `Bool`, float or int dtypes, exact shapes.
def lt(device: Types.Device, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "lt")
  _ = try comparable(a, b, "lt")
  ha = try handle_of(a)
  hb = try handle_of(b)
  one(device, Types.Dtype.Bool, a.shape) do Raw.lt(ha, hb) end
end

## `cond ? a : b` elementwise: `cond` is `Bool`, `a` and `b` share shape and
## dtype, every shape exact.
def where(device: Types.Device, cond: Types.Tensor, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "where")
  _ = try same_dtype(a, b, "where")
  return error("where needs a Bool condition, got #{TorchDtypes.name(cond.dtype)}") if cond.dtype != Types.Dtype.Bool
  return error("where needs a condition of shape #{a.shape}, got #{cond.shape}") if cond.shape != a.shape
  hc = try handle_of(cond)
  ha = try handle_of(a)
  hb = try handle_of(b)
  one(device, a.dtype, a.shape) do Raw.where(hc, ha, hb) end
end

## Elementwise clamp to `[lo, hi]`, float only.
def clamp(device: Types.Device, t: Types.Tensor, lo: Float, hi: Float) -> Result[Types.Tensor]
  _ = try ready(device, "clamp")
  return error("clamp needs lo <= hi, got #{lo} and #{hi}") if lo > hi
  _ = try float_only(t, "clamp")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.clamp(h, lo, hi) end
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

## The largest value per reduced block, float only; an empty tensor is an
## `error`, as in the reference.
def max(
  device: Types.Device,
  t: Types.Tensor,
  axes: Optional[List[Int]] = none,
  keep_dims: Bool = false,
) -> Result[Types.Tensor]
  _ = try ready(device, "max")
  _ = try float_only(t, "max")
  axs = try TorchShapes.normalize_axes(axes, t.shape.size, "max")
  h = try handle_of(t)
  one(device, t.dtype, TorchShapes.reduced_shape(t.shape, axs, keep_dims)) do Raw.max(h, axs, keep_dims) end
end

## The first position of the largest value per reduced block, as `I64`, with
## the tie rule of the reference: the lowest position wins.
def argmax(
  device: Types.Device,
  t: Types.Tensor,
  axes: Optional[List[Int]] = none,
  keep_dims: Bool = false,
) -> Result[Types.Tensor]
  _ = try ready(device, "argmax")
  _ = try float_only(t, "argmax")
  axs = try TorchShapes.normalize_axes(axes, t.shape.size, "argmax")
  h = try handle_of(t)
  one(device, Types.Dtype.I64, TorchShapes.reduced_shape(t.shape, axs, keep_dims)) do Raw.argmax(h, axs, keep_dims) end
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

## Drops size-1 dimensions: every one for `none`, or the named `axis`, which
## must be `1`.
def squeeze(device: Types.Device, t: Types.Tensor, axis: Optional[Int]) -> Result[Types.Tensor]
  _ = try ready(device, "squeeze")
  axs = case axis
    some(ax) ->
      return error("squeeze needs an axis in 0...#{t.shape.size}, got #{ax}") if ax < 0 or ax >= t.shape.size
      return error("squeeze needs a dimension of size 1 at #{ax}, got #{t.shape[ax]}") if t.shape[ax] != 1
      [ax]
    none -> []
  end

  h = try handle_of(t)
  one(device, t.dtype, TorchShapes.squeezed_shape(t.shape, axs)) do Raw.squeeze(h, axs) end
end

## Inserts a size-1 dimension at `axis`.
def unsqueeze(device: Types.Device, t: Types.Tensor, axis: Int) -> Result[Types.Tensor]
  _ = try ready(device, "unsqueeze")
  return error("unsqueeze needs an axis in 0...#{t.shape.size}, got #{axis}") if axis < 0 or axis > t.shape.size
  h = try handle_of(t)
  one(device, t.dtype, TorchShapes.unsqueezed_shape(t.shape, axis)) do Raw.unsqueeze(h, axis) end
end

## The `[starts[i], ends[i])` window per axis (no steps in M1).
def slice(device: Types.Device, t: Types.Tensor, starts: List[Int], ends: List[Int]) -> Result[Types.Tensor]
  _ = try ready(device, "slice")
  rank = t.shape.size
  return error("slice needs one start per axis, got #{starts} for #{t.shape}") if starts.size != rank
  return error("slice needs one end per axis, got #{ends} for #{t.shape}") if ends.size != rank
  shape: List[Int] = []
  for i in 0...rank
    if starts[i] < 0 or ends[i] < starts[i] or ends[i] > t.shape[i]
      return error("slice needs 0 <= start <= end <= #{t.shape[i]}, got #{starts[i]}..#{ends[i]}")
    end

    shape = shape.push(ends[i] - starts[i])
  end

  h = try handle_of(t)
  one(device, t.dtype, shape) do Raw.slice(h, starts, ends) end
end

## Joins `tensors` along `axis`: one rank, one dtype, every dimension but
## `axis` equal. The native call joins two, so the fold is here: concatenation
## along one axis is associative. One tensor is a `copy`, so the answer never
## aliases an argument — the reference builds a new list for it too.
def concat(device: Types.Device, tensors: List[Types.Tensor], axis: Int) -> Result[Types.Tensor]
  _ = try ready(device, "concat")
  return error("concat needs at least one tensor") if tensors.empty?
  first = tensors[0]
  rank = first.shape.size
  return error("concat needs an axis in 0...#{rank}, got #{axis}") if axis < 0 or axis >= rank
  return copy(device, first) if tensors.size == 1
  joined = first
  i = 1
  while i < tensors.size
    joined = try concat_two(device, joined, tensors[i], axis)
    i += 1
  end

  ok(joined)
end

## Stacks `tensors` along a new `axis`: every shape and dtype equal, through
## the same `unsqueeze` + `concat` the reference uses.
def stack(device: Types.Device, tensors: List[Types.Tensor], axis: Int) -> Result[Types.Tensor]
  _ = try ready(device, "stack")
  return error("stack needs at least one tensor") if tensors.empty?
  first = tensors[0]
  return error("stack needs an axis in 0...#{first.shape.size}, got #{axis}") if axis < 0 or axis > first.shape.size
  widened: List[Types.Tensor] = []
  for t in tensors
    return error("stack needs one shape, got #{first.shape} and #{t.shape}") if t.shape != first.shape
    widened = widened.push(try unsqueeze(device, t, axis))
  end

  concat(device, widened, axis)
end

## Views `t` as `shape`: trailing axes align, size-1 dimensions widen.
def broadcast_to(device: Types.Device, t: Types.Tensor, shape: List[Int]) -> Result[Types.Tensor]
  _ = try ready(device, "broadcast_to")
  _ = try TorchShapes.volume(shape)
  rank = t.shape.size
  return error("broadcast_to cannot drop axes, got #{t.shape} into #{shape}") if shape.size < rank
  lead = shape.size - rank
  for i in 0...rank
    dim = t.shape[i]
    want = shape[lead + i]
    return error("broadcast_to needs #{dim} to stay #{dim} or widen from 1, got #{want}") if dim != want and dim != 1
  end

  h = try handle_of(t)
  one(device, t.dtype, shape) do Raw.broadcast_to(h, shape) end
end

## Rows of `source` at integer `indices`: `indices.shape ++ source.shape[1:]`.
def gather(device: Types.Device, source: Types.Tensor, indices: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "gather")
  return error("gather needs a rank >= 1 source, got #{source.shape}") if source.shape.size < 1
  if !TorchDtypes.is_int_dtype(indices.dtype)
    return error("gather needs integer indices, got #{TorchDtypes.name(indices.dtype)}")
  end

  hs = try handle_of(source)
  hi = try handle_of(indices)
  one(device, source.dtype, TorchShapes.gathered_shape(source.shape, indices.shape)) do Raw.gather(hs, hi) end
end

## The inverse of `gather`: `updates` added into `rows` rows at integer
## `indices`; the answer is `[rows] ++ updates.shape[indices.rank:]`.
def scatter_add(device: Types.Device, indices: Types.Tensor, updates: Types.Tensor, rows: Int) -> Result[Types.Tensor]
  _ = try ready(device, "scatter_add")
  return error("scatter_add needs a row count of 0 or more, got #{rows}") if rows < 0
  if !TorchDtypes.is_int_dtype(indices.dtype)
    return error("scatter_add needs integer indices, got #{TorchDtypes.name(indices.dtype)}")
  end

  if !TorchDtypes.is_float_dtype(updates.dtype)
    return error("scatter_add needs float updates, got #{TorchDtypes.name(updates.dtype)}")
  end

  if updates.shape.take(indices.shape.size) != indices.shape
    return error("scatter_add needs updates of shape #{indices.shape} followed by a row, got #{updates.shape}")
  end

  hi = try handle_of(indices)
  hu = try handle_of(updates)
  shape = TorchShapes.scattered_shape(rows, indices.shape, updates.shape)
  one(device, updates.dtype, shape) do Raw.scatter_add(hi, hu, rows) end
end

## Reverses every axis: rank 2 swaps rows and columns, a rank below 2 is
## unchanged. A true axis permutation is derived from `slice`/`stack` later.
def transpose(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "transpose")
  h = try handle_of(t)
  one(device, t.dtype, t.shape.reverse) do Raw.transpose(h) end
end

## The axes of `t` in the order `axes` names, as a new row-major tensor.
def permute(device: Types.Device, t: Types.Tensor, axes: List[Int]) -> Result[Types.Tensor]
  _ = try ready(device, "permute")
  h = try handle_of(t)
  one(device, t.dtype, Types.permuted_shape(t.shape, axes)) do Raw.permute(h, axes) end
end

## Scaled dot-product attention (`at::scaled_dot_product_attention`); the
## kv heads are repeated up to the query heads first, since libtorch 2.4 has no
## grouped-query form.
def attention(
  device: Types.Device,
  q: Types.Tensor,
  k: Types.Tensor,
  v: Types.Tensor,
  mask: Optional[Types.Tensor],
  scale: Float,
) -> Result[Types.Tensor]
  _ = try ready(device, "attention")
  hq = try handle_of(q)
  hk = try handle_of(k)
  hv = try handle_of(v)
  shape = q.shape.take(3).push(v.shape[3])
  case mask
    some(m) ->
      hm = try handle_of(m)
      one(device, q.dtype, shape) do Raw.attention_masked(hq, hk, hv, hm, scale) end
    none -> one(device, q.dtype, shape) do Raw.attention(hq, hk, hv, scale) end
  end
end

## Rotary position embedding, as a composite of libtorch ops over cosine and
## sine tables made in doubles.
def rope(
  device: Types.Device,
  x: Types.Tensor,
  positions: Types.Tensor,
  base: Float,
  dims: Int,
  neox: Bool,
  inverse: Bool,
) -> Result[Types.Tensor]
  _ = try ready(device, "rope")
  hx = try handle_of(x)
  hp = try handle_of(positions)
  one(device, x.dtype, x.shape) do Raw.rope(hx, hp, base, dims, neox, inverse) end
end

## `t` in `dtype`: float to int truncates towards zero, anything nonzero
## becomes `true`, an int width that cannot hold a value is an `error`.
def to_dtype(device: Types.Device, t: Types.Tensor, dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, "to_dtype")
  h = try handle_of(t)
  one(device, dtype, t.shape) do Raw.to_dtype(h, TorchDtypes.code(dtype)) end
end

## The identity move: a copy that shares no buffer with `t`.
def copy(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "copy")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.copy(h) end
end

## `t` in row-major order: copied when libtorch holds a view, so the readback
## is the order the reference produces.
def contiguous(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  _ = try ready(device, "contiguous")
  h = try handle_of(t)
  one(device, t.dtype, t.shape) do Raw.contiguous(h) end
end

## The cross-correlation of `input` with `weight` over the last axis, moved by
## `stride`, with `padding` zeros on each side.
def conv1d(
  device: Types.Device,
  input: Types.Tensor,
  weight: Types.Tensor,
  stride: Int,
  padding: Int,
) -> Result[Types.Tensor]
  _ = try ready(device, "conv1d")
  _ = try same_dtype(input, weight, "conv1d")
  _ = try float_only(input, "conv1d")
  _ = try float_only(weight, "conv1d")
  rank = input.shape.size
  return error("conv1d needs a rank-2 or rank-3 input, got #{input.shape}") if rank < 2 or rank > 3
  return error("conv1d needs a rank-3 weight, got #{weight.shape}") if weight.shape.size != 3
  return error("conv1d needs stride >= 1, got #{stride}") if stride < 1
  return error("conv1d needs padding >= 0, got #{padding}") if padding < 0
  channels = input.shape[rank - 2]
  return error("conv1d needs #{channels} input channels, got #{weight.shape[1]}") if weight.shape[1] != channels
  length = input.shape[rank - 1]
  kernel = weight.shape[2]
  if length + 2 * padding < kernel
    return error("conv1d needs a window of #{kernel} positions in #{length} padded to #{length + 2 * padding}")
  end

  hi = try handle_of(input)
  hw = try handle_of(weight)
  out_shape = TorchShapes.conv1d_output_shape(input.shape, weight.shape[0], kernel, stride, padding)
  one(device, input.dtype, out_shape) do Raw.conv1d(hi, hw, stride, padding) end
end

## The mean over a window of `kernel` positions, moved by `stride`, with
## `padding` zeros on each side and a divisor of `kernel`.
def avg_pool1d(
  device: Types.Device,
  input: Types.Tensor,
  kernel: Int,
  stride: Int,
  padding: Int,
) -> Result[Types.Tensor]
  _ = try ready(device, "avg_pool1d")
  _ = try float_only(input, "avg_pool1d")
  rank = input.shape.size
  return error("avg_pool1d needs a rank-2 or rank-3 input, got #{input.shape}") if rank < 2 or rank > 3
  return error("avg_pool1d needs kernel >= 1, got #{kernel}") if kernel < 1
  return error("avg_pool1d needs stride >= 1, got #{stride}") if stride < 1
  return error("avg_pool1d needs padding >= 0, got #{padding}") if padding < 0
  if padding > kernel / 2
    return error("avg_pool1d needs padding <= #{kernel / 2} for a kernel of #{kernel}, got #{padding}")
  end

  length = input.shape[rank - 1]
  if length + 2 * padding < kernel
    return error("avg_pool1d needs a window of #{kernel} positions in #{length} padded to #{length + 2 * padding}")
  end

  h = try handle_of(input)
  out_shape = TorchShapes.avg_pool1d_output_shape(input.shape, kernel, stride, padding)
  one(device, input.dtype, out_shape) do Raw.avg_pool1d(h, kernel, stride, padding) end
end

private

## The native handle of `t`: the `Native` arm. A `Plain` tensor has no native
## handle, and asking this backend to read one is a `Result` error, not a
## halt: it means a tensor of the reference backend was labelled `Torch`.
def handle_of(t: Types.Tensor) -> Result[Raw.TensorHandle]
  case t.handle
    Types.Handle.Native(h) -> ok(h)
    Types.Handle.Plain(_) -> error("a Plain tensor has no native handle (backend #{Types.backend_name(t.backend)})")
    Types.Handle.Traced(_) -> error("a traced tensor has no native handle")
    Types.Handle.Ggml(_) -> error("a ggml tensor has no torch handle")
    Types.Handle.Xla(_) -> error("an XLA tensor has no torch handle")
  end
end

## The one device libtorch's CPU release serves, and the module it needs.
def ready(device: Types.Device, op: String) -> Result[Bool]
  case device
    Types.Device.Cpu -> usable(op)
    Types.Device.Vulkan(i) -> error("#{op}: the torch backend runs on Cpu only, got Vulkan(#{i})")
  end
end

## The native module is loaded: a checkout without libtorch says so here,
## before any call would halt with `RUNTIME-NATIVE-LOAD`. The message says
## where to look.
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

## A tensor of `shape` with every element `x`, made natively: no list of `x`
## crosses, so the element count has no cap.
def filled(op: String, device: Types.Device, shape: List[Int], dtype: Types.Dtype, x: Float) -> Result[Types.Tensor]
  _ = try usable(op)
  _ = try TorchShapes.volume(shape)
  Native.arena do
    h = try Raw.fill(shape, x, TorchDtypes.code(dtype))
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

## Two tensors that compare: one shape, one float or int dtype.
def comparable(a: Types.Tensor, b: Types.Tensor, op: String) -> Result[Bool]
  return error("#{op} needs two tensors of one shape, got #{a.shape} and #{b.shape}") if a.shape != b.shape
  _ = try same_dtype(a, b, op)
  if !TorchDtypes.is_float_dtype(a.dtype) and !TorchDtypes.is_int_dtype(a.dtype)
    return error("#{op} needs a float or int dtype, got #{TorchDtypes.name(a.dtype)}")
  end

  ok(true)
end

## Two tensors joined along `axis`: one rank, one dtype, every other dimension
## equal. `concat` folds these, and the native call re-checks them.
def concat_two(device: Types.Device, a: Types.Tensor, b: Types.Tensor, axis: Int) -> Result[Types.Tensor]
  _ = try same_dtype(a, b, "concat")
  return error("concat needs one rank, got #{a.shape} and #{b.shape}") if a.shape.size != b.shape.size
  for i in 0...a.shape.size
    if i != axis and a.shape[i] != b.shape[i]
      return error("concat needs dimension #{i} of size #{a.shape[i]}, got #{b.shape[i]}")
    end
  end

  ha = try handle_of(a)
  hb = try handle_of(b)
  one(device, a.dtype, TorchShapes.concatenated_shape(a.shape, b.shape, axis)) do Raw.concat(ha, hb, axis) end
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
