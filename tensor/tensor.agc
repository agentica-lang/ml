## The `tensor/` package: functional tensors over pluggable backends.
##
## The public API (`tensor/DESIGN.md` §§2–4, 7, 9). Every fallible op returns
## `Result` and reports `error`, never a halt; shape, dtype and backend
## mismatches are validated eagerly in these wrappers, before dispatch.
## Dispatch is by value (`t.backend`) through an exhaustive `case` over all
## four `Backend` arms (`tensor/DESIGN.md` §7): `Reference` runs, the other
## three answer `error("backend not available")` until their modules land
## (`torch/` is M2, `xla/`/`candle/` later).
##
## Rules this file enforces:
## - No operator overloading: `Tensor.add(a, b)` plus pipes, never `a + b`.
## - Device is explicit, never global: it rides in the `Backend` value.
## - No implicit broadcasting: every elementwise op needs exact shapes;
##   widen explicitly with `broadcast_to` first (the wrapper checks the
##   trailing-1 rule eagerly). Ranks, dimension sizes, dtype legality and
##   broadcast compatibility are all checked here (§9).
## - Dtypes: numeric compute is `F32`; `F16`/`F64` compute exactly in doubles
##   (a native `F32` backend will round on its side — see the PR body).
##   `Bool` masks and integer index tensors are first-class; `to_dtype`
##   truncates float to int, tests nonzero for `Bool`, and range-checks int
##   widths. Overflows and `NaN` are errors, never wrap-around.
## - Lifetimes (§8): the reference holds plain Agentica data and needs no
##   arena. Arenas are not faked for `Plain`: the first native backend
##   allocates its handles inside `Native.arena` per call/step instead.
## - There is no tracing and no `grad` in M1: every backward rule waits for
##   the graph of `tensor/DESIGN.md` §10 (`train/` is M4). Saying so here so
##   no caller mistakes a forward op for a differentiable one.
## - `conv1d`/`avg_pool1d` name the §3.1 surface but answer
##   `error("not implemented")`: no §3.2 op needs them, and §5 admits an op
##   only for a consumer that does.
## - The 45 names of §3.1 all exist below; the PR body lists which compute
##   and which are stubs, so the gap is explicit rather than hidden.

import Reference from "reference"
import Types from "types"

export Types.Backend
export Types.Data
export Types.Device
export Types.Dtype
export Types.Handle
export Types.Tensor
export Types.TopK
export Types.Values

## One step of the `beam_search` loop: the accumulated log-score and the
## tokens chosen so far.
struct Beam
  score: Float
  seq: List[Int]
end

## The shape of `t`: a field read, no backend call.
def shape(t: Types.Tensor) -> List[Int] = t.shape

## The dtype of `t`: a field read, no backend call.
def dtype(t: Types.Tensor) -> Types.Dtype = t.dtype

## The backend of `t`: a field read, no backend call.
def backend(t: Types.Tensor) -> Types.Backend = t.backend

## Creation: zeros of `shape`, `dtype` and `backend`.
def zeros(shape: List[Int], dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case backend
    Types.Backend.Reference -> Reference.zeros(shape, dtype)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Creation: ones of `shape`, `dtype` and `backend`.
def ones(shape: List[Int], dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case backend
    Types.Backend.Reference -> Reference.ones(shape, dtype)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Creation: `empty` is deterministic `zeros` over plain Agentica data
## (there is no uninitialized memory to expose; see `reference.agc`).
def empty(shape: List[Int], dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case backend
    Types.Backend.Reference -> Reference.empty(shape, dtype)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Creation: every element `value` (tagged for `dtype`).
def full(shape: List[Int], value: Types.Values, dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case backend
    Types.Backend.Reference -> Reference.full(shape, value, dtype)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Creation: zeros of the shape, dtype and backend of `t` (metadata only).
def like(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.like(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Creation: `start, start + step, ...` below `stop` (above for `step < 0`).
## Integer dtypes only: positions are indices (`tensor/DESIGN.md` §5).
def arange(start: Int, stop: Int, step: Int, dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  return error("arange needs an Int dtype, got #{Types.dtype_name(dtype)}") if !Types.is_int_dtype(dtype)
  return error("arange needs a nonzero step, got #{step}") if step == 0
  case backend
    Types.Backend.Reference -> Reference.arange(start, stop, step, dtype)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Creation: the one `value` as a rank-0 tensor.
def scalar_to_tensor(value: Types.Values, dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  case backend
    Types.Backend.Reference -> Reference.scalar_to_tensor(value, dtype)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Creation: row-major `data` of `shape`; the count must match.
def from_list(
  data: List[Types.Values],
  shape: List[Int],
  dtype: Types.Dtype,
  backend: Types.Backend,
) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  return error("from_list needs #{n} values for #{shape}, got #{data.size}") if data.size != n
  case backend
    Types.Backend.Reference -> Reference.from_list(data, shape, dtype)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Host readback: the values tagged with the tensor's dtype.
def to_list(t: Types.Tensor) -> Result[List[Types.Values]]
  case t.backend
    Types.Backend.Reference -> Reference.to_list(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `a + b`, float only, exact shapes.
def add(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(a, b, "add")
  _ = try same_shape(a, b, "add")
  _ = try same_dtype(a, b, "add")
  _ = try float_only(a, "add")
  case a.backend
    Types.Backend.Reference -> Reference.add(a, b)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `a - b`, float only, exact shapes.
def sub(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(a, b, "sub")
  _ = try same_shape(a, b, "sub")
  _ = try same_dtype(a, b, "sub")
  _ = try float_only(a, "sub")
  case a.backend
    Types.Backend.Reference -> Reference.sub(a, b)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `a * b`, float only, exact shapes.
def mul(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(a, b, "mul")
  _ = try same_shape(a, b, "mul")
  _ = try same_dtype(a, b, "mul")
  _ = try float_only(a, "mul")
  case a.backend
    Types.Backend.Reference -> Reference.mul(a, b)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `a / b`, float only, exact shapes; zero divisors are errors.
def div(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(a, b, "div")
  _ = try same_shape(a, b, "div")
  _ = try same_dtype(a, b, "div")
  _ = try float_only(a, "div")
  case a.backend
    Types.Backend.Reference -> Reference.div(a, b)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `-a`, float only.
def neg(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "neg")
  case t.backend
    Types.Backend.Reference -> Reference.neg(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `exp`, float only.
def exp(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "exp")
  case t.backend
    Types.Backend.Reference -> Reference.exp(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `log`, float only, `x <= 0` is an error.
def log(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "log")
  case t.backend
    Types.Backend.Reference -> Reference.log(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `sqrt`, float only, `x < 0` is an error.
def sqrt(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "sqrt")
  case t.backend
    Types.Backend.Reference -> Reference.sqrt(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `base ** exponent`, float only, exact shapes; `base <= 0`
## is an error (square with `mul`; see `reference.agc`).
def pow(base: Types.Tensor, exponent: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(base, exponent, "pow")
  _ = try same_shape(base, exponent, "pow")
  _ = try same_dtype(base, exponent, "pow")
  _ = try float_only(base, "pow")
  case base.backend
    Types.Backend.Reference -> Reference.pow(base, exponent)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `max(0, x)`, float only.
def relu(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "relu")
  case t.backend
    Types.Backend.Reference -> Reference.relu(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise sigmoid, float only.
def sigmoid(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "sigmoid")
  case t.backend
    Types.Backend.Reference -> Reference.sigmoid(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise tanh, float only.
def tanh(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "tanh")
  case t.backend
    Types.Backend.Reference -> Reference.tanh(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## The user-facing `gelu`: the `tanh` approximation from §3.1 (the `erf`
## form cannot be expressed over the primitives and is not offered).
def gelu_tanh_approx(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "gelu_tanh_approx")
  case t.backend
    Types.Backend.Reference -> Reference.gelu_tanh_approx(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `a == b` as `1.0`/`0.0`, float or int dtypes, exact shapes.
def eq(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(a, b, "eq")
  _ = try same_shape(a, b, "eq")
  _ = try same_dtype(a, b, "eq")
  _ = try float_or_int(a, "eq")
  case a.backend
    Types.Backend.Reference -> Reference.eq(a, b)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise `a < b` as `1.0`/`0.0`, float or int dtypes, exact shapes.
def lt(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(a, b, "lt")
  _ = try same_shape(a, b, "lt")
  _ = try same_dtype(a, b, "lt")
  _ = try float_or_int(a, "lt")
  case a.backend
    Types.Backend.Reference -> Reference.lt(a, b)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## `cond ? a : b` elementwise: `cond` is `Bool`, `a` and `b` share shape
## and dtype, every shape exact.
def where(cond: Types.Tensor, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(cond, a, "where")
  _ = try same_backend(cond, b, "where")
  _ = try same_shape(a, b, "where")
  _ = try same_dtype(a, b, "where")
  return error("where needs a Bool condition, got #{Types.dtype_name(cond.dtype)}") if cond.dtype != Types.Dtype.Bool
  return error("where needs a condition of shape #{a.shape}, got #{cond.shape}") if cond.shape != a.shape
  case a.backend
    Types.Backend.Reference -> Reference.where(cond, a, b)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Elementwise clamp to `[lo, hi]`, float only.
def clamp(t: Types.Tensor, lo: Float, hi: Float) -> Result[Types.Tensor]
  return error("clamp needs lo <= hi, got #{lo} and #{hi}") if lo > hi
  _ = try float_only(t, "clamp")
  case t.backend
    Types.Backend.Reference -> Reference.clamp(t, lo, hi)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Sums every element (`none`) or the named `axes`, float only.
def sum(t: Types.Tensor, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Types.Tensor]
  _ = try float_only(t, "sum")
  _ = try check_axes_option(axes, t.shape.size, "sum")
  case t.backend
    Types.Backend.Reference -> Reference.sum(t, axes, keep_dims)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## `sum` divided by the reduced count, float only.
def mean(t: Types.Tensor, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Types.Tensor]
  _ = try float_only(t, "mean")
  _ = try check_axes_option(axes, t.shape.size, "mean")
  case t.backend
    Types.Backend.Reference -> Reference.mean(t, axes, keep_dims)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## The largest value per reduced block, float only.
def max(t: Types.Tensor, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Types.Tensor]
  _ = try float_only(t, "max")
  _ = try check_axes_option(axes, t.shape.size, "max")
  case t.backend
    Types.Backend.Reference -> Reference.max(t, axes, keep_dims)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## The index of the largest value per reduced block, as `I64`.
def argmax(t: Types.Tensor, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Types.Tensor]
  _ = try float_only(t, "argmax")
  _ = try check_axes_option(axes, t.shape.size, "argmax")
  case t.backend
    Types.Backend.Reference -> Reference.argmax(t, axes, keep_dims)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## The rank-2 product `[rows, inner] . [inner, cols]`, float only.
def matmul(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(a, b, "matmul")
  _ = try same_dtype(a, b, "matmul")
  _ = try float_only(a, "matmul")
  return error("matmul needs rank-2 tensors, got #{a.shape} and #{b.shape}") if a.shape.size != 2 or b.shape.size != 2
  return error("matmul needs #{a.shape[1]} in the inner dimension, got #{b.shape[0]}") if a.shape[1] != b.shape[0]
  case a.backend
    Types.Backend.Reference -> Reference.matmul(a, b)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## The batched product `[B..., m, k] . [B..., k, n]`: one rank, one batch
## prefix (widen with `broadcast_to` first), float only.
def batch_matmul(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(a, b, "batch_matmul")
  _ = try same_dtype(a, b, "batch_matmul")
  _ = try float_only(a, "batch_matmul")
  return error("batch_matmul needs rank >= 2, got #{a.shape} and #{b.shape}") if a.shape.size < 2 or b.shape.size < 2
  return error("batch_matmul needs one rank, got #{a.shape} and #{b.shape}") if a.shape.size != b.shape.size
  rank = a.shape.size
  for i in 0...(rank - 2)
    return error("batch_matmul needs one batch shape, got #{a.shape} and #{b.shape}") if a.shape[i] != b.shape[i]
  end

  if a.shape[rank - 1] != b.shape[rank - 2]
    return error("batch_matmul needs #{a.shape[rank - 1]} in the inner dimension, got #{b.shape[rank - 2]}")
  end

  case a.backend
    Types.Backend.Reference -> Reference.batch_matmul(a, b)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## The transpose: reverses every axis (rank 2 swaps rows and columns).
## A true axis permutation is derived from `slice`/`stack` where needed.
def transpose(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> ok(Reference.transpose(t))
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## A tensor of `shape` with the same values: the volumes must match.
def reshape(t: Types.Tensor, shape: List[Int]) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case t.backend
    Types.Backend.Reference -> Reference.reshape(t, shape)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Drops size-1 dimensions: every one for `none`, or the named `axis`.
def squeeze(t: Types.Tensor, axis: Optional[Int] = none) -> Result[Types.Tensor]
  _ = try check_squeeze_axis(axis, t.shape, "squeeze")
  case t.backend
    Types.Backend.Reference -> Reference.squeeze(t, axis)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Inserts a size-1 dimension at `axis`.
def unsqueeze(t: Types.Tensor, axis: Int = 0) -> Result[Types.Tensor]
  return error("unsqueeze needs an axis in 0...#{t.shape.size}, got #{axis}") if axis < 0 or axis > t.shape.size
  case t.backend
    Types.Backend.Reference -> Reference.unsqueeze(t, axis)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## The `[starts[i], ends[i])` window per axis (no steps in M1).
def slice(t: Types.Tensor, starts: List[Int], ends: List[Int]) -> Result[Types.Tensor]
  _ = try check_slice(starts, ends, t.shape, "slice")
  case t.backend
    Types.Backend.Reference -> Reference.slice(t, starts, ends)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Joins `tensors` along `axis`: one shape but for `axis`, one dtype.
def concat(tensors: List[Types.Tensor], axis: Int) -> Result[Types.Tensor]
  return error("concat needs at least one tensor") if tensors.empty?
  first = tensors[0]
  rank = first.shape.size
  return error("concat needs an axis in 0...#{rank}, got #{axis}") if axis < 0 or axis >= rank
  for t in tensors
    _ = try same_backend(first, t, "concat")
  end

  case first.backend
    Types.Backend.Reference -> Reference.concat(tensors, axis)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Stacks `tensors` along a new `axis`: every shape and dtype equal.
def stack(tensors: List[Types.Tensor], axis: Int = 0) -> Result[Types.Tensor]
  return error("stack needs at least one tensor") if tensors.empty?
  first = tensors[0]
  return error("stack needs an axis in 0...#{first.shape.size}, got #{axis}") if axis < 0 or axis > first.shape.size
  for t in tensors
    _ = try same_backend(first, t, "stack")
  end

  case first.backend
    Types.Backend.Reference -> Reference.stack(tensors, axis)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Views `t` as `shape`: trailing axes align, size-1 dimensions widen.
def broadcast_to(t: Types.Tensor, shape: List[Int]) -> Result[Types.Tensor]
  _ = try check_broadcast(t.shape, shape, "broadcast_to")
  case t.backend
    Types.Backend.Reference -> Reference.broadcast_to(t, shape)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Rows of `source` at integer `indices`: `indices.shape ++ source.shape[1:]`.
def gather(source: Types.Tensor, indices: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(source, indices, "gather")
  return error("gather needs a rank >= 1 source, got #{source.shape}") if source.shape.size < 1
  if !Types.is_int_dtype(indices.dtype)
    return error("gather needs integer indices, got #{Types.dtype_name(indices.dtype)}")
  end

  case source.backend
    Types.Backend.Reference -> Reference.gather(source, indices)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Moves `t` to `dtype` (float to int truncates, anything nonzero is `true`).
def to_dtype(t: Types.Tensor, dtype: Types.Dtype) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.to_dtype(t, dtype)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Moves `t` to `device`: reference tensors live on `Cpu` only.
def to_device(t: Types.Tensor, device: Types.Device) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.to_device(t, device)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## The identity move: values are immutable, so a copy shares nothing.
def copy(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.copy(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## The identity move: the reference has no strides, every tensor is
## contiguous by construction.
def contiguous(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.contiguous(t)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Not in M1: a 1-D convolution stem waits for a consumer that needs it.
def conv1d(input: Types.Tensor, weight: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(input, weight, "conv1d")
  _ = try float_only(input, "conv1d")
  _ = try float_only(weight, "conv1d")
  case input.backend
    Types.Backend.Reference -> Reference.conv1d(input, weight)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Not in M1: `mean` over an explicit window covers the derived cases.
def avg_pool1d(input: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(input, "avg_pool1d")
  case input.backend
    Types.Backend.Reference -> Reference.avg_pool1d(input)
    Types.Backend.Torch(_) -> error("backend not available")
    Types.Backend.Xla(_) -> error("backend not available")
    Types.Backend.Candle(_) -> error("backend not available")
  end
end

## Derived: `softmax` over `axis` with max-subtraction, from §3.1 only.
def softmax(x: Types.Tensor, axis: Int) -> Result[Types.Tensor]
  rank = x.shape.size
  return error("softmax needs an axis in 0...#{rank}, got #{axis}") if axis < 0 or axis >= rank
  m = try max(x, some([axis]), true)
  shifted = try sub(x, try broadcast_to(m, x.shape))
  e = try exp(shifted)
  s = try sum(e, some([axis]), true)
  div(e, try broadcast_to(s, x.shape))
end

## Derived: `layer_norm` over the last axis, from §3.1 only (no affine).
def layer_norm(x: Types.Tensor, eps: Float) -> Result[Types.Tensor]
  return error("layer_norm needs eps > 0, got #{eps}") if eps <= 0.0
  rank = x.shape.size
  return error("layer_norm needs rank >= 1, got #{x.shape}") if rank < 1
  last = rank - 1
  mu = try mean(x, some([last]), true)
  centered = try sub(x, try broadcast_to(mu, x.shape))
  var = try mean(try mul(centered, centered), some([last]), true)
  eps_full = try full(x.shape, Types.Values.Float(eps), x.dtype, x.backend)
  denom = try sqrt(try add(try broadcast_to(var, x.shape), eps_full))
  div(centered, denom)
end

## Derived: `rms_norm` over the last axis, from §3.1 only (no affine).
def rms_norm(x: Types.Tensor, eps: Float) -> Result[Types.Tensor]
  return error("rms_norm needs eps > 0, got #{eps}") if eps <= 0.0
  rank = x.shape.size
  return error("rms_norm needs rank >= 1, got #{x.shape}") if rank < 1
  last = rank - 1
  ms = try mean(try mul(x, x), some([last]), true)
  eps_full = try full(x.shape, Types.Values.Float(eps), x.dtype, x.backend)
  denom = try sqrt(try add(try broadcast_to(ms, x.shape), eps_full))
  div(x, denom)
end

## Derived: the user-facing `gelu` is the `tanh` approximation (§3.2).
def gelu(x: Types.Tensor) -> Result[Types.Tensor] = gelu_tanh_approx(x)

## Derived: `silu(x) = x * sigmoid(x)`, from §3.1 only.
def silu(x: Types.Tensor) -> Result[Types.Tensor]
  x_sig = try sigmoid(x)
  mul(x, x_sig)
end

## Derived: the rank-1 product, literally `reshape` + `matmul`.
def dot(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(a, b, "dot")
  _ = try same_dtype(a, b, "dot")
  _ = try float_only(a, "dot")
  return error("dot needs rank-1 tensors, got #{a.shape} and #{b.shape}") if a.shape.size != 1 or b.shape.size != 1
  return error("dot needs one length, got #{a.shape} and #{b.shape}") if a.shape[0] != b.shape[0]
  n = a.shape[0]
  a2 = try reshape(a, [1, n])
  b2 = try reshape(b, [n, 1])
  c = try matmul(a2, b2)
  reshape(c, [])
end

## Derived: `x . w (+ b)`, with the bias widened by `broadcast_to`.
def linear(x: Types.Tensor, w: Types.Tensor, b: Optional[Types.Tensor] = none) -> Result[Types.Tensor]
  _ = try same_backend(x, w, "linear")
  _ = try same_dtype(x, w, "linear")
  _ = try float_only(x, "linear")
  return error("linear needs a rank-2 input, got #{x.shape}") if x.shape.size != 2
  return error("linear needs a rank-2 weight, got #{w.shape}") if w.shape.size != 2
  return error("linear needs #{x.shape[1]} in the inner dimension, got #{w.shape[0]}") if x.shape[1] != w.shape[0]
  y = try matmul(x, w)
  case b
    some(bias) ->
      _ = try same_backend(x, bias, "linear")
      _ = try same_dtype(x, bias, "linear")
      return error("linear needs a bias of shape [#{w.shape[1]}], got #{bias.shape}") if bias.shape != [w.shape[1]]
      add(y, try broadcast_to(bias, y.shape))
    none -> ok(y)
  end
end

## Derived: row lookup, `gather` with the rank the name promises.
def embedding(ids: Types.Tensor, table: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(ids, table, "embedding")
  return error("embedding needs integer ids, got #{Types.dtype_name(ids.dtype)}") if !Types.is_int_dtype(ids.dtype)
  _ = try float_only(table, "embedding")
  return error("embedding needs rank-1 ids, got #{ids.shape}") if ids.shape.size != 1
  return error("embedding needs a rank-2 table, got #{table.shape}") if table.shape.size != 2
  gather(table, ids)
end

## Derived: multi-head attention over §3.1 only, no fused kernel.
## `query`/`key`/`value` are `[batch, seq, width]` (`value` may differ in
## width); the heads split along the width without a transpose permutation:
## each `(batch, head)` pair runs rank-2 `matmul`/`softmax` on slices and the
## answers are joined with `concat`/`stack`. `mask`, when given, is an
## additive bias widened to `[batch, heads, q, k]`.
def multi_head_attention(
  query: Types.Tensor,
  key: Types.Tensor,
  value: Types.Tensor,
  heads: Int,
  mask: Optional[Types.Tensor] = none,
) -> Result[Types.Tensor]
  _ = try same_backend(query, key, "multi_head_attention")
  _ = try same_backend(query, value, "multi_head_attention")
  _ = try same_dtype(query, key, "multi_head_attention")
  _ = try same_dtype(query, value, "multi_head_attention")
  _ = try float_only(query, "multi_head_attention")
  if query.shape.size != 3 or key.shape.size != 3 or value.shape.size != 3
    return error("multi_head_attention needs rank-3 tensors, got #{query.shape}, #{key.shape} and #{value.shape}")
  end

  b = query.shape[0]
  sq = query.shape[1]
  d = query.shape[2]
  sk = key.shape[1]
  dk = key.shape[2]
  dv = value.shape[2]
  if key.shape[0] != b or value.shape[0] != b
    return error("multi_head_attention needs one batch, got #{b}, #{key.shape[0]} and #{value.shape[0]}")
  end

  if value.shape[1] != sk
    return error("multi_head_attention needs key length #{sk} to match value length #{value.shape[1]}")
  end

  return error("multi_head_attention needs key width #{d}, got #{dk}") if dk != d
  return error("multi_head_attention needs heads >= 1, got #{heads}") if heads < 1
  return error("multi_head_attention needs width #{d} divisible by #{heads} heads") if d % heads != 0
  return error("multi_head_attention needs value width #{dv} divisible by #{heads} heads") if dv % heads != 0
  if b < 1 or sq < 1 or sk < 1 or d < 1 or dv < 1
    return error("multi_head_attention needs nonempty dimensions, got #{query.shape}, #{key.shape} and #{value.shape}")
  end

  dh = d / heads
  dvh = dv / heads
  scale = 1.0 / Math.sqrt(dh.to_f)
  widened = case mask
    some(m) ->
      _ = try same_backend(query, m, "multi_head_attention")
      _ = try same_dtype(query, m, "multi_head_attention")
      _ = try float_only(m, "multi_head_attention")
      some(try broadcast_to(m, [b, heads, sq, sk]))
    none -> none
  end

  per_batch: List[Types.Tensor] = []
  for bb in 0...b
    head_ctx: List[Types.Tensor] = []
    for h in 0...heads
      qbh = try reshape(try slice(query, [bb, 0, h * dh], [bb + 1, sq, (h + 1) * dh]), [sq, dh])
      kbh = try reshape(try slice(key, [bb, 0, h * dh], [bb + 1, sk, (h + 1) * dh]), [sk, dh])
      scores = try matmul(qbh, try transpose(kbh))
      scaled = try mul(scores, try full(scores.shape, Types.Values.Float(scale), query.dtype, query.backend))
      biased = case widened
        some(w) -> try add(scaled, try reshape(try slice(w, [bb, h, 0, 0], [bb + 1, h + 1, sq, sk]), [sq, sk]))
        none -> scaled
      end

      probs = try softmax(biased, 1)
      vbh = try reshape(try slice(value, [bb, 0, h * dvh], [bb + 1, sk, (h + 1) * dvh]), [sk, dvh])
      head_ctx = head_ctx.push(try matmul(probs, vbh))
    end

    rows: List[Types.Tensor] = []
    for i in 0...sq
      parts: List[Types.Tensor] = []
      for h in 0...heads
        parts = parts.push(try reshape(try slice(head_ctx[h], [i, 0], [i + 1, dvh]), [dvh]))
      end

      rows = rows.push(try concat(parts, 0))
    end

    per_batch = per_batch.push(try stack(rows, 0))
  end

  stack(per_batch, 0)
end

## Derived: one greedy decode step: optional temperature scaling, `softmax`,
## then `argmax` to an `I64` rank-0 tensor.
def greedy_step(logits: Types.Tensor, temperature: Optional[Float] = none) -> Result[Types.Tensor]
  _ = try float_only(logits, "greedy_step")
  return error("greedy_step needs rank-1 logits, got #{logits.shape}") if logits.shape.size != 1
  scaled = case temperature
    some(t) ->
      return error("greedy_step needs temperature > 0, got #{t}") if t <= 0.0
      try mul(logits, try full(logits.shape, Types.Values.Float(1.0 / t), logits.dtype, logits.backend))
    none -> logits
  end

  probs = try softmax(scaled, 0)
  argmax(probs, none, false)
end

## Derived: `dropout` at inference is the identity (kept for `nn/` API
## symmetry); `p` is still checked to lie in `0..1`.
def dropout(x: Types.Tensor, p: Float) -> Result[Types.Tensor]
  return error("dropout needs p in 0..1, got #{p}") if p < 0.0 or p > 1.0
  ok(x)
end

## Derived: the `k` largest values of a rank-1 tensor, descending, with
## their positions as `I64`. Ties keep an unspecified order.
def top_k(x: Types.Tensor, k: Int) -> Result[Types.TopK]
  _ = try float_only(x, "top_k")
  return error("top_k needs rank-1 input, got #{x.shape}") if x.shape.size != 1
  n = x.shape[0]
  return error("top_k needs 1 <= k <= #{n}, got #{k}") if k < 1 or k > n
  raw = try to_list(x)
  vals: List[Float] = []
  for v in raw
    case v
      Types.Values.Float(f) -> vals = vals.push(f)
      _ -> return error("top_k reads a float tensor")
    end
  end

  order = (0...n).to_a.sort_by do |i: Int| 0.0 - vals[i] end
  chosen = order.take(k)
  top_vals: List[Types.Values] = []
  top_idx: List[Types.Values] = []
  for i in chosen
    top_vals = top_vals.push(Types.Values.Float(vals[i]))
    top_idx = top_idx.push(Types.Values.Int(i))
  end

  values = try from_list(top_vals, [k], x.dtype, x.backend)
  indices = try from_list(top_idx, [k], Types.Dtype.I64, x.backend)
  ok(Types.TopK(values: values, indices: indices))
end

## Derived: fixed-step beam search over `[steps, vocab]` logits, an Agentica
## loop over `top_k`-shaped sorting (no native `top_k` in v1). The answer is
## the `width` winning token rows as `I64 [width, steps]`; there is no end
## token and no length penalty in M1 (fixed steps, like the §3.2 loop).
def beam_search(logits: Types.Tensor, width: Int) -> Result[Types.Tensor]
  _ = try float_only(logits, "beam_search")
  return error("beam_search needs rank-2 logits, got #{logits.shape}") if logits.shape.size != 2
  steps = logits.shape[0]
  vocab = logits.shape[1]
  return error("beam_search needs at least one step, got #{logits.shape}") if steps < 1
  return error("beam_search needs width >= 1, got #{width}") if width < 1
  return error("beam_search needs width <= vocabulary #{vocab}, got #{width}") if width > vocab
  raw = try to_list(logits)
  table: List[Float] = []
  for v in raw
    case v
      Types.Values.Float(f) -> table = table.push(f)
      _ -> return error("beam_search reads a float tensor")
    end
  end

  beams: List[Beam] = [Beam(score: 0.0, seq: [])]
  for t in 0...steps
    cands: List[Beam] = []
    for beam in beams
      for v in 0...vocab
        cands = cands.push(Beam(score: beam.score + table[t * vocab + v], seq: beam.seq.push(v)))
      end
    end

    ranked = cands.sort_by do |c: Beam| 0.0 - c.score end
    beams = ranked.take(width)
  end

  out: List[Types.Values] = []
  for beam in beams
    for v in beam.seq
      out = out.push(Types.Values.Int(v))
    end
  end

  from_list(out, [width, steps], Types.Dtype.I64, logits.backend)
end

private

## Two tensors share their backend.
def same_backend(a: Types.Tensor, b: Types.Tensor, op: String) -> Result[Bool]
  if a.backend != b.backend
    return error(
      "#{op} needs tensors on one backend, got #{Types.backend_name(a.backend)} and #{Types.backend_name(b.backend)}",
    )
  end

  ok(true)
end

## Two tensors share their shape.
def same_shape(a: Types.Tensor, b: Types.Tensor, op: String) -> Result[Bool]
  return error("#{op} needs two tensors of one shape, got #{a.shape} and #{b.shape}") if a.shape != b.shape
  ok(true)
end

## Two tensors share their dtype.
def same_dtype(a: Types.Tensor, b: Types.Tensor, op: String) -> Result[Bool]
  if a.dtype != b.dtype
    return error("#{op} needs two #{Types.dtype_name(a.dtype)} tensors, got #{Types.dtype_name(b.dtype)}")
  end

  ok(true)
end

## `t` computes: float dtypes only.
def float_only(t: Types.Tensor, op: String) -> Result[Bool]
  return error("#{op} needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  ok(true)
end

## `t` compares: float or int dtypes.
def float_or_int(t: Types.Tensor, op: String) -> Result[Bool]
  if !Types.is_float_dtype(t.dtype) and !Types.is_int_dtype(t.dtype)
    return error("#{op} needs a float or int dtype, got #{Types.dtype_name(t.dtype)}")
  end

  ok(true)
end

## `none` is every axis; entries name distinct axes of a rank-`rank` tensor.
def check_axes_option(axes: Optional[List[Int]], rank: Int, op: String) -> Result[Bool]
  case axes
    some(given) -> check_axes(given, rank, op)
    none -> ok(true)
  end
end

## Every entry of `axes` names a distinct axis of a rank-`rank` tensor.
def check_axes(axes: List[Int], rank: Int, op: String) -> Result[Bool]
  for ax in axes
    return error("#{op} needs axes in 0...#{rank}, got #{ax}") if ax < 0 or ax >= rank
  end

  return error("#{op} names an axis twice, got #{axes}") if axes.uniq.size != axes.size
  ok(true)
end

## The named axis exists and is `1`, or every axis when `none`.
def check_squeeze_axis(axis: Optional[Int], shape: List[Int], op: String) -> Result[Bool]
  case axis
    some(ax) ->
      return error("#{op} needs an axis in 0...#{shape.size}, got #{ax}") if ax < 0 or ax >= shape.size
      return error("#{op} needs a dimension of size 1 at #{ax}, got #{shape[ax]}") if shape[ax] != 1
      ok(true)
    none -> ok(true)
  end
end

## One window per axis, every window inside its dimension.
def check_slice(starts: List[Int], ends: List[Int], shape: List[Int], op: String) -> Result[Bool]
  rank = shape.size
  return error("#{op} needs one start per axis, got #{starts} for #{shape}") if starts.size != rank
  return error("#{op} needs one end per axis, got #{ends} for #{shape}") if ends.size != rank
  for i in 0...rank
    if starts[i] < 0 or ends[i] < starts[i] or ends[i] > shape[i]
      return error("#{op} needs 0 <= start <= end <= #{shape[i]}, got #{starts[i]}..#{ends[i]}")
    end
  end

  ok(true)
end

## Trailing axes align, size-1 dimensions widen, ranks never shrink.
def check_broadcast(from: List[Int], to: List[Int], op: String) -> Result[Bool]
  _ = try Types.volume(to)
  rank = from.size
  out_rank = to.size
  return error("#{op} cannot drop axes, got #{from} into #{to}") if out_rank < rank
  lead = out_rank - rank
  for i in 0...rank
    dim = from[i]
    want = to[lead + i]
    return error("#{op} needs #{dim} to stay #{dim} or widen from 1, got #{want}") if dim != want and dim != 1
  end

  ok(true)
end
