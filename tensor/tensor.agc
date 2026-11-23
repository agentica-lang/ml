## The `tensor/` package: functional tensors over pluggable backends.
##
## The public API (`tensor/DESIGN.md` §§2–4, 7, 9). Every fallible op returns
## `Result` and reports `error`, never a halt; shape, dtype and backend
## mismatches are validated eagerly in these wrappers, before dispatch.
## Dispatch is by value (`t.backend`) through an exhaustive `case` over all
## three `Backend` arms (`tensor/DESIGN.md` §7): `Reference` runs in Agentica,
## `Torch` runs on libtorch through `src/torch.agc` and `src/native.agc`, and
## `Trace` records a node of a traced graph after the same checks
## (`src/graph.agc`, `trace.agc`, §10). With libtorch absent (the native
## module is `optional`) the `Torch` arm answers `error("backend not
## available")` too, and `available/1` says so before a call does.
##
## Rules this file enforces:
## - No operator overloading: `Tensor.add(a, b)` plus pipes, never `a + b`.
## - Device is explicit, never global: it rides in the `Backend` value.
## - No implicit broadcasting: every elementwise op needs exact shapes;
##   widen explicitly with `broadcast_to` first (the wrapper checks the
##   trailing-1 rule eagerly). Ranks, dimension sizes, dtype legality,
##   backend agreement and broadcast compatibility are checked in the wrapper
##   before dispatch (§9); element domains (divisors, `log`/`sqrt`/`pow`
##   inputs, index bounds) and value-count matches live in the backend that
##   owns the storage, which is the only place that can see them.
## - Reduction `axes` are a set: `none` is every axis, and a named list is
##   sorted ascending in the wrapper before dispatch. `argmax` numbers the
##   position inside a reduced block by that order, so sorting is what keeps
##   the backends breaking a tie between two axes on the same element.
## - Dtypes: numeric compute is `F32`; `F16`/`F64` compute exactly in doubles
##   (a native `F32` backend rounds on its side, which is the tolerance the
##   conformance suite quotes).
##   `Bool` masks and integer index tensors are first-class; `to_dtype`
##   truncates float to int, tests nonzero for `Bool`, and range-checks int
##   widths. Int overflows are errors, never wrap-around.
## - Finite values in, errors out, never a halt or a hang: non-finite storage
##   (`NaN`, `+-inf`) is refused with `error` at every entry point that could
##   halt or hang on it (`exp`, `log`, `sqrt`, `pow`, `sigmoid`, `tanh`,
##   `gelu_tanh_approx`, `top_k`). Saturating finite inputs stays legitimate
##   (`exp` of a large value is `+inf`), but non-finite values never enter
##   the transcendental core. This is a backend-agnostic input rule: the
##   wrappers document it, and each backend enforces it before touching its
##   own numbers (the reference enforces it in-backend, since a wrapper
##   cannot see stored values without a readback). `softmax`, `gelu`,
##   `silu`, `layer_norm`, `rms_norm` and `greedy_step` inherit the rule
##   through the primitives they are built from; `greedy_step` additionally
##   refuses a temperature so small that `1/t` overflows.
## - Lifetimes (§8): the reference holds plain Agentica data and needs no
##   arena. The native backend allocates its handles inside a `Native.arena`
##   per call, and a caller that runs a step wraps the step in one arena of its
##   own (`src/torch.agc`); arenas are never faked for `Plain`.
## - Every backend is forward-only. On `Trace` tensors an op records a node;
##   `trace.agc` derives the gradient graph and runs it (`tensor/DESIGN.md`
##   §10). Every op records; one with no backward rule yet makes
##   `gradients` answer an error naming it.
## - `conv1d`/`avg_pool1d` take only what `tensor/DESIGN.md` §3.1 admits
##   (stride and padding 1-D, no dilation and no groups); the wrapper
##   validates their window arithmetic before dispatch, so both backends
##   accept and refuse the same shapes.
## - The 45 names of §3.1 all exist below. `Reference` computes all of them;
##   the `Torch` arm computes everything but `gelu_tanh_approx`,
##   `batch_matmul` and `to_device`, and answers
##   `error("<op> is not implemented on the torch backend yet")` for those,
##   so the gap is a `Result` error naming the op, never a wrong number
##   (`README.md`, "What is implemented").

import Graph from "src/graph"
import Reference from "src/reference"
import Torch from "src/torch"
import Types from "src/types"

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

## True when `backend` can run on this machine: `Reference` always can, `Torch`
## when the native libtorch module is built (`[native.torch] optional = true` in
## `agentica.toml`). A caller that needs to work on both asks this instead of catching the
## `error("backend not available")` an op would answer. `Trace` is never
## available: it records ops and computes nothing (`trace.agc`).
def available(backend: Types.Backend) -> Bool
  case backend
    Types.Backend.Reference -> true
    Types.Backend.Torch(_) -> Torch.available()
    Types.Backend.Trace -> false
  end
end

## Creation: zeros of `shape`, `dtype` and `backend`.
def zeros(shape: List[Int], dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case backend
    Types.Backend.Reference -> Reference.zeros(shape, dtype)
    Types.Backend.Torch(device) -> Torch.zeros(device, shape, dtype)
    Types.Backend.Trace -> Graph.filled(shape, 0.0, dtype)
  end
end

## Creation: ones of `shape`, `dtype` and `backend`.
def ones(shape: List[Int], dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case backend
    Types.Backend.Reference -> Reference.ones(shape, dtype)
    Types.Backend.Torch(device) -> Torch.ones(device, shape, dtype)
    Types.Backend.Trace -> Graph.filled(shape, 1.0, dtype)
  end
end

## Creation: `empty` is deterministic `zeros` over plain Agentica data
## (there is no uninitialized memory to expose; see `src/reference.agc`).
def empty(shape: List[Int], dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case backend
    Types.Backend.Reference -> Reference.empty(shape, dtype)
    Types.Backend.Torch(device) -> Torch.empty(device, shape, dtype)
    Types.Backend.Trace -> Graph.filled(shape, 0.0, dtype)
  end
end

## Creation: every element `value` (tagged for `dtype`).
def full(shape: List[Int], value: Types.Values, dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case backend
    Types.Backend.Reference -> Reference.full(shape, value, dtype)
    Types.Backend.Torch(device) -> Torch.full(device, shape, value, dtype)
    Types.Backend.Trace -> Graph.fill(shape, value, dtype)
  end
end

## Creation: zeros of the shape, dtype and backend of `t` (metadata only).
def like(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.like(t)
    Types.Backend.Torch(device) -> Torch.like(device, t)
    Types.Backend.Trace -> Graph.filled(t.shape, 0.0, t.dtype)
  end
end

## Creation: `start, start + step, ...` below `stop` (above for `step < 0`).
## Integer dtypes only: positions are indices (`tensor/DESIGN.md` §5).
def arange(start: Int, stop: Int, step: Int, dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  return error("arange needs an Int dtype, got #{Types.dtype_name(dtype)}") if !Types.is_int_dtype(dtype)
  return error("arange needs a nonzero step, got #{step}") if step == 0
  case backend
    Types.Backend.Reference -> Reference.arange(start, stop, step, dtype)
    Types.Backend.Torch(device) -> Torch.arange(device, start, stop, step, dtype)
    Types.Backend.Trace -> ok(Graph.arange(start, stop, step, dtype))
  end
end

## Creation: the one `value` as a rank-0 tensor.
def scalar_to_tensor(value: Types.Values, dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  case backend
    Types.Backend.Reference -> Reference.scalar_to_tensor(value, dtype)
    Types.Backend.Torch(device) -> Torch.scalar_to_tensor(device, value, dtype)
    Types.Backend.Trace -> Graph.fill([], value, dtype)
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
    Types.Backend.Torch(device) -> Torch.from_list(device, data, shape, dtype)
    Types.Backend.Trace -> error("from_list makes no traced tensor: lift one with Trace.constant")
  end
end

## Host readback: the values tagged with the tensor's dtype.
def to_list(t: Types.Tensor) -> Result[List[Types.Values]]
  case t.backend
    Types.Backend.Reference -> Reference.to_list(t)
    Types.Backend.Torch(_) -> Torch.to_list(t)
    Types.Backend.Trace -> error("a traced tensor has no values: run it with Trace.eval")
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
    Types.Backend.Torch(device) -> Torch.add(device, a, b)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Add, [a, b], a.shape, a.dtype))
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
    Types.Backend.Torch(device) -> Torch.sub(device, a, b)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Sub, [a, b], a.shape, a.dtype))
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
    Types.Backend.Torch(device) -> Torch.mul(device, a, b)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Mul, [a, b], a.shape, a.dtype))
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
    Types.Backend.Torch(device) -> Torch.div(device, a, b)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Div, [a, b], a.shape, a.dtype))
  end
end

## Elementwise `-a`, float only.
def neg(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "neg")
  case t.backend
    Types.Backend.Reference -> Reference.neg(t)
    Types.Backend.Torch(device) -> Torch.neg(device, t)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Neg, [t], t.shape, t.dtype))
  end
end

## Elementwise `exp`, float only.
def exp(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "exp")
  case t.backend
    Types.Backend.Reference -> Reference.exp(t)
    Types.Backend.Torch(device) -> Torch.exp(device, t)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Exp, [t], t.shape, t.dtype))
  end
end

## Elementwise `log`, float only, `x <= 0` is an error.
def log(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "log")
  case t.backend
    Types.Backend.Reference -> Reference.log(t)
    Types.Backend.Torch(device) -> Torch.log(device, t)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Log, [t], t.shape, t.dtype))
  end
end

## Elementwise `sqrt`, float only, `x < 0` is an error.
def sqrt(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "sqrt")
  case t.backend
    Types.Backend.Reference -> Reference.sqrt(t)
    Types.Backend.Torch(device) -> Torch.sqrt(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Sqrt, [t], t)
  end
end

## Elementwise `base ** exponent`, float only, exact shapes; `base <= 0`
## is an error (square with `mul`; see `src/reference.agc`).
def pow(base: Types.Tensor, exponent: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(base, exponent, "pow")
  _ = try same_shape(base, exponent, "pow")
  _ = try same_dtype(base, exponent, "pow")
  _ = try float_only(base, "pow")
  case base.backend
    Types.Backend.Reference -> Reference.pow(base, exponent)
    Types.Backend.Torch(device) -> Torch.pow(device, base, exponent)
    Types.Backend.Trace -> Graph.same(Types.Op.Pow, [base, exponent], base)
  end
end

## Elementwise `max(0, x)`, float only.
def relu(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "relu")
  case t.backend
    Types.Backend.Reference -> Reference.relu(t)
    Types.Backend.Torch(device) -> Torch.relu(device, t)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Relu, [t], t.shape, t.dtype))
  end
end

## Elementwise sigmoid, float only.
def sigmoid(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "sigmoid")
  case t.backend
    Types.Backend.Reference -> Reference.sigmoid(t)
    Types.Backend.Torch(device) -> Torch.sigmoid(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Sigmoid, [t], t)
  end
end

## Elementwise tanh, float only.
def tanh(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "tanh")
  case t.backend
    Types.Backend.Reference -> Reference.tanh(t)
    Types.Backend.Torch(device) -> Torch.tanh(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Tanh, [t], t)
  end
end

## The user-facing `gelu`: the `tanh` approximation from §3.1 (the `erf`
## form cannot be expressed over the primitives and is not offered).
def gelu_tanh_approx(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "gelu_tanh_approx")
  case t.backend
    Types.Backend.Reference -> Reference.gelu_tanh_approx(t)
    Types.Backend.Torch(_) -> Torch.unsupported("gelu_tanh_approx")
    Types.Backend.Trace -> Graph.same(Types.Op.Gelu, [t], t)
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
    Types.Backend.Torch(device) -> Torch.eq(device, a, b)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Eq, [a, b], a.shape, Types.Dtype.Bool))
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
    Types.Backend.Torch(device) -> Torch.lt(device, a, b)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Lt, [a, b], a.shape, Types.Dtype.Bool))
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
    Types.Backend.Torch(device) -> Torch.where(device, cond, a, b)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Where, [cond, a, b], a.shape, a.dtype))
  end
end

## Elementwise clamp to `[lo, hi]`, float only.
def clamp(t: Types.Tensor, lo: Float, hi: Float) -> Result[Types.Tensor]
  return error("clamp needs lo <= hi, got #{lo} and #{hi}") if lo > hi
  _ = try float_only(t, "clamp")
  case t.backend
    Types.Backend.Reference -> Reference.clamp(t, lo, hi)
    Types.Backend.Torch(device) -> Torch.clamp(device, t, lo, hi)
    Types.Backend.Trace -> Graph.same(Types.Op.Clamp(lo, hi), [t], t)
  end
end

## Sums every element (`none`) or the named `axes`, float only. `axes` is a
## set: the wrapper hands the backend the ascending order, which is the order
## `argmax` numbers a reduced block by (`README.md`, "What is implemented").
def sum(t: Types.Tensor, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Types.Tensor]
  _ = try float_only(t, "sum")
  axs = try normalize_axes_option(axes, t.shape.size, "sum")
  case t.backend
    Types.Backend.Reference -> Reference.sum(t, axs, keep_dims)
    Types.Backend.Torch(device) -> Torch.sum(device, t, axs, keep_dims)
    Types.Backend.Trace -> ok(Graph.reduction(true, t, axs, keep_dims))
  end
end

## `sum` divided by the reduced count, float only.
def mean(t: Types.Tensor, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Types.Tensor]
  _ = try float_only(t, "mean")
  axs = try normalize_axes_option(axes, t.shape.size, "mean")
  case t.backend
    Types.Backend.Reference -> Reference.mean(t, axs, keep_dims)
    Types.Backend.Torch(device) -> Torch.mean(device, t, axs, keep_dims)
    Types.Backend.Trace -> ok(Graph.reduction(false, t, axs, keep_dims))
  end
end

## The largest value per reduced block, float only.
def max(t: Types.Tensor, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Types.Tensor]
  _ = try float_only(t, "max")
  axs = try normalize_axes_option(axes, t.shape.size, "max")
  case t.backend
    Types.Backend.Reference -> Reference.max(t, axs, keep_dims)
    Types.Backend.Torch(device) -> Torch.max(device, t, axs, keep_dims)
    Types.Backend.Trace -> ok(Graph.reduced(false, t, axs, keep_dims))
  end
end

## The index of the largest value per reduced block, as `I64`. The position is
## the row-major offset inside the reduced block, numbered by the ascending
## `axes`, and a tie keeps the lowest one — on both backends, because the
## wrapper sorts the axes before dispatch.
def argmax(t: Types.Tensor, axes: Optional[List[Int]] = none, keep_dims: Bool = false) -> Result[Types.Tensor]
  _ = try float_only(t, "argmax")
  axs = try normalize_axes_option(axes, t.shape.size, "argmax")
  case t.backend
    Types.Backend.Reference -> Reference.argmax(t, axs, keep_dims)
    Types.Backend.Torch(device) -> Torch.argmax(device, t, axs, keep_dims)
    Types.Backend.Trace -> ok(Graph.reduced(true, t, axs, keep_dims))
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
    Types.Backend.Torch(device) -> Torch.matmul(device, a, b)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Matmul, [a, b], [a.shape[0], b.shape[1]], a.dtype))
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
    Types.Backend.Torch(_) -> Torch.unsupported("batch_matmul")
    Types.Backend.Trace -> ok(Graph.batch_matmul(a, b))
  end
end

## The transpose: reverses every axis (rank 2 swaps rows and columns).
## A true axis permutation is derived from `slice`/`stack` where needed.
def transpose(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.transpose(t)
    Types.Backend.Torch(device) -> Torch.transpose(device, t)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Transpose, [t], t.shape.reverse, t.dtype))
  end
end

## A tensor of `shape` with the same values: the volumes must match.
def reshape(t: Types.Tensor, shape: List[Int]) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case t.backend
    Types.Backend.Reference -> Reference.reshape(t, shape)
    Types.Backend.Torch(device) -> Torch.reshape(device, t, shape)
    Types.Backend.Trace -> Graph.reshape(t, shape)
  end
end

## Drops size-1 dimensions: every one for `none`, or the named `axis`.
def squeeze(t: Types.Tensor, axis: Optional[Int] = none) -> Result[Types.Tensor]
  _ = try check_squeeze_axis(axis, t.shape, "squeeze")
  case t.backend
    Types.Backend.Reference -> Reference.squeeze(t, axis)
    Types.Backend.Torch(device) -> Torch.squeeze(device, t, axis)
    Types.Backend.Trace -> ok(Graph.squeeze(t, axis))
  end
end

## Inserts a size-1 dimension at `axis`.
def unsqueeze(t: Types.Tensor, axis: Int = 0) -> Result[Types.Tensor]
  return error("unsqueeze needs an axis in 0...#{t.shape.size}, got #{axis}") if axis < 0 or axis > t.shape.size
  case t.backend
    Types.Backend.Reference -> Reference.unsqueeze(t, axis)
    Types.Backend.Torch(device) -> Torch.unsqueeze(device, t, axis)
    Types.Backend.Trace -> ok(Graph.unsqueeze(t, axis))
  end
end

## The `[starts[i], ends[i])` window per axis (no steps in M1).
def slice(t: Types.Tensor, starts: List[Int], ends: List[Int]) -> Result[Types.Tensor]
  _ = try check_slice(starts, ends, t.shape, "slice")
  case t.backend
    Types.Backend.Reference -> Reference.slice(t, starts, ends)
    Types.Backend.Torch(device) -> Torch.slice(device, t, starts, ends)
    Types.Backend.Trace -> ok(Graph.slice(t, starts, ends))
  end
end

## Joins `tensors` along `axis`: one rank, one dtype, every dimension but
## `axis` equal. The wrapper checks all of it before dispatch (§9), so a
## backend only re-checks what keeps its own indexing total.
def concat(tensors: List[Types.Tensor], axis: Int) -> Result[Types.Tensor]
  return error("concat needs at least one tensor") if tensors.empty?
  first = tensors[0]
  rank = first.shape.size
  return error("concat needs an axis in 0...#{rank}, got #{axis}") if axis < 0 or axis >= rank
  for t in tensors
    _ = try same_backend(first, t, "concat")
    _ = try same_rank(first, t, "concat")
    _ = try same_dtype(first, t, "concat")
    _ = try same_other_dims(first, t, axis, "concat")
  end

  case first.backend
    Types.Backend.Reference -> Reference.concat(tensors, axis)
    Types.Backend.Torch(device) -> Torch.concat(device, tensors, axis)
    Types.Backend.Trace -> ok(Graph.concat(tensors, axis))
  end
end

## Stacks `tensors` along a new `axis`: every shape and dtype equal.
def stack(tensors: List[Types.Tensor], axis: Int = 0) -> Result[Types.Tensor]
  return error("stack needs at least one tensor") if tensors.empty?
  first = tensors[0]
  return error("stack needs an axis in 0...#{first.shape.size}, got #{axis}") if axis < 0 or axis > first.shape.size
  for t in tensors
    _ = try same_backend(first, t, "stack")
    _ = try same_shape(first, t, "stack")
    _ = try same_dtype(first, t, "stack")
  end

  case first.backend
    Types.Backend.Reference -> Reference.stack(tensors, axis)
    Types.Backend.Torch(device) -> Torch.stack(device, tensors, axis)
    Types.Backend.Trace -> ok(Graph.stack(tensors, axis))
  end
end

## Views `t` as `shape`: trailing axes align, size-1 dimensions widen.
def broadcast_to(t: Types.Tensor, shape: List[Int]) -> Result[Types.Tensor]
  _ = try check_broadcast(t.shape, shape, "broadcast_to")
  case t.backend
    Types.Backend.Reference -> Reference.broadcast_to(t, shape)
    Types.Backend.Torch(device) -> Torch.broadcast_to(device, t, shape)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.BroadcastTo, [t], shape, t.dtype))
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
    Types.Backend.Torch(device) -> Torch.gather(device, source, indices)
    Types.Backend.Trace -> ok(Graph.gather(source, indices))
  end
end

## Moves `t` to `dtype` (float to int truncates, anything nonzero is `true`).
def to_dtype(t: Types.Tensor, dtype: Types.Dtype) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.to_dtype(t, dtype)
    Types.Backend.Torch(device) -> Torch.to_dtype(device, t, dtype)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.ToDtype, [t], t.shape, dtype))
  end
end

## Moves `t` to `device`: reference tensors live on `Cpu` only.
def to_device(t: Types.Tensor, device: Types.Device) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.to_device(t, device)
    Types.Backend.Torch(_) -> Torch.unsupported("to_device")
    Types.Backend.Trace -> Graph.same(Types.Op.ToDevice(device), [t], t)
  end
end

## The identity move: values are immutable, so a copy shares nothing.
def copy(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.copy(t)
    Types.Backend.Torch(device) -> Torch.copy(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Copy, [t], t)
  end
end

## The identity move: the reference has no strides, every tensor is
## contiguous by construction.
def contiguous(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.contiguous(t)
    Types.Backend.Torch(device) -> Torch.contiguous(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Contiguous, [t], t)
  end
end

## `t` itself, as a constant of the gradient: the identity on every backend
## that computes, and on `Trace` a node whose backward rule is zero
## (`tensor/DESIGN.md` §10).
def stop_grad(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> ok(t)
    Types.Backend.Torch(_) -> ok(t)
    Types.Backend.Trace -> Graph.same(Types.Op.StopGrad, [t], t)
  end
end

## Convolution: the cross-correlation of `input` with `weight` along its last
## axis, moved by `stride`, with `padding` zeros on each side. No dilation and
## no groups: `tensor/DESIGN.md` §3.1 admits stride/padding-1D and leaves the
## rest to a consumer that needs it. `input` is `[C_in, L]` or
## `[N, C_in, L]`, `weight` is `[C_out, C_in, K]`, and the answer is the
## input's leading axes plus `[C_out, L_out]`, where
## `L_out = (L + 2*padding - K) / stride + 1`. The window arithmetic is
## validated here, before dispatch, so both backends refuse the same shapes.
def conv1d(input: Types.Tensor, weight: Types.Tensor, stride: Int = 1, padding: Int = 0) -> Result[Types.Tensor]
  _ = try same_backend(input, weight, "conv1d")
  _ = try same_dtype(input, weight, "conv1d")
  _ = try float_only(input, "conv1d")
  _ = try float_only(weight, "conv1d")
  rank = input.shape.size
  return error("conv1d needs a rank-2 or rank-3 input, got #{input.shape}") if rank < 2 or rank > 3
  return error("conv1d needs a rank-3 weight, got #{weight.shape}") if weight.shape.size != 3
  _ = try check_window(stride, padding, "conv1d")
  channels = input.shape[rank - 2]
  return error("conv1d needs #{channels} input channels, got #{weight.shape[1]}") if weight.shape[1] != channels
  _ = try check_window_fits(input.shape[rank - 1], weight.shape[2], padding, "conv1d")
  case input.backend
    Types.Backend.Reference -> Reference.conv1d(input, weight, stride, padding)
    Types.Backend.Torch(device) -> Torch.conv1d(device, input, weight, stride, padding)
    Types.Backend.Trace -> ok(Graph.conv1d(input, weight, stride, padding))
  end
end

## The mean over a window of `kernel` positions, moved by `stride`, with
## `padding` zeros on each side; the divisor is always `kernel`
## (`count_include_pad`), so a window that runs off the input averages over
## the zeros it padded with. `input` is `[C, L]` or `[N, C, L]` and the
## answer replaces `L` with `L_out = (L + 2*padding - kernel) / stride + 1`.
## Padding stops at half the kernel: that is the window the libtorch kernel
## admits, and the shared domain is validated here so both backends refuse the
## same ones.
def avg_pool1d(input: Types.Tensor, kernel: Int, stride: Int = 1, padding: Int = 0) -> Result[Types.Tensor]
  _ = try float_only(input, "avg_pool1d")
  rank = input.shape.size
  return error("avg_pool1d needs a rank-2 or rank-3 input, got #{input.shape}") if rank < 2 or rank > 3
  return error("avg_pool1d needs kernel >= 1, got #{kernel}") if kernel < 1
  _ = try check_window(stride, padding, "avg_pool1d")
  if padding > kernel / 2
    return error("avg_pool1d needs padding <= #{kernel / 2} for a kernel of #{kernel}, got #{padding}")
  end

  _ = try check_window_fits(input.shape[rank - 1], kernel, padding, "avg_pool1d")
  case input.backend
    Types.Backend.Reference -> Reference.avg_pool1d(input, kernel, stride, padding)
    Types.Backend.Torch(device) -> Torch.avg_pool1d(device, input, kernel, stride, padding)
    Types.Backend.Trace -> ok(Graph.avg_pool1d(input, kernel, stride, padding))
  end
end

## Derived: `softmax` over `axis` with max-subtraction, from §3.1 only. The
## maximum is a `stop_grad` constant: it cancels out of the value, so it
## needs no gradient.
def softmax(x: Types.Tensor, axis: Int) -> Result[Types.Tensor]
  rank = x.shape.size
  return error("softmax needs an axis in 0...#{rank}, got #{axis}") if axis < 0 or axis >= rank
  m = try stop_grad(try max(x, some([axis]), true))
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
      return error("greedy_step needs a finite temperature, got #{t}") if !Types.is_finite(t)
      return error("greedy_step needs temperature > 0, got #{t}") if t <= 0.0
      r = 1.0 / t
      return error("greedy_step needs a temperature whose reciprocal is finite, got #{t}") if !Types.is_finite(r)
      try mul(logits, try full(logits.shape, Types.Values.Float(r), logits.dtype, logits.backend))
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

  for i in 0...vals.size
    return error("top_k needs finite values, got a non-finite value at index #{i}") if !Types.is_finite(vals[i])
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

## Two tensors share their rank.
def same_rank(a: Types.Tensor, b: Types.Tensor, op: String) -> Result[Bool]
  return error("#{op} needs one rank, got #{a.shape} and #{b.shape}") if a.shape.size != b.shape.size
  ok(true)
end

## Two tensors agree on every dimension but `axis`.
def same_other_dims(a: Types.Tensor, b: Types.Tensor, axis: Int, op: String) -> Result[Bool]
  for i in 0...a.shape.size
    if i != axis and a.shape[i] != b.shape[i]
      return error("#{op} needs dimension #{i} of size #{a.shape[i]}, got #{b.shape[i]}")
    end
  end

  ok(true)
end

## The window of a convolution or a pooling moves by at least one position per
## step, with a padding that is not negative.
def check_window(stride: Int, padding: Int, op: String) -> Result[Bool]
  return error("#{op} needs stride >= 1, got #{stride}") if stride < 1
  return error("#{op} needs padding >= 0, got #{padding}") if padding < 0
  ok(true)
end

## The input is not empty, and padded by `padding` zeros on each side it still
## holds one whole window of `kernel` positions: below that the answer would
## have no position.
def check_window_fits(length: Int, kernel: Int, padding: Int, op: String) -> Result[Bool]
  return error("#{op} needs a nonempty input, got #{length} positions") if length < 1
  if length + 2 * padding < kernel
    return error("#{op} needs a window of #{kernel} positions in #{length} padded to #{length + 2 * padding}")
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
## The answer is what every backend takes: `none`, or the axes in ascending
## order. Reductions are order-independent, but `argmax` numbers the position
## inside a reduced block by the order of its axes, and the two backends walk
## the tensor differently, so `some([2, 0])` would number `(i0=0, i2=1)` and
## `(i0=1, i2=0)` the other way round on each side and break a tie between them
## on different elements. Sorting here is the one place the rule lives.
def normalize_axes_option(axes: Optional[List[Int]], rank: Int, op: String) -> Result[Optional[List[Int]]]
  case axes
    some(given) ->
      _ = try check_axes(given, rank, op)
      ok(some(given.sort))
    none -> ok(none)
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
