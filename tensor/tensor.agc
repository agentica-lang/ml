## The `tensor/` package: functional tensors over pluggable backends.
##
## The public API (`tensor/DESIGN.md` §§2–4, 7, 9). Every fallible op returns
## `Result` and reports `error`, never a halt; shape, dtype and backend
## mismatches are validated eagerly in these wrappers, before dispatch.
## Dispatch is by value (`t.backend`) through an exhaustive `case` over all
## four `Backend` arms (`tensor/DESIGN.md` §7): `Reference` runs in Agentica,
## `Torch` runs on libtorch through `src/torch.agc` and `src/native.agc`,
## `Trace` records a node of a traced graph after the same checks
## (`src/graph.agc`, `trace.agc`, §10), and `Ggml` creates and reads back
## tensors on ggml (`src/ggml.agc`) and answers `compiled_only` for every other
## op. With a native module absent (both are `optional`) its arm answers
## `error("backend not available")`, and `available/1` says so before a call
## does.
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
##   arena. The native backends allocate their handles inside a `Native.arena`
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
## - The 49 names of §3.1 all exist below. `Reference` computes all of them;
##   the `Torch` arm computes everything but `batch_matmul` and
##   `to_device`, and answers
##   `error("<op> is not implemented on the torch backend yet")` for those,
##   so the gap is a `Result` error naming the op, never a wrong number
##   (`README.md`, "What is implemented"). The `Ggml` arm creates, reads
##   back and copies between devices (`to_device`), and answers
##   `Ggml.compiled_only(op)` for every other primitive:
##   ggml runs ops only in a traced graph (`Trace`, `README.md`, "The ggml
##   backend").

import Ggml from "src/ggml"
import Graph from "src/graph"
import Reference from "src/reference"
import Torch from "src/torch"
import Types from "src/types"

export Types.Backend
export Types.Data
export Types.Device
export Types.Dtype
export Types.FloatFormat
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
## `agentica.toml`), `Ggml` when the native ggml module is built and, for
## `Vulkan(i)`, ggml sees that device. A caller that needs to work on both asks
## this instead of catching the `error("backend not available")` an op would
## answer. `Trace` is never available: it records ops and computes nothing
## (`trace.agc`).
def available(backend: Types.Backend) -> Bool
  case backend
    Types.Backend.Reference -> true
    Types.Backend.Torch(_) -> Torch.available()
    Types.Backend.Trace -> false
    Types.Backend.Ggml(device) -> Ggml.available(device)
  end
end

## Chooses fp16 (`true`) or F32 (`false`, the default) `mul_mat` for every
## Vulkan device of the ggml backend in this process. ggml fixes the choice
## at the process's first ggml call (a CPU one or `available` included), so
## a different choice after that is an error. On a build without Vulkan it
## changes nothing. fp16 is faster and changes gradients by up to about 10 %
## at MLP sizes (`tensor/README.md`, "The ggml backend").
def vulkan_fp16(enabled: Bool) -> Result[Void] = Ggml.vulkan_fp16(enabled)

## Creation: zeros of `shape`, `dtype` and `backend` (`F32` on `Reference` by default).
def zeros(
  shape: List[Int],
  dtype: Types.Dtype = Types.Dtype.F32,
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case backend
    Types.Backend.Reference -> Reference.zeros(shape, dtype)
    Types.Backend.Torch(device) -> Torch.zeros(device, shape, dtype)
    Types.Backend.Trace -> Graph.filled(shape, 0.0, dtype)
    Types.Backend.Ggml(device) -> Ggml.filled(device, shape, dtype, 0.0, "zeros")
  end
end

## Creation: ones of `shape`, `dtype` and `backend`.
def ones(
  shape: List[Int],
  dtype: Types.Dtype = Types.Dtype.F32,
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case backend
    Types.Backend.Reference -> Reference.ones(shape, dtype)
    Types.Backend.Torch(device) -> Torch.ones(device, shape, dtype)
    Types.Backend.Trace -> Graph.filled(shape, 1.0, dtype)
    Types.Backend.Ggml(device) -> Ggml.filled(device, shape, dtype, 1.0, "ones")
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
    Types.Backend.Ggml(device) -> Ggml.filled(device, shape, dtype, 0.0, "empty")
  end
end

## Creation: every element `value` (tagged for `dtype`).
def full(
  shape: List[Int],
  value: Types.Values,
  dtype: Types.Dtype = Types.Dtype.F32,
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case backend
    Types.Backend.Reference -> Reference.full(shape, value, dtype)
    Types.Backend.Torch(device) -> Torch.full(device, shape, value, dtype)
    Types.Backend.Trace -> Graph.fill(shape, value, dtype)
    Types.Backend.Ggml(device) -> Ggml.full(device, shape, value, dtype)
  end
end

## Creation: zeros of the shape, dtype and backend of `t` (metadata only).
def like(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.like(t)
    Types.Backend.Torch(device) -> Torch.like(device, t)
    Types.Backend.Trace -> Graph.filled(t.shape, 0.0, t.dtype)
    Types.Backend.Ggml(device) -> Ggml.filled(device, t.shape, t.dtype, 0.0, "like")
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
    Types.Backend.Ggml(device) -> Ggml.arange(device, start, stop, step, dtype)
  end
end

## Creation: the one `value` as a rank-0 tensor.
def scalar_to_tensor(value: Types.Values, dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  case backend
    Types.Backend.Reference -> Reference.scalar_to_tensor(value, dtype)
    Types.Backend.Torch(device) -> Torch.scalar_to_tensor(device, value, dtype)
    Types.Backend.Trace -> Graph.fill([], value, dtype)
    Types.Backend.Ggml(device) -> Ggml.full(device, [], value, dtype)
  end
end

## Creation: row-major `data` of `shape`; the count must match.
def from_list(
  data: List[Types.Values],
  shape: List[Int],
  dtype: Types.Dtype = Types.Dtype.F32,
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  return error("from_list needs #{n} values for #{shape}, got #{data.size}") if data.size != n
  case backend
    Types.Backend.Reference -> Reference.from_list(data, shape, dtype)
    Types.Backend.Torch(device) -> Torch.from_list(device, data, shape, dtype)
    Types.Backend.Trace -> error("from_list makes no traced tensor: lift one with Trace.constant")
    Types.Backend.Ggml(device) -> Ggml.from_list(device, data, shape, dtype)
  end
end

## Creation: an `F32` tensor of `shape` from the little-endian floats of `format`
## that start at byte `offset` of the file `path`, for weights: the backend reads
## the file itself, so no value crosses as an Agentica list. `F16` and `BF16`
## widen to `F32`. A short file, a bad offset or a missing file is an `error`.
## The reference reads the bytes in Agentica and is slow on a large file.
def from_file(
  path: String,
  offset: Int,
  shape: List[Int],
  format: Types.FloatFormat = Types.FloatFormat.F32,
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  return error("from_file needs an offset of 0 or more, got #{offset}") if offset < 0
  if n > (9223372036854775807 - offset) / Types.file_width(format)
    return error("from_file: #{n} values at offset #{offset} reach past what an Int counts")
  end

  case backend
    Types.Backend.Reference -> Reference.from_file(path, offset, shape, format)
    Types.Backend.Torch(device) -> Torch.from_file(device, path, offset, shape, format)
    Types.Backend.Trace -> error("from_file makes no traced tensor: lift one with Trace.constant")
    Types.Backend.Ggml(device) -> Ggml.from_file(device, path, offset, shape, format)
  end
end

## Creation: plain `values` as an `F32` tensor of `shape`, the short form of
## `from_list` for floats.
def floats(
  values: List[Float],
  shape: List[Int],
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  from_list(values.map { |x: Float| Types.Values.Float(x) }, shape, Types.Dtype.F32, backend)
end

## Host readback: the values tagged with the tensor's dtype.
def to_list(t: Types.Tensor) -> Result[List[Types.Values]]
  case t.backend
    Types.Backend.Reference -> Reference.to_list(t)
    Types.Backend.Torch(_) -> Torch.to_list(t)
    Types.Backend.Trace -> error("a traced tensor has no values: run it with Trace.eval")
    Types.Backend.Ggml(_) -> Ggml.to_list(t)
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("add")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("sub")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("mul")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("div")
  end
end

## Elementwise `-a`, float only.
def neg(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "neg")
  case t.backend
    Types.Backend.Reference -> Reference.neg(t)
    Types.Backend.Torch(device) -> Torch.neg(device, t)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Neg, [t], t.shape, t.dtype))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("neg")
  end
end

## Elementwise `exp`, float only.
def exp(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "exp")
  case t.backend
    Types.Backend.Reference -> Reference.exp(t)
    Types.Backend.Torch(device) -> Torch.exp(device, t)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Exp, [t], t.shape, t.dtype))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("exp")
  end
end

## Elementwise `log`, float only, `x <= 0` is an error.
def log(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "log")
  case t.backend
    Types.Backend.Reference -> Reference.log(t)
    Types.Backend.Torch(device) -> Torch.log(device, t)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Log, [t], t.shape, t.dtype))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("log")
  end
end

## Elementwise `sqrt`, float only, `x < 0` is an error.
def sqrt(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "sqrt")
  case t.backend
    Types.Backend.Reference -> Reference.sqrt(t)
    Types.Backend.Torch(device) -> Torch.sqrt(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Sqrt, [t], t)
    Types.Backend.Ggml(_) -> Ggml.compiled_only("sqrt")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("pow")
  end
end

## Elementwise `max(0, x)`, float only.
def relu(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "relu")
  case t.backend
    Types.Backend.Reference -> Reference.relu(t)
    Types.Backend.Torch(device) -> Torch.relu(device, t)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Relu, [t], t.shape, t.dtype))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("relu")
  end
end

## Elementwise sigmoid, float only.
def sigmoid(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "sigmoid")
  case t.backend
    Types.Backend.Reference -> Reference.sigmoid(t)
    Types.Backend.Torch(device) -> Torch.sigmoid(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Sigmoid, [t], t)
    Types.Backend.Ggml(_) -> Ggml.compiled_only("sigmoid")
  end
end

## Elementwise tanh, float only.
def tanh(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "tanh")
  case t.backend
    Types.Backend.Reference -> Reference.tanh(t)
    Types.Backend.Torch(device) -> Torch.tanh(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Tanh, [t], t)
    Types.Backend.Ggml(_) -> Ggml.compiled_only("tanh")
  end
end

## The user-facing `gelu`: the `tanh` approximation from §3.1 (the `erf`
## form cannot be expressed over the primitives and is not offered).
def gelu_tanh_approx(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "gelu_tanh_approx")
  case t.backend
    Types.Backend.Reference -> Reference.gelu_tanh_approx(t)
    Types.Backend.Torch(device) -> Torch.gelu_tanh_approx(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Gelu, [t], t)
    Types.Backend.Ggml(_) -> Ggml.compiled_only("gelu_tanh_approx")
  end
end

## The exact GELU, `x * (1 + erf(x / sqrt(2))) / 2`, as Whisper and BERT use
## it. Not a §3.1 primitive and inference only: `Trace` records it (so the
## ggml backend compiles it) but has no gradient for it (train with
## `gelu_tanh_approx`).
def gelu_erf(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "gelu_erf")
  case t.backend
    Types.Backend.Reference -> Reference.gelu_erf(t)
    Types.Backend.Torch(device) -> Torch.gelu_erf(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.GeluErf, [t], t)
    Types.Backend.Ggml(_) -> Ggml.compiled_only("gelu_erf")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("eq")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("lt")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("where")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("clamp")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("sum")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("mean")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("max")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("argmax")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("matmul")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("batch_matmul")
  end
end

## The transpose: reverses every axis (rank 2 swaps rows and columns).
## A true axis permutation is derived from `slice`/`stack` where needed.
def transpose(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.transpose(t)
    Types.Backend.Torch(device) -> Torch.transpose(device, t)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Transpose, [t], t.shape.reverse, t.dtype))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("transpose")
  end
end

## The axes of `t` in the order `axes` names: output axis `i` is input axis
## `axes[i]`, so `permute(t, [0, 2, 1, 3])` swaps the middle two of a rank-4
## tensor. `axes` is a permutation of `0...rank`; the answer is a new
## row-major tensor.
def permute(t: Types.Tensor, axes: List[Int]) -> Result[Types.Tensor]
  return error("permute needs one axis per dimension of #{t.shape}, got #{axes}") if axes.size != t.shape.size
  _ = try check_axes(axes, t.shape.size, "permute")
  case t.backend
    Types.Backend.Reference -> Reference.permute(t, axes)
    Types.Backend.Torch(device) -> Torch.permute(device, t, axes)
    Types.Backend.Trace -> ok(Graph.permute(t, axes))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("permute")
  end
end

## A tensor of `shape` with the same values: the volumes must match.
def reshape(t: Types.Tensor, shape: List[Int]) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  case t.backend
    Types.Backend.Reference -> Reference.reshape(t, shape)
    Types.Backend.Torch(device) -> Torch.reshape(device, t, shape)
    Types.Backend.Trace -> Graph.reshape(t, shape)
    Types.Backend.Ggml(_) -> Ggml.compiled_only("reshape")
  end
end

## Drops size-1 dimensions: every one for `none`, or the named `axis`.
def squeeze(t: Types.Tensor, axis: Optional[Int] = none) -> Result[Types.Tensor]
  _ = try check_squeeze_axis(axis, t.shape, "squeeze")
  case t.backend
    Types.Backend.Reference -> Reference.squeeze(t, axis)
    Types.Backend.Torch(device) -> Torch.squeeze(device, t, axis)
    Types.Backend.Trace -> ok(Graph.squeeze(t, axis))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("squeeze")
  end
end

## Inserts a size-1 dimension at `axis`.
def unsqueeze(t: Types.Tensor, axis: Int = 0) -> Result[Types.Tensor]
  return error("unsqueeze needs an axis in 0...#{t.shape.size}, got #{axis}") if axis < 0 or axis > t.shape.size
  case t.backend
    Types.Backend.Reference -> Reference.unsqueeze(t, axis)
    Types.Backend.Torch(device) -> Torch.unsqueeze(device, t, axis)
    Types.Backend.Trace -> ok(Graph.unsqueeze(t, axis))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("unsqueeze")
  end
end

## The `[starts[i], ends[i])` window per axis (no steps in M1).
def slice(t: Types.Tensor, starts: List[Int], ends: List[Int]) -> Result[Types.Tensor]
  _ = try check_slice(starts, ends, t.shape, "slice")
  case t.backend
    Types.Backend.Reference -> Reference.slice(t, starts, ends)
    Types.Backend.Torch(device) -> Torch.slice(device, t, starts, ends)
    Types.Backend.Trace -> ok(Graph.slice(t, starts, ends))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("slice")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("concat")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("stack")
  end
end

## Views `t` as `shape`: trailing axes align, size-1 dimensions widen.
def broadcast_to(t: Types.Tensor, shape: List[Int]) -> Result[Types.Tensor]
  _ = try check_broadcast(t.shape, shape, "broadcast_to")
  case t.backend
    Types.Backend.Reference -> Reference.broadcast_to(t, shape)
    Types.Backend.Torch(device) -> Torch.broadcast_to(device, t, shape)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.BroadcastTo, [t], shape, t.dtype))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("broadcast_to")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("gather")
  end
end

## The inverse of `gather`: `updates` of shape `indices.shape ++ tail` added
## into `rows` zero rows of shape `tail`, each at the row its index names. A
## repeated index sums its rows; an index outside `[0, rows)` is an error.
## The answer is `[rows] ++ tail`.
def scatter_add(indices: Types.Tensor, updates: Types.Tensor, rows: Int) -> Result[Types.Tensor]
  _ = try same_backend(indices, updates, "scatter_add")
  return error("scatter_add needs a row count of 0 or more, got #{rows}") if rows < 0
  if !Types.is_int_dtype(indices.dtype)
    return error("scatter_add needs integer indices, got #{Types.dtype_name(indices.dtype)}")
  end

  _ = try float_only(updates, "scatter_add")
  if updates.shape.take(indices.shape.size) != indices.shape
    return error("scatter_add needs updates of shape #{indices.shape} followed by a row, got #{updates.shape}")
  end

  case updates.backend
    Types.Backend.Reference -> Reference.scatter_add(indices, updates, rows)
    Types.Backend.Torch(device) -> Torch.scatter_add(device, indices, updates, rows)
    Types.Backend.Trace -> ok(Graph.scatter_add(indices, updates, rows))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("scatter_add")
  end
end

## Moves `t` to `dtype` (float to int truncates, anything nonzero is `true`).
def to_dtype(t: Types.Tensor, dtype: Types.Dtype) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.to_dtype(t, dtype)
    Types.Backend.Torch(device) -> Torch.to_dtype(device, t, dtype)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.ToDtype, [t], t.shape, dtype))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("to_dtype")
  end
end

## Moves `t` to `device`: reference tensors live on `Cpu` only.
def to_device(t: Types.Tensor, device: Types.Device) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.to_device(t, device)
    Types.Backend.Torch(_) -> Torch.unsupported("to_device")
    Types.Backend.Trace -> Graph.same(Types.Op.ToDevice(device), [t], t)
    Types.Backend.Ggml(_) -> Ggml.to_device(t, device)
  end
end

## The identity move: values are immutable, so a copy shares nothing.
def copy(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.copy(t)
    Types.Backend.Torch(device) -> Torch.copy(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Copy, [t], t)
    Types.Backend.Ggml(_) -> Ggml.compiled_only("copy")
  end
end

## The identity move: the reference has no strides, every tensor is
## contiguous by construction.
def contiguous(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.contiguous(t)
    Types.Backend.Torch(device) -> Torch.contiguous(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Contiguous, [t], t)
    Types.Backend.Ggml(_) -> Ggml.compiled_only("contiguous")
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
    Types.Backend.Ggml(_) -> ok(t)
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("conv1d")
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
    Types.Backend.Ggml(_) -> Ggml.compiled_only("avg_pool1d")
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

## Scaled dot-product attention over `q` `[batch, heads, q_len, head_dim]`,
## `k` `[batch, kv_heads, k_len, head_dim]` and `v` `[batch, kv_heads, k_len,
## value_dim]`: `softmax(scale * q k^T + mask) v`, shaped `[batch, heads, q_len,
## value_dim]`. With grouped-query attention `kv_heads` divides `heads` and
## query head `h` reads kv head `h / (heads / kv_heads)`.
##
## `mask` is additive and shaped `[q_len, k_len]`, or `[batch, heads, q_len,
## k_len]` with `1` allowed for `batch` and for `heads`. `causal` adds the
## causal mask, built from `arange` and `lt` so it is the same on every
## backend: query `i` reads keys `0...i + k_len - q_len`, so the keys of a cache
## that holds earlier positions line up with the last queries. `scale` defaults
## to `1 / sqrt(head_dim)`.
def attention(
  q: Types.Tensor,
  k: Types.Tensor,
  v: Types.Tensor,
  mask: Optional[Types.Tensor] = none,
  scale: Optional[Float] = none,
  causal: Bool = false,
) -> Result[Types.Tensor]
  _ = try same_backend(q, k, "attention")
  _ = try same_backend(q, v, "attention")
  _ = try same_dtype(q, k, "attention")
  _ = try same_dtype(q, v, "attention")
  _ = try float_only(q, "attention")
  if q.shape.size != 4 or k.shape.size != 4 or v.shape.size != 4
    return error("attention needs rank-4 tensors, got #{q.shape}, #{k.shape} and #{v.shape}")
  end

  b = q.shape[0]
  heads = q.shape[1]
  sq = q.shape[2]
  d = q.shape[3]
  kv_heads = k.shape[1]
  sk = k.shape[2]
  if k.shape[0] != b or v.shape[0] != b
    return error("attention needs one batch, got #{b}, #{k.shape[0]} and #{v.shape[0]}")
  end

  return error("attention needs key width #{d}, got #{k.shape[3]}") if k.shape[3] != d
  if v.shape[1] != kv_heads or v.shape[2] != sk
    return error("attention needs the value to match the key in heads and length, got #{k.shape} and #{v.shape}")
  end

  if kv_heads < 1 or heads % kv_heads != 0
    return error("attention needs the #{kv_heads} kv heads to divide the #{heads} heads")
  end

  if b < 1 or heads < 1 or sq < 1 or sk < 1 or d < 1 or v.shape[3] < 1
    return error("attention needs nonempty dimensions, got #{q.shape}, #{k.shape} and #{v.shape}")
  end

  chosen = scale.unwrap_or(1.0 / Math.sqrt(d.to_f))
  return error("attention needs a finite scale, got #{chosen}") if !Types.is_finite(chosen)
  bias = try attention_bias(q, sk, mask, causal)
  case q.backend
    Types.Backend.Reference -> Reference.attention(q, k, v, bias, chosen)
    Types.Backend.Torch(device) -> Torch.attention(device, q, k, v, bias, chosen)
    Types.Backend.Trace -> ok(Graph.attention(q, k, v, bias, chosen))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("attention")
  end
end

## Rotary position embedding of `x` `[..., seq, width]`: the first `dims`
## columns of row `s` (`width` by default, an even number) are rotated in
## pairs by the angle `positions[s] * base^(-2i/dims)`, pair `i`; the columns
## past `dims` pass through. With `neox` pair `i` is columns `(i, i + dims/2)`,
## the layout of Hugging Face's `rotate_half`; without it, `(2i, 2i + 1)`, as
## the original Llama does. `positions` is an integer vector of length `seq`.
## With `inverse` every pair turns the other way, which undoes the rotation (and
## is the backward of `rope`).
def rope(
  x: Types.Tensor,
  positions: Types.Tensor,
  base: Float = 10000.0,
  dims: Optional[Int] = none,
  neox: Bool = true,
  inverse: Bool = false,
) -> Result[Types.Tensor]
  _ = try same_backend(x, positions, "rope")
  _ = try float_only(x, "rope")
  if !Types.is_int_dtype(positions.dtype)
    return error("rope needs integer positions, got #{Types.dtype_name(positions.dtype)}")
  end

  rank = x.shape.size
  return error("rope needs rank >= 2, got #{x.shape}") if rank < 2
  seq = x.shape[rank - 2]
  width = x.shape[rank - 1]
  return error("rope needs positions of shape [#{seq}], got #{positions.shape}") if positions.shape != [seq]
  rotated = dims.unwrap_or(width)
  if rotated < 2 or rotated > width or rotated % 2 != 0
    return error("rope needs an even dims in 2...#{width}, got #{rotated}")
  end

  return error("rope needs a finite base > 0, got #{base}") if !Types.is_finite(base) or base <= 0.0
  return error("rope needs nonempty dimensions, got #{x.shape}") if try Types.volume(x.shape) == 0
  case x.backend
    Types.Backend.Reference -> Reference.rope(x, positions, base, rotated, neox, inverse)
    Types.Backend.Torch(device) -> Torch.rope(device, x, positions, base, rotated, neox, inverse)
    Types.Backend.Trace -> ok(Graph.rope(x, positions, base, rotated, neox, inverse))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("rope")
  end
end

## Derived: multi-head attention over `[batch, seq, width]` tensors (`value`
## may differ in width), a thin wrapper over `attention`: the heads split
## along the width by `reshape` + `permute`, and join back the same way.
## `mask`, when given, is an additive bias: `[q, k]` as it is, anything else
## widened to `[batch, heads, q, k]`.
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

  widened = case mask
    some(m) ->
      _ = try same_backend(query, m, "multi_head_attention")
      _ = try same_dtype(query, m, "multi_head_attention")
      _ = try float_only(m, "multi_head_attention")
      fitted = m
      if m.shape != [sq, sk]
        fitted = try broadcast_to(m, [b, heads, sq, sk])
      end

      some(fitted)
    none -> none
  end

  split = [0, 2, 1, 3]
  q = try permute(try reshape(query, [b, sq, heads, d / heads]), split)
  k = try permute(try reshape(key, [b, sk, heads, d / heads]), split)
  v = try permute(try reshape(value, [b, sk, heads, dv / heads]), split)
  merged = try permute(try attention(q, k, v, widened), split)
  reshape(merged, [b, sq, dv])
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

## The additive mask `attention` hands its backend: `mask` checked against the
## shapes it widens to, plus the causal mask when `causal`; `none` when there
## is neither.
def attention_bias(
  q: Types.Tensor,
  sk: Int,
  mask: Optional[Types.Tensor],
  causal: Bool,
) -> Result[Optional[Types.Tensor]]
  b = q.shape[0]
  heads = q.shape[1]
  sq = q.shape[2]
  given: Optional[Types.Tensor] = none
  case mask
    some(m) ->
      _ = try same_backend(q, m, "attention")
      _ = try same_dtype(q, m, "attention")
      shape = m.shape
      fits = shape == [sq, sk]
      if shape.size == 4 and shape[2] == sq and shape[3] == sk
        fits = (shape[0] == 1 or shape[0] == b) and (shape[1] == 1 or shape[1] == heads)
      end

      return error("attention needs a mask of [#{sq}, #{sk}] or [batch, heads, #{sq}, #{sk}], got #{shape}") if !fits
      given = some(m)
    none -> nil
  end

  return ok(given) if !causal
  future = try causal_mask(q, sq, sk)
  case given
    some(m) ->
      return ok(some(try add(m, future))) if m.shape.size == 2
      ok(some(try add(m, try broadcast_to(try reshape(future, [1, 1, sq, sk]), m.shape))))
    none -> ok(some(future))
  end
end

## `[q_len, k_len]` of `0` where query `i` may read key `j` (`j <= i + k_len -
## q_len`) and `-1e9` where it may not, in `like`'s dtype, built from
## `arange` and `lt`.
def causal_mask(like: Types.Tensor, sq: Int, sk: Int) -> Result[Types.Tensor]
  rows = try arange(sk - sq, sk, 1, Types.Dtype.I32, like.backend)
  cols = try arange(0, sk, 1, Types.Dtype.I32, like.backend)
  later = try lt(
    try broadcast_to(try reshape(rows, [sq, 1]), [sq, sk]),
    try broadcast_to(try reshape(cols, [1, sk]), [sq, sk]),
  )

  blocked = try full([sq, sk], Types.Values.Float(-1.0e9), like.dtype, like.backend)
  mul(try to_dtype(later, like.dtype), blocked)
end
