## The reference backend: every §3.1 primitive over plain Agentica data.
##
## Pure Agentica, no native code, no `Resource`: a tensor is row-major
## (`List[Float]`, C order) behind `Types.Handle.Plain`, and every op below
## builds a new immutable `Types.Tensor`. This module is the oracle the
## conformance suite of `tensor/DESIGN.md` §11 checks every future backend
## against; `torch/` (M2) will implement this same operation set.
##
## Errors are `Result`/`error`, never halts: the public wrappers in
## `tensor.agc` validate eagerly (`tensor/DESIGN.md` §9) and this module keeps
## only the cheap structural checks that keep indexing total. Functions here
## take the caller to have validated ranks, dtypes and backends already.
##
## The transcendental core (`exp_ref`, `log_ref` and what derives from them)
## is hand-rolled because Agentica exposes no `exp`/`log`/`pow` on `Float`
## (only `Math.sqrt`): range reduction plus a Taylor series in doubles.
## Accuracy is ~1e-15 relative on ordinary inputs; out-of-domain inputs
## (`log` of `x <= 0`, `sqrt` of `x < 0`, `pow` of `b <= 0`) are errors, and
## overflow/underflow saturate to `+inf`/`0.0` instead of halting.
## Non-finite inputs (`NaN`, `+-inf`) are errors at every entry point that
## could halt or hang on them (`exp`, `log`, `sqrt`, `pow`, `sigmoid`,
## `tanh`, `gelu_tanh_approx`): saturation of finite inputs stays legitimate,
## but non-finite storage never enters the core.

import Types from "types"

## Natural log of two, double precision.
LN2 = 0.6931471805599453

## sqrt(2 / pi), the `gelu_tanh_approx` constant, double precision.
SQRT2_OVER_PI = 0.7978845608028654

## The `gelu_tanh_approx` cubic constant.
GELU_C = 0.044715

## Above this `exp_ref` answers `+inf` (as `1.0e308 * 10.0`).
EXP_MAX = 709.0

## Below this `exp_ref` answers `0.0`.
EXP_MIN = -745.0

## `exp(x)`, pure Agentica: range reduction to `x = k*ln2 + r` plus a
## degree-16 Taylor series in `r`, scaled back by `2^k`. Saturates instead
## of overflowing.
def exp_ref(x: Float) -> Float
  return 1.0e308 * 10.0 if x >= EXP_MAX
  return 0.0 if x <= EXP_MIN
  k = (x / LN2).round
  r = x - k.to_f * LN2
  term = 1.0
  total = 1.0
  for n in 1...17
    term = (term * r) / n.to_f
    total = total + term
  end

  scale = total
  if k >= 0
    for _down in 0...k
      scale = scale * 2.0
    end
  else
    for _up in 0...-k
      scale = scale / 2.0
    end
  end

  scale
end

## `log(x)` for `x > 0`, pure Agentica: dyadic normalization to `[1, 2)`
## plus the `atanh` series. Callers reject `x <= 0` first.
def log_ref(x: Float) -> Float
  m = x
  e = 0
  while m >= 2.0
    m = m / 2.0
    e += 1
  end

  while m < 1.0
    m = m * 2.0
    e -= 1
  end

  y = (m - 1.0) / (m + 1.0)
  total = 0.0
  power = y
  for n in 1...61
    if n % 2 == 1
      total = total + power / n.to_f
    end

    power = power * y
  end

  2.0 * total + e.to_f * LN2
end

## `tanh(x)`, pure Agentica, saturating past `|x| > 20`.
def tanh_ref(x: Float) -> Float
  return 1.0 if x >= 20.0
  return -1.0 if x <= -20.0
  e2 = exp_ref(2.0 * x)
  (e2 - 1.0) / (e2 + 1.0)
end

## `sigmoid(x)`, pure Agentica, saturating past `|x| > 30`.
def sigmoid_ref(x: Float) -> Float
  return 1.0 if x >= 30.0
  return 0.0 if x <= -30.0
  1.0 / (1.0 + exp_ref(-x))
end

## A `Reference` tensor of `shape` and `dtype` holding `values`.
def make(shape: List[Int], dtype: Types.Dtype, values: List[Float]) -> Types.Tensor
  Types.Tensor(
    shape: shape,
    dtype: dtype,
    backend: Types.Backend.Reference,
    handle: Types.Handle.Plain(Types.Data(values: values)),
  )
end

## The row-major values behind `t`. A `Native` handle is not this backend's to
## read: the public wrappers only dispatch `Reference` tensors here, and the
## arm answers `error` rather than pretending an empty tensor.
def data_of(t: Types.Tensor) -> Result[List[Float]]
  case t.handle
    Types.Handle.Plain(d) -> ok(d.values)
    Types.Handle.Native(_) ->
      error("the reference backend cannot read a native handle (#{Types.backend_name(t.backend)})")
  end
end

## Row-major strides of `shape`: `strides[i]` is the product of `shape[i+1..]`.
def strides_of(shape: List[Int]) -> List[Int]
  strides: List[Int] = []
  acc = 1
  for d in shape.reverse
    strides = strides.push(acc)
    acc = acc * d
  end

  strides.reverse
end

## The multi-index of linear offset `linear` in a tensor of `shape`.
## Callers only reach here with a positive volume, so no stride is zero.
def multi_of(linear: Int, shape: List[Int], strides: List[Int]) -> List[Int]
  index: List[Int] = []
  rest = linear
  for i in 0...shape.size
    s = strides[i]
    index = index.push(rest / s)
    rest = rest % s
  end

  index
end

## The linear offset of multi-index `index` under `strides`.
def offset_of(index: List[Int], strides: List[Int]) -> Int
  total = 0
  for i in 0...index.size
    total = total + index[i] * strides[i]
  end

  total
end

## `shape` with `axis` removed (used by reductions and `squeeze`).
def drop_axis(shape: List[Int], axis: Int) -> List[Int]
  kept: List[Int] = []
  for i in 0...shape.size
    if i != axis
      kept = kept.push(shape[i])
    end
  end

  kept
end

## Creation: a tensor of `shape` holding `value` everywhere.
def filled(shape: List[Int], dtype: Types.Dtype, value: Float) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  ok(make(shape, dtype, List.repeat(value, n)))
end

## Creation: every element zero.
def zeros(shape: List[Int], dtype: Types.Dtype) -> Result[Types.Tensor] = filled(shape, dtype, 0.0)

## Creation: every element one (`1.0`, `1`, `true` by storage rule).
def ones(shape: List[Int], dtype: Types.Dtype) -> Result[Types.Tensor] = filled(shape, dtype, 1.0)

## Creation: uninitialized memory has no meaning over plain Agentica data,
## so `empty` is deterministic `zeros` (documented, not hidden).
def empty(shape: List[Int], dtype: Types.Dtype) -> Result[Types.Tensor] = filled(shape, dtype, 0.0)

## Creation: every element `value`.
def full(shape: List[Int], value: Types.Values, dtype: Types.Dtype) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  x = try Types.value_to_float(value, dtype)
  ok(make(shape, dtype, List.repeat(x, n)))
end

## Creation: zeros of the shape, dtype and backend of `t` (metadata only).
def like(t: Types.Tensor) -> Result[Types.Tensor] = filled(t.shape, t.dtype, 0.0)

## Creation: `start, start + step, ...` below `stop` (above for `step < 0`).
## Integer dtypes only: positions are indices (`tensor/DESIGN.md` §5).
def arange(start: Int, stop: Int, step: Int, dtype: Types.Dtype) -> Result[Types.Tensor]
  return error("arange needs an Int dtype, got #{Types.dtype_name(dtype)}") if !Types.is_int_dtype(dtype)
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

  n = values.size
  ok(make([n], dtype, values))
end

## Creation: the one `value` as a rank-0 tensor.
def scalar_to_tensor(value: Types.Values, dtype: Types.Dtype) -> Result[Types.Tensor]
  x = try Types.value_to_float(value, dtype)
  ok(make([], dtype, [x]))
end

## Creation: row-major `data` of `shape`; the count must match.
def from_list(data: List[Types.Values], shape: List[Int], dtype: Types.Dtype) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  return error("from_list needs #{n} values for #{shape}, got #{data.size}") if data.size != n
  values: List[Float] = []
  for v in data
    values = values.push(try Types.value_to_float(v, dtype))
  end

  ok(make(shape, dtype, values))
end

## Readback: the values tagged with the tensor's dtype.
def to_list(t: Types.Tensor) -> Result[List[Types.Values]]
  out: List[Types.Values] = []
  for x in try data_of(t)
    out = out.push(try Types.float_to_value(x, t.dtype))
  end

  ok(out)
end

## One unary op over the values.
def map_one(t: Types.Tensor, f: Fn[Float, Float]) -> Result[Types.Tensor]
  xs = try data_of(t)
  ok(make(t.shape, t.dtype, xs.map do |x: Float| f.call(x) end))
end

## One binary op over two same-shape value lists.
def map_two(a: Types.Tensor, b: Types.Tensor, dtype: Types.Dtype, f: Fn[Float, Float, Float]) -> Result[Types.Tensor]
  xs = try data_of(a)
  ys = try data_of(b)
  out: List[Float] = []
  for i in 0...xs.size
    out = out.push(f.call(xs[i], ys[i]))
  end

  ok(make(a.shape, dtype, out))
end

## The cheap structural check every binary op repeats.
def check_pair(a: Types.Tensor, b: Types.Tensor, op: String) -> Result[Bool]
  return error("#{op} needs two tensors of one shape, got #{a.shape} and #{b.shape}") if a.shape != b.shape
  if a.dtype != b.dtype
    return error("#{op} needs two #{Types.dtype_name(a.dtype)} tensors, got #{Types.dtype_name(b.dtype)}")
  end

  ok(true)
end

## Every storage value is finite: non-finite inputs are `error`, never a
## halt or a hang (`log` of `+inf` would divide forever, `round` halts on
## `NaN`). The wrapper documents this as the backend-agnostic input rule;
## the reference enforces it here, and every native backend enforces the
## same rule before touching its own numbers.
def require_finite(xs: List[Float], op: String) -> Result[List[Float]]
  for i in 0...xs.size
    return error("#{op} needs finite values, got a non-finite value at index #{i}") if !Types.is_finite(xs[i])
  end

  ok(xs)
end

## Elementwise `a + b`, float only.
def add(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  try check_pair(a, b, "add")
  return error("add needs a float dtype, got #{Types.dtype_name(a.dtype)}") if !Types.is_float_dtype(a.dtype)
  out = try map_two(a, b, a.dtype) do |x: Float, y: Float| x + y end
  ok(out)
end

## Elementwise `a - b`, float only.
def sub(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  try check_pair(a, b, "sub")
  return error("sub needs a float dtype, got #{Types.dtype_name(a.dtype)}") if !Types.is_float_dtype(a.dtype)
  out = try map_two(a, b, a.dtype) do |x: Float, y: Float| x - y end
  ok(out)
end

## Elementwise `a * b`, float only.
def mul(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  try check_pair(a, b, "mul")
  return error("mul needs a float dtype, got #{Types.dtype_name(a.dtype)}") if !Types.is_float_dtype(a.dtype)
  out = try map_two(a, b, a.dtype) do |x: Float, y: Float| x * y end
  ok(out)
end

## Elementwise `a / b`, float only; a zero divisor is an error, never a halt.
def div(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  try check_pair(a, b, "div")
  return error("div needs a float dtype, got #{Types.dtype_name(a.dtype)}") if !Types.is_float_dtype(a.dtype)
  xs = try data_of(a)
  ys = try data_of(b)
  out: List[Float] = []
  for i in 0...xs.size
    return error("div by zero at index #{i}") if ys[i] == 0.0
    out = out.push(xs[i] / ys[i])
  end

  ok(make(a.shape, a.dtype, out))
end

## Elementwise `-a`, float only.
def neg(t: Types.Tensor) -> Result[Types.Tensor]
  return error("neg needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  out = try map_one(t) do |x: Float| -x end
  ok(out)
end

## Elementwise `exp`, float only, finite inputs only: saturation to `+inf`
## is a legitimate answer, but `NaN`/`inf` in never evaluates.
def exp(t: Types.Tensor) -> Result[Types.Tensor]
  return error("exp needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  xs = try require_finite(try data_of(t), "exp")
  out = xs.map do |x: Float| exp_ref(x) end
  ok(make(t.shape, t.dtype, out))
end

## Elementwise `log`, float only, finite inputs with `x > 0` only:
## `log` of `+inf` would divide forever, so non-finite inputs are errors.
def log(t: Types.Tensor) -> Result[Types.Tensor]
  return error("log needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  xs = try require_finite(try data_of(t), "log")
  out: List[Float] = []
  for x in xs
    return error("log needs x > 0, got #{x}") if x <= 0.0
    out = out.push(log_ref(x))
  end

  ok(make(t.shape, t.dtype, out))
end

## Elementwise `sqrt`, float only, finite inputs with `x >= 0` only.
def sqrt(t: Types.Tensor) -> Result[Types.Tensor]
  return error("sqrt needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  xs = try require_finite(try data_of(t), "sqrt")
  out: List[Float] = []
  for x in xs
    return error("sqrt needs x >= 0, got #{x}") if x < 0.0
    out = out.push(Math.sqrt(x))
  end

  ok(make(t.shape, t.dtype, out))
end

## Elementwise `base ** exponent`, float only, `base <= 0` is an error:
## the `exp`/`log` core cannot reach non-positive bases, so callers square
## with `mul`, and a general signed power waits for a `sign`-carrying rule.
def pow(base: Types.Tensor, exponent: Types.Tensor) -> Result[Types.Tensor]
  try check_pair(base, exponent, "pow")
  return error("pow needs a float dtype, got #{Types.dtype_name(base.dtype)}") if !Types.is_float_dtype(base.dtype)
  bs = try require_finite(try data_of(base), "pow")
  es = try require_finite(try data_of(exponent), "pow")
  out: List[Float] = []
  for i in 0...bs.size
    return error("pow needs base > 0, got #{bs[i]} at index #{i}") if bs[i] <= 0.0
    out = out.push(exp_ref(es[i] * log_ref(bs[i])))
  end

  ok(make(base.shape, base.dtype, out))
end

## Elementwise `max(0, x)`, float only.
def relu(t: Types.Tensor) -> Result[Types.Tensor]
  return error("relu needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  out = try map_one(t) do |x: Float|
    return 0.0 if x <= 0.0
    x
  end

  ok(out)
end

## Elementwise sigmoid, float only, finite inputs only.
def sigmoid(t: Types.Tensor) -> Result[Types.Tensor]
  return error("sigmoid needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  xs = try require_finite(try data_of(t), "sigmoid")
  out = xs.map do |x: Float| sigmoid_ref(x) end
  ok(make(t.shape, t.dtype, out))
end

## Elementwise tanh, float only, finite inputs only.
def tanh(t: Types.Tensor) -> Result[Types.Tensor]
  return error("tanh needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  xs = try require_finite(try data_of(t), "tanh")
  out = xs.map do |x: Float| tanh_ref(x) end
  ok(make(t.shape, t.dtype, out))
end

## The `gelu` approximation of §3.1: `0.5*x*(1+tanh(c*(x+0.044715*x^3))`,
## finite inputs only. The `erf` form is not offered: it cannot be
## expressed over §3.1.
def gelu_tanh_approx(t: Types.Tensor) -> Result[Types.Tensor]
  if !Types.is_float_dtype(t.dtype)
    return error("gelu_tanh_approx needs a float dtype, got #{Types.dtype_name(t.dtype)}")
  end

  xs = try require_finite(try data_of(t), "gelu_tanh_approx")
  out = xs.map do |x: Float|
    inner = SQRT2_OVER_PI * (x + GELU_C * x * x * x)
    0.5 * x * (1.0 + tanh_ref(inner))
  end

  ok(make(t.shape, t.dtype, out))
end

## Elementwise `a == b` as `1.0`/`0.0`, float or int dtypes.
def eq(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  try check_pair(a, b, "eq")
  if !Types.is_float_dtype(a.dtype) and !Types.is_int_dtype(a.dtype)
    return error("eq needs a float or int dtype, got #{Types.dtype_name(a.dtype)}")
  end

  out = try map_two(a, b, Types.Dtype.Bool) do |x: Float, y: Float|
    return 1.0 if x == y
    0.0
  end

  ok(out)
end

## Elementwise `a < b` as `1.0`/`0.0`, float or int dtypes.
def lt(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  try check_pair(a, b, "lt")
  if !Types.is_float_dtype(a.dtype) and !Types.is_int_dtype(a.dtype)
    return error("lt needs a float or int dtype, got #{Types.dtype_name(a.dtype)}")
  end

  out = try map_two(a, b, Types.Dtype.Bool) do |x: Float, y: Float|
    return 1.0 if x < y
    0.0
  end

  ok(out)
end

## `cond ? a : b` elementwise: `cond` is `Bool`, `a` and `b` share shape
## and dtype; every shape is exact, there is no implicit broadcast.
def where(cond: Types.Tensor, a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  try check_pair(a, b, "where")
  return error("where needs a Bool condition, got #{Types.dtype_name(cond.dtype)}") if cond.dtype != Types.Dtype.Bool
  return error("where needs a condition of shape #{a.shape}, got #{cond.shape}") if cond.shape != a.shape
  cs = try data_of(cond)
  xs = try data_of(a)
  ys = try data_of(b)
  out: List[Float] = []
  for i in 0...xs.size
    if cs[i] == 1.0
      out = out.push(xs[i])
    else
      out = out.push(ys[i])
    end
  end

  ok(make(a.shape, a.dtype, out))
end

## Elementwise clamp to `[lo, hi]`, float only.
def clamp(t: Types.Tensor, lo: Float, hi: Float) -> Result[Types.Tensor]
  return error("clamp needs lo <= hi, got #{lo} and #{hi}") if lo > hi
  return error("clamp needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  out = try map_one(t) do |x: Float| x.clamp(lo, hi) end
  ok(out)
end

## Reduction axes: `none` is every axis, otherwise each entry names an axis
## once; the order of the answer never depends on the order given.
def normalize_axes(axes: Optional[List[Int]], rank: Int, op: String) -> Result[List[Int]]
  case axes
    some(given) ->
      for ax in given
        return error("#{op} needs axes in 0...#{rank}, got #{ax}") if ax < 0 or ax >= rank
      end

      return error("#{op} names an axis twice, got #{given}") if given.uniq.size != given.size
      ok(given)
    none -> ok((0...rank).to_a)
  end
end

## The shape left after reducing `axes`: dropped, or `1` with `keep_dims`.
def reduced_shape(shape: List[Int], axes: List[Int], keep_dims: Bool) -> List[Int]
  out: List[Int] = []
  for i in 0...shape.size
    next if !keep_dims and axes.include?(i)
    if axes.include?(i)
      out = out.push(1)
    else
      out = out.push(shape[i])
    end
  end

  out
end

## The product of the reduced dimensions (the count `mean` divides by).
def reduced_count(shape: List[Int], axes: List[Int]) -> Int
  total = 1
  for ax in axes
    total = total * shape[ax]
  end

  total
end

## Sums every element into the reduced shape.
def sum_axes(t: Types.Tensor, axes: List[Int], keep_dims: Bool) -> Result[Types.Tensor]
  shape = t.shape
  out_shape = reduced_shape(shape, axes, keep_dims)
  in_strides = strides_of(shape)
  out_strides = strides_of(out_shape)
  n = try Types.volume(out_shape)
  acc: List[Float] = List.repeat(0.0, n)
  xs = try data_of(t)
  for linear in 0...xs.size
    index = multi_of(linear, shape, in_strides)
    out_index: List[Int] = []
    for i in 0...shape.size
      if axes.include?(i)
        if keep_dims
          out_index = out_index.push(0)
        end
      else
        out_index = out_index.push(index[i])
      end
    end

    o = offset_of(out_index, out_strides)
    acc = acc.set(o, acc[o] + xs[linear])
  end

  ok(make(out_shape, t.dtype, acc))
end

## Reduces `t` over `axes` (`none` is every element) to a float tensor.
def sum(t: Types.Tensor, axes: Optional[List[Int]], keep_dims: Bool) -> Result[Types.Tensor]
  return error("sum needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  axs = try normalize_axes(axes, t.shape.size, "sum")
  sum_axes(t, axs, keep_dims)
end

## `sum` divided by the reduced count; reducing zero elements is an error.
def mean(t: Types.Tensor, axes: Optional[List[Int]], keep_dims: Bool) -> Result[Types.Tensor]
  return error("mean needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  axs = try normalize_axes(axes, t.shape.size, "mean")
  count = reduced_count(t.shape, axs)
  return error("mean of zero elements is an error") if count == 0
  total = try sum_axes(t, axs, keep_dims)
  out: List[Float] = []
  for x in try data_of(total)
    out = out.push(x / count.to_f)
  end

  ok(make(total.shape, total.dtype, out))
end

## The largest value per reduced block; an empty block is an error.
def max(t: Types.Tensor, axes: Optional[List[Int]], keep_dims: Bool) -> Result[Types.Tensor]
  return error("max needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  axs = try normalize_axes(axes, t.shape.size, "max")
  return error("max of zero elements is an error") if try Types.volume(t.shape) == 0
  shape = t.shape
  out_shape = reduced_shape(shape, axs, keep_dims)
  in_strides = strides_of(shape)
  out_strides = strides_of(out_shape)
  n = try Types.volume(out_shape)
  best: List[Float] = List.repeat(0.0, n)
  seen: List[Bool] = List.repeat(false, n)
  xs = try data_of(t)
  for linear in 0...xs.size
    index = multi_of(linear, shape, in_strides)
    out_index: List[Int] = []
    for i in 0...shape.size
      if axs.include?(i)
        if keep_dims
          out_index = out_index.push(0)
        end
      else
        out_index = out_index.push(index[i])
      end
    end

    o = offset_of(out_index, out_strides)
    if !seen[o] or xs[linear] > best[o]
      best = best.set(o, xs[linear])
      seen = seen.set(o, true)
    end
  end

  ok(make(out_shape, t.dtype, best))
end

## The index of the largest value per reduced block as `I64`: for one axis
## the position along it, for several the flat row-major offset inside the
## reduced block; ties keep the first index. An empty block is an error.
def argmax(t: Types.Tensor, axes: Optional[List[Int]], keep_dims: Bool) -> Result[Types.Tensor]
  return error("argmax needs a float dtype, got #{Types.dtype_name(t.dtype)}") if !Types.is_float_dtype(t.dtype)
  axs = try normalize_axes(axes, t.shape.size, "argmax")
  return error("argmax of zero elements is an error") if try Types.volume(t.shape) == 0
  shape = t.shape
  out_shape = reduced_shape(shape, axs, keep_dims)
  in_strides = strides_of(shape)
  out_strides = strides_of(out_shape)
  sub_shape: List[Int] = []
  for ax in axs
    sub_shape = sub_shape.push(shape[ax])
  end

  sub_strides = strides_of(sub_shape)
  n = try Types.volume(out_shape)
  best: List[Float] = List.repeat(0.0, n)
  pos: List[Float] = List.repeat(0.0, n)
  seen: List[Bool] = List.repeat(false, n)
  xs = try data_of(t)
  for linear in 0...xs.size
    index = multi_of(linear, shape, in_strides)
    out_index: List[Int] = []
    sub_index: List[Int] = []
    for i in 0...shape.size
      if axs.include?(i)
        if keep_dims
          out_index = out_index.push(0)
        end

        sub_index = sub_index.push(index[i])
      else
        out_index = out_index.push(index[i])
      end
    end

    o = offset_of(out_index, out_strides)
    if !seen[o] or xs[linear] > best[o]
      best = best.set(o, xs[linear])
      pos = pos.set(o, offset_of(sub_index, sub_strides).to_f)
      seen = seen.set(o, true)
    end
  end

  ok(make(out_shape, Types.Dtype.I64, pos))
end

## The rank-2 product `[rows, inner] . [inner, cols]`, float only.
def matmul(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  return error("matmul needs rank-2 tensors, got #{a.shape} and #{b.shape}") if a.shape.size != 2 or b.shape.size != 2
  return error("matmul needs #{a.shape[1]} in the inner dimension, got #{b.shape[0]}") if a.shape[1] != b.shape[0]
  if !Types.is_float_dtype(a.dtype) or !Types.is_float_dtype(b.dtype)
    return error("matmul needs a float dtype, got #{Types.dtype_name(a.dtype)}")
  end

  if a.dtype != b.dtype
    return error("matmul needs two #{Types.dtype_name(a.dtype)} tensors, got #{Types.dtype_name(b.dtype)}")
  end

  rows = a.shape[0]
  inner = a.shape[1]
  cols = b.shape[1]
  xs = try data_of(a)
  ys = try data_of(b)
  out: List[Float] = []
  for r in 0...rows
    for c in 0...cols
      acc = 0.0
      for k in 0...inner
        acc = acc + xs[r * inner + k] * ys[k * cols + c]
      end

      out = out.push(acc)
    end
  end

  ok(make([rows, cols], a.dtype, out))
end

## The batched product `[B..., m, k] . [B..., k, n]`: the batch prefix is
## identical (widen with `broadcast_to` first), float only.
def batch_matmul(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  return error("batch_matmul needs rank >= 2, got #{a.shape} and #{b.shape}") if a.shape.size < 2 or b.shape.size < 2
  return error("batch_matmul needs one rank, got #{a.shape} and #{b.shape}") if a.shape.size != b.shape.size
  rank = a.shape.size
  for i in 0...(rank - 2)
    return error("batch_matmul needs one batch shape, got #{a.shape} and #{b.shape}") if a.shape[i] != b.shape[i]
  end

  m = a.shape[rank - 2]
  ka = a.shape[rank - 1]
  kb = b.shape[rank - 2]
  n = b.shape[rank - 1]
  return error("batch_matmul needs #{ka} in the inner dimension, got #{kb}") if ka != kb
  if !Types.is_float_dtype(a.dtype) or !Types.is_float_dtype(b.dtype)
    return error("batch_matmul needs a float dtype, got #{Types.dtype_name(a.dtype)}")
  end

  if a.dtype != b.dtype
    return error("batch_matmul needs two #{Types.dtype_name(a.dtype)} tensors, got #{Types.dtype_name(b.dtype)}")
  end

  batch_shape: List[Int] = []
  for i in 0...(rank - 2)
    batch_shape = batch_shape.push(a.shape[i])
  end

  batches = try Types.volume(batch_shape)
  plane_a = m * ka
  plane_b = kb * n
  xs = try data_of(a)
  ys = try data_of(b)
  out: List[Float] = []
  for batch in 0...batches
    for r in 0...m
      for c in 0...n
        acc = 0.0
        for k in 0...ka
          acc = acc + xs[batch * plane_a + r * ka + k] * ys[batch * plane_b + k * n + c]
        end

        out = out.push(acc)
      end
    end
  end

  ok(make(batch_shape.push(m).push(n), a.dtype, out))
end

## The transpose: reverses every axis (for rank 2 this swaps rows and
## columns, for rank 0/1 it is the identity). A true axis permutation is
## derived from `slice`/`stack` where needed (see `tensor.agc`).
def transpose(t: Types.Tensor) -> Result[Types.Tensor]
  shape = t.shape
  rank = shape.size
  xs = try data_of(t)
  return ok(make(shape, t.dtype, xs)) if rank < 2
  flipped = shape.reverse
  in_strides = strides_of(shape)
  out_strides = strides_of(flipped)
  out: List[Float] = []
  for linear in 0...xs.size
    index = multi_of(linear, flipped, out_strides)
    out = out.push(xs[offset_of(index.reverse, in_strides)])
  end

  ok(make(flipped, t.dtype, out))
end

## A tensor of `shape` with the same values: the volumes must match.
def reshape(t: Types.Tensor, shape: List[Int]) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  xs = try data_of(t)
  return error("reshape needs #{n} values for #{shape}, got #{xs.size}") if xs.size != n
  ok(make(shape, t.dtype, xs))
end

## Drops size-1 dimensions: every one for `none`, or the named `axis`,
## which must be `1`.
def squeeze(t: Types.Tensor, axis: Optional[Int]) -> Result[Types.Tensor]
  case axis
    some(ax) ->
      return error("squeeze needs an axis in 0...#{t.shape.size}, got #{ax}") if ax < 0 or ax >= t.shape.size
      return error("squeeze needs a dimension of size 1 at #{ax}, got #{t.shape[ax]}") if t.shape[ax] != 1
      ok(make(drop_axis(t.shape, ax), t.dtype, try data_of(t)))
    none ->
      kept: List[Int] = []
      for d in t.shape
        if d != 1
          kept = kept.push(d)
        end
      end

      ok(make(kept, t.dtype, try data_of(t)))
  end
end

## Inserts a size-1 dimension at `axis`.
def unsqueeze(t: Types.Tensor, axis: Int) -> Result[Types.Tensor]
  return error("unsqueeze needs an axis in 0...#{t.shape.size}, got #{axis}") if axis < 0 or axis > t.shape.size
  shape = t.shape
  out: List[Int] = []
  for i in 0...shape.size
    if i == axis
      out = out.push(1)
    end

    out = out.push(shape[i])
  end

  if axis == shape.size
    out = out.push(1)
  end

  ok(make(out, t.dtype, try data_of(t)))
end

## The `[starts[i], ends[i])` window per axis (no steps in M1).
def slice(t: Types.Tensor, starts: List[Int], ends: List[Int]) -> Result[Types.Tensor]
  rank = t.shape.size
  return error("slice needs one start per axis, got #{starts} for #{t.shape}") if starts.size != rank
  return error("slice needs one end per axis, got #{ends} for #{t.shape}") if ends.size != rank
  for i in 0...rank
    if starts[i] < 0 or ends[i] < starts[i] or ends[i] > t.shape[i]
      return error("slice needs 0 <= start <= end <= #{t.shape[i]}, got #{starts[i]}..#{ends[i]}")
    end
  end

  shape: List[Int] = []
  for i in 0...rank
    shape = shape.push(ends[i] - starts[i])
  end

  in_strides = strides_of(t.shape)
  out_strides = strides_of(shape)
  xs = try data_of(t)
  n = try Types.volume(shape)
  out: List[Float] = []
  for linear in 0...n
    index = multi_of(linear, shape, out_strides)
    full: List[Int] = []
    for i in 0...rank
      full = full.push(index[i] + starts[i])
    end

    out = out.push(xs[offset_of(full, in_strides)])
  end

  ok(make(shape, t.dtype, out))
end

## Joins `tensors` along `axis`: one rank, one dtype, every other dimension
## equal; the axis sizes add up.
def concat(tensors: List[Types.Tensor], axis: Int) -> Result[Types.Tensor]
  return error("concat needs at least one tensor") if tensors.empty?
  first = tensors[0]
  rank = first.shape.size
  return error("concat needs an axis in 0...#{rank}, got #{axis}") if axis < 0 or axis >= rank
  for t in tensors
    return error("concat needs one rank, got #{first.shape} and #{t.shape}") if t.shape.size != rank
    if t.dtype != first.dtype
      return error("concat needs one dtype, got #{Types.dtype_name(first.dtype)} and #{Types.dtype_name(t.dtype)}")
    end

    for i in 0...rank
      if i != axis and t.shape[i] != first.shape[i]
        return error("concat needs dimension #{i} of size #{first.shape[i]}, got #{t.shape[i]}")
      end
    end
  end

  shape = first.shape
  out_shape: List[Int] = []
  for i in 0...rank
    if i != axis
      out_shape = out_shape.push(shape[i])
    else
      total = 0
      for t in tensors
        total = total + t.shape[axis]
      end

      out_shape = out_shape.push(total)
    end
  end

  in_strides: List[List[Int]] = []
  for t in tensors
    in_strides = in_strides.push(strides_of(t.shape))
  end

  out_strides = strides_of(out_shape)
  n = try Types.volume(out_shape)
  datas: List[List[Float]] = []
  for t in tensors
    datas = datas.push(try data_of(t))
  end

  out: List[Float] = []
  for linear in 0...n
    index = multi_of(linear, out_shape, out_strides)
    pos = index[axis]
    picked = 0
    local = pos
    for k in 0...tensors.size
      width = tensors[k].shape[axis]
      if local < width
        picked = k
        break
      end

      local = local - width
    end

    full = index.set(axis, local)
    out = out.push(datas[picked][offset_of(full, in_strides[picked])])
  end

  ok(make(out_shape, first.dtype, out))
end

## Stacks `tensors` along a new `axis`: every shape, dtype and backend
## equal (the backend check lives in the wrapper).
def stack(tensors: List[Types.Tensor], axis: Int) -> Result[Types.Tensor]
  return error("stack needs at least one tensor") if tensors.empty?
  first = tensors[0]
  rank = first.shape.size
  return error("stack needs an axis in 0...#{rank}, got #{axis}") if axis < 0 or axis > rank
  for t in tensors
    return error("stack needs one shape, got #{first.shape} and #{t.shape}") if t.shape != first.shape
    if t.dtype != first.dtype
      return error("stack needs one dtype, got #{Types.dtype_name(first.dtype)} and #{Types.dtype_name(t.dtype)}")
    end
  end

  widened: List[Types.Tensor] = []
  for t in tensors
    widened = widened.push(try unsqueeze(t, axis))
  end

  concat(widened, axis)
end

## Views `t` as `shape`: trailing axes align, size-1 dimensions widen.
def broadcast_to(t: Types.Tensor, shape: List[Int]) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  rank = t.shape.size
  out_rank = shape.size
  return error("broadcast_to cannot drop axes, got #{t.shape} into #{shape}") if out_rank < rank
  lead = out_rank - rank
  for i in 0...rank
    dim = t.shape[i]
    want = shape[lead + i]
    return error("broadcast_to needs #{dim} to stay #{dim} or widen from 1, got #{want}") if dim != want and dim != 1
  end

  in_strides = strides_of(t.shape)
  out_strides = strides_of(shape)
  xs = try data_of(t)
  out: List[Float] = []
  for linear in 0...n
    index = multi_of(linear, shape, out_strides)
    full: List[Int] = []
    for i in 0...rank
      if t.shape[i] == 1
        full = full.push(0)
      else
        full = full.push(index[lead + i])
      end
    end

    out = out.push(xs[offset_of(full, in_strides)])
  end

  ok(make(shape, t.dtype, out))
end

## Rows of `source` at `indices`: an integer tensor of any int dtype, every
## index in range; the answer is `indices.shape ++ source.shape[1:]`.
def gather(source: Types.Tensor, indices: Types.Tensor) -> Result[Types.Tensor]
  return error("gather needs a rank >= 1 source, got #{source.shape}") if source.shape.size < 1
  if !Types.is_int_dtype(indices.dtype)
    return error("gather needs integer indices, got #{Types.dtype_name(indices.dtype)}")
  end

  rows = source.shape[0]
  tail: List[Int] = []
  for i in 1...source.shape.size
    tail = tail.push(source.shape[i])
  end

  out_shape = indices.shape + tail
  row = try Types.volume(tail)
  src = try data_of(source)
  out: List[Float] = []
  for x in try data_of(indices)
    i = try x.to_i
    return error("gather needs indices in 0...#{rows}, got #{i}") if i < 0 or i >= rows
    for k in 0...row
      out = out.push(src[i * row + k])
    end
  end

  ok(make(out_shape, source.dtype, out))
end

## Moves `t` to `dtype`: float to float keeps the value, float to int
## truncates toward zero, anything to `Bool` tests nonzero, and int widths
## are range-checked. Int targets error on overflow and `NaN`, never
## wrap-around; `Bool` keeps the nonzero rule for every input, so `NaN`
## is `true` (it is nonzero, as in torch) rather than an error.
def to_dtype(t: Types.Tensor, dtype: Types.Dtype) -> Result[Types.Tensor]
  out: List[Float] = []
  for x in try data_of(t)
    out = out.push(try convert_one(x, t.dtype, dtype))
  end

  ok(make(t.shape, dtype, out))
end

## Moves `t` to `device`: reference tensors live on `Cpu` only.
def to_device(t: Types.Tensor, device: Types.Device) -> Result[Types.Tensor]
  case device
    Types.Device.Cpu -> ok(make(t.shape, t.dtype, try data_of(t)))
    Types.Device.Cuda(n) -> error("reference tensors live on Cpu, not Cuda(#{n})")
  end
end

## The identity move: values are immutable, so a copy shares nothing.
def copy(t: Types.Tensor) -> Result[Types.Tensor]
  out: List[Float] = []
  for x in try data_of(t)
    out = out.push(x)
  end

  ok(make(t.shape, t.dtype, out))
end

## The identity move: the reference has no strides, every tensor is
## contiguous by construction.
def contiguous(t: Types.Tensor) -> Result[Types.Tensor] = copy(t)

## Convolution: the cross-correlation of `input` with `weight` along the last
## axis, an explicit loop over batch, output channel, position, input channel
## and kernel tap. `input` is `[C_in, L]` or `[N, C_in, L]`, `weight` is
## `[C_out, C_in, K]`, and the window moves by `stride` with `padding` zeros on
## each side; there is no dilation and there are no groups (`DESIGN.md` §3.1
## admits stride/padding-1D).
def conv1d(input: Types.Tensor, weight: Types.Tensor, stride: Int, padding: Int) -> Result[Types.Tensor]
  rank = input.shape.size
  return error("conv1d needs a rank-2 or rank-3 input, got #{input.shape}") if rank < 2 or rank > 3
  return error("conv1d needs a rank-3 weight, got #{weight.shape}") if weight.shape.size != 3
  return error("conv1d needs stride >= 1, got #{stride}") if stride < 1
  return error("conv1d needs padding >= 0, got #{padding}") if padding < 0
  if !Types.is_float_dtype(input.dtype) or input.dtype != weight.dtype
    return error("conv1d needs one float dtype, got #{Types.dtype_name(input.dtype)}")
  end

  n = 1
  if rank == 3
    n = input.shape[0]
  end

  in_channels = input.shape[rank - 2]
  length = input.shape[rank - 1]
  out_channels = weight.shape[0]
  kernel = weight.shape[2]
  return error("conv1d needs #{in_channels} input channels, got #{weight.shape[1]}") if weight.shape[1] != in_channels
  padded = length + 2 * padding
  return error("conv1d needs a window of #{kernel} positions in #{length} padded to #{padded}") if padded < kernel
  out_length = (padded - kernel) / stride + 1
  xs = try data_of(input)
  ws = try data_of(weight)
  out: List[Float] = []
  for bn in 0...n
    for co in 0...out_channels
      for o in 0...out_length
        acc = 0.0
        for ci in 0...in_channels
          for k in 0...kernel
            pos = o * stride + k - padding
            if pos >= 0 and pos < length
              acc = acc + xs[(bn * in_channels + ci) * length + pos] * ws[(co * in_channels + ci) * kernel + k]
            end
          end
        end

        out = out.push(acc)
      end
    end
  end

  shape: List[Int] = []
  if rank == 3
    shape = [n, out_channels, out_length]
  else
    shape = [out_channels, out_length]
  end

  ok(make(shape, input.dtype, out))
end

## The mean over a window of `kernel` positions, moved by `stride`, with
## `padding` zeros on each side, dividing by `kernel` (`count_include_pad`:
## a window off the edge averages over the zeros it was padded with). `input`
## is `[C, L]` or `[N, C, L]`; the last axis becomes
## `(L + 2*padding - kernel) / stride + 1`.
def avg_pool1d(input: Types.Tensor, kernel: Int, stride: Int, padding: Int) -> Result[Types.Tensor]
  rank = input.shape.size
  return error("avg_pool1d needs a rank-2 or rank-3 input, got #{input.shape}") if rank < 2 or rank > 3
  return error("avg_pool1d needs kernel >= 1, got #{kernel}") if kernel < 1
  return error("avg_pool1d needs stride >= 1, got #{stride}") if stride < 1
  return error("avg_pool1d needs padding >= 0, got #{padding}") if padding < 0
  if !Types.is_float_dtype(input.dtype)
    return error("avg_pool1d needs a float dtype, got #{Types.dtype_name(input.dtype)}")
  end

  if padding > kernel / 2
    return error("avg_pool1d needs padding <= #{kernel / 2} for a kernel of #{kernel}, got #{padding}")
  end

  n = 1
  if rank == 3
    n = input.shape[0]
  end

  channels = input.shape[rank - 2]
  length = input.shape[rank - 1]
  padded = length + 2 * padding
  return error("avg_pool1d needs a window of #{kernel} positions in #{length} padded to #{padded}") if padded < kernel
  out_length = (padded - kernel) / stride + 1
  xs = try data_of(input)
  out: List[Float] = []
  for bn in 0...n
    for c in 0...channels
      for o in 0...out_length
        acc = 0.0
        for k in 0...kernel
          pos = o * stride + k - padding
          if pos >= 0 and pos < length
            acc = acc + xs[(bn * channels + c) * length + pos]
          end
        end

        out = out.push(acc / kernel.to_f)
      end
    end
  end

  shape: List[Int] = []
  if rank == 3
    shape = [n, channels, out_length]
  else
    shape = [channels, out_length]
  end

  ok(make(shape, input.dtype, out))
end

private

## One value across the dtype move of `to_dtype`.
def convert_one(x: Float, from: Types.Dtype, to: Types.Dtype) -> Result[Float]
  return ok(x) if Types.is_float_dtype(to) and Types.is_float_dtype(from)
  return ok(x) if Types.is_float_dtype(to) and (Types.is_int_dtype(from) or from == Types.Dtype.Bool)
  if to == Types.Dtype.Bool
    return ok(0.0) if x == 0.0
    return ok(1.0)
  end

  i = try x.to_i
  try Types.check_int_range(i, to)
  ok(i.to_f)
end
