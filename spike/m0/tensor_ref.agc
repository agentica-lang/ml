## The reference side of the M0 spike: a tensor as plain Agentica data.
##
## This is **not** `tensor/`. It is the spike-sized stand-in that keeps the
## fields `tensor/DESIGN.md` §2 gives a `Tensor` — shape, dtype and the values
## — and the ops the demo needs, no more. `data` is row-major (C order) and
## holds `volume(shape)` floats; `shape == []` is a rank-0 scalar.
##
## Compute is Agentica `Float`, IEEE 754 double, so everything this backend
## makes is `F64`. The `F32` variant exists because the native side computes
## in `float`: the dtype the struct carries is a promise the FFI boundary
## cannot make (see `README.md`).

## The element type of the values in a `Tensor`.
enum Dtype
  F32
  F64
end

## A dense tensor: no backend, no handle, no strides, no views.
struct Tensor
  shape: List[Int]
  dtype: Dtype
  data: List[Float]
end

## The number of elements `shape` names: 1 for `[]`, the product of the
## dimensions otherwise. A negative dimension is an error.
def volume(shape: List[Int]) -> Result[Int]
  total = 1
  for d in shape
    return error("a shape holds no negative dimension, got #{shape}") if d < 0
    total = total * d
  end

  ok(total)
end

## A tensor of `shape`, every element zero.
def zeros(shape: List[Int]) -> Result[Tensor]
  n = try volume(shape)
  ok(Tensor(shape: shape, dtype: Dtype.F64, data: List.repeat(0.0, n)))
end

## A tensor of `shape`, every element one.
def ones(shape: List[Int]) -> Result[Tensor]
  n = try volume(shape)
  ok(Tensor(shape: shape, dtype: Dtype.F64, data: List.repeat(1.0, n)))
end

## A tensor of `shape` from row-major `data`; the count must match the shape.
def from_list(data: List[Float], shape: List[Int]) -> Result[Tensor]
  n = try volume(shape)
  return error("shape #{shape} holds #{n} values, got #{data.size}") if data.size != n

  ok(Tensor(shape: shape, dtype: Dtype.F64, data: data))
end

## The one Float of a rank-0 tensor (what `sum` gives), for the code that
## prints a loss.
def scalar(t: Tensor) -> Result[Float]
  return error("scalar takes a rank-0 tensor, got shape #{t.shape}") if !t.shape.empty?

  ok(t.data[0])
end

## Elementwise `a + b`; both must have one shape.
def add(a: Tensor, b: Tensor) -> Result[Tensor]
  return error("add takes two tensors of one shape, got #{a.shape} and #{b.shape}") if a.shape != b.shape
  values: List[Float] = []
  i = 0
  while i < a.data.size
    values = values.push(a.data[i] + b.data[i])
    i += 1
  end

  ok(Tensor(shape: a.shape, dtype: a.dtype, data: values))
end

## Elementwise `a * b`; both must have one shape. The backward rule of `relu`
## is the one caller in the spike: the upstream gradient times the mask.
def mul(a: Tensor, b: Tensor) -> Result[Tensor]
  return error("mul takes two tensors of one shape, got #{a.shape} and #{b.shape}") if a.shape != b.shape
  values: List[Float] = []
  i = 0
  while i < a.data.size
    values = values.push(a.data[i] * b.data[i])
    i += 1
  end

  ok(Tensor(shape: a.shape, dtype: a.dtype, data: values))
end

## The sum of every element, as a rank-0 tensor (no `axes`, no `keep_dims`: the
## spike has one reduction site and it reduces all of it).
def sum(t: Tensor) -> Tensor = Tensor(shape: [], dtype: t.dtype, data: [t.data.sum])

## `max(0, x)` elementwise. The input `y` is what `relu_backward` needs, so the
## caller keeps it: the spike's graph is a hand-written list of calls, not a
## traced value.
def relu(t: Tensor) -> Tensor
  values = t.data.map do |x|
    return 0.0 if x <= 0.0
    x
  end

  Tensor(shape: t.shape, dtype: t.dtype, data: values)
end

## The backward rule of `relu` at its input `y`: `dy` where `y > 0`, else zero.
def relu_backward(dy: Tensor, y: Tensor) -> Result[Tensor]
  return error("relu_backward takes one shape, got #{dy.shape} and #{y.shape}") if dy.shape != y.shape
  mask = Tensor(
    shape: y.shape,
    dtype: y.dtype,
    data: y.data.map do |x|
      return 0.0 if x <= 0.0
      1.0
    end,
  )

  mul(dy, mask)
end

## The rank-2 transpose. `tensor/DESIGN.md` §3.1 has it as a primitive; the
## backward of `matmul` is where the spike needs it.
def transpose(t: Tensor) -> Result[Tensor]
  return error("transpose takes a rank-2 tensor, got #{t.shape}") if t.shape.size != 2
  rows = t.shape[0]
  cols = t.shape[1]
  values: List[Float] = []
  for c in 0...cols
    for r in 0...rows
      values = values.push(t.data[r * cols + c])
    end
  end

  ok(Tensor(shape: [cols, rows], dtype: t.dtype, data: values))
end

## The rank-2 product `a · b`, `a` being `[rows, inner]` and `b` `[inner, cols]`.
def matmul(a: Tensor, b: Tensor) -> Result[Tensor]
  return error("matmul takes rank-2 tensors, got #{a.shape} and #{b.shape}") if a.shape.size != 2 or b.shape.size != 2
  return error("matmul needs #{a.shape[1]} in the inner dimension, got #{b.shape[0]}") if a.shape[1] != b.shape[0]
  rows = a.shape[0]
  inner = a.shape[1]
  cols = b.shape[1]
  values: List[Float] = []
  for r in 0...rows
    for c in 0...cols
      acc = 0.0
      for k in 0...inner
        acc = acc + a.data[r * inner + k] * b.data[k * cols + c]
      end

      values = values.push(acc)
    end
  end

  ok(Tensor(shape: [rows, cols], dtype: a.dtype, data: values))
end
