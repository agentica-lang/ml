## Shared types of the `tensor/` package (`tensor/DESIGN.md` §2).
##
## This module is the dependency leaf: it imports nothing, so both the
## reference backend (`reference.agc`) and the public API (`tensor.agc`) can
## use it without an import cycle. The public entry point is `tensor.agc`,
## which re-exports every type here; tests and future backends go through
## `Tensor.*`, never through this module directly.
##
## Storage rule (design of record: `tensor/DESIGN.md` §2): the reference
## backend holds plain Agentica data, row-major (C order), as `List[Float]`
## inside `Data`. Float-class dtypes hold the value itself; int-class dtypes
## hold exactly integral values; `Bool` holds `1.0`/`0.0`. `Values` is the
## dtype-tagged crossing type for `from_list`/`to_list` only.

enum Dtype
  F32
  F16
  F64
  I8
  I16
  I32
  I64
  U8
  Bool
end

enum Device
  Cpu
  Cuda(Int)
end

enum Backend
  Reference
  Torch(Device)
  Xla(Device)
  Candle(Device)
end

## Plain Agentica backing store: the row-major values of one tensor.
struct Data
  values: List[Float]
end

## How a tensor's values are held. M1 has the `Plain` arm only: the reference
## backend owns Agentica data and needs no arena. `Native(Resource)` arrives
## with the first native backend (`torch/`, M2), which will use
## `Native.arena` per call/step (`tensor/DESIGN.md` §8); arenas are never
## faked for `Plain`.
enum Handle
  Plain(Data)
end

## One dtype-tagged value: the element type of `from_list`/`to_list`.
enum Values
  Float(Float)
  Int(Int)
  Bool(Bool)
end

## An immutable tensor value. Every op returns a new `Tensor`.
struct Tensor
  shape: List[Int]
  dtype: Dtype
  backend: Backend
  handle: Handle
end

## The two results of `top_k`: `values` descending, `indices` their positions.
struct TopK
  values: Tensor
  indices: Tensor
end

## The number of elements `shape` names: `1` for `[]`, the product otherwise.
def volume(shape: List[Int]) -> Result[Int]
  total = 1
  for d in shape
    return error("a shape holds no negative dimension, got #{shape}") if d < 0
    total = total * d
  end

  ok(total)
end

## True for the dtypes that hold non-integer numerics.
def is_float_dtype(dtype: Dtype) -> Bool
  case dtype
    Dtype.F32 -> true
    Dtype.F16 -> true
    Dtype.F64 -> true
    _ -> false
  end
end

## True for the dtypes that hold integers.
def is_int_dtype(dtype: Dtype) -> Bool
  case dtype
    Dtype.I8 -> true
    Dtype.I16 -> true
    Dtype.I32 -> true
    Dtype.I64 -> true
    Dtype.U8 -> true
    _ -> false
  end
end

## The short name used in error messages.
def dtype_name(dtype: Dtype) -> String
  case dtype
    Dtype.F32 -> "F32"
    Dtype.F16 -> "F16"
    Dtype.F64 -> "F64"
    Dtype.I8 -> "I8"
    Dtype.I16 -> "I16"
    Dtype.I32 -> "I32"
    Dtype.I64 -> "I64"
    Dtype.U8 -> "U8"
    Dtype.Bool -> "Bool"
  end
end

## The short name used in error messages.
def device_name(device: Device) -> String
  case device
    Device.Cpu -> "Cpu"
    Device.Cuda(n) -> "Cuda(#{n})"
  end
end

## The short name used in error messages.
def backend_name(backend: Backend) -> String
  case backend
    Backend.Reference -> "Reference"
    Backend.Torch(device) -> "Torch(#{device_name(device)})"
    Backend.Xla(device) -> "Xla(#{device_name(device)})"
    Backend.Candle(device) -> "Candle(#{device_name(device)})"
  end
end

## The storage float of one tagged value: checks the variant fits `dtype`
## and, for int dtypes, that the integer fits the width.
def value_to_float(value: Values, dtype: Dtype) -> Result[Float]
  case value
    Values.Float(x) ->
      return error("a #{dtype_name(dtype)} tensor takes no Float value, got #{x}") if !is_float_dtype(dtype)
      ok(x)
    Values.Int(i) ->
      return error("a #{dtype_name(dtype)} tensor takes no Int value, got #{i}") if !is_int_dtype(dtype)
      try check_int_range(i, dtype)
      ok(i.to_f)
    Values.Bool(b) ->
      return error("a #{dtype_name(dtype)} tensor takes no Bool value") if dtype != Dtype.Bool
      if b
        ok(1.0)
      else
        ok(0.0)
      end
  end
end

## Tags one storage float with `dtype`: the inverse of `value_to_float`.
## Int dtypes need an integral value in range, `Bool` needs `1.0`/`0.0`.
def float_to_value(x: Float, dtype: Dtype) -> Result[Values]
  case dtype
    Dtype.F32 -> ok(Values.Float(x))
    Dtype.F16 -> ok(Values.Float(x))
    Dtype.F64 -> ok(Values.Float(x))
    Dtype.Bool ->
      return ok(Values.Bool(true)) if x == 1.0
      return ok(Values.Bool(false)) if x == 0.0
      error("a Bool tensor holds 1.0 or 0.0, got #{x}")
    _ ->
      i = try x.to_i
      return error("an integer tensor holds whole values, got #{x}") if i.to_f != x
      try check_int_range(i, dtype)
      ok(Values.Int(i))
  end
end

## `i` fits the width of the int `dtype` (callers check the class first).
def check_int_range(i: Int, dtype: Dtype) -> Result[Int]
  case dtype
    Dtype.I8 -> return error("I8 holds -128..127, got #{i}") if i < -128 or i > 127
    Dtype.I16 -> return error("I16 holds -32768..32767, got #{i}") if i < -32768 or i > 32767
    Dtype.I32 -> return error("I32 holds -2147483648..2147483647, got #{i}") if i < -2147483648 or i > 2147483647
    Dtype.U8 -> return error("U8 holds 0..255, got #{i}") if i < 0 or i > 255
    _ -> return ok(i)
  end

  ok(i)
end
