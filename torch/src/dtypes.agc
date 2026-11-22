## Dtype vocabulary of the torch backend: the predicates, the names used in
## error messages, the code that crosses to C, and the two conversions between
## a native storage value (a `Float`, as the boundary carries) and a
## dtype-tagged `Values`.
##
## `tensor/` keeps the same three tables in `tensor/src/types.agc`, which the
## `tensor/` package does not export: a backend validates its own storage
## crossing (`tensor/DESIGN.md` §9: "backends may re-check cheaply"), so the
## torch backend carries its own copy rather than reaching into another
## package's internals. The codes below are also in `torch/native/torch.c`
## (`torch_scalar_type`): one table, two files, and the conformance tests are
## what keeps them agreeing.

import Tensor from "pkg/tensor/tensor"

## True for the dtypes that hold non-integer numerics.
def is_float_dtype(dtype: Tensor.Dtype) -> Bool
  case dtype
    Tensor.Dtype.F32 -> true
    Tensor.Dtype.F16 -> true
    Tensor.Dtype.F64 -> true
    _ -> false
  end
end

## True for the dtypes that hold integers.
def is_int_dtype(dtype: Tensor.Dtype) -> Bool
  case dtype
    Tensor.Dtype.I8 -> true
    Tensor.Dtype.I16 -> true
    Tensor.Dtype.I32 -> true
    Tensor.Dtype.I64 -> true
    Tensor.Dtype.U8 -> true
    _ -> false
  end
end

## The short name used in error messages.
def name(dtype: Tensor.Dtype) -> String
  case dtype
    Tensor.Dtype.F32 -> "F32"
    Tensor.Dtype.F16 -> "F16"
    Tensor.Dtype.F64 -> "F64"
    Tensor.Dtype.I8 -> "I8"
    Tensor.Dtype.I16 -> "I16"
    Tensor.Dtype.I32 -> "I32"
    Tensor.Dtype.I64 -> "I64"
    Tensor.Dtype.U8 -> "U8"
    Tensor.Dtype.Bool -> "Bool"
  end
end

## The code the native layer reads: `torch_scalar_type` in `native/torch.c`
## maps it to a libtorch scalar type. The order is the one libtorch's CPU
## release offers for these nine, and it is a wire detail, not an API.
def code(dtype: Tensor.Dtype) -> Int
  case dtype
    Tensor.Dtype.F32 -> 0
    Tensor.Dtype.F16 -> 1
    Tensor.Dtype.F64 -> 2
    Tensor.Dtype.I8 -> 3
    Tensor.Dtype.I16 -> 4
    Tensor.Dtype.I32 -> 5
    Tensor.Dtype.I64 -> 6
    Tensor.Dtype.U8 -> 7
    Tensor.Dtype.Bool -> 8
  end
end

## `i` fits the width of the int `dtype` (callers check the class first).
def check_int_range(i: Int, dtype: Tensor.Dtype) -> Result[Int]
  case dtype
    Tensor.Dtype.I8 -> return error("I8 holds -128..127, got #{i}") if i < -128 or i > 127
    Tensor.Dtype.I16 -> return error("I16 holds -32768..32767, got #{i}") if i < -32768 or i > 32767
    Tensor.Dtype.I32 ->
      return error("I32 holds -2147483648..2147483647, got #{i}") if i < -2147483648 or i > 2147483647

    Tensor.Dtype.U8 -> return error("U8 holds 0..255, got #{i}") if i < 0 or i > 255
    _ -> return ok(i)
  end

  ok(i)
end

## The storage float of one tagged value: checks the variant fits `dtype`
## and, for int dtypes, that the integer fits the width.
def value_to_float(value: Tensor.Values, dtype: Tensor.Dtype) -> Result[Float]
  case value
    Tensor.Values.Float(x) ->
      return error("a #{name(dtype)} tensor takes no Float value, got #{x}") if !is_float_dtype(dtype)
      ok(x)
    Tensor.Values.Int(i) ->
      return error("a #{name(dtype)} tensor takes no Int value, got #{i}") if !is_int_dtype(dtype)
      try check_int_range(i, dtype)
      ok(i.to_f)
    Tensor.Values.Bool(b) ->
      return error("a #{name(dtype)} tensor takes no Bool value") if dtype != Tensor.Dtype.Bool
      if b
        ok(1.0)
      else
        ok(0.0)
      end
  end
end

## Tags one storage float with `dtype`: the inverse of `value_to_float`.
## Int dtypes need an integral value in range, `Bool` needs `1.0`/`0.0`.
def float_to_value(x: Float, dtype: Tensor.Dtype) -> Result[Tensor.Values]
  case dtype
    Tensor.Dtype.F32 -> ok(Tensor.Values.Float(x))
    Tensor.Dtype.F16 -> ok(Tensor.Values.Float(x))
    Tensor.Dtype.F64 -> ok(Tensor.Values.Float(x))
    Tensor.Dtype.Bool ->
      return ok(Tensor.Values.Bool(true)) if x == 1.0
      return ok(Tensor.Values.Bool(false)) if x == 0.0
      error("a Bool tensor holds 1.0 or 0.0, got #{x}")
    _ ->
      i = try x.to_i
      return error("an integer tensor holds whole values, got #{x}") if i.to_f != x
      try check_int_range(i, dtype)
      ok(Tensor.Values.Int(i))
  end
end
