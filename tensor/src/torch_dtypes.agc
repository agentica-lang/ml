## Dtype vocabulary of the libtorch backend: the predicates, the names used in
## error messages, the code that crosses to C, and the two conversions between
## a native storage value (a `Float`, as the boundary carries) and a
## dtype-tagged `Values`.
##
## `src/types.agc` keeps the same tables, and this module re-derives them over
## its own arm of the boundary rather than reaching across: a backend validates
## its own storage crossing (`tensor/DESIGN.md` §9, "backends may re-check
## cheaply"), and the code a dtype crosses as is a wire detail of this backend,
## not of the contract. The same codes are in `native/torch.c`
## (`scalar_type_of`): one table, two files, and the conformance tests are what
## keeps them agreeing.

import Types from "types"

## True for the dtypes that hold non-integer numerics.
def is_float_dtype(dtype: Types.Dtype) -> Bool
  case dtype
    Types.Dtype.F32 -> true
    Types.Dtype.F16 -> true
    Types.Dtype.F64 -> true
    _ -> false
  end
end

## True for the dtypes that hold integers.
def is_int_dtype(dtype: Types.Dtype) -> Bool
  case dtype
    Types.Dtype.I8 -> true
    Types.Dtype.I16 -> true
    Types.Dtype.I32 -> true
    Types.Dtype.I64 -> true
    Types.Dtype.U8 -> true
    _ -> false
  end
end

## The short name used in error messages.
def name(dtype: Types.Dtype) -> String
  case dtype
    Types.Dtype.F32 -> "F32"
    Types.Dtype.F16 -> "F16"
    Types.Dtype.F64 -> "F64"
    Types.Dtype.I8 -> "I8"
    Types.Dtype.I16 -> "I16"
    Types.Dtype.I32 -> "I32"
    Types.Dtype.I64 -> "I64"
    Types.Dtype.U8 -> "U8"
    Types.Dtype.Bool -> "Bool"
    Types.Dtype.Q8_0 -> "Q8_0"
  end
end

## The code the native layer reads: `torch_scalar_type` in `native/torch.c`
## maps it to a libtorch scalar type. The order is the one libtorch's CPU
## release offers for these nine, and it is a wire detail, not an API. `Q8_0`
## (a weight of the ggml backend) has no libtorch type: `-1`, which the C layer refuses.
def code(dtype: Types.Dtype) -> Int
  case dtype
    Types.Dtype.F32 -> 0
    Types.Dtype.F16 -> 1
    Types.Dtype.F64 -> 2
    Types.Dtype.I8 -> 3
    Types.Dtype.I16 -> 4
    Types.Dtype.I32 -> 5
    Types.Dtype.I64 -> 6
    Types.Dtype.U8 -> 7
    Types.Dtype.Bool -> 8
    Types.Dtype.Q8_0 -> -1
  end
end

## `i` fits the width of the int `dtype` (callers check the class first).
def check_int_range(i: Int, dtype: Types.Dtype) -> Result[Int]
  case dtype
    Types.Dtype.I8 -> return error("I8 holds -128..127, got #{i}") if i < -128 or i > 127
    Types.Dtype.I16 -> return error("I16 holds -32768..32767, got #{i}") if i < -32768 or i > 32767
    Types.Dtype.I32 -> return error("I32 holds -2147483648..2147483647, got #{i}") if i < -2147483648 or i > 2147483647

    Types.Dtype.U8 -> return error("U8 holds 0..255, got #{i}") if i < 0 or i > 255
    _ -> return ok(i)
  end

  ok(i)
end

## The storage float of one tagged value: checks the variant fits `dtype`
## and, for int dtypes, that the integer fits the width.
def value_to_float(value: Types.Values, dtype: Types.Dtype) -> Result[Float]
  case value
    Types.Values.Float(x) ->
      return error("a #{name(dtype)} tensor takes no Float value, got #{x}") if !is_float_dtype(dtype)
      ok(x)
    Types.Values.Int(i) ->
      return error("a #{name(dtype)} tensor takes no Int value, got #{i}") if !is_int_dtype(dtype)
      try check_int_range(i, dtype)
      ok(i.to_f)
    Types.Values.Bool(b) ->
      return error("a #{name(dtype)} tensor takes no Bool value") if dtype != Types.Dtype.Bool
      if b
        ok(1.0)
      else
        ok(0.0)
      end
  end
end

## Tags one storage float with `dtype`: the inverse of `value_to_float`.
## Int dtypes need an integral value in range, `Bool` needs `1.0`/`0.0`.
def float_to_value(x: Float, dtype: Types.Dtype) -> Result[Types.Values]
  case dtype
    Types.Dtype.F32 -> ok(Types.Values.Float(x))
    Types.Dtype.F16 -> ok(Types.Values.Float(x))
    Types.Dtype.F64 -> ok(Types.Values.Float(x))
    Types.Dtype.Bool ->
      return ok(Types.Values.Bool(true)) if x == 1.0
      return ok(Types.Values.Bool(false)) if x == 0.0
      error("a Bool tensor holds 1.0 or 0.0, got #{x}")
    _ ->
      i = try x.to_i
      return error("an integer tensor holds whole values, got #{x}") if i.to_f != x
      try check_int_range(i, dtype)
      ok(Types.Values.Int(i))
  end
end
