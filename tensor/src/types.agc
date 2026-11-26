## Shared types of the `tensor/` package (`tensor/DESIGN.md` §2).
##
## This module imports three things: `src/native.agc`, `src/ggml_native.agc`
## and `src/xla_native.agc`, the `extern module` declarations whose resources
## the `Native`, `Ggml` and `Xla` arms of `Handle` name. Agentica resources are nominal to the block that declares them, so
## the backend's handle type has to be visible here for a `Tensor` to carry one
## (`tensor/DESIGN.md` §1, §2, §7).
## Everything else about this module is unchanged: the reference backend
## (`src/reference.agc`) and the public API (`tensor.agc`) use it without an
## import cycle, and the public entry point stays `tensor.agc`, which
## re-exports every type here.
##
## Storage rule (design of record: `tensor/DESIGN.md` §2): the reference
## backend holds plain Agentica data, row-major (C order), as `List[Float]`
## inside `Data`. Float-class dtypes hold the value itself; int-class dtypes
## hold exactly integral values; `Bool` holds `1.0`/`0.0`. `Values` is the
## dtype-tagged crossing type for `from_list`/`to_list` only.

import GgmlRaw from "ggml_native"
import Raw from "native"
import XlaRaw from "xla_native"

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
  Q8_0
  Q4_K
  Q6_K
end

## The formats `Tensor.from_file` reads: little-endian IEEE binary32, binary16
## and bfloat16, and ggml's quantized blocks: Q8_0 (32 values: a binary16 scale
## and 32 signed bytes), Q4_K (256 values in 144 bytes) and Q6_K (256 values in
## 210). The tensor it makes is `F32` whichever one it reads, or a weight of the
## dtype of the same name that keeps the blocks of the file (ggml only). `U8` is
## for `Tensor.from_bytes` alone: one unsigned byte per value, `0..255` as an
## `F32`, such as the samples of an image.
enum FloatFormat
  F32
  F16
  BF16
  Q8_0
  Q4_K
  Q6_K
  U8
end

## The values and the bytes of one block of a quantized format (`block_of`).
struct Block
  values: Int
  bytes: Int
end

## Where a backend computes: the host CPU, or the Vulkan device of that index
## (ggml's numbering). `Reference` and `Torch` serve `Cpu` only.
enum Device
  Cpu
  Vulkan(Int)
end

## Where a weight read from a file lives. `Device` is the device's own memory; `Host` is, on a Vulkan device, its
## pinned host memory: the card reads the weight over the bus, and the device's memory keeps the rest. Elsewhere it is
## `Device`.
enum Placement
  Device
  Host
end

enum Backend
  Reference
  Torch(Device)
  Trace
  Ggml(Device)
  Xla(Device)
end

## Plain Agentica backing store: the row-major values of one tensor.
struct Data
  values: List[Float]
end

## How a tensor's values are held. `Plain` is the reference backend, which
## owns Agentica data and needs no arena; `Native` is a handle of the libtorch
## backend and `Ggml` one of the ggml backend, each owned by the task that made
## it and closed by the arena it was made in, `Native.close`, or the end of the
## task (`tensor/DESIGN.md` §8). Each handle is the resource of its module, so
## it is typed by that module's declaration and not by an opaque resource.
enum Handle
  Plain(Data)
  Native(Raw.TensorHandle)
  Traced(Node)
  Ggml(GgmlRaw.TensorHandle)
  Xla(XlaRaw.Buffer)
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

## What a traced node computes (`tensor/DESIGN.md` §10.2). `Input` is an
## argument of the traced function, `Constant` a tensor lifted into the trace,
## `Full` a fill of the node's shape; the shape-changing ops read their target
## from the node's tensor. Every op of §3.1 has a variant, so any traced
## function runs forward; the ones with no backward rule yet make `gradients`
## answer an error naming the op.
enum Op
  Input
  Constant(Tensor)
  Full(Values)
  Add
  Sub
  Mul
  Div
  Neg
  Exp
  Log
  Relu
  Matmul
  Transpose
  Sum(List[Int], Bool)
  Mean(List[Int], Bool)
  BroadcastTo
  Reshape
  Lt
  Where
  Sqrt
  Pow
  Sigmoid
  Tanh
  Gelu
  GeluErf
  LayerNorm(Float)
  RmsNorm(Float)
  AdamStep
  AdamPart(Int)
  SetRows
  TopK(Int)
  GatedDeltaRule
  ExpertMatmul
  Eq
  Clamp(Float, Float)
  Max(List[Int], Bool)
  Argmax(List[Int], Bool)
  BatchMatmul
  Squeeze(Optional[Int])
  Unsqueeze(Int)
  Slice(List[Int], List[Int])
  Concat(Int)
  Stack(Int)
  Gather
  ScatterAdd(Int)
  ToDtype
  ToDevice(Device)
  Copy
  Contiguous
  Conv1d(Int, Int)
  AvgPool1d(Int, Int, Int)
  Conv2d(Int, Int, Int)
  MaxPool2d(Int, Int, Int)
  AvgPool2d(Int, Int, Int)
  Arange(Int, Int, Int)
  StopGrad
  Permute(List[Int])
  Attention(Float)
  Rope(Float, Int, Bool, Bool)
end

## One node of a traced graph: unique `id` in the run, its op and its
## arguments, which are traced tensors themselves.
struct Node
  id: Ref
  op: Op
  args: List[Tensor]
end

## The two results of `gated_delta_rule`: the answers `[tokens, heads, width]` and the state after the last token
## `[heads, width, width]`.
struct DeltaRule
  out: Tensor
  state: Tensor
end

## The two results of `top_k`: `values` descending, `indices` their positions.
struct TopK
  values: Tensor
  indices: Tensor
end

## The number of elements `shape` names: `1` for `[]`, the product otherwise.
def volume(shape: List[Int]) -> Result[Int]
  for d in shape
    return error("a shape holds no negative dimension, got #{shape}") if d < 0
  end

  return ok(0) if shape.include?(0)
  total = 1
  for d in shape
    return error("a shape of #{shape} holds more elements than an Int counts") if total > 9223372036854775807 / d
    total = total * d
  end

  ok(total)
end

## The values and the bytes of one block of a quantized `format`: Q8_0 32 values in 34 bytes, Q4_K and Q6_K 256 values in
## 144 and 210. `none` for a format that is not quantized.
def block_of(format: FloatFormat) -> Optional[Block]
  case format
    FloatFormat.Q8_0 -> some(Block(values: 32, bytes: 34))
    FloatFormat.Q4_K -> some(Block(values: 256, bytes: 144))
    FloatFormat.Q6_K -> some(Block(values: 256, bytes: 210))
    _ -> none
  end
end

## The bytes of `n` values of `format`: a quantized range holds whole blocks.
def file_bytes(format: FloatFormat, n: Int) -> Result[Int]
  return error("#{n} values hold more bytes than an Int counts") if n > 2305843009213693951
  case block_of(format)
    some(b) ->
      return error("#{format_name(format)} holds whole blocks of #{b.values} values, got #{n}") if n % b.values != 0
      return ok((n / b.values) * b.bytes)
    none -> nil
  end

  case format
    FloatFormat.F32 -> ok(n * 4)
    FloatFormat.F16 -> ok(n * 2)
    FloatFormat.BF16 -> ok(n * 2)
    FloatFormat.U8 -> ok(n)
    _ -> error("#{format_name(format)} is a quantized format")
  end
end

## The short name used in error messages.
def format_name(format: FloatFormat) -> String
  case format
    FloatFormat.F32 -> "F32"
    FloatFormat.F16 -> "F16"
    FloatFormat.BF16 -> "BF16"
    FloatFormat.Q8_0 -> "Q8_0"
    FloatFormat.Q4_K -> "Q4_K"
    FloatFormat.Q6_K -> "Q6_K"
    FloatFormat.U8 -> "U8"
  end
end

## The code the native layers read: `FF_*` in `native/float_file.h`.
def file_code(format: FloatFormat) -> Int
  case format
    FloatFormat.F32 -> 0
    FloatFormat.F16 -> 1
    FloatFormat.BF16 -> 2
    FloatFormat.Q8_0 -> 3
    FloatFormat.U8 -> 4
    FloatFormat.Q4_K -> 5
    FloatFormat.Q6_K -> 6
  end
end

## What a loaded ggml tensor holds, for the native layer: 0 F32, 1 F16, 2 Q8_0, 3 Q4_K, 4 Q6_K (the type of a graph's
## leaf too).
def held_code(dtype: Dtype) -> Int
  case dtype
    Dtype.F16 -> 1
    Dtype.Q8_0 -> 2
    Dtype.Q4_K -> 3
    Dtype.Q6_K -> 4
    _ -> 0
  end
end

## True for the dtypes of ggml's quantized blocks: `Q8_0`, `Q4_K` and `Q6_K`.
def is_quantized_dtype(dtype: Dtype) -> Bool
  case dtype
    Dtype.Q8_0 -> true
    Dtype.Q4_K -> true
    Dtype.Q6_K -> true
    _ -> false
  end
end

## The file format whose blocks a quantized dtype keeps (`F32` for the others, which keep none).
def block_format(dtype: Dtype) -> FloatFormat
  case dtype
    Dtype.Q8_0 -> FloatFormat.Q8_0
    Dtype.Q4_K -> FloatFormat.Q4_K
    Dtype.Q6_K -> FloatFormat.Q6_K
    _ -> FloatFormat.F32
  end
end

## True for the dtypes a weight is kept in on ggml to save memory: `F16` and the quantized ones. Read-only, and only the
## weight of a matmul or the table of a gather (`tensor/DESIGN.md` §7).
def is_packed_dtype(dtype: Dtype) -> Bool = dtype == Dtype.F16 or is_quantized_dtype(dtype)

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
    Dtype.Q8_0 -> "Q8_0"
    Dtype.Q4_K -> "Q4_K"
    Dtype.Q6_K -> "Q6_K"
    Dtype.F64 -> "F64"
    Dtype.I8 -> "I8"
    Dtype.I16 -> "I16"
    Dtype.I32 -> "I32"
    Dtype.I64 -> "I64"
    Dtype.U8 -> "U8"
    Dtype.Bool -> "Bool"
  end
end

## True for ordinary values: false for `NaN` and both infinities.
## `(x - x) == 0.0` holds exactly for the finite cases (including `0.0`
## and denormals) and fails for the three non-finite ones, with no
## arithmetic a caller could halt on.
def is_finite(x: Float) -> Bool = x - x == 0.0

## The short name used in error messages.
def device_name(device: Device) -> String
  case device
    Device.Cpu -> "Cpu"
    Device.Vulkan(i) -> "Vulkan(#{i})"
  end
end

## The short name used in error messages.
def backend_name(backend: Backend) -> String
  case backend
    Backend.Reference -> "Reference"
    Backend.Torch(device) -> "Torch(#{device_name(device)})"
    Backend.Trace -> "Trace"
    Backend.Ggml(device) -> "Ggml(#{device_name(device)})"
    Backend.Xla(device) -> "Xla(#{device_name(device)})"
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
    Dtype.Q8_0 -> ok(Values.Float(x))
    Dtype.Q4_K -> ok(Values.Float(x))
    Dtype.Q6_K -> ok(Values.Float(x))
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
    Dtype.I8 -> return error("I8 holds -128..127, got #{i}") if !i.to_int8.ok?
    Dtype.I16 -> return error("I16 holds -32768..32767, got #{i}") if !i.to_int16.ok?
    Dtype.I32 -> return error("I32 holds -2147483648..2147483647, got #{i}") if !i.to_int32.ok?
    Dtype.U8 -> return error("U8 holds 0..255, got #{i}") if !i.to_uint8.ok?
    _ -> return ok(i)
  end

  ok(i)
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

## The name of `op` in error messages.
def op_name(op: Op) -> String
  case op
    Op.Input -> "input"
    Op.Constant(_) -> "constant"
    Op.Full(_) -> "full"
    Op.Add -> "add"
    Op.Sub -> "sub"
    Op.Mul -> "mul"
    Op.Div -> "div"
    Op.Neg -> "neg"
    Op.Exp -> "exp"
    Op.Log -> "log"
    Op.Relu -> "relu"
    Op.Matmul -> "matmul"
    Op.Transpose -> "transpose"
    Op.Sum(_, _) -> "sum"
    Op.Mean(_, _) -> "mean"
    Op.BroadcastTo -> "broadcast_to"
    Op.Reshape -> "reshape"
    Op.Lt -> "lt"
    Op.Where -> "where"
    Op.Sqrt -> "sqrt"
    Op.Pow -> "pow"
    Op.Sigmoid -> "sigmoid"
    Op.Tanh -> "tanh"
    Op.Gelu -> "gelu_tanh_approx"
    Op.GeluErf -> "gelu_erf"
    Op.LayerNorm(_) -> "layer_norm"
    Op.RmsNorm(_) -> "rms_norm"
    Op.AdamStep -> "adam_step"
    Op.AdamPart(_) -> "adam_part"
    Op.SetRows -> "set_rows"
    Op.TopK(_) -> "top_k"
    Op.GatedDeltaRule -> "gated_delta_rule"
    Op.ExpertMatmul -> "expert_matmul"
    Op.Eq -> "eq"
    Op.Clamp(_, _) -> "clamp"
    Op.Max(_, _) -> "max"
    Op.Argmax(_, _) -> "argmax"
    Op.BatchMatmul -> "batch_matmul"
    Op.Squeeze(_) -> "squeeze"
    Op.Unsqueeze(_) -> "unsqueeze"
    Op.Slice(_, _) -> "slice"
    Op.Concat(_) -> "concat"
    Op.Stack(_) -> "stack"
    Op.Gather -> "gather"
    Op.ScatterAdd(_) -> "scatter_add"
    Op.ToDtype -> "to_dtype"
    Op.ToDevice(_) -> "to_device"
    Op.Copy -> "copy"
    Op.Contiguous -> "contiguous"
    Op.Conv1d(_, _) -> "conv1d"
    Op.AvgPool1d(_, _, _) -> "avg_pool1d"
    Op.Conv2d(_, _, _) -> "conv2d"
    Op.MaxPool2d(_, _, _) -> "max_pool2d"
    Op.AvgPool2d(_, _, _) -> "avg_pool2d"
    Op.Arange(_, _, _) -> "arange"
    Op.Permute(_) -> "permute"
    Op.Attention(_) -> "attention"
    Op.Rope(_, _, _, _) -> "rope"
    Op.StopGrad -> "stop_grad"
  end
end

## `shape` with its axes in the order `axes` names: the shape a `permute`
## leaves. Output axis `i` is input axis `axes[i]`.
def permuted_shape(shape: List[Int], axes: List[Int]) -> List[Int] = axes.map { |a| shape[a] }

## The `axes` that undo `axes`: `permute(permute(t, axes), inverse_axes(axes))`
## is `t`.
def inverse_axes(axes: List[Int]) -> List[Int]
  out: List[Int] = axes.map { |_| 0 }
  for i in 0...axes.size
    out = out.set(axes[i], i)
  end

  out
end
