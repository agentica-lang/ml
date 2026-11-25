## The ggml backend of `tensor/`: tensors on ggml's CPU backend or on a Vulkan
## device, called from the `Backend.Ggml(device)` arms of `../tensor.agc`
## (`tensor/DESIGN.md` §2, §7, §8). It lives inside `tensor/` for the reason
## `src/torch.agc` does: `Handle.Ggml` names the resource of `ggml_native.agc`.
##
## It creates tensors, uploads them, reads them back and copies them between
## devices (`to_device`). ggml is fast when a whole computation is one
## compiled graph (a Vulkan dispatch costs about 2 ms,
## https://github.com/agentica-lang/ml/blob/bb056e75b3004706cffb7c851fa94b5e8a3b9253/spike/ggml-graph/README.md), so ops run only there, from a traced graph
## (`Trace`, `src/lower.agc`). Eagerly, every other primitive answers
## `compiled_only(op)`: an error naming the op, never a wrong number and
## never a slow op-by-op path (§7).
##
## Storage: ggml holds `F32` and `I32`. `Bool`, `I8`, `I16` and `U8` are held
## as `F32`, which represents each of their values exactly; `F16`, `Q8_0`,
## `F64` and `I64` are refused, as the lowering refuses them, except
## the weights `from_file` makes.
##
## Lifetimes (§8): one `Native.arena` per call, as in `src/torch.agc`.

import FileRange from "file_range"
import GgmlRaw from "ggml_native"
import Types from "types"

## True when `device` can hold ggml tensors here: the module is built and, for
## `Vulkan(i)`, ggml sees a Vulkan device `i`. Opens no device.
def available(device: Types.Device) -> Bool
  return false if !Native.load("ggml").ok?
  case device
    Types.Device.Cpu -> true
    Types.Device.Vulkan(i) -> i >= 0 and i < GgmlRaw.vulkan_devices()
  end
end

## The answer of every primitive the ggml backend has no eager form of.
def compiled_only(op: String) -> Result[Types.Tensor]
  error(
    "#{op} does not run eagerly on the ggml backend: ggml computes whole traced graphs " +
    "(Trace.eval, Trace.value_and_grad; tensor/DESIGN.md §7)",
  )
end

## Creation: every element `x`, a storage float already checked for `dtype`.
def filled(device: Types.Device, shape: List[Int], dtype: Types.Dtype, x: Float, op: String) -> Result[Types.Tensor]
  code = try ready(device, shape, dtype, op)
  int32 = dtype == Types.Dtype.I32
  made(op, device, shape, dtype) do GgmlRaw.filled(code, shape, x, int32) end
end

## Creation: every element `value` (tagged for `dtype`).
def full(device: Types.Device, shape: List[Int], value: Types.Values, dtype: Types.Dtype) -> Result[Types.Tensor]
  filled(device, shape, dtype, try Types.value_to_float(value, dtype), "full")
end

## Creation: `start, start + step, ...` below `stop` (above for `step < 0`).
def arange(device: Types.Device, start: Int, stop: Int, step: Int, dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, [], dtype, "arange")
  values: List[Float] = []
  v = start
  while (step > 0 and v < stop) or (step < 0 and v > stop)
    values = values.push(try Types.value_to_float(Types.Values.Int(v), dtype))
    v += step
  end

  upload(device, [values.size], dtype, values, "arange")
end

## Creation: row-major `F32` `values` of `shape`, without the `Values` tags
## `from_list` takes; the wrapper checked the count.
def floats(device: Types.Device, values: List[Float], shape: List[Int]) -> Result[Types.Tensor]
  upload(device, shape, Types.Dtype.F32, values, "from_list")
end

## Creation: row-major `data` of `shape`; the wrapper checked the count.
def from_list(
  device: Types.Device,
  data: List[Types.Values],
  shape: List[Int],
  dtype: Types.Dtype,
) -> Result[Types.Tensor]
  values: List[Float] = []
  for v in data
    values = values.push(try Types.value_to_float(v, dtype))
  end

  upload(device, shape, dtype, values, "from_list")
end

## Creation: a tensor of `shape` from the values of `format` at byte `offset`
## of the file `path`, read by C; on Vulkan through one host copy. It is `F32`,
## `F16` with `dtype: F16` (half the memory and the bandwidth of a weight; a
## value beyond 65504 is an error), or `Q8_0` with `dtype: Q8_0` (the blocks of
## a Q8_0 file as they are). With `TENSOR_LOAD=agentica` Agentica reads the
## range and C widens or copies the bytes.
def from_file(
  device: Types.Device,
  path: String,
  offset: Int,
  shape: List[Int],
  format: Types.FloatFormat,
  dtype: Types.Dtype,
) -> Result[Types.Tensor]
  code = try ready(device, shape, Types.Dtype.F32, "from_file")
  if dtype != Types.Dtype.F32 and !Types.is_packed_dtype(dtype)
    return error("from_file: a ggml tensor read from a file is F32, F16 or Q8_0, got #{Types.dtype_name(dtype)}")
  end

  held = Types.held_code(dtype)
  if FileRange.agentica_read?()
    n = try Types.volume(shape)
    bytes = try FileRange.read_range(path, offset, try Types.file_bytes(format, n))
    return made("from_file", device, shape, dtype) do
      GgmlRaw.from_bytes(code, bytes, shape, Types.file_code(format), held)
    end
  end

  made("from_file", device, shape, dtype) do GgmlRaw.load(code, path, offset, shape, Types.file_code(format), held) end
end

## Creation: an `F32` tensor of `shape` from `bytes`, which hold exactly its values as `format` writes them; C widens
## them into the buffer.
def from_bytes(device: Types.Device, bytes: Bytes, shape: List[Int], format: Types.FloatFormat) -> Result[Types.Tensor]
  code = try ready(device, shape, Types.Dtype.F32, "from_bytes")
  held = Types.held_code(Types.Dtype.F32)
  made("from_bytes", device, shape, Types.Dtype.F32) do
    GgmlRaw.from_bytes(code, bytes, shape, Types.file_code(format), held)
  end
end

## Host readback: the row-major values tagged with `t`'s dtype.
def to_list(t: Types.Tensor) -> Result[List[Types.Values]]
  _ = try usable("to_list")
  h = try handle_of(t)
  out: List[Types.Values] = []
  for x in try GgmlRaw.readback(h)
    out = out.push(try Types.float_to_value(x, t.dtype))
  end

  ok(out)
end

## Appends the `F32` values of `t` to the file `path`, written by C.
def append_file(t: Types.Tensor, path: String) -> Result[Void]
  _ = try usable("append_file")
  GgmlRaw.append_file(try handle_of(t), path)
end

## `t` copied to `device`: between ggml's CPU and a Vulkan device, or between
## two Vulkan devices.
def to_device(t: Types.Tensor, device: Types.Device) -> Result[Types.Tensor]
  code = try code_of(device, "to_device")
  h = try handle_of(t)
  made("to_device", device, t.shape, t.dtype) do GgmlRaw.transfer(h, code) end
end

## Chooses fp16 `mul_mat` (`true`, the default) or F32 (`false`) for every
## Vulkan device this process opens; an error once a ggml call fixed the
## other one. Without the ggml module there is nothing to choose and it
## does nothing.
def vulkan_fp16(enabled: Bool) -> Result[Void]
  return ok() if !Native.load("ggml").ok?
  GgmlRaw.vulkan_fp16(enabled)
end

## `count` values of `src_format` at byte `src_offset` of the file `src`, written as `dst_format` at byte `dst_offset` of
## the existing file `dst`.
def convert_file(
  src: String,
  src_offset: Int,
  src_format: Types.FloatFormat,
  count: Int,
  dst: String,
  dst_offset: Int,
  dst_format: Types.FloatFormat,
) -> Result[Void]
  _ = try usable("convert_file")
  GgmlRaw.convert_file(
    src,
    src_offset,
    Types.file_code(src_format),
    count,
    dst,
    dst_offset,
    Types.file_code(dst_format),
  )
end

## The device's code for the native calls: `-1` for `Cpu`, the index for
## `Vulkan`; the module must be loaded.
def code_of(device: Types.Device, op: String) -> Result[Int]
  _ = try usable(op)
  case device
    Types.Device.Cpu -> ok(-1)
    Types.Device.Vulkan(i) ->
      return error("#{op}: a Vulkan device index is >= 0, got #{i}") if i < 0
      ok(i)
  end
end

## The ggml handle of `t`.
def handle_of(t: Types.Tensor) -> Result[GgmlRaw.TensorHandle]
  case t.handle
    Types.Handle.Ggml(h) -> ok(h)
    _ -> error("a #{Types.backend_name(t.backend)} tensor has no ggml handle")
  end
end

## The `Tensor` a fresh handle makes.
def tensor_of(device: Types.Device, shape: List[Int], dtype: Types.Dtype, h: GgmlRaw.TensorHandle) -> Types.Tensor
  Types.Tensor(shape: shape, dtype: dtype, backend: Types.Backend.Ggml(device), handle: Types.Handle.Ggml(h))
end

private

def upload(
  device: Types.Device,
  shape: List[Int],
  dtype: Types.Dtype,
  values: List[Float],
  op: String,
) -> Result[Types.Tensor]
  code = try ready(device, shape, dtype, op)
  int32 = dtype == Types.Dtype.I32
  made(op, device, shape, dtype) do GgmlRaw.upload(code, shape, values, int32) end
end

## One native creation inside the arena of this call: the tensor leaves it as
## the block's value, and a native error names the op.
def made(
  op: String,
  device: Types.Device,
  shape: List[Int],
  dtype: Types.Dtype,
  call: Fn[Result[GgmlRaw.TensorHandle]],
) -> Result[Types.Tensor]
  Native.arena do
    case call.call()
      ok(h) -> ok(tensor_of(device, shape, dtype, h))
      error(e) -> error("#{op}: #{e.message}")
    end
  end
end

## The module is loaded, the dtype and rank fit ggml, and the device's code
## for the native calls: `-1` for `Cpu`, the index for `Vulkan`.
def ready(device: Types.Device, shape: List[Int], dtype: Types.Dtype, op: String) -> Result[Int]
  _ = try usable(op)
  _ = try storable(dtype, op)
  return error("#{op}: the ggml backend holds tensors of rank 4 at most, got #{shape}") if shape.size > 4
  code_of(device, op)
end

def storable(dtype: Types.Dtype, op: String) -> Result[Bool]
  case dtype
    Types.Dtype.F16 -> refused(dtype, op)
    Types.Dtype.Q8_0 -> refused(dtype, op)
    Types.Dtype.F64 -> refused(dtype, op)
    Types.Dtype.I64 -> refused(dtype, op)
    _ -> ok(true)
  end
end

def refused(dtype: Types.Dtype, op: String) -> Result[Bool]
  error("#{op}: the ggml backend holds no #{Types.dtype_name(dtype)} tensor: F32, I32, I16, I8, U8 and Bool only")
end

def usable(op: String) -> Result[Bool]
  opened = Native.load("ggml")
  return ok(true) if opened.ok?
  error(
    "#{op}: backend not available (#{opened.message}): run " +
    "`eval \"$(sh tools/build-engine.sh | grep '^export ')\"` in the shell that runs agentica " +
    "(tensor/README.md, \"The ggml backend\")",
  )
end
