## The ggml backend of `tensor/`: tensors on ggml's CPU backend or on a Vulkan
## device, called from the `Backend.Ggml(device)` arms of `../tensor.agc`
## (`tensor/DESIGN.md` §2, §7, §8). It lives inside `tensor/` for the reason
## `src/torch.agc` does: `Handle.Ggml` names the resource of `ggml_native.agc`.
##
## It creates tensors, uploads them and reads them back; it runs no op.
## ggml is fast when a whole step is one compiled graph (a Vulkan dispatch
## costs about 2 ms, spike/ggml-graph/README.md), so its ops will run there,
## from a traced graph (`Trace`). Until that compiler exists every other
## primitive answers `compiled_only(op)`: an error naming the op, never a
## wrong number and never a slow op-by-op path (§7).
##
## Storage: ggml holds `F32` and `I32`. `Bool`, `I8`, `I16` and `U8` are held
## as `F32`, which represents each of their values exactly; `F16`, `F64` and
## `I64` are refused, as the spike's lowering refuses them.
##
## Lifetimes (§8): one `Native.arena` per call, as in `src/torch.agc`.

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
    "#{op} does not run eagerly on the ggml backend: ggml computes whole traced graphs (Trace), " +
    "which tensor/ does not compile yet (tensor/DESIGN.md §7)",
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

## `t` copied to `device`: between ggml's CPU and a Vulkan device, or between
## two Vulkan devices.
def to_device(t: Types.Tensor, device: Types.Device) -> Result[Types.Tensor]
  code = try code_of(device, "to_device")
  h = try handle_of(t)
  made("to_device", device, t.shape, t.dtype) do GgmlRaw.transfer(h, code) end
end

## Chooses fp16 `mul_mat` (`true`) or F32 (`false`, the default) for every
## Vulkan device this process opens; an error once a device opened with the
## other one.
def prefer_fp16(enabled: Bool) -> Result[Void]
  _ = try usable("vulkan_fp16")
  GgmlRaw.prefer_fp16(enabled)
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
