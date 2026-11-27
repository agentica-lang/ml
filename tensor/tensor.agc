## The `tensor/` package: functional tensors over pluggable backends.
##
## The public API (`tensor/DESIGN.md` §§2–4, 7, 9). Every fallible op returns
## `Result` and reports `error`, never a halt; shape, dtype and backend
## mismatches are validated eagerly in these wrappers, before dispatch.
## Dispatch is by value (`t.backend`) through an exhaustive `case` over all
## five `Backend` arms (`tensor/DESIGN.md` §7): `Reference` runs in Agentica,
## `Torch` runs on libtorch through `src/torch.agc` and `src/native.agc`,
## `Trace` records a node of a traced graph after the same checks
## (`src/graph.agc`, `trace.agc`, §10), and `Ggml` creates and reads back
## tensors on ggml (`src/ggml.agc`) and answers `compiled_only` for every other
## op, as `Xla` does on XLA (`src/xla.agc`). With a native module absent (all
## are `optional`) its arm answers `error("backend not available")`, and `available/1` says so before a call
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
##   (`NaN`, `+-inf`; `exp` takes `-inf`, which is `0`) is refused with `error` at every entry point that could
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
## - `conv2d`, `max_pool2d` and `avg_pool2d` take `[N, C, H, W]` and one stride and padding for both axes.
## - `conv1d`/`avg_pool1d` take only what `tensor/DESIGN.md` §3.1 admits
##   (stride and padding 1-D, no dilation and no groups); the wrapper
##   validates their window arithmetic before dispatch, so both backends
##   accept and refuse the same shapes.
## - The 52 names of §3.1 all exist below. `Reference` computes all of them;
##   the `Torch` arm computes everything but `to_device`, and answers
##   `error("<op> is not implemented on the torch backend yet")` for it,
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
import Xla from "src/xla"

export Types.Backend
export Types.Data
export Types.DeltaRule
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

## The bytes a ggml device has: what is `free` now and its `total`.
struct DeviceMemory
  free: Int
  total: Int
end

## The `DeviceMemory` of the device `backend` runs on (`ggml_backend_dev_memory`): the card for `Ggml(Vulkan(i))`, the
## RAM of the host for `Ggml(Cpu)`. It opens no backend. Another backend, a missing Vulkan device or a missing ggml module
## is an error. A card that shares memory with the host (an iGPU) reports the part of it that the driver lends.
def device_memory(backend: Types.Backend) -> Result[DeviceMemory]
  case backend
    Types.Backend.Ggml(device) ->
      bytes = try Ggml.device_memory(device)
      ok(DeviceMemory(free: bytes[0], total: bytes[1]))
    _ -> error("device_memory: only a ggml backend has a device to ask, got #{Types.backend_name(backend)}")
  end
end

## The shape of `t`: a field read, no backend call.
def shape(t: Types.Tensor) -> List[Int] = t.shape

## The dtype of `t`: a field read, no backend call.
def dtype(t: Types.Tensor) -> Types.Dtype = t.dtype

## True for `dtype` of a weight that ggml keeps packed to save memory (`F16`, `Q8_0`, `Q4_K` or `Q6_K`): read-only, and
## only the weight of a product or the table of a lookup (`tensor/DESIGN.md` §7).
def packed?(dtype: Types.Dtype) -> Bool = Types.is_packed_dtype(dtype)

## The backend of `t`: a field read, no backend call.
def backend(t: Types.Tensor) -> Types.Backend = t.backend

## True when `backend` can run on this machine: `Reference` always can, `Torch`
## when the native libtorch module is built (`[native.torch] optional = true` in
## `agentica.toml`), `Ggml` when the native ggml module is built and, for
## `Vulkan(i)`, ggml sees that device, `Xla(Cpu)` when the XLA plugin loads
## (`tools/install_xla.agc`). A caller that needs to work on both asks
## this instead of catching the `error("backend not available")` an op would
## answer. `Trace` is never available: it records ops and computes nothing
## (`trace.agc`).
def available(backend: Types.Backend) -> Bool
  case backend
    Types.Backend.Reference -> true
    Types.Backend.Torch(_) -> Torch.available()
    Types.Backend.Trace -> false
    Types.Backend.Ggml(device) -> Ggml.available(device)
    Types.Backend.Xla(device) -> Xla.available(device)
  end
end

## Chooses fp16 (`true`, the default) or F32 (`false`) `mul_mat` for every
## Vulkan device of the ggml backend in this process. ggml fixes the choice
## at the process's first ggml call (a CPU one, or `available` of a Vulkan
## device, included), so a different choice after that is an error:
## `vulkan_fp16(false)` goes first in a check that must be exact. On a build without Vulkan or without the
## ggml module it changes nothing. fp16 is about 1.7 times faster for the products of a model and
## changes gradients by up to about 10 % at MLP sizes (`tensor/README.md`,
## "The ggml backend"), and it does not keep the ids of Hugging Face's
## greedy `generate`.
def vulkan_fp16(enabled: Bool) -> Result[Void] = Ggml.vulkan_fp16(enabled)

## Creation: zeros of `shape`, `dtype` and `backend` (`F32` on `Reference` by default).
def zeros(
  shape: List[Int],
  dtype: Types.Dtype = Types.Dtype.F32,
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  _ = try not_quantized(dtype, "zeros")
  case backend
    Types.Backend.Reference -> Reference.zeros(shape, dtype)
    Types.Backend.Torch(device) -> Torch.zeros(device, shape, dtype)
    Types.Backend.Trace -> Graph.filled(shape, 0.0, dtype)
    Types.Backend.Ggml(device) -> Ggml.filled(device, shape, dtype, 0.0, "zeros")
    Types.Backend.Xla(device) -> Xla.filled(device, shape, dtype, 0.0, "zeros")
  end
end

## Creation: ones of `shape`, `dtype` and `backend`.
def ones(
  shape: List[Int],
  dtype: Types.Dtype = Types.Dtype.F32,
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  _ = try not_quantized(dtype, "ones")
  case backend
    Types.Backend.Reference -> Reference.ones(shape, dtype)
    Types.Backend.Torch(device) -> Torch.ones(device, shape, dtype)
    Types.Backend.Trace -> Graph.filled(shape, 1.0, dtype)
    Types.Backend.Ggml(device) -> Ggml.filled(device, shape, dtype, 1.0, "ones")
    Types.Backend.Xla(device) -> Xla.filled(device, shape, dtype, 1.0, "ones")
  end
end

## Creation: `empty` is deterministic `zeros` over plain Agentica data
## (there is no uninitialized memory to expose; see `src/reference.agc`).
def empty(shape: List[Int], dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  _ = try Types.volume(shape)
  _ = try not_quantized(dtype, "empty")
  case backend
    Types.Backend.Reference -> Reference.empty(shape, dtype)
    Types.Backend.Torch(device) -> Torch.empty(device, shape, dtype)
    Types.Backend.Trace -> Graph.filled(shape, 0.0, dtype)
    Types.Backend.Ggml(device) -> Ggml.filled(device, shape, dtype, 0.0, "empty")
    Types.Backend.Xla(device) -> Xla.filled(device, shape, dtype, 0.0, "empty")
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
  _ = try not_quantized(dtype, "full")
  case backend
    Types.Backend.Reference -> Reference.full(shape, value, dtype)
    Types.Backend.Torch(device) -> Torch.full(device, shape, value, dtype)
    Types.Backend.Trace -> Graph.fill(shape, value, dtype)
    Types.Backend.Ggml(device) -> Ggml.full(device, shape, value, dtype)
    Types.Backend.Xla(device) -> Xla.full(device, shape, value, dtype)
  end
end

## Creation: zeros of the shape, dtype and backend of `t` (metadata only).
def like(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try not_quantized(t.dtype, "like")
  case t.backend
    Types.Backend.Reference -> Reference.like(t)
    Types.Backend.Torch(device) -> Torch.like(device, t)
    Types.Backend.Trace -> Graph.filled(t.shape, 0.0, t.dtype)
    Types.Backend.Ggml(device) -> Ggml.filled(device, t.shape, t.dtype, 0.0, "like")
    Types.Backend.Xla(device) -> Xla.filled(device, t.shape, t.dtype, 0.0, "like")
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
    Types.Backend.Xla(device) -> Xla.arange(device, start, stop, step, dtype)
  end
end

## Creation: the one `value` as a rank-0 tensor.
def scalar_to_tensor(value: Types.Values, dtype: Types.Dtype, backend: Types.Backend) -> Result[Types.Tensor]
  case backend
    Types.Backend.Reference -> Reference.scalar_to_tensor(value, dtype)
    Types.Backend.Torch(device) -> Torch.scalar_to_tensor(device, value, dtype)
    Types.Backend.Trace -> Graph.fill([], value, dtype)
    Types.Backend.Ggml(device) -> Ggml.full(device, [], value, dtype)
    Types.Backend.Xla(device) -> Xla.full(device, [], value, dtype)
  end
end

## Creation: row-major `data` of `shape`; the count must match. On XLA an `F16` value is rounded to nearest even and
## one beyond 65504, an infinity included, is an `error` naming its element; a NaN stays one.
def from_list(
  data: List[Types.Values],
  shape: List[Int],
  dtype: Types.Dtype = Types.Dtype.F32,
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  return error("from_list needs #{n} values for #{shape}, got #{data.size}") if data.size != n
  _ = try not_quantized(dtype, "from_list")
  case backend
    Types.Backend.Reference -> Reference.from_list(data, shape, dtype)
    Types.Backend.Torch(device) -> Torch.from_list(device, data, shape, dtype)
    Types.Backend.Trace -> error("from_list makes no traced tensor: lift one with Trace.constant")
    Types.Backend.Ggml(device) -> Ggml.from_list(device, data, shape, dtype)
    Types.Backend.Xla(device) -> Xla.from_list(device, data, shape, dtype)
  end
end

## Creation: an `F32` tensor of `shape` from the little-endian floats of `format`
## that start at byte `offset` of the file `path`, for weights: the backend reads
## the file itself, so no value crosses as an Agentica list. `F16` and `BF16`
## widen to `F32`. A short file, a bad offset or a missing file is an `error`.
## The reference reads the bytes in Agentica and is slow on a large file.
##
## With `dtype: F16` the tensor is a half weight: it keeps 16 bits per value
## (half the memory and the bandwidth), converted at load, and an
## `F32`, `F16` or `BF16` file value beyond 65504 is an `error`. With
## `format: Q8_0`, `Q4_K` or `Q6_K` the file holds ggml's blocks (32 values, or
## 256 for the K formats), and the tensor is `F32` (dequantized at load, on
## every backend) or, with the `dtype` of the same name, a quantized weight that
## keeps the blocks (a quarter to a seventh of the memory of `F32`; a rank-2
## weight of rows of whole blocks). Only the ggml backend holds a quantized
## weight, and only ggml and XLA an `F16` one, and a trace may use it only as the weight of
## `matmul` or `linear` and as the table of `gather` or `embedding`
## (`tensor/DESIGN.md` §7). It is for inference, with `linear(..., out_in:
## true)` for a checkpoint's `[out, in]` weight.
def from_file(
  path: String,
  offset: Int,
  shape: List[Int],
  format: Types.FloatFormat = Types.FloatFormat.F32,
  backend: Types.Backend = Types.Backend.Reference,
  dtype: Types.Dtype = Types.Dtype.F32,
) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  if format == Types.FloatFormat.U8
    return error("from_file reads F32, F16, BF16 or quantized files: U8 is for from_bytes")
  end

  if !Types.is_packed_dtype(dtype) and dtype != Types.Dtype.F32
    return error("from_file reads F32, F16, BF16 or a quantized dtype, got #{Types.dtype_name(dtype)}")
  end

  if Types.is_packed_dtype(dtype) and !holds_packed?(backend, dtype)
    return error(
      "from_file with dtype #{Types.dtype_name(dtype)} (packed weights) needs the ggml backend (or XLA for F16 and BF16), got #{Types.backend_name(backend)}",
    )
  end

  name = Types.dtype_name(dtype)
  if Types.is_quantized_dtype(dtype)
    if format != Types.block_format(dtype)
      return error("from_file with dtype #{name} keeps the blocks of a #{name} file, got a file of another format")
    end

    block = Types.block_of(format).unwrap!
    if shape.size != 2 or shape[1] % block.values != 0
      return error(
        "from_file with dtype #{name} reads a rank-2 weight whose rows hold whole blocks of #{block.values} values, got #{shape}",
      )
    end
  end

  if (dtype == Types.Dtype.F16 or dtype == Types.Dtype.BF16) and Types.block_of(format).some?
    fmt = Types.format_name(format)
    return error("from_file reads a #{fmt} file as F32 or keeps it as #{fmt}, not as #{Types.dtype_name(dtype)}")
  end

  return error("from_file needs an offset of 0 or more, got #{offset}") if offset < 0
  bytes = try Types.file_bytes(format, n)
  if bytes > 9223372036854775807 - offset
    return error("from_file: #{n} values at offset #{offset} reach past what an Int counts")
  end

  case backend
    Types.Backend.Reference -> Reference.from_file(path, offset, shape, format)
    Types.Backend.Torch(device) -> Torch.from_file(device, path, offset, shape, format)
    Types.Backend.Trace -> error("from_file makes no traced tensor: lift one with Trace.constant")
    Types.Backend.Ggml(device) -> Ggml.from_file(device, path, offset, shape, format, dtype)
    Types.Backend.Xla(device) -> Xla.from_file(device, path, offset, shape, format, dtype)
  end
end

## Creation: an `F32` tensor of `shape` from `data`, which holds exactly its values as `format` writes them (little
## endian, `F32` by default): what `from_file` makes of a range, for bytes that are in memory already, such as the
## pixels of an image. The backend widens the bytes itself, so no value becomes a list. `FloatFormat.U8` takes one byte
## per value and gives it as `0.0..255.0`, so the samples of a `U8` image cross into the backend as bytes, a quarter of their `F32` size, and the backend widens them to `F32`. A
## `data` of another size is an `error`.
def from_bytes(
  data: Bytes,
  shape: List[Int],
  format: Types.FloatFormat = Types.FloatFormat.F32,
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  size = try Types.file_bytes(format, n)
  return error("from_bytes needs #{size} bytes for #{shape}, got #{data.size}") if data.size != size
  case backend
    Types.Backend.Reference -> Reference.from_bytes(data, shape, format)
    Types.Backend.Torch(device) -> Torch.from_bytes(device, data, shape, format)
    Types.Backend.Trace -> error("from_bytes makes no traced tensor: lift one with Trace.constant")
    Types.Backend.Ggml(device) -> Ggml.from_bytes(device, data, shape, format)
    Types.Backend.Xla(device) -> Xla.from_bytes(device, data, shape, format)
  end
end

## Converting a range of one file into another, for writing a checkpoint: the `count` floats of `src_format` that
## start at byte `src_offset` of the file `src` are written as `dst_format` at byte `dst_offset` of `dst`, which must
## exist (a file that grows to the end of the range). C does it in chunks on several threads, so no tensor and no
## list of values exists. The same format is a copy of the
## bytes; `F32` or `BF16` to `F16` rounds to nearest even and refuses a value beyond 65504, `F32` or `F16` to `BF16`
## and `BF16` to `F32` round or widen exactly as `from_file` and the converters of llama.cpp do. Needs the ggml
## module; a bad range or a missing file is an `error`.
def convert_file(
  src: String,
  src_offset: Int,
  src_format: Types.FloatFormat,
  count: Int,
  dst: String,
  dst_offset: Int,
  dst_format: Types.FloatFormat,
) -> Result[Void]
  if src_offset < 0 or dst_offset < 0
    return error("convert_file needs offsets of 0 or more, got #{src_offset} and #{dst_offset}")
  end

  return error("convert_file needs a count of 0 or more, got #{count}") if count < 0
  if (
    Types.block_of(src_format).some? or
    Types.block_of(dst_format).some? or
    src_format == Types.FloatFormat.U8 or
    dst_format == Types.FloatFormat.U8
  )
    return error("convert_file converts F32, F16 and BF16 ranges, not quantized or U8 ones")
  end

  Ggml.convert_file(src, src_offset, src_format, count, dst, dst_offset, dst_format)
end

## Creation: plain `values` as an `F32` tensor of `shape`, the short form of
## `from_list` for floats.
def floats(
  values: List[Float],
  shape: List[Int],
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  n = try Types.volume(shape)
  return error("floats needs #{n} values for #{shape}, got #{values.size}") if values.size != n
  case backend
    Types.Backend.Torch(device) -> Torch.floats(device, values, shape)
    Types.Backend.Ggml(device) -> Ggml.floats(device, values, shape)
    Types.Backend.Xla(device) -> Xla.floats(device, values, shape)
    _ -> from_list(values.map { |x: Float| Types.Values.Float(x) }, shape, Types.Dtype.F32, backend)
  end
end

## Creation: packed `values` (4 bytes an item) as an `F32` tensor of `shape`,
## for large data that is `F32` already, such as audio. Torch, ggml and XLA take
## `Float`s across their native calls, so they get the list widened, one
## transient copy of 8 bytes an item; the reference converts as it builds.
def floats32(
  values: List[Float32],
  shape: List[Int],
  backend: Types.Backend = Types.Backend.Reference,
) -> Result[Types.Tensor]
  case backend
    Types.Backend.Torch(_) -> floats(values.map { |x: Float32| x.to_f }, shape, backend)
    Types.Backend.Ggml(_) -> floats(values.map { |x: Float32| x.to_f }, shape, backend)
    Types.Backend.Xla(_) -> floats(values.map { |x: Float32| x.to_f }, shape, backend)
    _ ->
      n = try Types.volume(shape)
      return error("floats32 needs #{n} values for #{shape}, got #{values.size}") if values.size != n
      from_list(values.map { |x: Float32| Types.Values.Float(x.to_f) }, shape, Types.Dtype.F32, backend)
  end
end

## Host readback: the values tagged with the tensor's dtype.
def to_list(t: Types.Tensor) -> Result[List[Types.Values]]
  case t.backend
    Types.Backend.Reference -> Reference.to_list(t)
    Types.Backend.Torch(_) -> Torch.to_list(t)
    Types.Backend.Trace -> error("a traced tensor has no values: run it with Trace.eval")
    Types.Backend.Ggml(_) -> Ggml.to_list(t)
    Types.Backend.Xla(_) -> Xla.to_list(t)
  end
end

## Appends the values of the `F32` tensor `t`, row-major and little-endian, to the file `path`, which must exist: what
## `from_file` reads back with `FloatFormat.F32`, for writing a checkpoint. The native backends write from C, so no
## value crosses as an Agentica list; the reference encodes them in Agentica and is slow on a large tensor.
def append_file(t: Types.Tensor, path: String) -> Result[Void]
  return error("append_file writes an F32 tensor, got #{Types.dtype_name(t.dtype)}") if t.dtype != Types.Dtype.F32

  case t.backend
    Types.Backend.Reference -> Reference.append_file(t, path)
    Types.Backend.Torch(_) -> Torch.append_file(t, path)
    Types.Backend.Trace -> error("a traced tensor has no values: run it with Trace.eval")
    Types.Backend.Ggml(_) -> Ggml.append_file(t, path)
    Types.Backend.Xla(_) -> Xla.append_file(t, path)
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
    Types.Backend.Xla(_) -> Xla.compiled_only("add")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("sub")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("mul")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("div")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("neg")
  end
end

## Elementwise `exp`, float only. `exp(-inf)` is `0` (a masked score); `NaN` and `+inf` are errors.
def exp(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "exp")
  case t.backend
    Types.Backend.Reference -> Reference.exp(t)
    Types.Backend.Torch(device) -> Torch.exp(device, t)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Exp, [t], t.shape, t.dtype))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("exp")
    Types.Backend.Xla(_) -> Xla.compiled_only("exp")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("log")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("sqrt")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("pow")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("relu")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("sigmoid")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("tanh")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("gelu_tanh_approx")
  end
end

## The exact GELU, `x * (1 + erf(x / sqrt(2))) / 2`, as Whisper and BERT use
## it. Not a §3.1 primitive: `Trace` records it (so the ggml backend compiles
## it) and differentiates it.
def gelu_erf(t: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(t, "gelu_erf")
  case t.backend
    Types.Backend.Reference -> Reference.gelu_erf(t)
    Types.Backend.Torch(device) -> Torch.gelu_erf(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.GeluErf, [t], t)
    Types.Backend.Ggml(_) -> Ggml.compiled_only("gelu_erf")
    Types.Backend.Xla(_) -> Xla.compiled_only("gelu_erf")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("eq")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("lt")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("where")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("clamp")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("sum")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("mean")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("max")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("argmax")
  end
end

## The rank-2 product `[rows, inner] . [inner, cols]`, float only.
def matmul(a: Types.Tensor, b: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(a, b, "matmul")
  _ = try weight_dtype(a, b, "matmul")
  _ = try float_only(a, "matmul")
  return error("matmul needs rank-2 tensors, got #{a.shape} and #{b.shape}") if a.shape.size != 2 or b.shape.size != 2
  return error("matmul needs #{a.shape[1]} in the inner dimension, got #{b.shape[0]}") if a.shape[1] != b.shape[0]
  case a.backend
    Types.Backend.Reference -> Reference.matmul(a, b)
    Types.Backend.Torch(device) -> Torch.matmul(device, a, b)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.Matmul, [a, b], [a.shape[0], b.shape[1]], a.dtype))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("matmul")
    Types.Backend.Xla(_) -> Xla.compiled_only("matmul")
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
    Types.Backend.Torch(device) -> Torch.batch_matmul(device, a, b)
    Types.Backend.Trace -> ok(Graph.batch_matmul(a, b))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("batch_matmul")
    Types.Backend.Xla(_) -> Xla.compiled_only("batch_matmul")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("transpose")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("permute")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("reshape")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("squeeze")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("unsqueeze")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("slice")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("concat")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("stack")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("broadcast_to")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("gather")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("scatter_add")
  end
end

## `cache` `[..., rows, width]` (rank 2 to 4) with its rows `positions[i]` replaced by the rows `fresh[..., i, :]`
## (`fresh` is `[..., n, width]`, `positions` an `I32` `[n]`). The positions must be distinct: a repeat is unspecified.
## An index outside `[0, rows)` is an error. On every backend but `Ggml` the answer is a new tensor. In a trace that
## runs on `Ggml` it is the cache written in place (`Trace.run`): `cache` must be an input of the function, read by no
## other node, and the result must be an output, which is the very tensor given; `cache` is consumed (see
## `Trace.run`), even when the run fails, and the cache must be `F32`, or `F16` for `F32` rows (ggml rounds them; a value beyond 65504 becomes infinite, as in llama.cpp); the written cache is not checked for finite
## values. `Trace.eval` writes a copy. No gradient.
def set_rows(cache: Types.Tensor, fresh: Types.Tensor, positions: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(cache, fresh, "set_rows")
  _ = try same_backend(cache, positions, "set_rows")
  rank = cache.shape.size
  return error("set_rows needs a cache of rank 2 to 4, got #{cache.shape}") if rank < 2 or rank > 4
  _ = try float_only(cache, "set_rows")
  if fresh.dtype != cache.dtype and !f16_cache?(cache, fresh)
    return error("set_rows needs rows of #{Types.dtype_name(cache.dtype)}, got #{Types.dtype_name(fresh.dtype)}")
  end

  if positions.shape.size != 1 or positions.dtype != Types.Dtype.I32
    return error("set_rows needs positions of I32 [n], got #{Types.dtype_name(positions.dtype)} #{positions.shape}")
  end

  want = cache.shape.set(rank - 2, positions.shape[0])
  if fresh.shape != want
    return error("set_rows needs rows of shape #{want} for a cache #{cache.shape}, got #{fresh.shape}")
  end

  case cache.backend
    Types.Backend.Trace -> Graph.same(Types.Op.SetRows, [cache, fresh, positions], cache)
    Types.Backend.Ggml(_) -> Ggml.compiled_only("set_rows")
    Types.Backend.Xla(_) -> Xla.compiled_only("set_rows")
    _ -> set_rows_composed(cache, fresh, positions)
  end
end

## The indices of the `k` largest elements of the float vector `x`, or of each row of the matrix `x`, an `I32` `[k]`
## (`[rows, k]`), from the largest down on every backend but `Ggml`, where `ggml_top_k` gives them in no particular
## order. Among equal values the ones kept are not specified; the reference and `Xla` keep the lowest indices. In a trace it is one node, which `Ggml` runs on the
## device: only the ids cross to the host, not the whole vector. A row has at most 2^24 elements (ids are exact in `F32`
## below that). No gradient.
def top_ids(x: Types.Tensor, k: Int) -> Result[Types.Tensor]
  _ = try float_only(x, "top_ids")
  return error("top_ids needs a vector or a matrix, got #{x.shape}") if x.shape.size < 1 or x.shape.size > 2
  width = x.shape[x.shape.size - 1]
  return error("top_ids needs k from 1 to #{width}, got #{k}") if k < 1 or k > width
  shape = x.shape.set(x.shape.size - 1, k)
  case x.backend
    Types.Backend.Trace -> ok(Graph.record(Types.Op.TopK(k), [x], shape, Types.Dtype.I32))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("top_ids")
    Types.Backend.Xla(_) -> Xla.compiled_only("top_ids")
    _ ->
      values = try to_list(x)
      scores: List[Float] = []
      for v in values
        scores = scores.push(try Types.value_to_float(v, x.dtype))
      end

      chosen: List[Types.Values] = []
      for r in 0...(scores.size / width)
        largest = (0...width).to_a.sort_by { |i| 0.0 - scores[r * width + i] }
        chosen = chosen + largest.take(k).map { |i| Types.Values.Int(i) }
      end

      from_list(chosen, shape, Types.Dtype.I32, x.backend)
  end
end

## The product of each pair `(token, use)` with the expert that `ids` names: `out[t, u] = weights[ids[t, u]] x[t, u]`,
## the core of a mixture of experts (ggml runs it as `ggml_mul_mat_id`).
## - `weights`: `[experts, rows, inner]`, one `[rows, inner]` matrix per expert as a checkpoint stores it.
## - `x`: `[tokens, uses, inner]`, or `[tokens, 1, inner]` for one input that every use of the token shares.
## - `ids`: `I32 [tokens, uses]`, each in `0...experts` (an id outside is an `error`).
## - The answer is `[tokens, uses, rows]`.
## `F32` only. In a trace it is one node: `Ggml` runs `ggml_mul_mat_id` (CPU and Vulkan), and `XLA` gathers
## the selected weights and runs a batched product. The eager implementation loops over the pairs. No gradient.
def expert_matmul(weights: Types.Tensor, x: Types.Tensor, ids: Types.Tensor) -> Result[Types.Tensor]
  try check_expert_matmul(weights, x, ids)
  out_shape = [ids.shape[0], ids.shape[1], weights.shape[1]]
  case weights.backend
    Types.Backend.Trace -> ok(Graph.record(Types.Op.ExpertMatmul, [weights, x, ids], out_shape, weights.dtype))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("expert_matmul")
    Types.Backend.Xla(_) -> Xla.compiled_only("expert_matmul")
    _ -> expert_matmul_composed(weights, x, ids)
  end
end

## An error unless the three tensors are what `expert_matmul` takes.
def check_expert_matmul(weights: Types.Tensor, x: Types.Tensor, ids: Types.Tensor) -> Result[Void]
  _ = try same_backend(weights, x, "expert_matmul")
  _ = try same_backend(weights, ids, "expert_matmul")
  _ = try same_dtype(weights, x, "expert_matmul")
  if weights.dtype != Types.Dtype.F32
    return error("expert_matmul needs F32 tensors, got #{Types.dtype_name(weights.dtype)}")
  end

  if weights.shape.size != 3 or !all_positive(weights.shape)
    return error("expert_matmul needs weights of [experts, rows, inner], got #{weights.shape}")
  end

  if ids.dtype != Types.Dtype.I32 or ids.shape.size != 2 or !all_positive(ids.shape)
    return error("expert_matmul needs ids of I32 [tokens, uses], got #{Types.dtype_name(ids.dtype)} #{ids.shape}")
  end

  tokens = ids.shape[0]
  uses = ids.shape[1]
  inner = weights.shape[2]
  if x.shape.size != 3 or x.shape[0] != tokens or x.shape[2] != inner or (x.shape[1] != uses and x.shape[1] != 1)
    return error("expert_matmul needs x of [#{tokens}, #{uses} or 1, #{inner}], got #{x.shape}")
  end

  ok()
end

## True when every extent of `shape` is at least 1.
def all_positive(shape: List[Int]) -> Bool
  for d in shape
    return false if d < 1
  end

  true
end

## The loop `expert_matmul` stands for, over the ops of this module, on any backend that computes.
def expert_matmul_composed(weights: Types.Tensor, x: Types.Tensor, ids: Types.Tensor) -> Result[Types.Tensor]
  experts = weights.shape[0]
  rows = weights.shape[1]
  inner = weights.shape[2]
  tokens = ids.shape[0]
  uses = ids.shape[1]
  picks: List[Int] = []
  for v in try to_list(ids)
    case v
      Types.Values.Int(i) -> picks = picks.push(i)
      _ -> return error("expert_matmul needs integer ids")
    end
  end

  answers: List[Types.Tensor] = []
  for t in 0...tokens
    for u in 0...uses
      e = picks[t * uses + u]
      return error("expert_matmul needs ids from 0 to #{experts - 1}, got #{e}") if e < 0 or e >= experts
      w = try reshape(try slice(weights, [e, 0, 0], [e + 1, rows, inner]), [rows, inner])
      j = u
      if x.shape[1] == 1
        j = 0
      end

      v = try reshape(try slice(x, [t, j, 0], [t + 1, j + 1, inner]), [inner, 1])
      answers = answers.push(try reshape(try matmul(w, v), [1, 1, rows]))
    end
  end

  reshape(try concat(answers, 0), [tokens, uses, rows])
end

## The gated delta rule over `tokens` steps, the recurrence of the linear attention of Qwen3.5 (`q` and `k` already
## L2-normalized): per head, `state = exp(g) * state`, then `state = state + k (beta * (v - state k))^T`, and the
## answer is `state q / sqrt(width)`: the op scales `q` itself, so a caller does not.
## - `q`, `k`, `v`: `[tokens, heads, width]`; `g` (the log of the decay, normally at most 0; not checked) and `beta`: `[tokens, heads]`.
## - `state`: `[heads, width, width]`, the transpose of the matrix of the paper (row `j` is the value axis, column
##   `i` the key axis), the layout ggml has: the zeros to start, or the `state` of an earlier call to go on.
## - `F32` only. A token with `g = 0` and `beta = 0` leaves the state as it was, which is how a padded prompt keeps the
##   state of its last real token.
## In a trace it is one node: `Ggml` runs `ggml_gated_delta_net` (CPU, and Vulkan for widths 16, 32, 64 and 128),
## while `XLA` compiles a StableHLO loop.
## The other backends run the recurrence token by token over the ops of this module, so a long sequence is slow there.
## No gradient.
def gated_delta_rule(
  q: Types.Tensor,
  k: Types.Tensor,
  v: Types.Tensor,
  g: Types.Tensor,
  beta: Types.Tensor,
  state: Types.Tensor,
) -> Result[Types.DeltaRule]
  try check_delta_rule(q, k, v, g, beta, state)
  tokens = q.shape[0]
  heads = q.shape[1]
  width = q.shape[2]
  case q.backend
    Types.Backend.Trace ->
      packed = Graph.record(
        Types.Op.GatedDeltaRule,
        [q, k, v, g, beta, state],
        [tokens + width, heads * width],
        Types.Dtype.F32,
      )

      out = try reshape(try slice(packed, [0, 0], [tokens, heads * width]), [tokens, heads, width])
      held = try reshape(try slice(packed, [tokens, 0], [tokens + width, heads * width]), [heads, width, width])
      ok(Types.DeltaRule(out: out, state: held))
    Types.Backend.Ggml(_) -> error(Ggml.compiled_only("gated_delta_rule").message)
    Types.Backend.Xla(_) -> error(Xla.compiled_only("gated_delta_rule").message)
    _ -> gated_delta_composed(q, k, v, g, beta, state)
  end
end

## What a `gated_delta_rule` node answers: the answers as `tokens` rows of `heads * width`, then the state as `width`
## rows of the same width (what ggml's op writes).
def gated_delta_packed(
  q: Types.Tensor,
  k: Types.Tensor,
  v: Types.Tensor,
  g: Types.Tensor,
  beta: Types.Tensor,
  state: Types.Tensor,
) -> Result[Types.Tensor]
  r = try gated_delta_composed(q, k, v, g, beta, state)
  tokens = q.shape[0]
  width = q.shape[2]
  cols = q.shape[1] * width
  concat([try reshape(r.out, [tokens, cols]), try reshape(r.state, [width, cols])], 0)
end

## An error unless the six tensors are what `gated_delta_rule` takes.
def check_delta_rule(
  q: Types.Tensor,
  k: Types.Tensor,
  v: Types.Tensor,
  g: Types.Tensor,
  beta: Types.Tensor,
  state: Types.Tensor,
) -> Result[Void]
  for t in [k, v, g, beta, state]
    _ = try same_backend(q, t, "gated_delta_rule")
    _ = try same_dtype(q, t, "gated_delta_rule")
  end

  return error("gated_delta_rule needs F32 tensors, got #{Types.dtype_name(q.dtype)}") if q.dtype != Types.Dtype.F32

  if q.shape.size != 3 or q.shape[0] < 1 or q.shape[1] < 1 or q.shape[2] < 1
    return error("gated_delta_rule needs q of [tokens, heads, width], got #{q.shape}")
  end

  if k.shape != q.shape or v.shape != q.shape
    return error("gated_delta_rule needs q, k and v of one shape, got #{q.shape}, #{k.shape} and #{v.shape}")
  end

  per_head = [q.shape[0], q.shape[1]]
  if g.shape != per_head or beta.shape != per_head
    return error("gated_delta_rule needs g and beta of #{per_head}, got #{g.shape} and #{beta.shape}")
  end

  if state.shape != [q.shape[1], q.shape[2], q.shape[2]]
    return error("gated_delta_rule needs a state of #{[q.shape[1], q.shape[2], q.shape[2]]}, got #{state.shape}")
  end

  ok()
end

## The recurrence of `gated_delta_rule` one token at a time, on any backend that computes.
def gated_delta_composed(
  q: Types.Tensor,
  k: Types.Tensor,
  v: Types.Tensor,
  g: Types.Tensor,
  beta: Types.Tensor,
  state: Types.Tensor,
) -> Result[Types.DeltaRule]
  tokens = q.shape[0]
  heads = q.shape[1]
  width = q.shape[2]
  matrix = [heads, width, width]
  scale = try full([heads, width], Types.Values.Float(1.0 / Math.sqrt(width.to_f)), q.dtype, q.backend)
  held = state
  outs: List[Types.Tensor] = []
  for t in 0...tokens
    qt = try reshape(try slice(q, [t, 0, 0], [t + 1, heads, width]), [heads, 1, width])
    kt = try broadcast_to(try reshape(try slice(k, [t, 0, 0], [t + 1, heads, width]), [heads, 1, width]), matrix)
    vt = try reshape(try slice(v, [t, 0, 0], [t + 1, heads, width]), [heads, width])
    decay = try exp(try reshape(try slice(g, [t, 0], [t + 1, heads]), [heads, 1, 1]))
    rate = try broadcast_to(try reshape(try slice(beta, [t, 0], [t + 1, heads]), [heads, 1]), [heads, width])
    held = try mul(held, try broadcast_to(decay, matrix))
    seen = try sum(try mul(held, kt), some([2]), false)
    change = try mul(try sub(vt, seen), rate)
    held = try add(held, try mul(try broadcast_to(try reshape(change, [heads, width, 1]), matrix), kt))
    answer = try mul(try sum(try mul(held, try broadcast_to(qt, matrix)), some([2]), false), scale)
    outs = outs.push(try reshape(answer, [1, heads, width]))
  end

  ok(Types.DeltaRule(out: try concat(outs, 0), state: held))
end

## The ops `set_rows` stands for: the cache with the written rows zeroed, plus the rows scattered into place.
def set_rows_composed(cache: Types.Tensor, fresh: Types.Tensor, positions: Types.Tensor) -> Result[Types.Tensor]
  rank = cache.shape.size
  rows = cache.shape[rank - 2]
  axes = [rank - 2] + (0...rank).select { |i| i != rank - 2 }
  back = (0...rank).map { |i| axes.index(i).unwrap! }
  moved = try permute(cache, axes)
  width = try Types.volume(moved.shape.drop(1))
  placed = try scatter_add(positions, try reshape(try permute(fresh, axes), [positions.shape[0], width]), rows)
  ones_in = try ones([positions.shape[0], 1], cache.dtype, cache.backend)
  kept = try sub(try ones([rows, 1], cache.dtype, cache.backend), try scatter_add(positions, ones_in, rows))
  held = try mul(try reshape(moved, [rows, width]), try broadcast_to(kept, [rows, width]))
  permute(try reshape(try add(held, placed), moved.shape), back)
end

## Moves `t` to `dtype` (float to int truncates, anything nonzero is `true`).
def to_dtype(t: Types.Tensor, dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try not_quantized(dtype, "to_dtype")
  case t.backend
    Types.Backend.Reference -> Reference.to_dtype(t, dtype)
    Types.Backend.Torch(device) -> Torch.to_dtype(device, t, dtype)
    Types.Backend.Trace -> ok(Graph.record(Types.Op.ToDtype, [t], t.shape, dtype))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("to_dtype")
    Types.Backend.Xla(_) -> Xla.compiled_only("to_dtype")
  end
end

## Moves `t` to `device`: reference tensors live on `Cpu` only.
def to_device(t: Types.Tensor, device: Types.Device) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.to_device(t, device)
    Types.Backend.Torch(_) -> Torch.unsupported("to_device")
    Types.Backend.Trace -> Graph.same(Types.Op.ToDevice(device), [t], t)
    Types.Backend.Ggml(_) -> Ggml.to_device(t, device)
    Types.Backend.Xla(_) -> Xla.to_device(t, device)
  end
end

## The identity move: values are immutable, so a copy shares nothing.
def copy(t: Types.Tensor) -> Result[Types.Tensor]
  case t.backend
    Types.Backend.Reference -> Reference.copy(t)
    Types.Backend.Torch(device) -> Torch.copy(device, t)
    Types.Backend.Trace -> Graph.same(Types.Op.Copy, [t], t)
    Types.Backend.Ggml(_) -> Ggml.compiled_only("copy")
    Types.Backend.Xla(_) -> Xla.compiled_only("copy")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("contiguous")
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
    Types.Backend.Xla(_) -> ok(t)
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
    Types.Backend.Xla(_) -> Xla.compiled_only("conv1d")
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
    Types.Backend.Xla(_) -> Xla.compiled_only("avg_pool1d")
  end
end

## Convolution over the last two axes: `input` is `[N, C_in, H, W]`, `weight` is `[C_out, C_in / groups, KH, KW]`,
## and the window moves by `stride` with `padding` zeros on each side, on both axes. The answer is `[N, C_out, H_out,
## W_out]`, where `H_out = (H + 2*padding - KH) / stride + 1` and `W_out` likewise. With `groups` above 1 the
## channels split into that many groups, and each group of output channels reads its own group of input channels.
## No dilation.
def conv2d(
  input: Types.Tensor,
  weight: Types.Tensor,
  stride: Int = 1,
  padding: Int = 0,
  groups: Int = 1,
) -> Result[Types.Tensor]
  _ = try same_backend(input, weight, "conv2d")
  _ = try same_dtype(input, weight, "conv2d")
  _ = try float_only(input, "conv2d")
  _ = try float_only(weight, "conv2d")
  return error("conv2d needs a rank-4 input, got #{input.shape}") if input.shape.size != 4
  return error("conv2d needs a rank-4 weight, got #{weight.shape}") if weight.shape.size != 4
  for d in input.shape + weight.shape
    return error("conv2d needs a nonempty input and weight, got #{input.shape} and #{weight.shape}") if d < 1
  end

  _ = try check_window(stride, padding, "conv2d")
  return error("conv2d needs groups >= 1, got #{groups}") if groups < 1
  channels = input.shape[1]
  if channels % groups != 0 or weight.shape[0] % groups != 0
    return error(
      "conv2d needs groups #{groups} to divide the #{channels} input and #{weight.shape[0]} output channels",
    )
  end

  if weight.shape[1] != channels / groups
    return error("conv2d needs #{channels / groups} input channels per group, got #{weight.shape[1]}")
  end

  _ = try check_window_fits(input.shape[2], weight.shape[2], padding, "conv2d")
  _ = try check_window_fits(input.shape[3], weight.shape[3], padding, "conv2d")
  case input.backend
    Types.Backend.Reference -> Reference.conv2d(input, weight, stride, padding, groups)
    Types.Backend.Torch(device) -> Torch.conv2d(device, input, weight, stride, padding, groups)
    Types.Backend.Trace -> ok(Graph.conv2d(input, weight, stride, padding, groups))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("conv2d")
    Types.Backend.Xla(_) -> Xla.compiled_only("conv2d")
  end
end

## The maximum over each square window of `kernel` by `kernel` positions of `input` (`[N, C, H, W]`), moved by
## `stride`; the zeros of `padding` never win (a padded position is skipped). The answer is `[N, C, H_out, W_out]`.
## Padding stops at half the kernel, as in `avg_pool1d`.
def max_pool2d(input: Types.Tensor, kernel: Int, stride: Int = 1, padding: Int = 0) -> Result[Types.Tensor]
  _ = try check_pool2d(input, kernel, stride, padding, "max_pool2d")
  case input.backend
    Types.Backend.Reference -> Reference.max_pool2d(input, kernel, stride, padding)
    Types.Backend.Torch(device) -> Torch.max_pool2d(device, input, kernel, stride, padding)
    Types.Backend.Trace -> ok(Graph.max_pool2d(input, kernel, stride, padding))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("max_pool2d")
    Types.Backend.Xla(_) -> Xla.compiled_only("max_pool2d")
  end
end

## The mean over each square window of `kernel` by `kernel` positions of `input` (`[N, C, H, W]`), moved by `stride`,
## with `padding` zeros on each side; the divisor is always `kernel * kernel`, as in `avg_pool1d`.
def avg_pool2d(input: Types.Tensor, kernel: Int, stride: Int = 1, padding: Int = 0) -> Result[Types.Tensor]
  _ = try check_pool2d(input, kernel, stride, padding, "avg_pool2d")
  case input.backend
    Types.Backend.Reference -> Reference.avg_pool2d(input, kernel, stride, padding)
    Types.Backend.Torch(device) -> Torch.avg_pool2d(device, input, kernel, stride, padding)
    Types.Backend.Trace -> ok(Graph.avg_pool2d(input, kernel, stride, padding))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("avg_pool2d")
    Types.Backend.Xla(_) -> Xla.compiled_only("avg_pool2d")
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

## `layer_norm` over the last axis, no affine: `(x - mean) / sqrt(variance + eps)`. In a trace it is one node, `LayerNorm`,
## so that ggml can run it as one norm; `eps` must then be a float the node holds (finite, within `F32`'s range, and not 0
## once rounded to `F32`). A trace that takes a gradient still records that one node in the forward; only the backward is
## composed of §3.1's ops. On every other backend it is the derived composition of those ops (`layer_norm_composed`).
def layer_norm(x: Types.Tensor, eps: Float) -> Result[Types.Tensor]
  return error("layer_norm needs a finite eps > 0, got #{eps}") if !Types.is_finite(eps) or eps <= 0.0
  return error("layer_norm needs rank >= 1, got #{x.shape}") if x.shape.size < 1
  _ = try float_only(x, "layer_norm")
  case x.backend
    Types.Backend.Trace ->
      if eps > 3.4028234663852886e38 or Float32.to_bits(eps.to_float32) == 0
        return error("layer_norm needs an eps that is neither above F32's range nor 0 in F32, got #{eps}")
      end

      Graph.same(Types.Op.LayerNorm(eps), [x], x)
    _ -> layer_norm_composed(x, eps)
  end
end

## `rms_norm` over the last axis, no affine: `x / sqrt(mean(x * x) + eps)`. Like `layer_norm`, in a trace it is one node,
## `RmsNorm`, so that ggml can run it as one norm (and fuse the product by a weight after it); `eps` must then be a float
## the node holds. On every other backend it is the derived composition of §3.1's ops (`rms_norm_composed`).
def rms_norm(x: Types.Tensor, eps: Float) -> Result[Types.Tensor]
  return error("rms_norm needs a finite eps > 0, got #{eps}") if !Types.is_finite(eps) or eps <= 0.0
  return error("rms_norm needs rank >= 1, got #{x.shape}") if x.shape.size < 1
  _ = try float_only(x, "rms_norm")
  case x.backend
    Types.Backend.Trace ->
      if eps > 3.4028234663852886e38 or Float32.to_bits(eps.to_float32) == 0
        return error("rms_norm needs an eps that is neither above F32's range nor 0 in F32, got #{eps}")
      end

      Graph.same(Types.Op.RmsNorm(eps), [x], x)
    _ -> rms_norm_composed(x, eps)
  end
end

## The update kernel of Adam and AdamW, which `Optim` uses: call `Optim`, not this. From a weight `p`, its gradient `g`,
## the first and second moments `m` and `v`, and `hyper`, a rank-1 tensor of 7 numbers (the learning rate, `beta1`,
## `beta2`, `eps`, the weight decay, then `1 / (1 - beta1^t)` and `1 / (1 - beta2^t)`), it answers the new `[p, m, v]`:
## `m = beta1 m + (1 - beta1) g`, `v = beta2 v + (1 - beta2) g^2`,
## `p = p (1 - lr decay) - lr (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)`. In a trace of `F32` tensors of
## rank 3 or less it is one node, which ggml runs as `ggml_opt_step_adamw` on copies of the arguments (they stay
## unchanged); every other case is the composition of the ops above (`Trace.stablehlo` writes that composition, as it
## has no emitter for the node). In a function compiled by `Trace.compile`, the moments `m` and `v` are updated in
## place, and their results are those very tensors (as the cache of a `set_rows`), when they are inputs that nothing
## else reads and both results are outputs: they are consumed. `Trace.eval` and `eval_many` never do.
def adam_step(
  p: Types.Tensor,
  g: Types.Tensor,
  m: Types.Tensor,
  v: Types.Tensor,
  hyper: Types.Tensor,
) -> Result[List[Types.Tensor]]
  for t in [g, m, v, hyper]
    _ = try same_backend(p, t, "adam_step")
    _ = try same_dtype(p, t, "adam_step")
  end

  _ = try float_only(p, "adam_step")
  for t in [g, m, v]
    _ = try same_shape(p, t, "adam_step")
  end

  return error("adam_step needs 7 hyper-parameters, got shape #{hyper.shape}") if hyper.shape != [7]
  case p.backend
    Types.Backend.Trace ->
      if p.dtype == Types.Dtype.F32 and p.shape.size <= 3
        stepped = Graph.adam_step([p, g, m, v, hyper], p)
        return ok((0...3).map { |i| Graph.adam_part(stepped, i, p.shape) })
      end

      adam_composed(p, g, m, v, hyper)
    _ -> adam_eager(p, g, m, v, hyper)
  end
end

## Derived: `adam_step` from the ops of §3.1, over the numbers of `adam_coefficients`. It is not a user API: it is
## public only for `Trace.stablehlo`, which writes an `adam_step` node as these ops. Call `adam_step`.
def adam_composed(
  p: Types.Tensor,
  g: Types.Tensor,
  m: Types.Tensor,
  v: Types.Tensor,
  hyper: Types.Tensor,
) -> Result[List[Types.Tensor]]
  c = try adam_coefficients(hyper, p)
  first = try add(try mul(m, c[1]), try mul(g, c[2]))
  second = try add(try mul(v, c[3]), try mul(try mul(g, g), c[4]))
  denominator = try add(try sqrt(try mul(second, c[8])), c[5])
  step = try mul(c[0], try div(try mul(first, c[7]), denominator))
  ok([try sub(try mul(p, c[6]), step), first, second])
end

## Derived: the user-facing `gelu` is the `tanh` approximation (§3.2).
def gelu(x: Types.Tensor) -> Result[Types.Tensor] = gelu_tanh_approx(x)

## Derived: `silu(x) = x * sigmoid(x)`, from §3.1 only.
def silu(x: Types.Tensor) -> Result[Types.Tensor]
  x_sig = try sigmoid(x)
  mul(x, x_sig)
end

## Derived: `softplus(x) = log(1 + exp(x))`, written as `max(x, 0) + log(1 + exp(-|x|))` so that a large `x` does not
## overflow `exp`. `|x|` is `2 relu(x) - x`, so that at `x = 0`, where the derivative of `relu` is taken as `0`, the
## gradient is still `sigmoid(0)`.
def softplus(x: Types.Tensor) -> Result[Types.Tensor]
  _ = try float_only(x, "softplus")
  above = try relu(x)
  size = try sub(try add(above, above), x)
  one = try ones(x.shape, x.dtype, x.backend)
  add(above, try log(try add(one, try exp(try neg(size)))))
end

## Derived: `x` over its last axis divided by `sqrt(sum(x * x) + eps)`, the L2 normalization of the linear attention of
## Qwen3.5 (`eps` inside the root: a row of zeros stays zeros).
def l2_norm(x: Types.Tensor, eps: Float) -> Result[Types.Tensor]
  return error("l2_norm needs a finite eps > 0, got #{eps}") if !Types.is_finite(eps) or eps <= 0.0
  return error("l2_norm needs rank >= 1, got #{x.shape}") if x.shape.size < 1
  _ = try float_only(x, "l2_norm")
  squares = try sum(try mul(x, x), some([x.shape.size - 1]), true)
  floor = try full(squares.shape, Types.Values.Float(eps), x.dtype, x.backend)
  div(x, try broadcast_to(try sqrt(try add(squares, floor)), x.shape))
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

## Derived: `x . w (+ b)`, with the bias widened by `broadcast_to`. With
## `out_in`, `w` is `[out, in]` as a checkpoint stores it and the product is
## `x . wᵀ`: libtorch reads `w` as it is, ggml too once the graph is lowered,
## and the reference and a trace take its transpose.
def linear(
  x: Types.Tensor,
  w: Types.Tensor,
  b: Optional[Types.Tensor] = none,
  out_in: Bool = false,
) -> Result[Types.Tensor]
  _ = try same_backend(x, w, "linear")
  _ = try weight_dtype(x, w, "linear")
  _ = try float_only(x, "linear")
  return error("linear needs a rank-2 input, got #{x.shape}") if x.shape.size != 2
  return error("linear needs a rank-2 weight, got #{w.shape}") if w.shape.size != 2
  inner = w.shape[0]
  outer = w.shape[1]
  if out_in
    inner = w.shape[1]
    outer = w.shape[0]
  end

  return error("linear needs #{x.shape[1]} in the inner dimension, got #{inner}") if x.shape[1] != inner
  y = try product(x, w, out_in)
  case b
    some(bias) ->
      _ = try same_backend(x, bias, "linear")
      _ = try same_dtype(x, bias, "linear")
      return error("linear needs a bias of shape [#{outer}], got #{bias.shape}") if bias.shape != [outer]
      add(y, try broadcast_to(bias, y.shape))
    none -> ok(y)
  end
end

## Derived: row lookup, `gather` with the rank the name promises.
def embedding(ids: Types.Tensor, table: Types.Tensor) -> Result[Types.Tensor]
  _ = try same_backend(ids, table, "embedding")
  return error("embedding needs integer ids, got #{Types.dtype_name(ids.dtype)}") if !Types.is_int_dtype(ids.dtype)
  if !Types.is_quantized_dtype(table.dtype)
    _ = try float_only(table, "embedding")
  end

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
## k_len]` with `1` allowed for `batch` and for `heads`; it takes finite values
## and `-inf`, and it has a gradient. `causal` adds the causal mask, built from
## `arange` and `lt` so it is the same on every backend: query `i` reads keys
## `0..i + k_len - q_len` (inclusive), so the keys of a cache that holds earlier
## positions line up with the last queries; with `q_len > k_len` the first
## queries would have no key to read, which is an error. `scale` defaults to
## `1 / sqrt(head_dim)`.
##
## `q`, `k` and `v` are finite, and a query with every key at `-inf` is an
## error on every backend, never a `NaN`. To hide a position, add a large finite
## negative number (the dtype's lowest, as Hugging Face does for padding) or
## `-inf`; the causal mask uses `-1e9` (`-65504` for `F16`). The reference
## builds the causal mask with `full`, which holds at most 1,048,576 elements,
## so on `Reference` `q_len * k_len` stops there; the other backends have no
## such limit.
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
  _ = try same_dtype(k, v, "attention")
  if k.dtype != q.dtype and !(k.dtype == Types.Dtype.F16 and q.dtype == Types.Dtype.F32 and compiled?(q))
    return error("attention needs #{Types.dtype_name(q.dtype)} keys and values, got #{Types.dtype_name(k.dtype)}")
  end

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
  case q.backend
    Types.Backend.Ggml(_) -> return Ggml.compiled_only("attention")
    Types.Backend.Xla(_) -> return Xla.compiled_only("attention")
    _ -> nil
  end

  return error("attention: causal needs q_len <= k_len, got #{sq} queries over #{sk} keys") if causal and sq > sk
  bias = try attention_bias(q, sk, mask, causal)
  case q.backend
    Types.Backend.Reference -> Reference.attention(q, k, v, bias, chosen)
    Types.Backend.Torch(device) -> Torch.attention(device, q, k, v, bias, chosen)
    Types.Backend.Trace -> ok(Graph.attention(q, k, v, bias, chosen))
    Types.Backend.Ggml(_) -> Ggml.compiled_only("attention")
    Types.Backend.Xla(_) -> Xla.compiled_only("attention")
  end
end

## Rotary position embedding of `x` `[..., seq, width]`: the first `dims`
## columns of row `s` (`width` by default, an even number) are rotated in
## pairs by the angle `positions[s] * base^(-2i/dims)`, pair `i`; the columns
## past `dims` pass through. With `neox` pair `i` is columns `(i, i + dims/2)`,
## the layout of Hugging Face's `rotate_half`; without it, `(2i, 2i + 1)`, as
## the original Llama does. `positions` is an integer vector of length `seq`
## (on ggml any integer dtype but `I64`), shared by every batch and head.
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
    Types.Backend.Xla(_) -> Xla.compiled_only("rope")
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

## ggml holds every packed dtype but `BF16`; XLA holds `F16` and `BF16`.
def holds_packed?(backend: Types.Backend, dtype: Types.Dtype) -> Bool
  case backend
    Types.Backend.Ggml(_) -> dtype != Types.Dtype.BF16
    Types.Backend.Xla(_) -> dtype == Types.Dtype.F16 or dtype == Types.Dtype.BF16
    _ -> false
  end
end

## What `body` makes, with every other tensor it made freed on return.
def staged(body: Fn[Result[Types.Tensor]]) -> Result[Types.Tensor] = Native.arena do body.call() end

## `t` times `k` in every element; the tensor of `k`s is freed on return.
def scaled_by(t: Types.Tensor, k: Float) -> Result[Types.Tensor]
  staged do
    factor = try full(t.shape, Types.Values.Float(k), t.dtype, t.backend)
    mul(t, factor)
  end
end

## `adam_composed` on a backend that computes: the same ops and numbers, made one statement at a time, each in an arena
## of its own. A coefficient is a tensor of the shape of `p`, and the nine at once and the temporaries of the whole
## step are more than a word table of MiniLM can hold beside its gradients.
def adam_eager(
  p: Types.Tensor,
  g: Types.Tensor,
  m: Types.Tensor,
  v: Types.Tensor,
  hyper: Types.Tensor,
) -> Result[List[Types.Tensor]]
  x = try host_floats(hyper)
  case p.backend
    Types.Backend.Torch(device) -> adam_native(device, p, g, m, v, x)
    _ -> adam_staged(p, g, m, v, x)
  end
end

## The libtorch kernel, which takes `F32`: one pass, its three outputs the views of one stacked tensor. Another dtype
## is composed.
def adam_native(
  device: Types.Device,
  p: Types.Tensor,
  g: Types.Tensor,
  m: Types.Tensor,
  v: Types.Tensor,
  x: List[Float],
) -> Result[List[Types.Tensor]]
  return adam_staged(p, g, m, v, x) if p.dtype != Types.Dtype.F32

  stack = try Torch.adam_step(device, p, g, m, v, x)
  ok([try part_of(stack, 0, p.shape), try part_of(stack, 1, p.shape), try part_of(stack, 2, p.shape)])
end

## Output `i` of a stacked step, as the view of its rows.
def part_of(stack: Types.Tensor, i: Int, shape: List[Int]) -> Result[Types.Tensor]
  starts = [i] + shape.map { |_: Int| 0 }
  ends = [i + 1] + shape
  reshape(try slice(stack, starts, ends), shape)
end

## `adam_composed` one statement at a time, each in its own arena, so a step holds a few tensors, not one per node.
def adam_staged(
  p: Types.Tensor,
  g: Types.Tensor,
  m: Types.Tensor,
  v: Types.Tensor,
  x: List[Float],
) -> Result[List[Types.Tensor]]
  first = try staged do
    keep = try scaled_by(m, x[1])
    add(keep, try scaled_by(g, 1.0 - x[1]))
  end

  second = try staged do
    keep = try scaled_by(v, x[2])
    squared = try mul(g, g)
    add(keep, try scaled_by(squared, 1.0 - x[2]))
  end

  denominator = try staged do
    root = try sqrt(try scaled_by(second, x[6]))
    add(root, try full(p.shape, Types.Values.Float(x[3]), p.dtype, p.backend))
  end

  step = try staged do
    ratio = try div(try scaled_by(first, x[5]), denominator)
    scaled_by(ratio, x[0])
  end

  stepped = try staged do
    decayed = try scaled_by(p, 1.0 - x[0] * x[4])
    sub(decayed, step)
  end

  ok([stepped, first, second])
end

## `x . w`, or `x . wᵀ` when `w` is `[out, in]`.
def product(x: Types.Tensor, w: Types.Tensor, out_in: Bool) -> Result[Types.Tensor]
  return matmul(x, w) if !out_in
  case x.backend
    Types.Backend.Torch(device) -> Torch.linear(device, x, w)
    _ -> matmul(x, try transpose(w))
  end
end

## The ops `layer_norm` stands for, from §3.1 only: the value on every backend that computes, and the graph a backward
## rule differentiates.
def layer_norm_composed(x: Types.Tensor, eps: Float) -> Result[Types.Tensor]
  last = x.shape.size - 1
  mu = try mean(x, some([last]), true)
  centered = try sub(x, try broadcast_to(mu, x.shape))
  var = try mean(try mul(centered, centered), some([last]), true)
  eps_full = try full(x.shape, Types.Values.Float(eps), x.dtype, x.backend)
  denom = try sqrt(try add(try broadcast_to(var, x.shape), eps_full))
  div(centered, denom)
end

## `[lr, beta1, 1 - beta1, beta2, 1 - beta2, eps, 1 - lr decay, correct1, correct2]` as tensors of the shape of `p`.
## A computing backend holds `hyper`: its 7 numbers are read once and the constants are made on the host, as the
## one-node form reads them; a trace slices `hyper`, which costs nothing when it runs.
def adam_coefficients(hyper: Types.Tensor, p: Types.Tensor) -> Result[List[Types.Tensor]]
  case p.backend
    Types.Backend.Trace ->
      at: List[Types.Tensor] = []
      for i in 0...7
        at = at.push(try hyper_at(hyper, i, p))
      end

      one = try full(p.shape, Types.Values.Float(1.0), p.dtype, p.backend)
      ok([
        at[0],
        at[1],
        try sub(one, at[1]),
        at[2],
        try sub(one, at[2]),
        at[3],
        try sub(one, try mul(at[0], at[4])),
        at[5],
        at[6],
      ])
    _ ->
      x = try host_floats(hyper)
      numbers = [x[0], x[1], 1.0 - x[1], x[2], 1.0 - x[2], x[3], 1.0 - x[0] * x[4], x[5], x[6]]
      out: List[Types.Tensor] = []
      for n in numbers
        out = out.push(try full(p.shape, Types.Values.Float(n), p.dtype, p.backend))
      end

      ok(out)
  end
end

## The number `i` of the traced `hyper` as a tensor of `like`'s shape.
def hyper_at(hyper: Types.Tensor, i: Int, like: Types.Tensor) -> Result[Types.Tensor]
  broadcast_to(try reshape(try slice(hyper, [i], [i + 1]), []), like.shape)
end

def host_floats(t: Types.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try to_list(t)
    case v
      Types.Values.Float(f) -> out = out.push(f)
      _ -> return error("adam_step takes float hyper-parameters")
    end
  end

  ok(out)
end

## The ops `rms_norm` stands for, from §3.1 only.
def rms_norm_composed(x: Types.Tensor, eps: Float) -> Result[Types.Tensor]
  last = x.shape.size - 1
  ms = try mean(try mul(x, x), some([last]), true)
  eps_full = try full(x.shape, Types.Values.Float(eps), x.dtype, x.backend)
  denom = try sqrt(try add(try broadcast_to(ms, x.shape), eps_full))
  div(x, denom)
end

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

## `x` and the weight `w` of a product share their dtype, or `w` is a half or
## quantized weight (`F16`, `Q8_0`, `Q4_K` or `Q6_K`, from `from_file` with that `dtype`) of an
## `F32` `x` in a trace: ggml reads the weight and answers `F32`, and XLA widens it in the program.
def weight_dtype(x: Types.Tensor, w: Types.Tensor, op: String) -> Result[Bool]
  case x.backend
    Types.Backend.Trace -> return ok(true) if x.dtype == Types.Dtype.F32 and Types.is_packed_dtype(w.dtype)
    _ -> nil
  end

  same_dtype(x, w, op)
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

## What `max_pool2d` and `avg_pool2d` share: a float `[N, C, H, W]` input and a window that fits it.
def check_pool2d(input: Types.Tensor, kernel: Int, stride: Int, padding: Int, op: String) -> Result[Bool]
  _ = try float_only(input, op)
  return error("#{op} needs a rank-4 input, got #{input.shape}") if input.shape.size != 4
  return error("#{op} needs a nonempty input, got #{input.shape}") if input.shape[0] < 1 or input.shape[1] < 1
  return error("#{op} needs kernel >= 1, got #{kernel}") if kernel < 1
  _ = try check_window(stride, padding, op)
  if padding > kernel / 2
    return error("#{op} needs padding <= #{kernel / 2} for a kernel of #{kernel}, got #{padding}")
  end

  _ = try check_window_fits(input.shape[2], kernel, padding, op)
  check_window_fits(input.shape[3], kernel, padding, op)
end

## The quantized dtypes are dtypes of weights read by `from_file`: no other op makes a tensor of one.
def not_quantized(dtype: Types.Dtype, op: String) -> Result[Bool]
  if Types.is_quantized_dtype(dtype)
    return error("#{op} makes no #{Types.dtype_name(dtype)} tensor: only from_file reads one (tensor/DESIGN.md §7)")
  end

  ok(true)
end

## An `F16` cache `set_rows` writes `F32` rows into, which only a graph (a trace or `Ggml`) does.
def f16_cache?(cache: Types.Tensor, fresh: Types.Tensor) -> Bool
  return false if cache.dtype != Types.Dtype.F16 or fresh.dtype != Types.Dtype.F32
  compiled?(cache)
end

## `t` belongs to a trace or to `Ggml`, where an `F16` key, value or cache is read without a widened copy.
def compiled?(t: Types.Tensor) -> Bool
  case t.backend
    Types.Backend.Trace -> true
    Types.Backend.Ggml(_) -> true
    _ -> false
  end
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

  lowest = -1.0e9
  if like.dtype == Types.Dtype.F16
    lowest = -65504.0
  end

  blocked = try full([sq, sk], Types.Values.Float(lowest), like.dtype, like.backend)
  mul(try to_dtype(later, like.dtype), blocked)
end
