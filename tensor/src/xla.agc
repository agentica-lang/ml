## The XLA backend of `tensor/`: tensors in PJRT buffers of XLA's CPU plugin, called from the `Backend.Xla(device)`
## arms of `../tensor.agc` (`tensor/DESIGN.md` §7). It lives inside `tensor/` for the reason `src/ggml.agc` does:
## `Handle.Xla` names the resource of `xla_native.agc`.
##
## XLA computes whole programs, so ops run only there: `Trace.compile` emits the traced function as StableHLO
## (`src/stablehlo.agc`) and the plugin compiles it once. Eagerly the backend creates tensors, reads them back and
## passes `stop_grad` through; every other primitive answers `compiled_only(op)`.
##
## The plugin is not built here and not linked: `native/xla.c` loads it with `dlopen` at the first use, from the ml
## cache, where `tools/install_xla.agc` puts the pinned wheel of PyPI (`WHEELS`).
##
## Storage: `F32`, `F64`, `I32`, `I64`, `I16`, `I8`, `U8` and `Bool`; `F16` and `Q8_0` are refused. Lifetimes (§8):
## one `Native.arena` per call, as in `src/ggml.agc`.

import Reference from "reference"
import Types from "types"
import XlaRaw from "xla_native"

## The member of a wheel that is the plugin.
PLUGIN = "xla_plugins/xla_cpu_pjrt/xla_cpu_pjrt.so"

## One wheel of `xla-cpu-pjrt` 0.0.1 on PyPI: the machine it is for, where it is and its sha256.
struct Wheel
  arch: String
  url: String
  sha256: String
end

## A function compiled by XLA: the program, the tensors that follow the inputs as arguments (what the function read
## through `Trace.constant`) and the shape and dtype of every output.
struct Program
  program: XlaRaw.Program
  constants: List[Types.Tensor]
  outs: List[Types.Tensor]
  ## The StableHLO text and the inputs it updates in place: a run after `Trace.close` compiles it again.
  text: String
  donated: List[Int]
  ## The program answers one more output, true when an index of a `gather` is outside its rows
  ## (`Stablehlo.Module.checked`).
  checked: Bool
end

## The pinned plugins, one per machine (Linux only). `tools/install_xla.agc` downloads one, and `plugin_path` finds
## it by its sha256.
def wheels() -> List[Wheel]
  [
    Wheel(
      arch: "x86_64",
      url: (
        "https://files.pythonhosted.org/packages/5d/ca/5487881b30d07396c168749ead36f57dbc4538cafdc535943afc81b3e5a2/xla_cpu_pjrt-0.0.1-py3-none-manylinux_2_27_x86_64.whl"
      ),
      sha256: "55837b6b7b47daa717f2d9a5a03563ea0ceaa338d06a81a1e2d3f060d960decc",
    ),
    Wheel(
      arch: "aarch64",
      url: (
        "https://files.pythonhosted.org/packages/c9/2d/2c6703356beded886d95f9e37adb49aeca7b21e7617f2ff59893f038f61e/xla_cpu_pjrt-0.0.1-py3-none-manylinux_2_27_aarch64.whl"
      ),
      sha256: "8f613be6e891858f40d64f9799319502fff42689661944fc18a6ffd8210b30c9",
    ),
  ]
end

## The wheel of this machine; an error on a system that has none.
def wheel() -> Result[Wheel]
  return error("the XLA backend needs its native module, which is not built (#{loaded_message()})") if !loaded?()
  arch = XlaRaw.arch()
  if System.os() == "linux"
    for w in wheels()
      return ok(w) if w.arch == arch
    end
  end

  error("no XLA plugin for #{System.os()} #{arch}: the pinned wheel has Linux x86_64 and aarch64 only")
end

## The folder of the plugins: `xla/` of `AGENTICA_ML_CACHE`, else of `$XDG_CACHE_HOME/agentica-ml`, else of
## `~/.cache/agentica-ml`.
def cache_dir() -> Result[String]
  root = set_env("AGENTICA_ML_CACHE")
  if root.empty?
    xdg = set_env("XDG_CACHE_HOME")
    if xdg.empty?
      home = set_env("HOME")
      return error("no cache folder: set AGENTICA_ML_CACHE or HOME") if home.empty?
      xdg = try Path.join([home, ".cache"])
    end

    root = try Path.join([xdg, "agentica-ml"])
  end

  Path.join([root, "xla"])
end

## The folder the wheel `w` is extracted into, named by its sha256.
def wheel_dir(w: Wheel) -> Result[String] = Path.join([try cache_dir(), w.sha256])

## Where the plugin of this machine is once installed.
def plugin_path() -> Result[String] = Path.join([try wheel_dir(try wheel()), PLUGIN])

## True when `device` can hold XLA tensors here: the module is built, the device is the CPU, and the plugin loads.
def available(device: Types.Device) -> Bool
  return false if device != Types.Device.Cpu
  usable("available").ok?
end

## The answer of every primitive the XLA backend has no eager form of.
def compiled_only(op: String) -> Result[Types.Tensor]
  error(
    "#{op} does not run eagerly on the XLA backend: XLA runs whole compiled functions (Trace.compile; tensor/DESIGN.md §7)",
  )
end

## Creation: every element `x`, a storage float already checked for `dtype`.
def filled(device: Types.Device, shape: List[Int], dtype: Types.Dtype, x: Float, op: String) -> Result[Types.Tensor]
  code = try ready(device, dtype, op)
  made(op, shape, dtype) do XlaRaw.filled(shape, x, code) end
end

## Creation: every element `value` (tagged for `dtype`).
def full(device: Types.Device, shape: List[Int], value: Types.Values, dtype: Types.Dtype) -> Result[Types.Tensor]
  filled(device, shape, dtype, try Types.value_to_float(value, dtype), "full")
end

## Creation: `start, start + step, ...` below `stop` (above for `step < 0`), made on the reference and uploaded.
def arange(device: Types.Device, start: Int, stop: Int, step: Int, dtype: Types.Dtype) -> Result[Types.Tensor]
  _ = try ready(device, dtype, "arange")
  uploaded(device, try Reference.arange(start, stop, step, dtype))
end

## Creation: row-major `F32` `values` of `shape`; the wrapper checked the count.
def floats(device: Types.Device, values: List[Float], shape: List[Int]) -> Result[Types.Tensor]
  code = try ready(device, Types.Dtype.F32, "from_list")
  made("from_list", shape, Types.Dtype.F32) do XlaRaw.upload(shape, values, code) end
end

## Creation: row-major `data` of `shape`; the wrapper checked the count.
def from_list(
  device: Types.Device,
  data: List[Types.Values],
  shape: List[Int],
  dtype: Types.Dtype,
) -> Result[Types.Tensor]
  code = try ready(device, dtype, "from_list")
  values: List[Float] = []
  for v in data
    values = values.push(try Types.value_to_float(v, dtype))
  end

  made("from_list", shape, dtype) do XlaRaw.upload(shape, values, code) end
end

## Creation: an `F32` tensor from a file, read by the reference and uploaded.
def from_file(
  device: Types.Device,
  path: String,
  offset: Int,
  shape: List[Int],
  format: Types.FloatFormat,
) -> Result[Types.Tensor]
  _ = try ready(device, Types.Dtype.F32, "from_file")
  uploaded(device, try Reference.from_file(path, offset, shape, format))
end

## Host readback: the row-major values tagged with `t`'s dtype.
def to_list(t: Types.Tensor) -> Result[List[Types.Values]]
  read = XlaRaw.readback(try handle_of(t))
  return error("to_list: #{read.message}") if !read.ok?
  out: List[Types.Values] = []
  for x in read.unwrap!
    out = out.push(try Types.float_to_value(x, t.dtype))
  end

  ok(out)
end

## Appends the `F32` values of `t` to the file `path`, through the reference.
def append_file(t: Types.Tensor, path: String) -> Result[Void]
  Reference.append_file(try Reference.from_list(try to_list(t), t.shape, t.dtype), path)
end

## `t` on `device`: the CPU is the only device, so `t` itself.
def to_device(t: Types.Tensor, device: Types.Device) -> Result[Types.Tensor]
  _ = try cpu(device, "to_device")
  ok(t)
end

## The XLA handle of `t`, which must not be closed.
def handle_of(t: Types.Tensor) -> Result[XlaRaw.Buffer]
  case t.handle
    Types.Handle.Xla(h) ->
      return error("the XLA tensor is closed") if Native.closed?(h)
      ok(h)
    _ -> error("a #{Types.backend_name(t.backend)} tensor has no XLA buffer")
  end
end

## The StableHLO `text` compiled, for outputs like the traced `outs`; `constants` must be XLA tensors, and `donated`
## are the inputs the text aliases to an output. The program comes from the cache of the process when the same text
## was compiled lately.
def compile(
  text: String,
  constants: List[Types.Tensor],
  outs: List[Types.Tensor],
  donated: List[Int],
  checked: Bool,
) -> Result[Program]
  _ = try usable("compile")
  for c in constants
    _ = try handle_of(c)
  end

  extra = 0
  if checked
    extra = 1
  end

  p = try Native.arena do XlaRaw.compile(text, outs.size + extra, donated) end
  ok(Program(program: p, constants: constants, outs: outs, text: text, donated: donated, checked: checked))
end

## One run of `p` over `args` (the inputs, checked by the caller): a new tensor per output.
## A run after `Trace.close` takes the program from the cache, or compiles it again.
def execute(p: Program, args: List[Types.Tensor]) -> Result[List[Types.Tensor]]
  if Native.closed?(p.program)
    again = try compile(p.text, p.constants, p.outs, p.donated, p.checked)
    values = execute(again, args)
    Native.close(again.program)
    return values
  end

  buffers = try Native.arena do run_call(p.program, args + p.constants) end
  if p.checked and !buffers.empty?
    flag = buffers.last
    buffers = buffers.take(buffers.size - 1)
    read = XlaRaw.readback(flag)
    Native.close(flag)
    if !read.ok? or read.unwrap! != [0.0]
      for b in buffers
        Native.close(b)
      end

      return error("gather: an index is outside the rows of its source") if read.ok?
      return error("run: #{read.message}")
    end
  end

  return error("the run gave #{buffers.size} outputs, the program has #{p.outs.size}") if buffers.size != p.outs.size
  ok((0...buffers.size).map { |i| tensor_of(p.outs[i].shape, p.outs[i].dtype, buffers[i]) })
end

## How many programs the process has compiled, the misses of its cache, for a test; 0 without the module.
def compiles() -> Int
  return 0 if !loaded?()
  XlaRaw.compiles()
end

## How many XLA buffers are alive in the process, for a test; 0 without the module.
def buffers() -> Int
  return 0 if !loaded?()
  XlaRaw.buffers()
end

## The plugin loaded, or an error that says how to install it.
def usable(op: String) -> Result[Bool] = open_at(try plugin_path(), op)

## The plugin at `path` loaded, or an error that names `op` and says how to install it.
def open_at(path: String, op: String) -> Result[Bool]
  case XlaRaw.open(path)
    ok(_) -> ok(true)
    error(e) ->
      error(
        "#{op}: backend not available (#{e.message}): run `agentica tools/install_xla.agc` in tensor/ " +
        "(tensor/README.md, \"The XLA backend\")",
      )
  end
end

private

def loaded?() -> Bool = Native.load("xla").ok?

def loaded_message() -> String
  case Native.load("xla")
    ok(_) -> ""
    error(e) -> e.message
  end
end

## The value of the environment variable `name`, or "" when it is not set.
def set_env(name: String) -> String
  case System.env(name)
    ok(v) -> v
    error(_) -> ""
  end
end

def run_call(program: XlaRaw.Program, args: List[Types.Tensor]) -> Result[List[XlaRaw.Buffer]]
  c = XlaRaw.call(program)
  for t in args
    try XlaRaw.arg(c, try handle_of(t))
  end

  XlaRaw.execute(c)
end

def tensor_of(shape: List[Int], dtype: Types.Dtype, h: XlaRaw.Buffer) -> Types.Tensor
  Types.Tensor(shape: shape, dtype: dtype, backend: Types.Backend.Xla(Types.Device.Cpu), handle: Types.Handle.Xla(h))
end

## The reference tensor `t` as an XLA tensor.
def uploaded(device: Types.Device, t: Types.Tensor) -> Result[Types.Tensor]
  from_list(device, try Reference.to_list(t), t.shape, t.dtype)
end

## One native creation inside the arena of this call; a native error names the op.
def made(op: String, shape: List[Int], dtype: Types.Dtype, call: Fn[Result[XlaRaw.Buffer]]) -> Result[Types.Tensor]
  Native.arena do
    case call.call()
      ok(h) -> ok(tensor_of(shape, dtype, h))
      error(e) -> error("#{op}: #{e.message}")
    end
  end
end

## The plugin is loaded, the device is the CPU, and the dtype's code for the native calls.
def ready(device: Types.Device, dtype: Types.Dtype, op: String) -> Result[Int]
  _ = try cpu(device, op)
  _ = try usable(op)
  code_of(dtype, op)
end

def cpu(device: Types.Device, op: String) -> Result[Bool]
  return ok(true) if device == Types.Device.Cpu
  error("#{op}: the XLA backend runs on the CPU only, got #{Types.device_name(device)}")
end

## The dtype codes of `native/xla.c`.
def code_of(dtype: Types.Dtype, op: String) -> Result[Int]
  case dtype
    Types.Dtype.F32 -> ok(0)
    Types.Dtype.F64 -> ok(1)
    Types.Dtype.I32 -> ok(2)
    Types.Dtype.I64 -> ok(3)
    Types.Dtype.I16 -> ok(4)
    Types.Dtype.I8 -> ok(5)
    Types.Dtype.U8 -> ok(6)
    Types.Dtype.Bool -> ok(7)
    _ -> error("#{op}: the XLA backend holds no #{Types.dtype_name(dtype)} tensor")
  end
end
