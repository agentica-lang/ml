## The native side of the XLA backend: the calls of `native/xla.c`, a thin C layer over the PJRT C API of a plugin that
## it loads with `dlopen` at run time (`tensor/DESIGN.md` §7). Nothing of XLA is linked at build time.
##
## This file is the FFI boundary and nothing else, as `ggml_native.agc` is for ggml. `src/xla.agc` is the backend.
## After a change to the block, from the package folder:
##
## ```sh
## agentica deps/native-gen/cli.agc src/xla_native.agc --output native/xla_native.h
## ```
##
## A dtype crosses as a code of `src/xla.agc` (`code_of`). The calls copy, compile or compute, so they run on the
## compute pool.

extern module "xla"
  ## One PJRT buffer on the CPU device of the plugin, destroyed by the free hook.
  resource Buffer
  ## One program compiled by the plugin (a PJRT loaded executable), destroyed by the free hook.
  resource Program
  ## The machine this module was built for: `x86_64`, `aarch64` or `other`. It loads no plugin.
  def arch() -> String
  ## Loads the plugin at `path` and makes its client, once for the process; a later call with the same path does
  ## nothing, and with another path is an error. A failure leaves no client, so a later call may try again. On
  ## x86_64 a CPU without AVX is an error, before anything loads.
  compute def open(path: String) -> Result[Void]
  ## A buffer of the row-major `shape` and the dtype `dtype` holding `data`, one value per element.
  compute def upload(shape: List[Int], data: List[Float], dtype: Int) -> Result[Buffer]
  ## A buffer of `shape` from the range at `offset` of the file `path`, which `format` (a `Types.file_code`) holds:
  ## the call reads and widens it in C, and checks the range against the file. `dtype` is the code of `F32` (0) or
  ## `F16` (8): a half buffer is made from the widened values, and a value beyond 65504 is an error.
  compute def load(path: String, offset: Int, shape: List[Int], format: Int, dtype: Int) -> Result[Buffer]
  ## The same from `data`, the bytes of the range that Agentica read: they must be exactly the range.
  compute def from_bytes(data: Bytes, shape: List[Int], format: Int, dtype: Int) -> Result[Buffer]
  ## A buffer of `shape` and `dtype` with every element `value`, filled in C.
  compute def filled(shape: List[Int], value: Float, dtype: Int) -> Result[Buffer]
  ## Host readback: the row-major values as doubles, which the caller tags with the tensor's dtype.
  compute def readback(b: Buffer) -> Result[List[Float]]
  ## The StableHLO module `text` compiled for one CPU device; it must return `outputs` tensors. `donated` are the
  ## inputs the text aliases to an output (`tf.aliasing_output`): a run gives up those buffers, and keeps the others.
  ## A program of the same text compiled lately comes from the cache of the process (32 programs), compiled once.
  compute def compile(text: String, outputs: Int, donated: List[Int]) -> Result[Program]
  ## The arguments of one run of a program, collected by `arg`: it holds each buffer until it is freed, so a buffer
  ## closed meanwhile is still there for the run.
  resource Call
  ## An empty call of `p`.
  def call(p: Program) -> Call
  ## Appends `b` as the next argument of `c`.
  def arg(c: Call, b: Buffer) -> Result[Void]
  ## Runs the program of `c` over its arguments, which must be all of them, in order: a new buffer per output. The
  ## call runs once: after the run it holds no buffer and no program, whatever the outputs keep it for.
  compute def execute(c: Call) -> Result[List[Buffer]]
  ## How many buffers are alive in the process, for a test.
  def buffers() -> Int
  ## How many programs the process has compiled (the misses of the cache), for a test.
  def compiles() -> Int
end
