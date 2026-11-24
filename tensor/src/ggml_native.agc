## The native side of the ggml backend: the calls of `native/ggml.c`, a thin C
## layer over the pinned ggml submodule (`third_party/ggml`), behind the
## `extern module` block below.
##
## This file is the FFI boundary and nothing else, as `native.agc` is for
## libtorch. `src/ggml.agc` is the backend: it validates shapes, dtypes and
## devices before a call here and answers `error` when the module is not
## loaded. It lives in `tensor/` for the reason the torch module does: the
## `Handle.Ggml` arm of `Tensor` has to name `TensorHandle`
## (`tensor/DESIGN.md` §1, §7; agentica-lang/agentica#1918 is the way out).
##
## The `.agc` is the one source of the signatures; after a change to the block,
## from the package folder:
##
## ```sh
## agentica deps/native-gen/cli.agc src/ggml_native.agc --output native/ggml_native.h
## ```
##
## A device crosses as an `Int`: `-1` is ggml's CPU backend and `i >= 0` the
## Vulkan device `i`. A tensor is stored as F32, or as I32 when `int32` says
## so (`src/ggml.agc` picks it from the dtype). The calls may allocate, copy or
## compute on a device, so they run on the compute pool.

extern module "ggml"
  ## One tensor in a ggml backend buffer of its device: the state owns a
  ## ggml context holding the tensor's metadata and the buffer holding its
  ## data, both freed by the free hook. Nothing else points at them.
  resource TensorHandle
  ## How many Vulkan devices this build of ggml sees, not counting a CPU
  ## device (a software rasterizer): 0 when ggml was built without its Vulkan
  ## backend (`tools/build-engine.sh cpu`). It opens no device.
  compute def vulkan_devices() -> Int
  ## A tensor of the row-major `shape` (rank 4 at most) on `device`, holding
  ## `data`, one value per element.
  compute def upload(device: Int, shape: List[Int], data: List[Float], int32: Bool) -> Result[TensorHandle]
  ## An F32 tensor (F16 when `half`) of the row-major `shape` on `device` from
  ## the little-endian floats of `format` (a code of `Types.file_code`) that
  ## start at byte `offset` of the file `path`. C reads the file itself, so no
  ## value crosses as a list; a range past the end of the file is an error. A
  ## half tensor converts the values a chunk at a time and refuses one that F16
  ## does not hold (beyond 65504).
  compute def load(
    device: Int,
    path: String,
    offset: Int,
    shape: List[Int],
    format: Int,
    half: Bool,
  ) -> Result[TensorHandle]
  ## The same from `data`, the bytes of the range that Agentica read
  ## (`TENSOR_LOAD=agentica`): the call reads them in place and widens them into
  ## the buffer. `data` must be exactly the range, or the call is an error.
  compute def from_bytes(
    device: Int,
    data: Bytes,
    shape: List[Int],
    format: Int,
    half: Bool,
  ) -> Result[TensorHandle]
  ## A tensor of `shape` on `device` with every element `value`, filled in C,
  ## so a large fill never crosses as a list.
  compute def filled(device: Int, shape: List[Int], value: Float, int32: Bool) -> Result[TensorHandle]
  ## Host readback: the row-major values as doubles, which the caller tags
  ## with the tensor's dtype.
  compute def readback(t: TensorHandle) -> Result[List[Float]]
  ## A copy of `t` on `device`, of the same shape and storage.
  compute def transfer(t: TensorHandle, device: Int) -> Result[TensorHandle]

  ## One compiled graph of the structural cache, held by one evaluation. The
  ## graph itself lives in the C cache for the process; the lease marks it
  ## in use, and its free hook gives it back (or drops it if it never
  ## compiled). So two tasks never share one graph's leaves at once.
  resource Lease
  ## A compiled graph cached under `key` on `device` that no one holds, or a
  ## new empty one with room for `capacity` nodes, to build with `node` and
  ## `compile`. A new graph with `keep` false takes no cache slot: it is
  ## dropped when its lease is freed.
  compute def acquire(device: Int, key: String, capacity: Int, keep: Bool) -> Result[Lease]
  ## True when the leased graph is compiled: a cache hit.
  compute def ready(lease: Lease) -> Bool
  ## Appends one node and answers its index: `op` is a code of
  ## `native/ggml_graph.c`, `args` earlier indices, `shape` the row-major
  ## shape the trace recorded, `ints` the op's integer parameters.
  compute def node(lease: Lease, op: Int, args: List[Int], shape: List[Int], ints: List[Int]) -> Result[Int]
  ## Builds the graph of `outputs`, allocates its fills and checks every
  ## node against the backend. A failed compile leaves the graph unusable.
  compute def compile(lease: Lease, outputs: List[Int]) -> Result[Void]
  ## Makes `t` the leaf `slot` for the next run (leaves count in the order
  ## they were sent): an F32 tensor is read in place, another one is
  ## converted into the leaf's own memory now. The graph holds `t`'s memory
  ## until the lease ends, so a `t` that is closed meanwhile does not leave
  ## the run without its data.
  compute def bind(lease: Lease, slot: Int, t: TensorHandle) -> Result[Void]
  ## A fresh tensor that the next run writes output `output` into, stored as
  ## I32 when `int32` says so.
  compute def output(lease: Lease, output: Int, int32: Bool) -> Result[TensorHandle]
  ## Makes `t`, a tensor that exists, the tensor that the next run writes
  ## output `output` into, in place of a fresh one: an F32 tensor of the same
  ## element count is written directly. `t` must not be a leaf of this run.
  compute def output_into(lease: Lease, output: Int, t: TensorHandle) -> Result[Void]
  ## Keeps the compiled graph of `lease` in the structural cache for as long
  ## as the `Pin` lives, however many other graphs the cache takes in: the
  ## graph is exempt from eviction. The pin ends when its owner closes it or
  ## ends; the graph then stays in the cache as any other. The lease must hold
  ## a compiled graph that has a slot of the cache.
  resource Pin
  compute def pin(lease: Lease) -> Result[Pin]
  ## Sets the fills (one value each, in the order they were sent) and runs
  ## the graph into the tensors `output` made; every leaf and output must be
  ## set since the last run. An error when an output is not finite.
  compute def run(lease: Lease, fills: List[Float]) -> Result[Void]
  ## Writes the `count` values of `src_format` (a code of `Types.file_code`) that start at byte `src_offset` of the
  ## file `src` into the existing file `dst` at byte `dst_offset`, as `dst_format`. The same format is a byte copy;
  ## F32 to F16 rounds to nearest even and refuses a value beyond 65504, and to BF16 it rounds to nearest even. C
  ## does it in chunks on several threads, so no value crosses as a list. Past the end of `dst` the file grows.
  compute def convert_file(
    src: String,
    src_offset: Int,
    src_format: Int,
    count: Int,
    dst: String,
    dst_offset: Int,
    dst_format: Int,
  ) -> Result[Void]
  ## How many graphs this process has compiled: the cache's misses.
  compute def compiles() -> Int
  ## Chooses fp16 (`true`) or F32 (`false`, the default) `mul_mat` for every
  ## Vulkan device this process opens. ggml reads it once, when it first
  ## enumerates its devices, so a different choice after any other call of
  ## this module (a CPU one included) is an error.
  compute def vulkan_fp16(enabled: Bool) -> Result[Void]
end
