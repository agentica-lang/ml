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
  ## `compile`.
  compute def acquire(device: Int, key: String, capacity: Int) -> Result[Lease]
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
  ## they were sent): an F32 tensor is read in place, so the caller keeps
  ## `t` alive until `run` returns; another one is converted into the
  ## leaf's own memory now.
  compute def bind(lease: Lease, slot: Int, t: TensorHandle) -> Result[Void]
  ## A fresh tensor that the next run writes output `output` into, stored as
  ## I32 when `int32` says so.
  compute def output(lease: Lease, output: Int, int32: Bool) -> Result[TensorHandle]
  ## Sets the fills (one value each, in the order they were sent) and runs
  ## the graph into the tensors `output` made; every leaf and output must be
  ## set since the last run. An error when an output is not finite.
  compute def run(lease: Lease, fills: List[Float]) -> Result[Void]
  ## How many graphs this process has compiled: the cache's misses.
  compute def compiles() -> Int
  ## Chooses fp16 (`true`) or F32 (`false`, the default) `mul_mat` for every
  ## Vulkan device this process opens. ggml reads it once, when it first
  ## enumerates its devices, so a different choice after any other call of
  ## this module is an error.
  compute def prefer_fp16(enabled: Bool) -> Result[Void]
end
