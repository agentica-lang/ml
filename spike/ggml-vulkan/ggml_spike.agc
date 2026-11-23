## The FFI boundary of the ggml-vulkan spike: a thin C layer over the pinned
## ggml submodule at `third_party/ggml`, behind
## `extern module "ggmlspike"`.
##
## This file is the one source of the signatures. From this folder, after a
## change to the block below:
##
## ```sh
## agentica deps/native-gen/cli.agc ggml_spike.agc --output native/ggmlspike_native.h
## agentica deps/native-gen/cli.agc ggml_spike.agc --output native/ggmlspike_native.h --check
## ```
##
## The module is the boundary and nothing else: one ggml call per function,
## the checks ggml would otherwise do silently wrong (ggml `add` broadcasts),
## and the readback. There is no `Backend` enum and no dispatch here — that is
## `tensor/`'s job (see `tensor/DESIGN.md` §7), and this spike exists to
## measure what such a backend would cost.
##
## The backend a tensor lives on is a `Backend` resource, never a guess:
## `backend_vulkan` fails unless `ggml_backend_vk_init` returns a backend that
## `ggml_backend_is_vk` accepts and whose device is not a CPU. A run that
## silently falls back to CPU is the failure this module exists to catch, so
## the type of the device is asserted in C and printed by `describe`.

extern module "ggmlspike"
  ## An open backend: ggml's CPU backend with a thread count, or the iGPU
  ## through ggml's Vulkan backend. The state owns the `ggml_backend_t`.
  resource Backend
  ## A F32 tensor whose state owns its ggml context and backend buffer. The
  ## `Backend` it was made on must outlive it (the spike keeps backends open
  ## for the whole run).
  resource Tensor

  ## Every device ggml currently registers, one `"name — description (type)"`
  ## per entry, in ggml's order. Reading it opens nothing.
  compute def devices() -> Result[List[String]]
  ## ggml's CPU backend, with `threads` threads for both the graph and batch
  ## paths.
  compute def backend_cpu(threads: Int) -> Result[Backend]
  ## The iGPU through ggml's Vulkan backend (`ggml_backend_vk_init(0)`),
  ## failing unless the backend really is Vulkan and the device is not a CPU.
  compute def backend_vulkan() -> Result[Backend]
  ## What the backend really is, read back from ggml: name, device
  ## description and device type. The type is the assertion of this spike.
  compute def describe(backend: Backend) -> Result[String]

  ## A `shape` tensor of F32 filled with the deterministic pattern of
  ## `tensor/tools/bench.agc`, so every backend gets the same values.
  compute def create(backend: Backend, shape: List[Int]) -> Result[Tensor]
  ## Elementwise `a + b`, both of one shape; error when they differ.
  compute def add(a: Tensor, b: Tensor) -> Result[Tensor]
  ## `max(0, x)`, elementwise.
  compute def relu(a: Tensor) -> Result[Tensor]
  ## ggml's `mul_mat(a, b)`, which is `a · bᵀ`: for `a` of `[k, m]` and `b` of
  ## `[k, n]` (row-major rows), the result is `[m, n]`.
  compute def mul_mat(a: Tensor, b: Tensor) -> Result[Tensor]
  ## One dispatch round trip with no op in it: an empty graph computed on the
  ## backend. This is the fixed cost every op above pays before it does work.
  compute def empty_graph(backend: Backend) -> Result[Void]

  ## The sum of every element, read back across the boundary as one `Float`:
  ## a full device-to-host copy plus a host sum, with no `List` decode.
  compute def checksum(t: Tensor) -> Result[Float]
  ## Every element, read back as a `List[Float]` (the slow path M0 measured at
  ## ~20 ns per value; `checksum` is the cost of the copy alone).
  compute def readback(t: Tensor) -> Result[List[Float]]
  ## The bytes the backend buffer holding `t` actually has, as ggml reports
  ## them. The only memory number this spike can report honestly.
  compute def allocated(t: Tensor) -> Result[Int]
  ## The device's free and total memory in bytes, as ggml reports them: on a
  ## Vega iGPU both are host memory.
  compute def memory(backend: Backend) -> Result[List[Int]]
end
