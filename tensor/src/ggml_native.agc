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
## so (`src/ggml.agc` picks it from the dtype). Every call but `vulkan_devices`
## may allocate or copy device memory, so they run on the compute pool.

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
end
