## The native side of the libtorch backend: the thirteen calls of
## `native/torch.c`, a thin C++ layer over libtorch's C++ API (as `tch-rs` is),
## behind the `extern module` block below.
##
## This file is the FFI boundary and nothing else. `src/torch.agc` is the
## backend: it validates shapes, dtypes and devices before a call here
## (`tensor/DESIGN.md` §9) and answers `error` when the module is not loaded.
## Nothing in this file knows what a `Tensor` is.
##
## The module lives in `tensor/` and not in a package of its own because a
## `resource` is nominal to the block that declares it: the `Handle.Native` arm
## of `Tensor` (`src/types.agc`) has to name `TensorHandle` here, and
## `tensor.agc` has to dispatch to these calls (`tensor/DESIGN.md` §7).
## agentica-lang/agentica#1918 is the change that would let the handle be
## opaque and this module move out again.
##
## The `.agc` is the one source of the signatures. native-gen comes from
## `agentica install` (see `../agentica.lock`); from the package folder, after a
## change to the `extern module` block:
##
## ```sh
## agentica deps/native-gen/cli.agc src/native.agc --output native/torch_native.h
## agentica deps/native-gen/cli.agc src/native.agc --output native/torch_native.h --check
## ```
##
## libtorch is a system dependency, not a copy: `tools/build-torch.sh` finds it
## through `TORCH_DIR`, puts its headers in `CPLUS_INCLUDE_PATH` and its
## libraries in `LIBRARY_PATH`, and sets `CC` to the C++ driver the build table
## cannot name. Without it the module is `optional`: the run warns,
## `Tensor.available(Backend.Torch(Device.Cpu))` says no, and the dispatch
## answers `error("backend not available")` (`README.md`, "Local setup").
##
## What crosses: `Int`, `Float`, `Bool`, `List` and `Result`, and the resource.
## A shape, a dtype and a device cross as nothing but what carries them: the
## shape is a `List[Int]`, the dtype is one of the codes of
## `src/torch_dtypes.agc`, and libtorch on CPU has no device to name. Every call
## runs on the compute pool, because every one of them blocks (`compute def`).

extern module "torch"
  ## A tensor on the libtorch CPU backend. The state owns a `torch::Tensor`
  ## made with `new`; `native/torch.c` deletes it in the free hook.
  resource TensorHandle
  ## A tensor of `shape` from row-major `data` in the dtype `dtype` names
  ## (`TorchDtypes.code`). The values cross as doubles; an integer dtype must hold
  ## integral values in range, which the callers of `create` check first.
  compute def create(shape: List[Int], data: List[Float], dtype: Int) -> Result[TensorHandle]
  ## Elementwise `a + b`, two tensors of one shape.
  compute def add(a: TensorHandle, b: TensorHandle) -> Result[TensorHandle]
  ## Elementwise `a * b`, two tensors of one shape.
  compute def mul(a: TensorHandle, b: TensorHandle) -> Result[TensorHandle]
  ## The rank-2 product `[rows, inner] . [inner, cols]`.
  compute def matmul(a: TensorHandle, b: TensorHandle) -> Result[TensorHandle]
  ## `max(0, x)`, elementwise.
  compute def relu(a: TensorHandle) -> Result[TensorHandle]
  ## `exp(x)`, elementwise, finite inputs only: a non-finite value is an
  ## error here rather than a `NaN` in the result.
  compute def exp(a: TensorHandle) -> Result[TensorHandle]
  ## `log(x)`, elementwise, finite positive inputs only.
  compute def log(a: TensorHandle) -> Result[TensorHandle]
  ## The sum over `axes`, dropping them or keeping them as `1` as `keep_dims`
  ## asks. `[]` is no reduction at all, never "every axis".
  compute def sum(a: TensorHandle, axes: List[Int], keep_dims: Bool) -> Result[TensorHandle]
  ## The mean over `axes`, same conventions as `sum`; zero reduced elements
  ## is an error, as in the reference, not a `NaN`.
  compute def mean(a: TensorHandle, axes: List[Int], keep_dims: Bool) -> Result[TensorHandle]
  ## The same values with `shape`; the element counts must agree.
  compute def reshape(a: TensorHandle, shape: List[Int]) -> Result[TensorHandle]
  ## Reverses every axis, as the reference's `transpose` does (rank 2 swaps
  ## rows and columns; below rank 2 nothing changes).
  compute def transpose(a: TensorHandle) -> Result[TensorHandle]
  ## `a` in the dtype `dtype` names: float to int truncates towards zero and
  ## range-checks, anything nonzero becomes `true`, an int width that cannot
  ## hold a value is an error.
  compute def to_dtype(a: TensorHandle, dtype: Int) -> Result[TensorHandle]
  ## Host readback: the row-major values as storage doubles, which the caller
  ## tags with the tensor's dtype. `I64` values past 2^53 lose precision.
  compute def readback(a: TensorHandle) -> Result[List[Float]]
end
