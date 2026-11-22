## The native side of the libtorch backend: the thirty-six calls of
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
  ## Elementwise `a - b`, two tensors of one shape.
  compute def sub(a: TensorHandle, b: TensorHandle) -> Result[TensorHandle]
  ## Elementwise `a / b`, two tensors of one shape; a zero divisor is an
  ## error, never an infinity that travels on as a value.
  compute def div(a: TensorHandle, b: TensorHandle) -> Result[TensorHandle]
  ## Elementwise `-x`.
  compute def neg(a: TensorHandle) -> Result[TensorHandle]
  ## The rank-2 product `[rows, inner] . [inner, cols]`.
  compute def matmul(a: TensorHandle, b: TensorHandle) -> Result[TensorHandle]
  ## `max(0, x)`, elementwise.
  compute def relu(a: TensorHandle) -> Result[TensorHandle]
  ## `exp(x)`, elementwise, finite inputs only: a non-finite value is an
  ## error here rather than a `NaN` in the result.
  compute def exp(a: TensorHandle) -> Result[TensorHandle]
  ## `log(x)`, elementwise, finite positive inputs only.
  compute def log(a: TensorHandle) -> Result[TensorHandle]
  ## `sqrt(x)`, elementwise, finite inputs with `x >= 0` only.
  compute def sqrt(a: TensorHandle) -> Result[TensorHandle]
  ## `base ** exponent`, elementwise over two tensors of one shape, finite
  ## inputs with `base > 0` only.
  compute def pow(base: TensorHandle, exponent: TensorHandle) -> Result[TensorHandle]
  ## Sigmoid, elementwise, finite inputs only.
  compute def sigmoid(a: TensorHandle) -> Result[TensorHandle]
  ## Tanh, elementwise, finite inputs only.
  compute def tanh(a: TensorHandle) -> Result[TensorHandle]
  ## `a == b` elementwise as a `Bool` tensor, two tensors of one shape.
  compute def eq(a: TensorHandle, b: TensorHandle) -> Result[TensorHandle]
  ## `a < b` elementwise as a `Bool` tensor, two tensors of one shape.
  compute def lt(a: TensorHandle, b: TensorHandle) -> Result[TensorHandle]
  ## `cond ? a : b`: a `Bool` condition and the two same-shape, same-dtype
  ## tensors it picks between.
  compute def where(cond: TensorHandle, a: TensorHandle, b: TensorHandle) -> Result[TensorHandle]
  ## Elementwise clamp to `[lo, hi]`, `lo <= hi`.
  compute def clamp(a: TensorHandle, lo: Float, hi: Float) -> Result[TensorHandle]
  ## The sum over `axes`, dropping them or keeping them as `1` as `keep_dims`
  ## asks. `[]` is no reduction at all, never "every axis".
  compute def sum(a: TensorHandle, axes: List[Int], keep_dims: Bool) -> Result[TensorHandle]
  ## The mean over `axes`, same conventions as `sum`; zero reduced elements
  ## is an error, as in the reference, not a `NaN`.
  compute def mean(a: TensorHandle, axes: List[Int], keep_dims: Bool) -> Result[TensorHandle]
  ## The largest value over `axes`, same conventions as `sum`; an empty tensor
  ## is an error, as the reference says, never `-inf`.
  compute def max(a: TensorHandle, axes: List[Int], keep_dims: Bool) -> Result[TensorHandle]
  ## The first position of the largest value over `axes`, as `I64`; `[]`
  ## reduces nothing and answers zeros, as the reference does.
  compute def argmax(a: TensorHandle, axes: List[Int], keep_dims: Bool) -> Result[TensorHandle]
  ## The same values with `shape`; the element counts must agree.
  compute def reshape(a: TensorHandle, shape: List[Int]) -> Result[TensorHandle]
  ## Drops the size-1 axes `axes` names, or every size-1 axis for `[]`.
  compute def squeeze(a: TensorHandle, axes: List[Int]) -> Result[TensorHandle]
  ## Inserts a size-1 axis at `axis`.
  compute def unsqueeze(a: TensorHandle, axis: Int) -> Result[TensorHandle]
  ## The `[starts[i], ends[i])` window per axis, row-major in the result.
  compute def slice(a: TensorHandle, starts: List[Int], ends: List[Int]) -> Result[TensorHandle]
  ## `a` and `b` joined along `axis`: one rank, one dtype, every other
  ## dimension equal.
  compute def concat(a: TensorHandle, b: TensorHandle, axis: Int) -> Result[TensorHandle]
  ## `a` widened to `shape`: trailing axes align, size-1 axes widen.
  compute def broadcast_to(a: TensorHandle, shape: List[Int]) -> Result[TensorHandle]
  ## The rows of `a` at the integer tensor `indices`:
  ## `indices.dims ++ a.dims[1:]`.
  compute def gather(a: TensorHandle, indices: TensorHandle) -> Result[TensorHandle]
  ## Reverses every axis, as the reference's `transpose` does (rank 2 swaps
  ## rows and columns; below rank 2 nothing changes).
  compute def transpose(a: TensorHandle) -> Result[TensorHandle]
  ## `a` in the dtype `dtype` names: float to int truncates towards zero and
  ## range-checks, anything nonzero becomes `true`, an int width that cannot
  ## hold a value is an error.
  compute def to_dtype(a: TensorHandle, dtype: Int) -> Result[TensorHandle]
  ## A copy that shares no buffer with `a`.
  compute def copy(a: TensorHandle) -> Result[TensorHandle]
  ## `a` in row-major order, copied when it is not already.
  compute def contiguous(a: TensorHandle) -> Result[TensorHandle]
  ## The cross-correlation of `input` with `weight` over the last axis, moved
  ## by `stride`, with `padding` zeros on each side.
  compute def conv1d(
    input: TensorHandle,
    weight: TensorHandle,
    stride: Int,
    padding: Int,
  ) -> Result[TensorHandle]
  ## The mean over a window of `kernel` positions, moved by `stride`, with
  ## `padding` zeros on each side and a divisor of `kernel`.
  compute def avg_pool1d(input: TensorHandle, kernel: Int, stride: Int, padding: Int) -> Result[TensorHandle]
  ## Host readback: the row-major values as storage doubles, which the caller
  ## tags with the tensor's dtype. `I64` values past 2^53 lose precision.
  compute def readback(a: TensorHandle) -> Result[List[Float]]
end
