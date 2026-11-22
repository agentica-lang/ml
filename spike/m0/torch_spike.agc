## The native side of the M0 spike: a thin C++ layer over libtorch's C++ API,
## behind `extern module "torchspike"`.
##
## The `.agc` is the one source of the signatures. native-gen comes from
## `agentica install` (see `agentica.lock`); from this folder, after a change
## to the `extern module` block:
##
## ```sh
## agentica deps/native-gen/cli.agc torch_spike.agc --output native/torchspike_native.h
## agentica deps/native-gen/cli.agc torch_spike.agc --output native/torchspike_native.h --check
## ```
##
## libtorch is a system dependency, not a copy: `tools/build-torch.sh` finds it
## through `TORCH_DIR`, puts its headers in `CPLUS_INCLUDE_PATH` and its
## libraries in `LIBRARY_PATH`, and sets `CC` to the C++ driver the build
## table cannot name. Without it the module is `optional`: the run warns and
## the reference side runs alone (`README.md`, "The native side").
##
## The module exposes what the demo needs and nothing else: `create`, `add`,
## `matmul`, `relu` and `sum`. `sum` answers a bare `Float`, not a tensor: a
## rank-0 result would need a readback call of its own, and the spike spends
## its five functions on the ops (see the README for what that costs).

import TensorRef from "tensor_ref"

extern module "torchspike"
  ## A tensor on the libtorch CPU backend. The native state owns the
  ## `torch::Tensor`; nothing here can name a shape, a dtype or a device.
  resource TorchTensor
  ## A tensor of `shape` from row-major `data`. The values cross as doubles
  ## and are stored as `float32`, the dtype libtorch computes in.
  compute def create(shape: List[Int], data: List[Float]) -> Result[TorchTensor]
  ## Elementwise `a + b`, both of one shape.
  compute def add(a: TorchTensor, b: TorchTensor) -> Result[TorchTensor]
  ## The rank-2 product of `[rows, inner]` and `[inner, cols]`.
  compute def matmul(a: TorchTensor, b: TorchTensor) -> Result[TorchTensor]
  ## `max(0, x)`, elementwise.
  compute def relu(a: TorchTensor) -> Result[TorchTensor]
  ## The sum of every element, read back across the boundary as one `Float`.
  compute def sum(a: TorchTensor) -> Result[Float]
end

## f(w) = sum(relu(matmul(x, w))) on libtorch, for a reference `x` and `w`
## whose values cross as data. One `Native.arena` per call: the four native
## tensors die at the end of the block and only the `Float` leaves.
def loss(x: TensorRef.Tensor, w: TensorRef.Tensor) -> Result[Float]
  Native.arena do
    tx = try create(x.shape, x.data)
    tw = try create(w.shape, w.data)
    y = try matmul(tx, tw)
    r = try relu(y)
    sum(r)
  end
end
