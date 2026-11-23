## Shared helpers of the `train/` tests (not a test file itself).

import Loss from "../loss"
import Nn from "pkg/nn/nn"
import Optim from "../optim"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)

## The XOR run of `fit_test` and `torch_fit_test`: a 2-8-1 relu MLP from
## `nn/`, made from seed 1, trained by Adam (`lr` 0.2) on the four XOR rows
## for 20 full-batch epochs with binary cross-entropy. `XOR_HISTORY` is its
## loss per epoch on the reference: the golden the reference reproduces and
## libtorch follows. Twenty epochs is what a test file's step budget allows
## (each costs about 75,000 VM steps on the reference).
XOR_EPOCHS = 20
XOR_HISTORY = [
  0.668955092647425,
  0.6489128214903377,
  0.5862082263575338,
  0.525739794815681,
  0.4731294383849625,
  0.42645841227892767,
  0.34001898287146354,
  0.2814594298613612,
  0.22671094650797502,
  0.18799976020761108,
  0.1469886765196035,
  0.11305588210587045,
  0.08508408201158438,
  0.06389667138419038,
  0.05135472401537906,
  0.04180874658056082,
  0.03518219015236515,
  0.02932017078944909,
  0.024213729307330817,
  0.020488959894194907,
]

def tensor(
  data: List[Float],
  shape: List[Int],
  dtype: Tensor.Dtype = Tensor.Dtype.F64,
  backend: Tensor.Backend = REF,
) -> Result[Tensor.Tensor]
  Tensor.from_list(data.map { |x| Tensor.Values.Float(x) }, shape, dtype, backend)
end

def floats_of(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end

## `got` matches `want` within `abs + rel * |want|`, element by element.
def expect_close(got: List[Float], want: List[Float], abs: Float, rel: Float, what: String) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    gap = (got[i] - want[i]).abs
    return error("#{what}: value #{i} is #{got[i]}, expected #{want[i]}") if gap > abs + rel * want[i].abs
  end

  ok()
end

def xor(backend: Tensor.Backend, on_epoch: Optional[Fn[Train.Epoch, Void]] = none) -> Result[Train.Fit]
  up = try Nn.init_dense(2, 8, 1, backend)
  down = try Nn.init_dense(8, 1, 2, backend)
  params = [up.w, up.b.unwrap!, down.w, down.b.unwrap!]
  adam = Optim.Optimizer.Adam(Optim.Adam(lr: 0.2))
  Train.fit(&xor_loss, params, [try xor_batch(backend)], adam, XOR_EPOCHS, on_epoch)
end

def xor_batch(backend: Tensor.Backend) -> Result[Train.Batch]
  ok(
    Train.Batch(
      inputs: try tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2], Tensor.Dtype.F32, backend),
      targets: try tensor([0.0, 1.0, 1.0, 0.0], [4, 1], Tensor.Dtype.F32, backend),
    ),
  )
end

## The logits of the MLP whose parameters are `ps`.
def xor_model(ps: List[Tensor.Tensor], x: Tensor.Tensor) -> Result[Tensor.Tensor]
  up = Nn.Dense(w: ps[0], b: some(ps[1]))
  down = Nn.Dense(w: ps[2], b: some(ps[3]))
  Nn.mlp(Nn.Mlp(up: up, down: down, activation: Nn.Activation.Relu), x)
end

def xor_loss(ps: List[Tensor.Tensor], b: Train.Batch) -> Result[Tensor.Tensor]
  Loss.binary_cross_entropy(try xor_model(ps, b.inputs), b.targets)
end
