## `fit`'s own contract, apart from learning (that is `fit_test`).

import Data from "../data"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Optim from "../optim"
import Train from "../train"

private

## A loss that is not a scalar fails the step, and the error says where.
def test_a_failed_step_names_its_epoch_and_batch() -> Result[Void]
  rows = try Data.dataset(try S.tensor([1.0, 2.0, 3.0, 4.0], [4]), try S.tensor([0.0, 0.0, 0.0, 0.0], [4]))
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1))
  got = Train.fit(&not_scalar, [try S.tensor([1.0, 1.0], [2])], Data.Loader(dataset: rows, batch_size: 2), sgd, 1)
  try Test.assert(!got.ok?, "a non-scalar loss is an error")
  Test.assert(got.message.start_with?("epoch 1, batch 0: "), "got #{got.message}")
end

## Five rows in batches of two: a batch of two and a batch of one, so a step is compiled for each shape and reused by
## the other epochs, and the loss curve is the reference's.
def test_batches_of_two_shapes_follow_the_reference() -> Result[Void]
  want = try regression(S.REF)
  for backend in try S.ggml_backends()
    got = try regression(backend)
    try S.expect_close(got, want, 0.000001, 0.00001, "loss history on #{backend}")
  end

  ok()
end

## A line fitted by SGD to 5 rows, 3 epochs: the mean loss of each epoch.
def regression(backend: Tensor.Backend) -> Result[List[Float]]
  xs = try S.tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5], Tensor.Dtype.F32, backend)
  ys = try S.tensor([2.0, 4.0, 6.0, 8.0, 10.0], [5], Tensor.Dtype.F32, backend)
  rows = try Data.dataset(xs, ys)
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.01))
  w = try S.tensor([0.5], [1], Tensor.Dtype.F32, backend)
  fit = try Train.fit(&squares, [w], Data.Loader(dataset: rows, batch_size: 2), sgd, 3)
  ok(fit.history)
end

## The mean squared error of `w x` against the targets.
def squares(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  gap = try Tensor.sub(try Tensor.mul(b.inputs, try Tensor.broadcast_to(ps[0], Tensor.shape(b.inputs))), b.targets)
  Tensor.mean(try Tensor.mul(gap, gap))
end

def not_scalar(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor] = Tensor.mul(ps[0], b.inputs)
