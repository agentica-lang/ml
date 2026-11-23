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

def not_scalar(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor] = Tensor.mul(ps[0], b.inputs)
