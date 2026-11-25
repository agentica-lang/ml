## `fit`'s own contract, apart from learning (that is `fit_test`).

import Data from "../data"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Optim from "../optim"
import Train from "../train"

## The mean loss of each epoch of `regression`, computed in doubles from `w = 0.5` with the update `w -= 0.01 * gradient`.
MEANS = [19.8544921875, 2.519814262390139, 0.3197998647853728]

private

## A loss that is not a scalar fails the step, and the error says where.
def test_a_failed_step_names_its_epoch_and_batch() -> Result[Void]
  rows = try Data.dataset(try S.tensor([1.0, 2.0, 3.0, 4.0], [4]), try S.tensor([0.0, 0.0, 0.0, 0.0], [4]))
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.1))
  got = Train.fit(&not_scalar, [try S.tensor([1.0, 1.0], [2])], Data.Loader(dataset: rows, batch_size: 2), sgd, 1)
  try Test.assert(!got.ok?, "a non-scalar loss is an error")
  Test.assert(got.message.start_with?("epoch 1, batch 0: "), "got #{got.message}")
end

## Five rows in batches of two, unshuffled: a batch of two, another of two and a batch of one. A step is compiled
## for each of the two shapes (the loss is traced twice) and reused by the other epochs, and the mean loss of every
## epoch is the closed form: a first batch baked into the trace would give other numbers.
def test_batches_of_two_shapes_follow_the_closed_form() -> Result[Void]
  for backend in [S.REF] + try S.ggml_backends()
    calls: Channel[Int] = Channel.new(16)
    got = try regression(backend, calls)
    try S.expect_close(got, MEANS, 0.000001, 0.00001, "loss history on #{backend}")
    try Test.eq(traced(calls), 2)
  end

  ok()
end

## A shuffled loader on ggml, whose batches `fit` gathers in the step, makes the batches of the reference (the seeded
## order, a last batch of one row) and leaves the dataset's tensors the caller's.
def test_a_shuffled_loader_on_ggml_trains_as_the_reference_and_keeps_its_dataset() -> Result[Void]
  want = try shuffled(S.REF)
  for backend in try S.ggml_backends()
    xs = try S.tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5], Tensor.Dtype.F32, backend)
    got = try shuffled_over(backend, xs)
    try S.expect_close(got, want, 0.000001, 0.00001, "loss history on #{backend}")
    try S.expect_close(try S.floats_of(xs), [1.0, 2.0, 3.0, 4.0, 5.0], 0.0, 0.0, "the dataset on #{backend}")
  end

  ok()
end

def shuffled(backend: Tensor.Backend) -> Result[List[Float]]
  shuffled_over(backend, try S.tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5], Tensor.Dtype.F32, backend))
end

def shuffled_over(backend: Tensor.Backend, xs: Tensor.Tensor) -> Result[List[Float]]
  ys = try S.tensor([2.0, 4.0, 6.0, 8.0, 10.0], [5], Tensor.Dtype.F32, backend)
  w = try S.tensor([0.5], [1], Tensor.Dtype.F32, backend)
  loader = Data.Loader(dataset: try Data.dataset(xs, ys), batch_size: 2, seed: 3)
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.01))
  ok((try Train.fit(&squares, [w], loader, sgd, 3)).history)
end

## A line fitted by SGD to 5 rows (`y = 2 x`), 3 epochs: the mean loss of each epoch.
def regression(backend: Tensor.Backend, calls: Channel[Int]) -> Result[List[Float]]
  xs = try S.tensor([1.0, 2.0, 3.0, 4.0, 5.0], [5], Tensor.Dtype.F32, backend)
  ys = try S.tensor([2.0, 4.0, 6.0, 8.0, 10.0], [5], Tensor.Dtype.F32, backend)
  rows = try Data.dataset(xs, ys)
  sgd = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.01))
  w = try S.tensor([0.5], [1], Tensor.Dtype.F32, backend)
  loader = Data.Loader(dataset: rows, batch_size: 2, shuffle: false)
  fit = try Train.fit(->(ps: List[Tensor.Tensor], b: Data.Batch) { counted(calls, ps, b) }, [w], loader, sgd, 3)
  ok(fit.history)
end

## `squares`, noting the call: a loss is called once when `fit` traces it.
def counted(calls: Channel[Int], ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  try calls.send(1, 0)
  squares(ps, b)
end

## How many calls were noted.
def traced(calls: Channel[Int]) -> Int
  n = 0
  for _ in 0...16
    if calls.recv(0).ok?
      n = n + 1
    end
  end

  n
end

## The mean squared error of `w x` against the targets.
def squares(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  gap = try Tensor.sub(try Tensor.mul(b.inputs, try Tensor.broadcast_to(ps[0], Tensor.shape(b.inputs))), b.targets)
  Tensor.mean(try Tensor.mul(gap, gap))
end

def not_scalar(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor] = Tensor.mul(ps[0], b.inputs)
