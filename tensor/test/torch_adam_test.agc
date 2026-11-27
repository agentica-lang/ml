## `Tensor.adam_step` on libtorch is one native kernel: one pass over each tensor, the three outputs in one allocation
## (the step is not in place: the weight, the gradient and the moments it was given stay what they were). It follows
## the arithmetic of `Tensor.adam_composed` in `F32`, but not bit for bit: the compiler may fuse a multiply and an add
## and the vector division and root differ in the last place. The two agree to 1e-6 of the value (F32 carries 1e-7),
## which is what the checks below allow. Run from `tensor/`.

import Conform from "torch_support"
import Tensor from "../tensor"
import Torch from "../src/torch"

WIDTH = 5000
TENSOR_BYTES = 20000.0

## Tensors beyond the four it reads that one step may hold at its peak: its output, a stack of three.
LIVE_LIMIT = 3.5

## Adam with no decay at its 3rd step; AdamW at its 40th with a rate the schedule has moved and a decay of 0.01; and a
## weight a decay mask excludes (decay 0 in the AdamW row). `[lr, beta1, beta2, eps, decay, 1 / (1 - beta1^t), 1 / (1 - beta2^t)]`.
ADAM = [0.001, 0.9, 0.999, 0.00000001, 0.0, 3.690036900369005, 333.66688900003714]
ADAMW = [0.0005, 0.9, 0.999, 0.00000001, 0.01, 1.0213, 1.0408]
MASKED = [0.0005, 0.9, 0.999, 0.00000001, 0.0, 1.0213, 1.0408]

private

## A deterministic spread of values in `-scale...scale` that no two tensors share: `seed` shifts the phase.
def spread(seed: Int, scale: Float) -> List[Float]
  (0...WIDTH).map { |i| scale * Math.sin((i * 7 + seed * 131).to_f * 0.37) }
end

def operands(backend: Tensor.Backend, hyper: List[Float]) -> Result[List[Tensor.Tensor]]
  make = ->(values: List[Float], shape: List[Int]) { Tensor.floats(values, shape, backend) }
  ok([
    try make.call(spread(1, 1.0), [50, 100]),
    try make.call(spread(2, 0.1), [50, 100]),
    try make.call(spread(3, 0.01), [50, 100]),
    try make.call(spread(4, 0.0001).map { |x: Float| x.abs }, [50, 100]),
    try make.call(hyper, [7]),
  ])
end

def stepped(args: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  Tensor.adam_step(args[0], args[1], args[2], args[3], args[4])
end

def composed(args: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  Tensor.adam_composed(args[0], args[1], args[2], args[3], args[4])
end

def same(got: Tensor.Tensor, want: Tensor.Tensor, what: String) -> Result[Void]
  ours = try Conform.floats_of(got)
  theirs = try Conform.floats_of(want)
  try Test.eq(ours.size, theirs.size)
  for i in 0...theirs.size
    gap = (ours[i] - theirs[i]).abs
    if gap > 0.000001 * theirs[i].abs + 0.000000001
      return error("#{what}: value #{i} is #{ours[i]}, the composition says #{theirs[i]}")
    end
  end

  ok()
end

def test_the_kernel_agrees_with_the_composition() -> Result[Void]
  return Conform.absent() if !Conform.available()

  for hyper in [ADAM, ADAMW, MASKED]
    args = try operands(Conform.TORCH, hyper)
    got = try stepped(args)
    want = try composed(args)
    for i in 0...3
      try same(got[i], want[i], "output #{i} of #{hyper}")
    end
  end

  ok()
end

def test_the_kernel_holds_only_its_output() -> Result[Void]
  return Conform.absent() if !Conform.available()

  args = try operands(Conform.TORCH, ADAMW)
  before = try Torch.memory(true)
  got = try stepped(args)
  after = try Torch.memory(false)
  try Test.assert(
    after[1] - before[0] <= LIVE_LIMIT * TENSOR_BYTES,
    "the step held #{(after[1] - before[0]) / TENSOR_BYTES} tensors besides its inputs",
  )

  try Test.eq(got.size, 3)
  ok()
end

def test_the_arguments_are_not_changed() -> Result[Void]
  return Conform.absent() if !Conform.available()

  args = try operands(Conform.TORCH, ADAMW)
  again = try operands(Conform.TORCH, ADAMW)
  _ = try stepped(args)
  for i in 0...4
    try same(args[i], again[i], "argument #{i}")
  end

  ok()
end

def test_a_gradient_that_is_not_finite_is_an_error() -> Result[Void]
  return Conform.absent() if !Conform.available()

  args = try operands(Conform.TORCH, ADAMW)
  big = try Tensor.full([50, 100], Tensor.Values.Float(1000.0), Tensor.Dtype.F32, Conform.TORCH)
  bad = try Tensor.exp(try Tensor.mul(args[1], big))
  try Test.assert(!stepped([args[0], bad, args[2], args[3], args[4]]).ok?, "an infinite gradient must fail")
  ok()
end

def test_the_step_chains_through_the_views_it_made() -> Result[Void]
  return Conform.absent() if !Conform.available()

  args = try operands(Conform.TORCH, ADAMW)
  first = try stepped(args)
  fed = [first[0], args[1], first[1], first[2], args[4]]
  got = try stepped(fed)
  want = try composed(fed)
  for i in 0...3
    try same(got[i], want[i], "output #{i} of the second step")
  end

  ok()
end

def test_a_double_weight_and_an_empty_one_step() -> Result[Void]
  return Conform.absent() if !Conform.available()

  for dtype in [Tensor.Dtype.F64, Tensor.Dtype.F16]
    ones = try Tensor.full([2, 3], Tensor.Values.Float(0.5), dtype, Conform.TORCH)
    values = ADAMW.map { |x: Float| Tensor.Values.Float(x) }
    hyper = try Tensor.from_list(values, [7], dtype, Conform.TORCH)
    got = try Tensor.adam_step(ones, ones, ones, ones, hyper)
    try Test.eq(Tensor.dtype(got[0]), dtype)
    want = try Tensor.adam_composed(ones, ones, ones, ones, hyper)
    try same(got[0], want[0], "a #{dtype} weight")
  end

  empty = try Tensor.zeros([0], Tensor.Dtype.F32, Conform.TORCH)
  hyper = try Tensor.floats(ADAMW, [7], Conform.TORCH)
  got = try Tensor.adam_step(empty, empty, empty, empty, hyper)
  try Test.eq(Tensor.shape(got[0]), [0])
  Test.eq(Tensor.shape(got[2]), [0])
end
