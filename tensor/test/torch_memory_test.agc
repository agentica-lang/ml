## What a traced run on libtorch keeps. `Trace` runs a graph node by node, and a node's value is dead once its last
## user ran: a long chain whose backward needs none of its intermediates must peak at a few tensors, not at one per
## node (`src/torch.agc` `memory` counts the bytes of the live libtorch storages). Run from `tensor/`.

import Conform from "torch_support"
import Tensor from "../tensor"
import Torch from "../src/torch"
import Trace from "../trace"

## The elements of each tensor of the chains, and how many links they have.
WIDTH = 1000
LINKS = 40

## The bytes of one tensor of the chains.
TENSOR_BYTES = 4000.0

## Tensors of the chain a run may hold at its peak: the input, the link being made and the one before it, the
## constants, and the outputs. The chain would hold `LINKS` of them if it kept every node.
LIVE_LIMIT = 10.0

## The same for a chain of Adam steps, whose own step holds its coefficients and the moments it makes.
ADAM_LIVE_LIMIT = 20.0

private

## What a run answered, the most bytes the live tensors held while it ran, and the bytes they hold when it is over,
## both beyond what they held before it.
struct Measured
  outs: List[Tensor.Tensor]
  peak: Float
  left: Float
end

def ramp() -> List[Float] = (0...WIDTH).to_a.map { |i| 0.001 * i.to_f }

## `x` through `LINKS` links of `(x + shift) * scale`, whose derivative is the constant `scale`: nothing of the forward
## pass is needed by the backward.
def chain(x: Tensor.Tensor, shift: Tensor.Tensor, scale: Tensor.Tensor) -> Result[Tensor.Tensor]
  y = x
  for _ in 0...LINKS
    y = try Tensor.mul(try Tensor.add(y, Trace.constant(shift)), Trace.constant(scale))
  end

  ok(y)
end

def measured(body: Fn[Result[List[Tensor.Tensor]]]) -> Result[Measured]
  before = try Torch.memory(true)
  outs = try body.call()
  after = try Torch.memory(false)
  ok(Measured(outs: outs, peak: after[1] - before[0], left: after[0] - before[0]))
end

## `got` (libtorch, `F32`) is `want` (the reference, in doubles) within 1e-5 of it: forty links round forty times.
def near(got: Tensor.Tensor, want: Tensor.Tensor, what: String) -> Result[Void]
  ours = try Conform.floats_of(got)
  theirs = try Conform.floats_of(want)
  try Test.eq(ours.size, theirs.size)
  for i in 0...theirs.size
    gap = (ours[i] - theirs[i]).abs
    return error("#{what}: value #{i} is #{ours[i]}, the reference says #{theirs[i]}") if gap > 0.00001 * theirs[i].abs
  end

  ok()
end

def operands(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  make = ->(values: List[Float]) { Tensor.floats(values, [WIDTH], backend) }
  ok([try make.call(ramp()), try make.call(List.repeat(0.5, WIDTH)), try make.call(List.repeat(0.99, WIDTH))])
end

def test_a_forward_run_holds_a_few_tensors_not_one_per_node() -> Result[Void]
  return Conform.absent() if !Conform.available()

  t = try operands(Conform.TORCH)
  run = try measured do
    out = try Trace.eval([t[0]]) do |xs| chain(xs[0], t[1], t[2]) end
    ok([out])
  end

  try Test.assert(run.peak <= LIVE_LIMIT * TENSOR_BYTES, "the run held #{run.peak / TENSOR_BYTES} tensors")
  try Test.assert(run.left <= TENSOR_BYTES, "the run left #{run.left / TENSOR_BYTES} tensors, one is its answer")
  r = try operands(Conform.REF)
  want = try Trace.eval([r[0]]) do |xs| chain(xs[0], r[1], r[2]) end
  near(run.outs[0], want, "the chain")
end

def test_a_gradient_holds_a_few_tensors_not_one_per_node() -> Result[Void]
  return Conform.absent() if !Conform.available()

  t = try operands(Conform.TORCH)
  run = try measured do Trace.grad([t[0]]) do |xs| Tensor.sum(try chain(xs[0], t[1], t[2])) end end
  try Test.assert(run.peak <= LIVE_LIMIT * TENSOR_BYTES, "the run held #{run.peak / TENSOR_BYTES} tensors")
  r = try operands(Conform.REF)
  want = try Trace.grad([r[0]]) do |xs| Tensor.sum(try chain(xs[0], r[1], r[2])) end
  near(run.outs[0], want[0], "the gradient of the chain")
end

## The Adam step on libtorch is made of many ops, each a tensor of the shape of the parameter.
def adam_links(xs: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  p = xs[0]
  m = xs[2]
  v = xs[3]
  for _ in 0...LINKS
    stepped = try Tensor.adam_step(p, xs[1], m, v, xs[4])
    p = stepped[0]
    m = stepped[1]
    v = stepped[2]
  end

  ok(p)
end

def adam_operands(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  make = ->(values: List[Float], shape: List[Int]) { Tensor.floats(values, shape, backend) }
  hyper = [0.001, 0.9, 0.999, 0.00000001, 0.0, 1.0, 1.0]
  ok([
    try make.call(ramp(), [WIDTH]),
    try make.call(List.repeat(0.1, WIDTH), [WIDTH]),
    try make.call(List.repeat(0.0, WIDTH), [WIDTH]),
    try make.call(List.repeat(0.0, WIDTH), [WIDTH]),
    try make.call(hyper, [7]),
  ])
end

def test_a_chain_of_adam_steps_frees_what_each_step_made() -> Result[Void]
  return Conform.absent() if !Conform.available()

  t = try adam_operands(Conform.TORCH)
  run = try measured do
    out = try Trace.eval(t) do |xs| adam_links(xs) end
    ok([out])
  end

  try Test.assert(run.peak <= ADAM_LIVE_LIMIT * TENSOR_BYTES, "the run held #{run.peak / TENSOR_BYTES} tensors")
  moved = try Conform.floats_of(run.outs[0])
  try Test.assert(
    moved[WIDTH - 1] < ramp()[WIDTH - 1] - 0.03,
    "forty steps of lr 0.001 moved the last value to #{moved[WIDTH - 1]}",
  )

  ok()
end
