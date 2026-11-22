## Lifetimes (`tensor/DESIGN.md` §8): one `Native.arena` per call or step, and
## what a caller may rely on afterwards. The arena itself closes the newest
## resource first when it ends; these checks pin down the two outcomes a caller
## can see from Agentica, so a backend that freed too eagerly fails here.

import Support from "support"
import Torch from "../torch"

## Two tensors that leave an arena together: a struct field is one of the
## shapes the keep rule reaches.
struct Pair
  left: Torch.Tensor
  right: Torch.Tensor
end

private

## One step in one arena: `left` and `right` both come out, `right` built from
## `left`, so both have to survive the block that made them.
def make_pair() -> Result[Pair]
  Native.arena do
    left = try Torch.ones(Support.CPU, [1, 2], Torch.Dtype.F32)
    right = try Torch.mul(Support.CPU, left, left)
    ok(Pair(left: left, right: right))
  end
end

## A three-op step in one arena: the intermediates are not reachable from the
## value, so the arena closes them and what comes out has to be intact.
def step(x: Torch.Tensor, y: Torch.Tensor) -> Result[Torch.Tensor]
  Native.arena do
    doubled = try Torch.mul(Support.CPU, x, y)
    shifted = try Torch.add(Support.CPU, doubled, doubled)
    Torch.reshape(Support.CPU, shifted, [4])
  end
end

def test_a_result_leaves_the_arena_that_made_it() -> Result[Void]
  return Support.absent() if !Support.available()

  t = try Torch.zeros(Support.CPU, [2, 2], Torch.Dtype.F32)
  try Test.assert(!Torch.closed(t), "a live tensor reads as closed")
  same = try Native.arena do Torch.add(Support.CPU, t, t) end

  try Test.assert(!Torch.closed(same), "the result of an arena was closed with it")
  try Test.assert(!Torch.closed(t), "an op's arena closed its argument")
  Support.expect_floats(same, [0.0, 0.0, 0.0, 0.0], "the tensor that left the arena")
end

def test_a_struct_field_leaves_the_arena_too() -> Result[Void]
  return Support.absent() if !Support.available()

  pair = try make_pair()
  try Test.assert(!Torch.closed(pair.left), "the left field was closed")
  try Test.assert(!Torch.closed(pair.right), "the right field was closed")
  try Test.eq(Torch.shape(pair.left), [1, 2])
  Support.expect_floats(pair.right, [1.0, 1.0], "the field that left the arena")
end

def test_a_whole_step_in_one_arena_reads_back_correctly() -> Result[Void]
  return Support.absent() if !Support.available()

  x = try Support.torch_f32([1.0, 2.0, 3.0, 4.0], [2, 2])
  y = try Support.torch_f32([0.5, 0.5, 0.5, 0.5], [2, 2])
  out = try step(x, y)

  try Test.eq(Torch.shape(out), [4])
  try Test.assert(!Torch.closed(out), "the step's result was closed")
  Support.expect_floats(out, [1.0, 2.0, 3.0, 4.0], "the step's result")
end
