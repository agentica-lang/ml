## Lifetimes (`tensor/DESIGN.md` §8): one `Native.arena` per call or step, what
## a caller may rely on afterwards, and the two `Handle` arms living in one
## process. The arena closes the newest resource first when it ends; these
## checks pin down what is observable from Agentica, so a backend that freed too
## eagerly, or a `Plain` tensor that grew a handle, fails here.

import Conform from "torch_support"
import Tensor from "../tensor"

## Two tensors that leave an arena together: a struct field is one of the
## shapes the keep rule reaches.
struct Pair
  left: Tensor.Tensor
  right: Tensor.Tensor
end

private

## One step in one arena: `left` and `right` both come out, `right` built from
## `left`, so both have to survive the block that made them.
def make_pair() -> Result[Pair]
  Native.arena do
    left = try Tensor.ones([1, 2], Tensor.Dtype.F32, Conform.TORCH)
    right = try Tensor.mul(left, left)
    ok(Pair(left: left, right: right))
  end
end

## A three-op step in one arena: the intermediates are not reachable from the
## value, so the arena closes them and what comes out has to be intact.
def step(x: Tensor.Tensor, y: Tensor.Tensor) -> Result[Tensor.Tensor]
  Native.arena do
    doubled = try Tensor.mul(x, y)
    shifted = try Tensor.add(doubled, doubled)
    Tensor.reshape(shifted, [4])
  end
end

## `ones + ones` reshaped, with the sum sent to `seen` on the way.
def doubled_into(seen: Channel[Tensor.Tensor]) -> Result[Tensor.Tensor]
  x = try Tensor.ones([2, 2], Tensor.Dtype.F32, Conform.TORCH)
  doubled = try Tensor.add(x, x)
  _ = seen.send(doubled, 0)
  Tensor.reshape(doubled, [4])
end

def test_a_result_leaves_the_arena_that_made_it() -> Result[Void]
  return Conform.absent() if !Conform.available()

  t = try Tensor.zeros([2, 2], Tensor.Dtype.F32, Conform.TORCH)
  try Test.assert(!Conform.native_closed(t), "a live tensor reads as closed")
  same = try Native.arena do Tensor.add(t, t) end

  try Test.assert(!Conform.native_closed(same), "the result of an arena was closed with it")
  try Test.assert(!Conform.native_closed(t), "an op's arena closed its argument")
  Conform.expect_floats(same, [0.0, 0.0, 0.0, 0.0], "the tensor that left the arena")
end

## The libtorch ops are `AN_FRESH`: the result an arena keeps owns its memory,
## so the intermediates it came from still close with the arena. One goes out
## through `seen`, which the arena's value does not reach, to look at it.
def test_a_kept_result_does_not_keep_its_intermediates() -> Result[Void]
  return Conform.absent() if !Conform.available()

  seen: Channel[Tensor.Tensor] = Channel.new(1)
  out = try Native.arena do doubled_into(seen) end
  middle = try seen.recv(0)
  try Test.assert(Conform.native_closed(middle), "the intermediate was kept with the result")
  try Test.assert(!Conform.native_closed(out), "the result was closed with the arena")
  Conform.expect_floats(out, [2.0, 2.0, 2.0, 2.0], "the result of the arena")
end

def test_a_struct_field_leaves_the_arena_too() -> Result[Void]
  return Conform.absent() if !Conform.available()

  pair = try make_pair()
  try Test.assert(!Conform.native_closed(pair.left), "the left field was closed")
  try Test.assert(!Conform.native_closed(pair.right), "the right field was closed")
  try Test.eq(Tensor.shape(pair.left), [1, 2])
  Conform.expect_floats(pair.right, [1.0, 1.0], "the field that left the arena")
end

def test_a_whole_step_in_one_arena_reads_back_correctly() -> Result[Void]
  return Conform.absent() if !Conform.available()

  x = try Conform.on_torch([1.0, 2.0, 3.0, 4.0], [2, 2])
  y = try Conform.on_torch([0.5, 0.5, 0.5, 0.5], [2, 2])
  out = try step(x, y)

  try Test.eq(Tensor.shape(out), [4])
  try Test.assert(!Conform.native_closed(out), "the step's result was closed")
  Conform.expect_floats(out, [1.0, 2.0, 3.0, 4.0], "the step's result")
end

def test_closing_a_native_handle_is_observable() -> Result[Void]
  return Conform.absent() if !Conform.available()

  t = try Conform.on_torch([1.0], [1])
  try Test.assert(!Conform.native_closed(t), "a fresh handle reads as closed")
  case t.handle
    Tensor.Handle.Native(h) -> Native.close(h)
    Tensor.Handle.Plain(_) -> puts("")
    Tensor.Handle.Traced(_) -> puts("")
    Tensor.Handle.Ggml(_) -> puts("")
    Tensor.Handle.Xla(_) -> puts("")
  end

  Test.assert(Conform.native_closed(t), "a closed handle must read as closed")
end

def test_native_and_plain_tensors_live_in_one_process() -> Result[Void]
  return Conform.absent() if !Conform.available()

  data = [1.0, 2.0, 3.0]
  ours = try Conform.on_torch(data, [3])
  theirs = try Conform.on_ref(data, [3])
  try Test.assert(Tensor.backend(ours) == Conform.TORCH, "the torch tensor names another backend")
  try Test.assert(Tensor.backend(theirs) == Conform.REF, "the reference tensor names another backend")

  # The same call, dispatched by value, twice in the same process.
  try Conform.agrees(try Tensor.exp(ours), try Tensor.exp(theirs), "exp on both backends")

  # A native step leaves the plain tensor alone, and its storage still reads.
  try Conform.agrees(try Tensor.add(ours, ours), try Tensor.add(theirs, theirs), "2 + 2 on both sides")

  # Mixing the two backends is an error, never a silent move between them.
  Test.error?(Tensor.add(ours, theirs))
end
