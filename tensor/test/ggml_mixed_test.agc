## Layers across a Vulkan device and the CPU: a graph over tensors on both. A leaf on `Ggml(Cpu)` under a graph on a
## Vulkan device makes the graph run through a scheduler over the two backends, and the node that reads a CPU weight or
## cache runs on the CPU. Its values are those of the same graph with every tensor on the device, an output of a CPU node
## is a CPU tensor, and a cache on the CPU is written in place. Without a Vulkan device these checks pass by asserting
## nothing.

import GgmlRaw from "../src/ggml_native"
import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

DEVICE = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
HOME = Tensor.Backend.Ggml(Tensor.Device.Cpu)
WIDTH = 128

## Ggml and a Vulkan device are there.
def mixed_present() -> Result[Bool]
  return ok(false) if !try Support.ggml_present()
  ok(Tensor.available(DEVICE))
end

## What a test asserts where there is no Vulkan device.
def absent() -> Result[Void]
  Test.assert(!Tensor.available(DEVICE), "no Vulkan device: the mixed placement was not run")
end

private

def floats(xs: List[Float], shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, backend)
end

def counting(n: Int) -> List[Float] = (0...n).map { |i| ((i * 7) % 11 - 5).to_f / 8.0 }

def specs_of(ts: List[Tensor.Tensor]) -> List[Trace.Spec] = ts.map { |t| Trace.spec_of(t) }

## `relu(x w^T)` and `x * 2`: the first reads the weight, the second only the input.
def model(xs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  y = try Tensor.linear(xs[0], xs[1], none, true)
  ok([try Tensor.relu(y), try Tensor.add(xs[0], xs[0])])
end

def inputs(weight_on: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([try floats(counting(12), [3, 4], DEVICE), try floats(counting(32), [8, 4], weight_on)])
end

## `rows` rows of input and a weight on `weight_on`.
def fn_inputs(rows: Int, weight_on: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([try floats(counting(rows * 4), [rows, 4], DEVICE), try floats(counting(32), [8, 4], weight_on)])
end

def values(ts: List[Tensor.Tensor]) -> Result[List[List[Float]]]
  out: List[List[Float]] = []
  for t in ts
    out = out.push(try Support.floats_of(t))
  end

  ok(out)
end

def test_a_cpu_weight_gives_the_values_of_the_device_weight() -> Result[Void]
  return absent() if !try mixed_present()

  args = try inputs(DEVICE)
  want = try values(try Trace.eval_many(args) do |xs| model(xs) end)
  args = try inputs(HOME)
  got = try Trace.eval_many(args) do |xs| model(xs) end
  try Test.eq(try values(got), want)
  try Test.eq(Tensor.backend(got[0]), HOME)
  Test.eq(Tensor.backend(got[1]), DEVICE)
end

## One graph structure, run with its weight on the device, then on the CPU, then on the device: each run places itself
## by the weight it is given.
def test_the_same_graph_is_placed_again_when_a_leaf_changes_sides() -> Result[Void]
  return absent() if !try mixed_present()

  want = try values(try Trace.eval_many(try inputs(DEVICE)) do |xs| model(xs) end)
  for side in [HOME, DEVICE, HOME, HOME, DEVICE]
    got = try Trace.eval_many(try inputs(side)) do |xs| model(xs) end
    try Test.eq(try values(got), want)
    try Test.eq(Tensor.backend(got[0]), side)
  end

  ok()
end

## `into` takes a tensor on the side of the node that writes the output, and refuses one on the other.
def test_an_output_given_lives_on_the_side_of_its_node() -> Result[Void]
  return absent() if !try mixed_present()

  args = try inputs(HOME)
  compiled = try Trace.compile(specs_of(args)) do |xs| model(xs) end
  on_cpu = try floats(counting(24), [3, 8], HOME)
  on_device = try floats(counting(12), [3, 4], DEVICE)
  out = try Trace.run(compiled, args, [some(on_cpu), some(on_device)])
  want = try values(try Trace.eval_many(try inputs(DEVICE)) do |xs| model(xs) end)
  try Test.eq(try values(out), want)
  try Test.error?(Trace.run(compiled, args, [some(on_device), none]))
  Test.error?(Trace.run(compiled, args, [none, some(on_cpu)]))
end

## A cache on the CPU is written in place by rows that come from the device, and the tensor given is the result.
def test_a_cpu_cache_is_written_in_place_by_rows_from_the_device() -> Result[Void]
  return absent() if !try mixed_present()

  args = [
    try floats(counting(24), [2, 4, 3], HOME),
    try floats([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], [2, 1, 3], DEVICE),
    try Tensor.from_list([Tensor.Values.Int(2)], [1], Tensor.Dtype.I32, DEVICE),
  ]

  compiled = try Trace.compile(specs_of(args)) do |xs| ok([try Tensor.set_rows(xs[0], xs[1], xs[2])]) end

  out = try Trace.run(compiled, args)
  got = try Support.floats_of(args[0])
  try Test.eq(got.drop(6).take(3), [-1.0, -2.0, -3.0])
  try Test.eq(got.drop(18).take(3), [-4.0, -5.0, -6.0])
  try Test.eq(got.take(6), counting(24).take(6))
  try Test.eq(Tensor.backend(out[0]), HOME)
  Test.eq(try Support.floats_of(out[0]), got)
end

## A long input against a small weight on the CPU: the activations are many times the bytes of the weight, and the node
## still runs where the weight is, so the weight is never copied to the device.
def test_a_long_input_does_not_pull_a_cpu_weight_to_the_device() -> Result[Void]
  return absent() if !try mixed_present()

  longer = [try floats(counting(400), [100, 4], DEVICE), try floats(counting(32), [8, 4], HOME)]
  want = try values(try Trace.eval_many([longer[0], try floats(counting(32), [8, 4], DEVICE)]) do |xs| model(xs) end)
  got = try Trace.eval_many(longer) do |xs| model(xs) end
  try Test.eq(try values(got), want)
  try Test.eq(Tensor.backend(got[0]), HOME)
  Test.eq(Tensor.backend(got[1]), DEVICE)
end

## Two mixed graphs of different sizes take turns on the one scheduler of the device: each run is planned for its own
## graph, so neither reads the memory of the other.
def test_two_mixed_graphs_take_turns_on_the_scheduler() -> Result[Void]
  return absent() if !try mixed_present()

  want_small = try values(try Trace.eval_many(try fn_inputs(3, DEVICE)) do |xs| model(xs) end)
  want_large = try values(try Trace.eval_many(try fn_inputs(100, DEVICE)) do |xs| model(xs) end)
  for rows in [3, 100, 3, 100, 100, 3]
    got = try Trace.eval_many(try fn_inputs(rows, HOME)) do |xs| model(xs) end
    want = want_large
    if rows == 3
      want = want_small
    end

    try Test.eq(try values(got), want)
  end

  ok()
end

## The residual stream: a row looked up in a large table on the device, then added to the product with a weight on the
## CPU. The sum belongs to the CPU, so the layers after it do not cross the bus at every step.
def residual(xs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  e = try Tensor.gather(xs[0], xs[1])
  y = try Tensor.linear(e, xs[2], none, true)
  ok([try Tensor.add(e, y)])
end

def residual_inputs(weight_on: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ids = try Tensor.from_list([Tensor.Values.Int(7)], [1], Tensor.Dtype.I32, DEVICE)
  ok([try floats(counting(300 * 64), [300, 64], DEVICE), ids, try floats(counting(64 * 64), [64, 64], weight_on)])
end

def test_a_sum_of_a_device_row_and_a_cpu_product_runs_on_the_cpu() -> Result[Void]
  return absent() if !try mixed_present()

  want = try values(try Trace.eval_many(try residual_inputs(DEVICE)) do |xs| residual(xs) end)
  got = try Trace.eval_many(try residual_inputs(HOME)) do |xs| residual(xs) end
  try Test.eq(try values(got), want)
  try Test.eq(Tensor.backend(got[0]), HOME)
  swapped = try Trace.eval_many(try residual_inputs(HOME)) do |xs|
    e = try Tensor.gather(xs[0], xs[1])
    ok([try Tensor.add(try Tensor.linear(e, xs[2], none, true), e)])
  end

  try Test.eq(try values(swapped), want)
  Test.eq(Tensor.backend(swapped[0]), HOME)
end

## A model of `blocks` residual blocks and a head: inputs are the ids, the table, the head, then per block a norm weight
## and a matrix. The first `on_device` blocks have their matrix on the device, the others on the CPU.
def stack(xs: List[Tensor.Tensor], blocks: Int) -> Result[List[Tensor.Tensor]]
  h = try Tensor.gather(xs[1], xs[0])
  for k in 0...blocks
    normed = try Tensor.mul(try Tensor.rms_norm(h, 0.00001), xs[3 + 2 * k])
    h = try Tensor.add(h, try Tensor.linear(normed, xs[4 + 2 * k], none, true))
  end

  ok([try Tensor.linear(h, xs[2], none, true)])
end

def stack_inputs(blocks: Int, on_device: Int) -> Result[List[Tensor.Tensor]]
  ids = try Tensor.from_list([Tensor.Values.Int(7)], [1], Tensor.Dtype.I32, DEVICE)
  args = [
    ids,
    try floats(counting(300 * WIDTH), [300, WIDTH], DEVICE),
    try floats(counting(300 * WIDTH), [300, WIDTH], DEVICE),
  ]

  for k in 0...blocks
    side = HOME
    if k < on_device
      side = DEVICE
    end

    args = args.push(try floats(counting(WIDTH), [1, WIDTH], DEVICE))
    args = args.push(try floats(counting(WIDTH * WIDTH), [WIDTH, WIDTH], side))
  end

  ok(args)
end

## The splits of the scheduler in one run of the stack, and its values against the same stack with all on the device.
def stack_splits(blocks: Int, on_device: Int) -> Result[Int]
  want = try values(try Trace.eval_many(try stack_inputs(blocks, blocks)) do |xs| stack(xs, blocks) end)
  got = try Trace.eval_many(try stack_inputs(blocks, on_device)) do |xs| stack(xs, blocks) end
  have = try values(got)
  for t in 0...want.size
    for k in 0...want[t].size
      gap = have[t][k] - want[t][k]
      if gap < 0.0
        gap = 0.0 - gap
      end

      try Test.assert(
        gap <= 0.001 * (1.0 + want[t][k].abs),
        "value #{k} is #{have[t][k]}, the device gives #{want[t][k]}",
      )
    end
  end

  ok(GgmlRaw.mixed_splits())
end

## Layers on the CPU do not make the scheduler change backend at every layer: the splits are a few, whatever the number
## of blocks, with all the blocks on the CPU (n of 0) and with the first ones on the device (n of 14 in the model).
def test_the_splits_do_not_grow_with_the_blocks_on_the_cpu() -> Result[Void]
  return absent() if !try mixed_present()

  four = try stack_splits(4, 0)
  eight = try stack_splits(6, 0)
  try Test.assert(four <= 4, "4 blocks on the CPU made #{four} splits")
  try Test.eq(eight, four)
  two_four = try stack_splits(4, 2)
  two_eight = try stack_splits(6, 2)
  try Test.assert(two_four <= 4, "2 blocks on the device and 2 on the CPU made #{two_four} splits")
  Test.eq(two_eight, two_four)
end

def message(r: Result[List[Tensor.Tensor]]) -> String
  case r
    ok(_) -> ""
    error(e) -> e.message
  end
end

## A value that is not finite and an index outside the table are refused as they are on the device, whichever side makes
## them.
def test_the_checks_of_a_mixed_graph_refuse_what_a_device_graph_refuses() -> Result[Void]
  return absent() if !try mixed_present()

  for side in [DEVICE, HOME]
    logs = Trace.eval_many(try fn_inputs(3, side)) do |xs|
      ok([try Tensor.log(try Tensor.linear(xs[0], xs[1], none, true))])
    end

    try Test.assert(message(logs) != "", "the log of a product with negative values was answered")
    if side == HOME
      same = Trace.eval_many(try fn_inputs(3, DEVICE)) do |xs|
        ok([try Tensor.log(try Tensor.linear(xs[0], xs[1], none, true))])
      end

      try Test.eq(message(logs), message(same))
    end
  end

  bad = [
    try floats(counting(300 * 64), [300, 64], HOME),
    try Tensor.from_list([Tensor.Values.Int(999)], [1], Tensor.Dtype.I32, DEVICE),
  ]

  from_cpu = Trace.eval_many(bad) do |xs| ok([try Tensor.gather(xs[0], xs[1])]) end
  from_device = Trace.eval_many([try floats(counting(300 * 64), [300, 64], DEVICE), bad[1]]) do |xs|
    ok([try Tensor.gather(xs[0], xs[1])])
  end

  try Test.assert(message(from_cpu) != "", "an index outside the table was answered")
  Test.eq(message(from_cpu), message(from_device))
end

## A graph with every leaf on the device never goes through the scheduler; one with a leaf on the CPU does.
def test_only_a_graph_with_a_cpu_leaf_goes_through_the_scheduler() -> Result[Void]
  return absent() if !try mixed_present()

  before = GgmlRaw.mixed_runs()
  try Trace.eval_many(try inputs(DEVICE)) do |xs| model(xs) end
  try Test.eq(GgmlRaw.mixed_runs(), before)
  try Trace.eval_many(try inputs(HOME)) do |xs| model(xs) end
  Test.eq(GgmlRaw.mixed_runs(), before + 1)
end
