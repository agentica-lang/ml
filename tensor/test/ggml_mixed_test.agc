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
  Test.eq(Tensor.backend(got[0]), HOME)
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
