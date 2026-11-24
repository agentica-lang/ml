## Random short traces over a weight that ggml holds as F32, F16 or Q8_0 (`Tensor.from_file(..., dtype:)`): each trace is a
## few ops (transpose, reshape, slice, embedding, linear, matmul, add, mul, the norms) chained on an input or on the weight
## itself, and the same calls run on the reference with the weight as F32. A trace either answers within the dtype's
## bound of the reference, or ggml refuses it with an error; it never crashes (a heap overflow kills the test process) and
## never answers something else. The weight is read as `[3, 32]` (3 blocks of 32 for Q8_0).
##
## The bound is derived, per output element, by carrying an error bound beside the value through the trace on the reference:
## structural ops (transpose, reshape, slice, embedding) move it, `add` keeps it, `mul` scales it by the absolute operand,
## a product by the weight sums it through the absolute weights and adds the rounding of that product: F32's order of
## summation (1e-6 of the sum of absolute terms), F16's rounding of every activation (2^-11), Q8_0's rounding of a row of 32
## activations to a block (`amax * (1/254 + 2^-11)`, `ggml_q8_test.agc`). A norm gets a first-order bound from the error of
## its input (`1.5` times it), and where the error of a row is not small against its spread (the norm could amplify it past
## a first-order bound) the bound is dropped for that row, so those outputs are checked for shape, dtype and finiteness only.
## The answer of ggml must be finite and within `1.5` times the bound of the reference's.
##
## What ggml may refuse is told per weight: with F32 it refuses nothing the reference runs, and with F16 and Q8_0 the
## fixed-seed sample of `CASES` traces runs at least `floor` of them on ggml. The ops, the bound and the shared code are in
## `prop_packed_support.agc`; every pair of ops, with the number ggml runs pinned, is in `prop_packed_pairs_*_test.agc`
## (a file each: `agentica test` gives a file its step budget).
## Without the ggml module the properties pass by asserting nothing, unless `TENSOR_REQUIRE_GGML` is set (CI sets it).
##
## A failure prints the shrunk trace and its seed: `Prop.Config(seed: some(N))` replays it.

import Packed from "prop_packed_support"
import Prop from "pkg/prop/prop"
import Support from "prop_support"

private

def dyadic() -> Prop.Gen[Float] = Prop.map(Prop.int(-64, 64)) do |n: Int| n.to_f / 8.0 end

def step() -> Prop.Gen[Packed.Step]
  Prop.map(Prop.pair(Prop.int(0, 63), Prop.int(0, 7))) do |p: Prop.Pair[Int, Int]|
    Packed.Step(op: p.first, arg: p.second)
  end
end

## 2 to 4 steps, from the input or from the weight.
def trails() -> Prop.Gen[Packed.Trail]
  inputs = Prop.pair(Prop.non_empty_list(dyadic(), 8), Prop.non_empty_list(Prop.int(0, 2), 3))
  shape = Prop.pair(Prop.int(0, 1), Prop.pair(Prop.pair(step(), step()), Prop.list(step(), 2)))
  Prop.map(
    Prop.pair(inputs, shape),
  ) do |p: Prop.Pair[Prop.Pair[List[Float], List[Int]], Prop.Pair[Int, Prop.Pair[Prop.Pair[Packed.Step, Packed.Step], List[Packed.Step]]]]|
    Packed.Trail(
      x: Support.cycle(p.first.first, 2 * Packed.COLS),
      ids: (0...3).map { |i: Int| p.first.second[i % p.first.second.size] },
      start: p.second.first,
      steps: [p.second.second.first.first, p.second.second.first.second] + p.second.second.second,
    )
  end
end

## `Packed.CASES` random traces, shrunk if one fails, and then the same number from a fixed seed, of which ggml runs at least
## `kind.floor`.
def holds_for(kind: Packed.Kind) -> Result[Void]
  try Packed.with_file("random-#{kind.name}") do |rig|
    for device in try Packed.devices()
      try Prop.check(trails(), Prop.Config(runs: Packed.CASES)) do |trail: Packed.Trail|
        _ = try Packed.compare(rig, kind, device, trail)
        ok()
      end

      ran = 0
      for trail in Prop.sample(trails(), Packed.SEED, Packed.CASES)
        if try Packed.compare(rig, kind, device, trail)
          ran += 1
        end
      end

      try Test.assert(ran >= kind.floor, "ggml ran #{ran} of #{Packed.CASES} traces with a #{kind.name} weight")
    end

    ok()
  end

  ok()
end

def test_traces_over_an_f32_weight_match_the_reference() -> Result[Void] = holds_for(Packed.kinds()[0])

def test_traces_over_an_f16_weight_match_the_reference_or_are_refused() -> Result[Void] = holds_for(Packed.kinds()[1])

def test_traces_over_a_q8_0_weight_match_the_reference_or_are_refused() -> Result[Void] = holds_for(Packed.kinds()[2])
