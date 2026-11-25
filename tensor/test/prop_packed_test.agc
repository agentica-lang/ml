## Random short traces (transpose, reshape, slice, embedding, linear, matmul, add, mul, the norms) over a weight that
## ggml holds as F32, F16 or Q8_0 (`Tensor.from_file(..., dtype:)`), against the same calls on the reference with the
## weight as F32. A trace answers within the dtype's bound of the reference (derived per element along the trace, in
## `prop_packed_support.agc`) or ggml refuses it with an error; it never crashes and never answers something else.
## What ggml may refuse is told per weight, and every pair of ops is in `prop_packed_pairs_*_test.agc`. A failure prints
## the shrunk trace and its seed: `Prop.Config(seed: some(N))` replays it.

import Packed from "prop_packed_support"
import Prop from "pkg/prop/prop"
import Support from "prop_support"

private

type Inputs = Prop.Pair[List[Float], List[Int]]

type Steps = Prop.Pair[Prop.Pair[Packed.Step, Packed.Step], List[Packed.Step]]

type Drawn = Prop.Pair[Inputs, Prop.Pair[Int, Steps]]

def dyadic() -> Prop.Gen[Float] = Prop.map(Prop.int(-64, 64)) do |n: Int| n.to_f / 8.0 end

def step() -> Prop.Gen[Packed.Step]
  Prop.map(Prop.pair(Prop.int(0, 63), Prop.int(0, 7))) do |p: Prop.Pair[Int, Int]|
    Packed.Step(op: p.first, arg: p.second)
  end
end

## An input and ids, where the trace starts (the input or the weight) and 2 to 4 steps.
def trails() -> Prop.Gen[Packed.Trail]
  inputs = Prop.pair(Prop.non_empty_list(dyadic(), 8), Prop.non_empty_list(Prop.int(0, 2), 3))
  first_two = Prop.pair(step(), step())
  steps = Prop.pair(first_two, Prop.list(step(), 2))
  Prop.map(Prop.pair(inputs, Prop.pair(Prop.int(0, 1), steps)), &trail_of)
end

def trail_of(drawn: Drawn) -> Packed.Trail
  values = drawn.first.first
  ids = drawn.first.second
  start = drawn.second.first
  steps = drawn.second.second
  Packed.Trail(
    x: Support.cycle(values, 2 * Packed.COLS),
    ids: (0...3).map { |i: Int| ids[i % ids.size] },
    start: start,
    steps: [steps.first.first, steps.first.second] + steps.second,
  )
end

## `Support.RUNS` random traces, shrunk if one fails, and then the same number from a fixed seed: ggml runs exactly
## `kind.sampled` of them (all, with F32), and at least `kind.tight[2]` of their outputs are checked with a real bound.
def holds_for(kind: Packed.Kind) -> Result[Void]
  Packed.with_file("random-#{kind.name}") do |rig|
    for device in try Packed.devices()
      try Prop.check(trails(), Prop.Config(runs: Support.RUNS)) do |trail: Packed.Trail|
        _ = try Packed.compare(rig, kind, device, trail)
        ok()
      end

      ran = 0
      tight = 0
      loose = 0
      for trail in Prop.sample(trails(), Packed.SEED, Support.RUNS)
        outcome = try Packed.compare(rig, kind, device, trail)
        tight += outcome.tight
        loose += outcome.loose
        if outcome.ran
          ran += 1
        end
      end

      want = kind.sampled
      if kind.name == "F32"
        want = Support.RUNS
      end

      try Test.assert(ran == want, "ggml ran #{ran} of #{Support.RUNS} traces with a #{kind.name} weight, not #{want}")
      try Test.assert(
        tight >= kind.tight[2],
        "#{tight} outputs checked with a real bound and #{loose} loosely, at least #{kind.tight[2]} tight",
      )
    end

    ok()
  end
end

def test_traces_over_an_f32_weight_match_the_reference() -> Result[Void] = holds_for(Packed.kinds()[0])

def test_traces_over_an_f16_weight_match_the_reference_or_are_refused() -> Result[Void] = holds_for(Packed.kinds()[1])

def test_traces_over_a_q8_0_weight_match_the_reference_or_are_refused() -> Result[Void] = holds_for(Packed.kinds()[2])
