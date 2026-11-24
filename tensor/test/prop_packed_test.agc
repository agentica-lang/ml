## Random short traces over a weight that ggml holds as F32, F16 or Q8_0 (`Tensor.from_file(..., dtype:)`): each trace is a
## few ops (transpose, reshape, slice, embedding, linear, matmul, add, mul, the norms) chained on an input or on the weight
## itself, and the same calls run on the reference with the weight as F32. A trace either answers within the dtype's
## bound of the reference, or ggml refuses it with an error; it never crashes (a heap overflow kills the test process) and
## never answers something else. The weight is read as `[3, 32]` (3 blocks of 32 for Q8_0).
##
## The bound is a fraction of the sum of the absolute terms: the trace runs once more on the reference with the absolute
## values of the inputs, and an output may differ by `tolerance` times that, plus `tolerance` times the largest output.
## F32 differs by the order of the sum; F16 by the rounding of the activations to F16 (2^-11); Q8_0 by the rounding of the
## activations to Q8_0 blocks (1/254 and the F16 scale, `ggml_q8_test.agc`), a few times over for a chain of products.
## Without the ggml module the properties pass by asserting nothing, unless `TENSOR_REQUIRE_GGML` is set (CI sets it).
##
## A failure prints the shrunk trace and its seed: `Prop.Config(seed: some(N))` replays it.

import Prop from "pkg/prop/prop"
import Support from "prop_support"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Device.Cpu
REF = Tensor.Backend.Reference
F32 = Tensor.Dtype.F32
Q8_0 = Tensor.FloatFormat.Q8_0

ROWS = 3
COLS = 32
## The F32 weights follow 3 padding bytes; the Q8_0 blocks of the same values follow them.
Q8_OFFSET = 3 + ROWS * COLS * 4
## Cases per property: the fast tier.
CASES = 60
## Ops a pair is made of: `op` indexes the ops that fit, modulo their number, and no shape has more.
OPS = 10
## How many of the sampled traces ggml must have run, so a refusal of everything does not pass.
RAN_AT_LEAST = 8

## The binary16 bits of each block's scale: 1/8, 1/16, 1/4.
SCALES = [0x3000, 0x2C00, 0x3400]
VALUES = [0.125, 0.0625, 0.25]

private

## One op of a trace: `op` picks among the ops that fit the shape in hand, `arg` picks a variant of it.
struct Step
  op: Int
  arg: Int
end

## An input, the ids of an embedding, where the trace starts (0 the input, 1 the weight itself) and its steps.
struct Trail
  x: List[Float]
  ids: List[Int]
  start: Int
  steps: List[Step]
end

struct Env
  x: Tensor.Tensor
  w: Tensor.Tensor
  ids: Tensor.Tensor
end

## The file, and the weights and their absolute values, built once.
struct Rig
  path: String
  w: List[Float]
  mag: List[Float]
end

## What ggml keeps the weight as: the dtype and where its bytes are.
struct Kind
  name: String
  dtype: Tensor.Dtype
  format: Tensor.FloatFormat
  offset: Int
  tolerance: Float
end

def devices() -> Result[List[Tensor.Device]]
  return ok([CPU]) if Tensor.available(Tensor.Backend.Ggml(CPU))
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok([]) if !flag.ok?
  return ok([]) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end

def f32_bytes(x: Float) -> List[Int]
  bits = Float.to_bits32(x)
  [bits % 256, (bits / 256) % 256, (bits / 65536) % 256, bits / 16777216]
end

def byte_of(b: Int, j: Int) -> Int = (j * 7 + b * 13) % 17 - 8

## The weights, row-major: the scale of the block times its byte, small multiples that F16 and Q8_0 hold exactly.
def weights() -> List[Float]
  out: List[Float] = []
  for b in 0...ROWS
    for j in 0...COLS
      out = out.push(VALUES[b] * byte_of(b, j).to_f)
    end
  end

  out
end

## The file: 3 padding bytes, the F32 weights, then the same weights in 3 blocks of Q8_0.
def file_bytes() -> Result[Bytes]
  out = [9, 9, 9] + weights().flat_map { |x: Float| f32_bytes(x) }
  for b in 0...ROWS
    out = out.push(SCALES[b] % 256)
    out = out.push(SCALES[b] / 256)
    for j in 0...COLS
      out = out.push((byte_of(b, j) + 256) % 256)
    end
  end

  Bytes.from_list(out)
end

def with_file(body: Fn[Rig, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-packed-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  path = "#{dir}/weights.bin"
  try File.open(path, "write", 5000) do |f|
    try File.write_bytes(f, try file_bytes(), 5000)
    ok()
  end

  w = weights()
  result = body.call(Rig(path: path, w: w, mag: abs_all(w)))
  _ = Path.remove(dir, 5000)
  result
end

def kinds() -> List[Kind]
  [
    Kind(name: "F32", dtype: F32, format: Tensor.FloatFormat.F32, offset: 3, tolerance: 0.00001),
    Kind(name: "F16", dtype: Tensor.Dtype.F16, format: Tensor.FloatFormat.F32, offset: 3, tolerance: 0.004),
    Kind(name: "Q8_0", dtype: Tensor.Dtype.Q8_0, format: Q8_0, offset: Q8_OFFSET, tolerance: 0.05),
  ]
end

## The ops that fit a tensor of this shape, by name.
def options(shape: List[Int]) -> List[String]
  if shape.size == 1
    return ["unsqueeze", "norm", "head", "fold"].select { |n: String| n != "fold" or shape[0] % COLS == 0 }
  end

  a = shape[0]
  b = shape[1]
  out = ["transpose", "flatten", "rows", "cols", "norm"]
  if a >= 3
    out = out.push("embedding")
  end

  if b == COLS
    out = out.push("linear_out_in")
  end

  if b == COLS
    out = out.push("matmul_wt")
  end

  if b == ROWS
    out = out.push("matmul_w")
  end

  if b == ROWS
    out = out.push("linear_in_out")
  end

  if a == COLS
    out = out.push("w_matmul")
  end

  if shape == [2, COLS]
    out = out.push("add_x").push("mul_x")
  end

  if shape == [ROWS, COLS]
    out = out.push("add_w").push("mul_w")
  end

  out
end

## Half of `n`, at least 1, from the start or the end as `arg` says.
def window(n: Int, arg: Int) -> List[Int]
  keep = [n / 2, 1].max.unwrap!
  return [0, keep] if arg % 2 == 0

  [n - keep, n]
end

def apply(step: Step, cur: Tensor.Tensor, env: Env) -> Result[Tensor.Tensor]
  shape = Tensor.shape(cur)
  names = options(shape)
  name = names[step.op % names.size]
  case name
    "transpose" -> Tensor.transpose(cur)
    "flatten" -> Tensor.reshape(cur, [shape[0] * shape[1]])
    "unsqueeze" -> Tensor.reshape(cur, [1, shape[0]])
    "fold" -> Tensor.reshape(cur, [shape[0] / COLS, COLS])
    "head" -> Tensor.slice(cur, [0], [[shape[0] / 2, 1].max.unwrap!])
    "rows" ->
      r = window(shape[0], step.arg)
      Tensor.slice(cur, [r[0], 0], [r[1], shape[1]])
    "cols" ->
      c = window(shape[1], step.arg)
      Tensor.slice(cur, [0, c[0]], [shape[0], c[1]])
    "norm" -> normed(cur, step.arg)
    "embedding" -> Tensor.embedding(env.ids, cur)
    "linear_out_in" -> Tensor.linear(cur, env.w, none, true)
    "linear_in_out" -> Tensor.linear(cur, env.w, none, false)
    "matmul_wt" -> Tensor.matmul(cur, try Tensor.transpose(env.w))
    "matmul_w" -> Tensor.matmul(cur, env.w)
    "w_matmul" -> Tensor.matmul(env.w, cur)
    "add_x" -> Tensor.add(cur, env.x)
    "mul_x" -> Tensor.mul(cur, env.x)
    "add_w" -> Tensor.add(cur, env.w)
    "mul_w" -> Tensor.mul(cur, env.w)
    _ -> error("no op named #{name}")
  end
end

## `rms_norm` or `layer_norm`, as `arg` says.
def normed(t: Tensor.Tensor, arg: Int) -> Result[Tensor.Tensor]
  return Tensor.rms_norm(t, 0.00001) if arg % 2 == 0

  Tensor.layer_norm(t, 0.00001)
end

def run(trail: Trail, env: Env) -> Result[Tensor.Tensor]
  cur = env.x
  if trail.start == 1
    cur = env.w
  end

  for step in trail.steps
    cur = try apply(step, cur, env)
  end

  Tensor.to_dtype(cur, F32)
end

## The ops of a trace by name, as the shapes of the reference's run pick them.
def names_of(trail: Trail, env: Env) -> Result[String]
  cur = env.x
  if trail.start == 1
    cur = env.w
  end

  names: List[String] = []
  for step in trail.steps
    ops = options(Tensor.shape(cur))
    names = names.push(ops[step.op % ops.size])
    cur = try apply(step, cur, env)
  end

  ok(names.join(", "))
end

def reference_env(x: List[Float], ids: List[Int], w: List[Float]) -> Result[Env]
  ok(
    Env(
      x: try Tensor.floats(x, [2, COLS], REF),
      w: try Tensor.floats(w, [ROWS, COLS], REF),
      ids: try Tensor.from_list(ids.map { |i: Int| Tensor.Values.Int(i) }, [3], Tensor.Dtype.I32, REF),
    ),
  )
end

def ggml_env(path: String, kind: Kind, trail: Trail, device: Tensor.Device) -> Result[Env]
  backend = Tensor.Backend.Ggml(device)
  ok(
    Env(
      x: try Tensor.floats(trail.x, [2, COLS], backend),
      w: try Tensor.from_file(path, kind.offset, [ROWS, COLS], kind.format, backend, kind.dtype),
      ids: try Tensor.from_list(trail.ids.map { |i: Int| Tensor.Values.Int(i) }, [3], Tensor.Dtype.I32, backend),
    ),
  )
end

def abs_all(xs: List[Float]) -> List[Float] = xs.map { |v: Float| v.abs }

## The trace on ggml against the reference: ok(true) if it ran and matched, ok(false) if ggml refused it (or the reference
## did, and there is nothing to compare), an error if the answers differ.
def compare(rig: Rig, kind: Kind, device: Tensor.Device, trail: Trail) -> Result[Bool]
  ref_env = try reference_env(trail.x, trail.ids, rig.w)
  case run(trail, ref_env)
    error(_) -> ok(false)
    ok(ref) -> against(rig, kind, device, trail, ref, ref_env)
  end
end

def against(
  rig: Rig,
  kind: Kind,
  device: Tensor.Device,
  trail: Trail,
  ref: Tensor.Tensor,
  ref_env: Env,
) -> Result[Bool]
  want = try Support.numbers(ref)
  env = try ggml_env(rig.path, kind, trail, device)
  ran = Trace.eval_many([env.x, env.w, env.ids], keep: false) do |a|
    out = try run(trail, Env(x: a[0], w: a[1], ids: a[2]))
    ok([out])
  end

  case ran
    error(_) -> ok(false)
    ok(outs) ->
      got = outs[0]
      try Test.eq(Tensor.dtype(got), F32)
      if Tensor.shape(got) != Tensor.shape(ref)
        return error("shape #{Tensor.shape(got)}, expected #{Tensor.shape(ref)}")
      end

      values = try Support.numbers(got)
      mag = try run(trail, try reference_env(abs_all(trail.x), trail.ids, rig.mag))
      scale = try Support.numbers(mag)
      peak = want.reduce(0.0) do |acc: Float, v: Float| [acc, v.abs].max.unwrap! end
      for i in 0...want.size
        bound = kind.tolerance * (scale[i].abs + peak) + 0.000001
        if (values[i] - want[i]).abs > bound
          return error(
            "output #{i} is #{values[i]}, the reference's #{want[i]} (bound #{bound}) with a #{kind.name} weight: #{try names_of(trail, ref_env)}",
          )
        end
      end

      ok(true)
  end
end

def dyadic() -> Prop.Gen[Float] = Prop.map(Prop.int(-64, 64)) do |n: Int| n.to_f / 8.0 end

def step() -> Prop.Gen[Step]
  Prop.map(Prop.pair(Prop.int(0, 63), Prop.int(0, 7))) do |p: Prop.Pair[Int, Int]| Step(op: p.first, arg: p.second) end
end

## 2 to 4 steps, from the input or from the weight.
def trails() -> Prop.Gen[Trail]
  inputs = Prop.pair(Prop.non_empty_list(dyadic(), 8), Prop.non_empty_list(Prop.int(0, 2), 3))
  shape = Prop.pair(Prop.int(0, 1), Prop.pair(Prop.pair(step(), step()), Prop.list(step(), 2)))
  Prop.map(
    Prop.pair(inputs, shape),
  ) do |p: Prop.Pair[Prop.Pair[List[Float], List[Int]], Prop.Pair[Int, Prop.Pair[Prop.Pair[Step, Step], List[Step]]]]|
    Trail(
      x: Support.cycle(p.first.first, 2 * COLS),
      ids: (0...3).map { |i: Int| p.first.second[i % p.first.second.size] },
      start: p.second.first,
      steps: [p.second.second.first.first, p.second.second.first.second] + p.second.second.second,
    )
  end
end

def holds_for(kind: Kind) -> Result[Void]
  try with_file do |rig|
    for device in try devices()
      try Prop.check(trails(), Prop.Config(runs: CASES)) do |trail: Trail|
        _ = try compare(rig, kind, device, trail)
        ok()
      end
    end

    ok()
  end

  ok()
end

def test_traces_over_an_f32_weight_match_the_reference_or_are_refused() -> Result[Void] = holds_for(kinds()[0])

def test_traces_over_an_f16_weight_match_the_reference_or_are_refused() -> Result[Void] = holds_for(kinds()[1])

def test_traces_over_a_q8_0_weight_match_the_reference_or_are_refused() -> Result[Void] = holds_for(kinds()[2])

## Every pair of ops, from the input and from the weight itself, each ending as F32: the combinations a random trace seldom
## reaches (the transposed table of a gather, the reshape or slice of a packed weight). At least `RAN_AT_LEAST` of them
## run on ggml, so a refusal of everything does not pass.
def pairs_for(kind: Kind) -> Result[Void]
  try with_file do |rig|
    for device in try devices()
      ran = 0
      for start in 0...2
        for first in 0...OPS
          for second in 0...OPS
            steps = [Step(op: first, arg: 0), Step(op: second, arg: 1)]
            trail = Trail(
              x: Support.cycle([1.0, -0.5, 2.0, 0.25], 2 * COLS),
              ids: [2, 0, 1],
              start: start,
              steps: steps,
            )

            if try compare(rig, kind, device, trail)
              ran += 1
            end
          end
        end
      end

      try Test.assert(
        ran >= RAN_AT_LEAST,
        "ggml ran #{ran} of #{2 * OPS * OPS} pairs of ops with a #{kind.name} weight",
      )
    end

    ok()
  end

  ok()
end

def test_every_pair_of_ops_over_an_f32_weight_matches_the_reference_or_is_refused() -> Result[Void]
  pairs_for(kinds()[0])
end

def test_every_pair_of_ops_over_an_f16_weight_matches_the_reference_or_is_refused() -> Result[Void]
  pairs_for(kinds()[1])
end

def test_every_pair_of_ops_over_a_q8_0_weight_matches_the_reference_or_is_refused() -> Result[Void]
  pairs_for(kinds()[2])
end
