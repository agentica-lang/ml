## The traces of `prop_packed_test.agc` and `prop_packed_pairs_*_test.agc`: ops chosen by the shape in hand, run on ggml
## with the weight held as F32, F16 or Q8_0 and on the reference, with the error bound derived along the trace. Not a test
## file: only a `*_test.agc` may import `prop`, and the generators are in it.
##
## See `prop_packed_test.agc` for what the bound covers.

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
## The seed of the sample whose runs are counted.
SEED = 20261124
## Ops a pair is made of: `op` indexes the ops that fit, modulo their number, and no shape has more.
OPS = 11
## The F32 rounding of a sum, relative to the sum of its absolute terms: an ulp is 1.2e-7, the order of the sum differs.
NOISE = 0.000001
## The factor between the derived bound and what is allowed.
SAFETY = 1.5
## The eps of the norms.
EPS = 0.00001
## A bound that says nothing: the row of a norm whose input error is not small.
UNKNOWN = 1000000000000000000000000000000.0
## The rounding of an F16 activation, relative.
HALF_STEP = 0.00048828125
## The rounding of a row of 32 activations to a Q8_0 block, relative to the largest: half a step, and the F16 scale.
BLOCK_STEP = 1.0 / 254.0 + 0.00048828125

## The binary16 bits of each block's scale: 1/8, 1/16, 1/4.
SCALES = [0x3000, 0x2C00, 0x3400]
VALUES = [0.125, 0.0625, 0.25]

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
  ## Relative rounding of each activation of a product, and of a row to a block, as above.
  rounding: Float
  block: Float
  ## Of the 800 pairs of ops, how many ggml runs (the rest it refuses); the sampled traces run at least `floor` of `CASES`.
  pairs: List[Int]
  floor: Int
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

## `label` keeps the directories of the files of a parallel run apart (each starts from the same seed).
def with_file(label: String, body: Fn[Rig, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-packed-#{label}-#{Random.int(0, 999999999)}"
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
    Kind(
      name: "F32",
      dtype: F32,
      format: Tensor.FloatFormat.F32,
      offset: 3,
      rounding: 0.0,
      block: 0.0,
      pairs: [207, 201],
      floor: 60,
    ),
    Kind(
      name: "F16",
      dtype: Tensor.Dtype.F16,
      format: Tensor.FloatFormat.F32,
      offset: 3,
      rounding: HALF_STEP,
      block: 0.0,
      pairs: [205, 2],
      floor: 25,
    ),
    Kind(
      name: "Q8_0",
      dtype: Tensor.Dtype.Q8_0,
      format: Q8_0,
      offset: Q8_OFFSET,
      rounding: 0.0,
      block: BLOCK_STEP,
      pairs: [201, 1],
      floor: 23,
    ),
  ]
end

## The ops that fit a tensor of this shape, by name.
def options(shape: List[Int]) -> List[String]
  if shape.size == 1
    return ["unsqueeze", "norm", "head", "fold", "widen"].select { |n: String| n != "fold" or shape[0] % COLS == 0 }
  end

  a = shape[0]
  b = shape[1]
  out = ["transpose", "flatten", "rows", "cols", "norm", "widen"]
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

def choose(step: Step, shape: List[Int]) -> String
  names = options(shape)
  names[step.op % names.size]
end

def apply(step: Step, cur: Tensor.Tensor, env: Env) -> Result[Tensor.Tensor]
  apply_named(choose(step, Tensor.shape(cur)), step, cur, env)
end

def apply_named(name: String, step: Step, cur: Tensor.Tensor, env: Env) -> Result[Tensor.Tensor]
  shape = Tensor.shape(cur)
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
    "widen" -> Tensor.to_dtype(cur, F32)
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

## The tensor the trace starts from, then the tensor after each step.
def chain(trail: Trail, env: Env) -> Result[List[Tensor.Tensor]]
  cur = env.x
  if trail.start == 1
    cur = env.w
  end

  out = [cur]
  for step in trail.steps
    cur = try apply(step, cur, env)
    out = out.push(cur)
  end

  ok(out)
end

def run(trail: Trail, env: Env) -> Result[Tensor.Tensor]
  steps = try chain(trail, env)
  Tensor.to_dtype(steps[steps.size - 1], F32)
end

## The ops of a trace by name, as the shapes of the reference's run pick them.
def names_of(trail: Trail, env: Env) -> Result[String]
  cur = env.x
  if trail.start == 1
    cur = env.w
  end

  names: List[String] = []
  for step in trail.steps
    names = names.push(choose(step, Tensor.shape(cur)))
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

def finite?(x: Float) -> Bool = x - x == 0.0

def abs_of(t: Tensor.Tensor) -> Result[Tensor.Tensor]
  Tensor.floats(abs_all(try Support.numbers(t)), Tensor.shape(t), REF)
end

def product?(name: String) -> Bool
  ["linear_out_in", "linear_in_out", "matmul_wt", "matmul_w", "w_matmul"].include?(name)
end

## The error bound of the output of a step, from the bound `err` of its input `cur` (the reference's values).
def propagate(
  name: String,
  step: Step,
  cur: Tensor.Tensor,
  err: Tensor.Tensor,
  kind: Kind,
  abs_env: Env,
) -> Result[Tensor.Tensor]
  return ok(err) if name == "add_x" or name == "add_w"
  return norm_bound(cur, err, step.arg) if name == "norm"
  moved = try apply_named(name, step, err, abs_env)
  return ok(moved) if !product?(name)

  absolute = try Support.numbers(try apply_named(name, step, try abs_of(cur), abs_env))
  out = try Support.numbers(moved)
  for i in 0...out.size
    out = out.set(i, out[i] + absolute[i] * (kind.rounding + NOISE))
  end

  if kind.block > 0.0 and (name == "linear_out_in" or name == "matmul_wt")
    out = try with_block_rounding(out, cur, err, kind, abs_env)
  end

  Tensor.floats(out, Tensor.shape(moved), REF)
end

## Q8_0 rounds each row of 32 activations of the product to a block: `block` times the largest of the row (and its error),
## times the sum of the absolute weights of each output column.
def with_block_rounding(
  out: List[Float],
  cur: Tensor.Tensor,
  err: Tensor.Tensor,
  kind: Kind,
  abs_env: Env,
) -> Result[List[Float]]
  values = try Support.numbers(cur)
  spread = try Support.numbers(err)
  weights = try Support.numbers(abs_env.w)
  rows = Tensor.shape(cur)[0]
  for i in 0...rows
    largest = 0.0
    for k in 0...COLS
      largest = [largest, values[i * COLS + k].abs + spread[i * COLS + k]].max.unwrap!
    end

    for j in 0...ROWS
      total = 0.0
      for k in 0...COLS
        total += weights[j * COLS + k]
      end

      out = out.set(i * ROWS + j, out[i * ROWS + j] + kind.block * largest * total)
    end
  end

  ok(out)
end

## First-order bound of `rms_norm` or `layer_norm` (`arg` even or odd) by row: the input error moves the output by
## `r * d` and its scale `r = 1 / sqrt(mean(u^2) + eps)` by `|out| * mean(|u| d) / (mean(u^2) + eps)`. A row whose error
## is not small against its spread (more than 5% of it) is UNKNOWN.
def norm_bound(cur: Tensor.Tensor, err: Tensor.Tensor, arg: Int) -> Result[Tensor.Tensor]
  shape = Tensor.shape(cur)
  cols = shape[shape.size - 1]
  rows = Support.volume(shape) / cols
  values = try Support.numbers(cur)
  spread = try Support.numbers(err)
  out = spread
  for i in 0...rows
    mean = 0.0
    mean_err = 0.0
    if arg % 2 == 1
      for k in 0...cols
        mean += values[i * cols + k] / cols.to_f
        mean_err += spread[i * cols + k] / cols.to_f
      end
    end

    square = 0.0
    for k in 0...cols
      square += ((values[i * cols + k] - mean) * (values[i * cols + k] - mean)) / cols.to_f
    end

    variance = square + EPS
    r = 1.0 / Math.sqrt(variance)
    drift = 0.0
    worst = 0.0
    for k in 0...cols
      d = spread[i * cols + k] + mean_err
      drift += ((values[i * cols + k] - mean).abs * d) / cols.to_f
      worst = [worst, d].max.unwrap!
    end

    for k in 0...cols
      d = spread[i * cols + k] + mean_err
      answer = (values[i * cols + k] - mean).abs * r
      bound = SAFETY * (r * d + (answer * drift) / variance) + NOISE * answer
      if worst * r > 0.05
        bound = UNKNOWN
      end

      out = out.set(i * cols + k, bound)
    end
  end

  Tensor.floats(out, shape, REF)
end

## The error bound of the output of the trace, element by element.
def bounds(trail: Trail, kind: Kind, rig: Rig, ref_env: Env, steps: List[Tensor.Tensor]) -> Result[List[Float]]
  abs_env = Env(
    x: try Tensor.floats(abs_all(trail.x), [2, COLS], REF),
    w: try Tensor.floats(rig.mag, [ROWS, COLS], REF),
    ids: ref_env.ids,
  )

  err = try Tensor.zeros(Tensor.shape(steps[0]), F32, REF)
  for i in 0...trail.steps.size
    name = choose(trail.steps[i], Tensor.shape(steps[i]))
    err = try propagate(name, trail.steps[i], steps[i], err, kind, abs_env)
  end

  Support.numbers(err)
end

## The trace on ggml against the reference: ok(true) if it ran and matched, ok(false) if ggml refused it, an error if the
## answers differ, if the reference refuses what `options` offered (it fits), or if either answers a value that is not
## finite.
def compare(rig: Rig, kind: Kind, device: Tensor.Device, trail: Trail) -> Result[Bool]
  ref_env = try reference_env(trail.x, trail.ids, rig.w)
  steps = try chain(trail, ref_env)
  ref = try Tensor.to_dtype(steps[steps.size - 1], F32)
  want = try Support.numbers(ref)
  for v in want
    return error("the reference answers #{v}: #{try names_of(trail, ref_env)}") if !finite?(v)
  end

  env = try ggml_env(rig.path, kind, trail, device)
  ran = Trace.eval_many([env.x, env.w, env.ids], keep: false) do |a|
    out = try run(trail, Env(x: a[0], w: a[1], ids: a[2]))
    ok([out])
  end

  case ran
    error(e) -> refused(kind, trail, ref_env, e.message)
    ok(outs) -> matches(rig, kind, trail, ref_env, outs[0], ref, steps)
  end
end

## With F32 ggml refuses nothing that the reference runs.
def refused(kind: Kind, trail: Trail, ref_env: Env, message: String) -> Result[Bool]
  return ok(false) if kind.name != "F32"

  error("ggml refused a trace of the reference on an F32 weight (#{try names_of(trail, ref_env)}): #{message}")
end

def matches(
  rig: Rig,
  kind: Kind,
  trail: Trail,
  ref_env: Env,
  got: Tensor.Tensor,
  ref: Tensor.Tensor,
  steps: List[Tensor.Tensor],
) -> Result[Bool]
  return error("dtype #{Tensor.dtype(got)}, expected F32: #{try names_of(trail, ref_env)}") if Tensor.dtype(got) != F32

  return error("shape #{Tensor.shape(got)}, expected #{Tensor.shape(ref)}") if Tensor.shape(got) != Tensor.shape(ref)

  values = try Support.numbers(got)
  want = try Support.numbers(ref)
  spread = try bounds(trail, kind, rig, ref_env, steps)
  for i in 0...want.size
    bound = SAFETY * spread[i] + NOISE * want[i].abs + NOISE
    if !finite?(values[i]) or !((values[i] - want[i]).abs <= bound)
      return error(
        "output #{i} is #{values[i]}, the reference's #{want[i]} (bound #{bound}) with a #{kind.name} weight: #{try names_of(trail, ref_env)}",
      )
    end
  end

  ok(true)
end

## Every pair of ops, from the input and from the weight itself, in both variants of each, ending as F32: the
## combinations a random trace seldom reaches (the transposed table of a gather, the reshape or slice of a packed weight).
## ggml runs exactly `kind.pairs` of them.
## The shape after the op `name` on `shape` (the shapes `options` offers it for).
def shape_after(name: String, shape: List[Int]) -> List[Int]
  n = shape[0]
  case name
    "transpose" -> [shape[1], shape[0]]
    "flatten" -> [shape[0] * shape[1]]
    "unsqueeze" -> [1, n]
    "fold" -> [n / COLS, COLS]
    "head" -> [[n / 2, 1].max.unwrap!]
    "rows" -> [window(n, 0)[1] - window(n, 0)[0], shape[1]]
    "cols" -> [n, window(shape[1], 0)[1] - window(shape[1], 0)[0]]
    "embedding" -> [3, shape[1]]
    "linear_out_in" -> [n, ROWS]
    "matmul_wt" -> [n, ROWS]
    "matmul_w" -> [n, COLS]
    "linear_in_out" -> [n, COLS]
    "w_matmul" -> [ROWS, shape[1]]
    _ -> shape
  end
end

## Only the slices and the norms read `arg`: the variants of the other ops are the same op.
def variants(name: String) -> Int
  return 2 if name == "rows" or name == "cols" or name == "norm"

  1
end

## Every pair of ops, from the input (`start` 0) or the weight itself (1), in every variant of each, ending as F32: the
## combinations a random trace seldom reaches (the transposed table of a gather, the reshape or slice of a packed weight).
## ggml runs exactly `kind.pairs[start]` of them.
def pairs_for(kind: Kind, start: Int) -> Result[Void]
  try with_file("#{kind.name}-#{start}") do |rig|
    for device in try devices()
      ran = 0
      total = 0
      shape = [2, COLS]
      if start == 1
        shape = [ROWS, COLS]
      end

      for first in 0...OPS
        one = choose(Step(op: first, arg: 0), shape)
        for second in 0...OPS
          two = choose(Step(op: second, arg: 0), shape_after(one, shape))
          for a in 0...variants(one)
            for b in 0...variants(two)
              trail = Trail(
                x: Support.cycle([1.0, -0.5, 2.0, 0.25], 2 * COLS),
                ids: [2, 0, 1],
                start: start,
                steps: [Step(op: first, arg: a), Step(op: second, arg: b)],
              )

              total += 1
              if try compare(rig, kind, device, trail)
                ran += 1
              end
            end
          end
        end
      end

      want = kind.pairs[start]
      try Test.assert(
        ran == want,
        "ggml ran #{ran} of #{total} pairs of ops from #{start} with a #{kind.name} weight, not #{want}",
      )
    end

    ok()
  end

  ok()
end
