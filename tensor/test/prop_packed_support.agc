## The traces of `prop_packed_test.agc` and `prop_packed_pairs_*_test.agc`: ops chosen by the shape in hand, run on
## ggml with the weight held as F32, F16 or Q8_0 and on the reference, with the error bound derived along the trace.
## Not a test file: only a `*_test.agc` may import `prop`, and the generators are in it. `prop_packed_test.agc` says
## what the bound covers.
##
## ggml may refuse a trace over a packed weight, and only with the messages that fail closed (`fail_closed?`): any
## other error is a failure, so a lowering bug or a shape error does not hide as a refusal.

import Support from "prop_support"
import Tensor from "../tensor"
import Trace from "../trace"

## Cases per property: the fast tier.
CASES = 60
## The seed of the sample whose runs are counted.
SEED = 20261124
## The width of the input and of the weight.
COLS = 32

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

## The file, and the weights and their absolute values, built once.
struct Rig
  path: String
  w: List[Float]
  mag: List[Float]
end

## What one trace did on ggml: whether it ran, and how many outputs were checked against a derived bound (`tight`) or
## only against the range of a norm (`loose`: a trace with a norm whose input error is not small, or whose first-order
## bound is above that range).
struct Outcome
  ran: Bool
  tight: Int
  loose: Int
end

## What ggml keeps the weight as, and what it is expected to refuse.
struct Kind
  name: String
  dtype: Tensor.Dtype
  format: Tensor.FloatFormat
  offset: Int
  ## Relative rounding of each activation of a product, and of a row to a block.
  rounding: Float
  block: Float
  ## The pairs of ops from the input that ggml refuses (`first,second`, `;` between them); with F32 there are none.
  refused: String
  ## The pairs of ops from the weight that ggml runs, and how many of the sampled traces it runs.
  from_weight: String
  sampled: Int
  ## The least number of outputs checked with a real bound: in the sweep from the input, from the weight, and in the
  ## sample.
  tight: List[Int]
end

def devices() -> Result[List[Tensor.Device]]
  return ok([CPU]) if Tensor.available(Tensor.Backend.Ggml(CPU))
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok([]) if !flag.ok?
  return ok([]) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
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
      refused: "",
      from_weight: "",
      sampled: 0,
      tight: [5060, 10686, 2525],
    ),
    Kind(
      name: "F16",
      dtype: Tensor.Dtype.F16,
      format: Tensor.FloatFormat.F32,
      offset: 3,
      rounding: HALF_STEP,
      block: 0.0,
      refused: "transpose,w_matmul",
      from_weight: (
        "widen,transpose;widen,flatten;widen,rows;widen,cols;widen,norm;widen,widen;widen,embedding;" +
        "widen,linear_out_in;widen,matmul_wt;embedding,widen;embedding,embedding"
      ),
      sampled: 31,
      tight: [5054, 946, 968],
    ),
    Kind(
      name: "Q8_0",
      dtype: Tensor.Dtype.Q8_0,
      format: Q8_0,
      offset: Q8_OFFSET,
      rounding: 0.0,
      block: BLOCK_STEP,
      refused: (
        "transpose,w_matmul;linear_out_in,matmul_w;linear_out_in,linear_in_out;" +
        "matmul_wt,matmul_w;matmul_wt,linear_in_out"
      ),
      from_weight: "embedding,widen",
      sampled: 25,
      tight: [4774, 96, 603],
    ),
  ]
end

## The trace on ggml against the reference: it ran and matched, or ggml refused it with a message that fails closed. An
## error if the answers differ, if the reference refuses what `options` offered (it fits), if either answers a value
## that is not finite, or if ggml fails any other way.
def compare(rig: Rig, kind: Kind, device: Tensor.Device, trail: Trail) -> Result[Outcome]
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

## Every pair of ops, from the input (`start` 0) or the weight itself (1), in every variant of each, ending as F32: the
## combinations a random trace seldom reaches (the transposed table of a gather, the reshape or slice of a packed
## weight). With F32 ggml runs all of them; with F16 and Q8_0 the refused pairs from the input are `kind.refused`, and
## ggml runs `kind.from_weight` of those from the weight.
def pairs_for(kind: Kind, start: Int) -> Result[Void]
  try with_file("#{kind.name}-#{start}") do |rig|
    for device in try devices()
      ref_env = try reference_env(sweep_x(), SWEEP_IDS, rig.w)
      first_shape = Tensor.shape(try start_of(start, ref_env))
      ran = 0
      total = 0
      tight = 0
      loose = 0
      refused: List[String] = []
      ran_pairs: List[String] = []
      for first in 0...options(first_shape).size
        one = Step(op: first, arg: 0)
        second_shape = Tensor.shape(try apply(one, try start_of(start, ref_env), ref_env))
        for second in 0...options(second_shape).size
          two = Step(op: second, arg: 0)
          pair = "#{choose(one, first_shape)},#{choose(two, second_shape)}"
          for a in 0...variants(choose(one, first_shape))
            for b in 0...variants(choose(two, second_shape))
              trail = Trail(
                x: sweep_x(),
                ids: SWEEP_IDS,
                start: start,
                steps: [Step(op: first, arg: a), Step(op: second, arg: b)],
              )

              total += 1
              outcome = try compare(rig, kind, device, trail)
              tight += outcome.tight
              loose += outcome.loose
              if outcome.ran
                ran += 1
                if !ran_pairs.include?(pair)
                  ran_pairs = ran_pairs.push(pair)
                end
              elsif !refused.include?(pair)
                refused = refused.push(pair)
              end
            end
          end
        end
      end

      try pins(kind, start, ran, total, tight, loose, refused.join(";"), ran_pairs.join(";"))
    end

    ok()
  end

  ok()
end

private

CPU = Tensor.Device.Cpu
REF = Tensor.Backend.Reference
F32 = Tensor.Dtype.F32
Q8_0 = Tensor.FloatFormat.Q8_0

ROWS = 3
## The F32 weights follow 3 padding bytes; the Q8_0 blocks of the same values follow them.
Q8_OFFSET = 3 + ROWS * COLS * 4
## The ids of the pairs.
SWEEP_IDS = [2, 0, 1]
## The F32 rounding of a sum, relative to the sum of its absolute terms: an ulp is 1.2e-7, the order of the sum
## differs.
NOISE = 0.000001
## The factor between the derived bound and what is allowed.
SAFETY = 1.5
## The eps of the norms.
EPS = 0.00001
## A row of a norm is loose when its input error exceeds this share of its spread: the norm could amplify it past a
## first-order bound.
LOOSE_SHARE = 0.05
## The rounding of an F16 activation, relative.
HALF_STEP = 0.00048828125
## The rounding of a row of 32 activations to a Q8_0 block, relative to the largest: half a step, and the F16 scale.
BLOCK_STEP = 1.0 / 254.0 + 0.00048828125

## The binary16 bits of each block's scale: 1/8, 1/16, 1/4.
SCALES = [0x3000, 0x2C00, 0x3400]
VALUES = [0.125, 0.0625, 0.25]
## What ggml says when it refuses a trace over a packed weight (`half_operands` and `not_half_weight` of
## `src/lower.agc`, `packed_misuse` of `native/ggml_graph.c`), and the checks of `Tensor.*` on the dtypes of its
## operands.
FAIL_CLOSED = [
  "a packed weight goes only into",
  "which ggml holds only for a packed weight",
  "a quantized or F16 weight goes only into",
  "tensors, got",
  "needs a float dtype, got",
]

struct Env
  x: Tensor.Tensor
  w: Tensor.Tensor
  ids: Tensor.Tensor
end

## The error bound carried beside a value, and whether a norm made it loose.
struct Carried
  err: Tensor.Tensor
  loose: Bool
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

## The input of the pairs.
def sweep_x() -> List[Float] = Support.cycle([1.0, -0.5, 2.0, 0.25], 2 * COLS)

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
    out = out.push("linear_out_in").push("matmul_wt")
  end

  if b == ROWS
    out = out.push("matmul_w").push("linear_in_out")
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

## Only the slices and the norms read `arg`: the variants of the other ops are the same op.
def variants(name: String) -> Int
  return 2 if name == "rows" or name == "cols" or name == "norm"

  1
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

## The tensor a trace starts from: the input, or the weight itself.
def start_of(start: Int, env: Env) -> Result[Tensor.Tensor]
  return ok(env.w) if start == 1

  ok(env.x)
end

## The tensor the trace starts from, then the tensor after each step.
def chain(trail: Trail, env: Env) -> Result[List[Tensor.Tensor]]
  cur = try start_of(trail.start, env)
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
  steps = try chain(trail, env)
  names: List[String] = []
  for i in 0...trail.steps.size
    names = names.push(choose(trail.steps[i], Tensor.shape(steps[i])))
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
  ids = trail.ids.map { |i: Int| Tensor.Values.Int(i) }
  ok(
    Env(
      x: try Tensor.floats(trail.x, [2, COLS], backend),
      w: try Tensor.from_file(path, kind.offset, [ROWS, COLS], kind.format, backend, kind.dtype),
      ids: try Tensor.from_list(ids, [3], Tensor.Dtype.I32, backend),
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

## The error bound of the output of a step, from the bound `carried` of its input `cur` (the reference's values).
def propagate(
  name: String,
  step: Step,
  cur: Tensor.Tensor,
  carried: Carried,
  kind: Kind,
  abs_env: Env,
) -> Result[Carried]
  err = carried.err
  return ok(carried) if name == "add_x" or name == "add_w"
  return norm_bound(cur, carried, step.arg) if name == "norm"
  moved = try apply_named(name, step, err, abs_env)
  return ok(Carried(err: moved, loose: carried.loose)) if !product?(name)

  absolute = try Support.numbers(try apply_named(name, step, try abs_of(cur), abs_env))
  out = try Support.numbers(moved)
  for i in 0...out.size
    out = out.set(i, out[i] + absolute[i] * (kind.rounding + NOISE))
  end

  if kind.block > 0.0 and (name == "linear_out_in" or name == "matmul_wt")
    out = try with_block_rounding(out, cur, err, kind, abs_env)
  end

  ok(Carried(err: try Tensor.floats(out, Tensor.shape(moved), REF), loose: carried.loose))
end

## Q8_0 rounds each row of 32 activations of the product to a block: `block` times the largest of the row (and its
## error), times the sum of the absolute weights of each output column.
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

## Bound of `rms_norm` or `layer_norm` (`arg` even or odd) by row. First order: the input error moves the output by `r
## * d` and its scale `r = 1 / sqrt(mean(u^2) + eps)` by `|out| * mean(|u| d) / (mean(u^2) + eps)`. A priori: the
## output of a row has `sum(out^2) <= cols`, so both answers are within `sqrt(cols)` and differ by at most `2 *
## sqrt(cols)`. A row whose input error is more than `LOOSE_SHARE` of its spread, or whose first-order bound is above
## the a priori one, takes the a priori bound and makes the trace loose.
def norm_bound(cur: Tensor.Tensor, carried: Carried, arg: Int) -> Result[Carried]
  shape = Tensor.shape(cur)
  cols = shape[shape.size - 1]
  rows = Support.volume(shape) / cols
  values = try Support.numbers(cur)
  spread = try Support.numbers(carried.err)
  limit = 2.0 * Math.sqrt(cols.to_f)
  out = spread
  loose = carried.loose
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
      if worst * r > LOOSE_SHARE or bound > limit
        bound = limit
        loose = true
      end

      out = out.set(i * cols + k, bound)
    end
  end

  ok(Carried(err: try Tensor.floats(out, shape, REF), loose: loose))
end

## The error bound of the output of the trace, element by element, and whether it is loose.
def bounds(trail: Trail, kind: Kind, rig: Rig, ref_env: Env, steps: List[Tensor.Tensor]) -> Result[Carried]
  abs_env = Env(
    x: try Tensor.floats(abs_all(trail.x), [2, COLS], REF),
    w: try Tensor.floats(rig.mag, [ROWS, COLS], REF),
    ids: ref_env.ids,
  )

  carried = Carried(err: try Tensor.zeros(Tensor.shape(steps[0]), F32, REF), loose: false)
  for i in 0...trail.steps.size
    name = choose(trail.steps[i], Tensor.shape(steps[i]))
    carried = try propagate(name, trail.steps[i], steps[i], carried, kind, abs_env)
  end

  ok(carried)
end

def fail_closed?(message: String) -> Bool = FAIL_CLOSED.any? { |part: String| message.include?(part) }

## With F32 ggml refuses nothing that the reference runs, and with the others only what fails closed.
def refused(kind: Kind, trail: Trail, ref_env: Env, message: String) -> Result[Outcome]
  names = try names_of(trail, ref_env)
  return error("ggml refused a trace of the reference on an F32 weight (#{names}): #{message}") if kind.name == "F32"

  if !fail_closed?(message)
    return error("ggml failed on a #{kind.name} weight without failing closed (#{names}): #{message}")
  end

  ok(Outcome(ran: false, tight: 0, loose: 0))
end

def matches(
  rig: Rig,
  kind: Kind,
  trail: Trail,
  ref_env: Env,
  got: Tensor.Tensor,
  ref: Tensor.Tensor,
  steps: List[Tensor.Tensor],
) -> Result[Outcome]
  names = try names_of(trail, ref_env)
  return error("dtype #{Tensor.dtype(got)}, expected F32: #{names}") if Tensor.dtype(got) != F32
  if Tensor.shape(got) != Tensor.shape(ref)
    return error("shape #{Tensor.shape(got)}, expected #{Tensor.shape(ref)}: #{names}")
  end

  values = try Support.numbers(got)
  want = try Support.numbers(ref)
  carried = try bounds(trail, kind, rig, ref_env, steps)
  spread = try Support.numbers(carried.err)
  for i in 0...want.size
    bound = SAFETY * spread[i] + NOISE * want[i].abs + NOISE
    if !finite?(values[i]) or !((values[i] - want[i]).abs <= bound)
      what = "with a #{kind.name} weight: #{names}"
      return error("output #{i} is #{values[i]}, the reference's #{want[i]} (bound #{bound}) #{what}")
    end
  end

  return ok(Outcome(ran: true, tight: 0, loose: want.size)) if carried.loose

  ok(Outcome(ran: true, tight: want.size, loose: 0))
end

## What a sweep must have found for `kind` (see `pairs_for`): the pairs refused, and the pairs run (`ran_pairs`).
def pins(
  kind: Kind,
  start: Int,
  ran: Int,
  total: Int,
  tight: Int,
  loose: Int,
  refused: String,
  ran_pairs: String,
) -> Result[Void]
  what = "#{kind.name} weight, pairs from #{start}"
  least = kind.tight[start]
  try Test.assert(
    tight >= least,
    "#{tight} outputs checked with a real bound and #{loose} loosely, at least #{least} tight: #{what}",
  )

  return Test.assert(ran == total, "ggml ran #{ran} of #{total}: #{what}") if kind.name == "F32"
  return Test.assert(refused == kind.refused, "ggml refused #{refused}, not #{kind.refused}: #{what}") if start == 0

  Test.assert(ran_pairs == kind.from_weight, "ggml ran #{ran_pairs}, not #{kind.from_weight}: #{what}")
end
