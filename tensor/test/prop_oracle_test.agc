## The reference is the oracle (`../DESIGN.md` §11): the same call on generated
## tensors must answer the same shape, dtype and values on libtorch, within the
## tolerance `F32` justifies (`../README.md`, "Tolerances, and why"), and each
## backend must agree with a naive `conv1d`/`avg_pool1d` written as plain loops.
##
## `Prop.oracle` compares with exact equality, so a tolerance needs a law of its
## own: every property here is a `Prop.check` whose law returns a plain `error`
## (never `Test.eq`), so the counterexample is shrunk. Where libtorch is not
## built, the comparison with it asserts that absence (`Conform.absent()`), and
## the naive-loop checks still run on the reference.

import Conform from "torch_support"
import Prop from "pkg/prop/prop"
import Support from "prop_support"
import Tensor from "../tensor"

## Libtorch's vectorised math is within an ulp or two of the correctly rounded
## value, and an ulp of `F32` is 1.2e-7 relative: one order looser than the
## default, still far below any wrong operand or a doubled `exp`.
LOOSE = 0.000001

## `layer_norm` and `rms_norm` subtract a mean (or square a sum) and divide by a
## square root in `F32`: the absolute error of an output near zero is about the
## `F32` epsilon times the input magnitude over the spread, a few 1e-7, so both
## the relative tolerance and the absolute floor are 1e-5.
NORM = 0.00001

## For the ops whose dyadic inputs make them exact in `F32`: `Support.near?` then
## asks for equality.
EXACT = 0.0

SHAPES = [[1], [5], [2, 3], [3, 2], [1, 4], [4, 1], [2, 2, 2]]
GRIDS = [[2, 3], [3, 2], [1, 4], [4, 1], [2, 2]]

private

struct Twins
  a: Support.Spec
  b: Support.Spec
end

## One window of a convolution or a pool: the sizes, and the input and weights.
struct Window
  batched: Bool
  batch: Int
  channels_in: Int
  channels_out: Int
  length: Int
  kernel: Int
  stride: Int
  padding: Int
  input: List[Float]
  weight: List[Float]
end

def dyadic() -> Prop.Gen[Float] = Prop.map(Prop.int(-64, 64)) do |n: Int| n.to_f / 8.0 end

def positive() -> Prop.Gen[Float] = Prop.map(Prop.int(1, 64)) do |n: Int| n.to_f / 8.0 end

def nonzero() -> Prop.Gen[Float] = Prop.map(Prop.int(-64, 63)) do |n: Int| away_from_zero(n).to_f / 8.0 end

def away_from_zero(n: Int) -> Int
  return n + 1 if n >= 0

  n
end

## Whole numbers from -2 to 2, so `eq` and `lt` meet ties.
def small() -> Prop.Gen[Float] = Prop.map(Prop.int(-2, 2)) do |n: Int| n.to_f end

def specs(shapes: List[List[Int]], values: Prop.Gen[Float]) -> Prop.Gen[Support.Spec]
  pattern = Prop.pair(Prop.one_of(shapes), Prop.non_empty_list(values, 12))
  Prop.map(pattern) do |p: Prop.Pair[List[Int], List[Float]]|
    Support.Spec(shape: p.first, data: Support.cycle(p.second, Support.volume(p.first)))
  end
end

## Two tensors of one shape, from two value generators.
def twins(shapes: List[List[Int]], left: Prop.Gen[Float], right: Prop.Gen[Float]) -> Prop.Gen[Twins]
  lists = Prop.pair(Prop.non_empty_list(left, 12), Prop.non_empty_list(right, 12))
  pattern = Prop.pair(Prop.one_of(shapes), lists)
  Prop.map(pattern) do |p: Prop.Pair[List[Int], Prop.Pair[List[Float], List[Float]]]|
    n = Support.volume(p.first)
    Twins(
      a: Support.Spec(shape: p.first, data: Support.cycle(p.second.first, n)),
      b: Support.Spec(shape: p.first, data: Support.cycle(p.second.second, n)),
    )
  end
end

def run[T](gen: Prop.Gen[T], law: Fn[T, Result[Void]]) -> Result[Void]
  Prop.check(gen, Prop.Config(runs: Support.RUNS), law)
end

def unary(op: String, t: Tensor.Tensor) -> Result[Tensor.Tensor]
  case op
    "neg" -> Tensor.neg(t)
    "exp" -> Tensor.exp(t)
    "tanh" -> Tensor.tanh(t)
    "sigmoid" -> Tensor.sigmoid(t)
    "relu" -> Tensor.relu(t)
    "log" -> Tensor.log(t)
    "sqrt" -> Tensor.sqrt(t)
    "silu" -> Tensor.silu(t)
    "softmax rows" -> Tensor.softmax(t, 1)
    "softmax columns" -> Tensor.softmax(t, 0)
    "layer_norm" -> Tensor.layer_norm(t, 0.00001)
    _ -> Tensor.rms_norm(t, 0.00001)
  end
end

def binary(op: String, a: Tensor.Tensor, b: Tensor.Tensor) -> Result[Tensor.Tensor]
  case op
    "add" -> Tensor.add(a, b)
    "sub" -> Tensor.sub(a, b)
    "mul" -> Tensor.mul(a, b)
    "div" -> Tensor.div(a, b)
    "pow" -> Tensor.pow(a, b)
    "eq" -> Tensor.eq(a, b)
    "lt" -> Tensor.lt(a, b)
    _ -> Tensor.matmul(a, b)
  end
end

## `op` on libtorch against `op` on the reference.
def unary_agrees(op: String, spec: Support.Spec, rel: Float, floor: Float) -> Result[Void]
  ours = try unary(op, try Support.tensor(Conform.TORCH, spec))
  theirs = try unary(op, try Support.tensor(Conform.REF, spec))
  Support.same(ours, theirs, op, rel, floor)
end

def binary_agrees(op: String, a: Support.Spec, b: Support.Spec, rel: Float) -> Result[Void]
  ours = try binary(op, try Support.tensor(Conform.TORCH, a), try Support.tensor(Conform.TORCH, b))
  theirs = try binary(op, try Support.tensor(Conform.REF, a), try Support.tensor(Conform.REF, b))
  Support.same(ours, theirs, op, rel)
end

def unary_ops_agree(
  ops: List[String],
  gen: Prop.Gen[Support.Spec],
  rel: Float,
  floor: Float = Conform.ABS,
) -> Result[Void]
  return Conform.absent() if !Conform.available()

  run(gen) do |spec: Support.Spec|
    for op in ops
      try unary_agrees(op, spec, rel, floor)
    end

    ok()
  end
end

def binary_ops_agree(ops: List[String], gen: Prop.Gen[Twins], rel: Float) -> Result[Void]
  return Conform.absent() if !Conform.available()

  run(gen) do |p: Twins|
    for op in ops
      try binary_agrees(op, p.a, p.b, rel)
    end

    ok()
  end
end

def test_exact_unary_ops_agree_with_the_reference() -> Result[Void]
  unary_ops_agree(["neg", "relu"], specs(SHAPES, dyadic()), EXACT)
end

def test_transcendental_ops_agree_with_the_reference() -> Result[Void]
  unary_ops_agree(["exp", "tanh", "sigmoid", "silu"], specs(SHAPES, dyadic()), LOOSE)
end

def test_log_and_sqrt_agree_with_the_reference() -> Result[Void]
  unary_ops_agree(["log", "sqrt"], specs(SHAPES, positive()), LOOSE)
end

def test_softmax_agrees_with_the_reference() -> Result[Void]
  unary_ops_agree(["softmax rows", "softmax columns"], specs(GRIDS, dyadic()), LOOSE)
end

def test_normalizations_agree_with_the_reference() -> Result[Void]
  unary_ops_agree(["layer_norm", "rms_norm"], specs(GRIDS, dyadic()), NORM, NORM)
end

def test_add_sub_and_mul_agree_with_the_reference() -> Result[Void]
  binary_ops_agree(["add", "sub", "mul"], twins(SHAPES, dyadic(), dyadic()), EXACT)
end

def test_div_agrees_with_the_reference() -> Result[Void]
  binary_ops_agree(["div"], twins(SHAPES, dyadic(), nonzero()), Conform.REL)
end

def test_pow_agrees_with_the_reference() -> Result[Void]
  binary_ops_agree(["pow"], twins(SHAPES, positive(), dyadic()), LOOSE)
end

def test_comparisons_agree_with_the_reference() -> Result[Void]
  binary_ops_agree(["eq", "lt"], twins(SHAPES, small(), small()), EXACT)
end

def test_matmul_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  sizes = Prop.one_of([[1, 1, 1], [2, 3, 2], [3, 1, 4], [4, 2, 1], [2, 2, 3]])
  gen = Prop.pair(sizes, Prop.pair(Prop.non_empty_list(dyadic(), 12), Prop.non_empty_list(dyadic(), 12)))
  run(gen) do |p: Prop.Pair[List[Int], Prop.Pair[List[Float], List[Float]]]|
    m = p.first[0]
    k = p.first[1]
    n = p.first[2]
    left = Support.Spec(shape: [m, k], data: Support.cycle(p.second.first, m * k))
    right = Support.Spec(shape: [k, n], data: Support.cycle(p.second.second, k * n))
    binary_agrees("matmul", left, right, EXACT)
  end
end

def test_clamp_agrees_with_the_reference() -> Result[Void]
  return Conform.absent() if !Conform.available()

  gen = Prop.pair(specs(SHAPES, dyadic()), Prop.pair(dyadic(), dyadic()))
  run(gen) do |p: Prop.Pair[Support.Spec, Prop.Pair[Float, Float]]|
    try Prop.assume(p.second.first <= p.second.second)
    ours = try Tensor.clamp(try Support.tensor(Conform.TORCH, p.first), p.second.first, p.second.second)
    theirs = try Tensor.clamp(try Support.tensor(Conform.REF, p.first), p.second.first, p.second.second)
    Support.same(ours, theirs, "clamp", EXACT)
  end
end

## One value of a tape, wrapping around: a few integers set every size of a case
## and then its data, and the tape shrinks to zeros, which is the smallest case.
def at(tape: List[Int], index: Int) -> Int = tape[index].abs

## Eight integers set the sizes of a case; the rest of the tape (and its zero
## padding, when it shrinks below eight) is data.
def tapes() -> Prop.Gen[List[Int]]
  Prop.map(Prop.list(Prop.int(-8, 8), 24)) do |tape: List[Int]| tape + List.repeat(0, (8 - tape.size).clamp(0, 8)) end
end

def conv_window(tape: List[Int]) -> Window
  batched = at(tape, 0) % 2 == 0
  batch = 1
  if batched
    batch = 1 + at(tape, 1) % 2
  end

  channels_in = 1 + at(tape, 2) % 2
  channels_out = 1 + at(tape, 3) % 2
  length = 1 + at(tape, 4) % 7
  kernel = 1 + at(tape, 5) % 3
  values = tape.map do |n: Int| n.to_f / 8.0 end
  Window(
    batched: batched,
    batch: batch,
    channels_in: channels_in,
    channels_out: channels_out,
    length: length,
    kernel: kernel,
    stride: 1 + at(tape, 6) % 3,
    padding: at(tape, 7) % 3,
    input: Support.cycle(values, batch * channels_in * length),
    weight: Support.cycle(values.reverse, channels_out * channels_in * kernel),
  )
end

## `[N, C_in, L]`, or `[C_in, L]` without the batch axis.
def input_shape(w: Window) -> List[Int]
  return [w.batch, w.channels_in, w.length] if w.batched

  [w.channels_in, w.length]
end

def conv_length(w: Window) -> Int = (w.length + 2 * w.padding - w.kernel) / w.stride + 1

## `x[n][c][i]` of the zero-padded input, where `i` counts from the padding.
def padded(w: Window, n: Int, c: Int, i: Int) -> Float
  at_input = i - w.padding
  return 0.0 if at_input < 0 or at_input >= w.length

  w.input[(n * w.channels_in + c) * w.length + at_input]
end

## The convolution as its definition: for each output position, the sum over
## input channels and kernel taps.
def naive_conv(w: Window) -> List[Float]
  out: List[Float] = []
  for n in 0...w.batch
    for co in 0...w.channels_out
      for o in 0...conv_length(w)
        total = 0.0
        for ci in 0...w.channels_in
          for k in 0...w.kernel
            tap = w.weight[(co * w.channels_in + ci) * w.kernel + k]
            total += tap * padded(w, n, ci, o * w.stride + k)
          end
        end

        out = out.push(total)
      end
    end
  end

  out
end

## The mean of each window of the zero-padded input, over `kernel` positions
## whether or not they are padding.
def naive_pool(w: Window) -> List[Float]
  out: List[Float] = []
  for n in 0...w.batch
    for c in 0...w.channels_in
      for o in 0...conv_length(w)
        total = 0.0
        for k in 0...w.kernel
          total += padded(w, n, c, o * w.stride + k)
        end

        out = out.push(total / w.kernel.to_f)
      end
    end
  end

  out
end

def output_shape(w: Window, channels: Int) -> List[Int]
  return [w.batch, channels, conv_length(w)] if w.batched

  [channels, conv_length(w)]
end

def test_conv1d_is_the_naive_sum_of_taps() -> Result[Void]
  run(Prop.map(tapes()) do |tape: List[Int]| conv_window(tape) end) do |w: Window|
    try Prop.assume(w.length + 2 * w.padding >= w.kernel)
    shape = input_shape(w)
    want = naive_conv(w)
    for backend in Support.backends()
      x = try Support.tensor(backend, Support.Spec(shape: shape, data: w.input))
      k = try Support.tensor(backend, Support.Spec(shape: [w.channels_out, w.channels_in, w.kernel], data: w.weight))
      got = try Tensor.conv1d(x, k, w.stride, w.padding)
      try Support.holds(got, output_shape(w, w.channels_out), want, "conv1d on #{backend}")
    end

    ok()
  end
end

def test_avg_pool1d_is_the_naive_window_mean() -> Result[Void]
  run(Prop.map(tapes()) do |tape: List[Int]| conv_window(tape) end) do |w: Window|
    try Prop.assume(w.length + 2 * w.padding >= w.kernel and w.padding <= w.kernel / 2)
    shape = input_shape(w)
    want = naive_pool(w)
    for backend in Support.backends()
      x = try Support.tensor(backend, Support.Spec(shape: shape, data: w.input))
      got = try Tensor.avg_pool1d(x, w.kernel, w.stride, w.padding)
      try Support.holds(got, output_shape(w, w.channels_in), want, "avg_pool1d on #{backend}")
    end

    ok()
  end
end

def test_the_torch_side_of_the_oracle_laws_is_compared_or_absent() -> Result[Void]
  Support.torch_is_compared_or_absent()
end

def scalar(backend: Tensor.Backend, x: Float) -> Result[Tensor.Tensor]
  Support.tensor(backend, Support.Spec(shape: [1], data: [x]))
end

## `exp(100)` is finite in the reference's doubles (libtorch's `F32` answers
## `+inf`, which `test/torch_elementwise_test.agc` pins).
def test_the_reference_computes_exp_past_the_f32_range() -> Result[Void]
  ref = try Tensor.exp(try scalar(Conform.REF, 100.0))
  Support.holds(ref, [1], [2.688117141816139e43], "exp(100) on the reference", LOOSE)
end

## `near?` is what keeps an infinity from agreeing with a large finite value, and
## makes two NaNs agree.
def test_near_treats_two_nans_as_equal_and_exact_means_exact() -> Result[Void]
  infinity = 1.0e308 * 10.0
  nan = infinity - infinity
  try Test.assert(Support.near?(nan, nan, LOOSE), "NaN against NaN")
  try Test.assert(!Support.near?(nan, 1.0, LOOSE), "NaN against a number")
  try Test.assert(Support.near?(infinity, infinity, EXACT), "infinity against itself")
  try Test.assert(!Support.near?(infinity, -infinity, LOOSE), "opposite infinities")
  try Test.assert(!Support.near?(infinity, 1.0e308, LOOSE), "an infinity agreed with a finite value")
  try Test.assert(!Support.near?(1000.0, 1000.00005, EXACT), "exact allows no gap")
  Test.assert(Support.near?(1000.0, 1000.00005, LOOSE), "loose allows a relative gap")
end
