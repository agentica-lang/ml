## The reductions over every ordering of `axes`, against a naive reduction.
##
## `axes` is a set, and `argmax` numbers the position inside a reduced block by
## the order of its axes, so `some([2, 0])` and `some([0, 2])` must be the same
## call (`../README.md`, "Reduction `axes` are a set"). That bug lived in the
## order, so the check is exhaustive: two rank-3 shapes ([2, 1, 3] has three
## different axis sizes, [2, 3, 2] repeats one), every ordered subset of their
## axes (and `none`), and three value patterns, on every available backend, for
## `sum`, `mean`, `max` and `argmax` with and without `keep_dims`. The naive
## reduction below is the oracle.

import Prop from "pkg/prop/prop"
import Support from "prop_support"
import Tensor from "../tensor"

## How the values of a shape are laid out (`value_at`): all equal, ascending
## (one maximum, at the end) and a repeating pattern with many ties.
PATTERNS = [0, 1, 2]

private

## What a reduction answers: the shape, and the value per kept position in
## row-major order.
struct Reduced
  shape: List[Int]
  values: List[Float]
end

## One op on one shape per test because a test file's step budget is 1M steps x
## (tests + 1) and the reference is slow. Merge them into one test per shape
## when agentica#1917 makes the step budget configurable.
def test_sum_over_every_ordering_of_the_axes_of_a_2_1_3_tensor() -> Result[Void] = exhaust("sum", [2, 1, 3])

def test_mean_over_every_ordering_of_the_axes_of_a_2_1_3_tensor() -> Result[Void] = exhaust("mean", [2, 1, 3])

def test_max_over_every_ordering_of_the_axes_of_a_2_1_3_tensor() -> Result[Void] = exhaust("max", [2, 1, 3])

def test_argmax_over_every_ordering_of_the_axes_of_a_2_1_3_tensor() -> Result[Void] = exhaust("argmax", [2, 1, 3])

def test_sum_over_every_ordering_of_the_axes_of_a_2_3_2_tensor() -> Result[Void] = exhaust("sum", [2, 3, 2])

def test_mean_over_every_ordering_of_the_axes_of_a_2_3_2_tensor() -> Result[Void] = exhaust("mean", [2, 3, 2])

def test_max_over_every_ordering_of_the_axes_of_a_2_3_2_tensor() -> Result[Void] = exhaust("max", [2, 3, 2])

def test_argmax_over_every_ordering_of_the_axes_of_a_2_3_2_tensor() -> Result[Void] = exhaust("argmax", [2, 3, 2])

def test_the_torch_side_is_compared_or_absent() -> Result[Void] = Support.torch_is_compared_or_absent()

## `op` over every ordering of every subset of the axes of `shape` (and `none`),
## for every pattern. Each test has a step budget and the reference is slow, so
## a test is one op on one shape.
def exhaust(op: String, shape: List[Int]) -> Result[Void]
  gen = Prop.pair(Prop.one_of(arrangements([0, 1, 2])), Prop.one_of(PATTERNS))
  Prop.exhaustive(gen, 1) do |c: Prop.Pair[Optional[List[Int]], Int]| law(op, shape, c.first, c.second) end
end

def law(op: String, shape: List[Int], axes: Optional[List[Int]], pattern: Int) -> Result[Void]
  n = Support.volume(shape)
  data: List[Float] = []
  for i in 0...n
    data = data.push(value_at(pattern, i))
  end

  spec = Support.Spec(shape: shape, data: data)
  for backend in Support.backends()
    t = try Support.tensor(backend, spec)
    for keep in [false, true]
      got = try reduce(op, t, axes, keep)
      want = naive(op, shape, data, ascending(axes, shape.size), keep)
      try Support.holds(got, want.shape, want.values, "#{op} over #{axes} keep_dims #{keep} on #{backend}")
    end
  end

  ok()
end

def reduce(op: String, t: Tensor.Tensor, axes: Optional[List[Int]], keep: Bool) -> Result[Tensor.Tensor]
  case op
    "sum" -> Tensor.sum(t, axes, keep)
    "mean" -> Tensor.mean(t, axes, keep)
    "max" -> Tensor.max(t, axes, keep)
    _ -> Tensor.argmax(t, axes, keep)
  end
end

def value_at(pattern: Int, i: Int) -> Float
  case pattern
    0 -> 0.0
    1 -> i.to_f
    _ -> ((i * 5 + 2) % 3).to_f
  end
end

## `none` (every axis), then every way to pick some of `pool` in any order.
def arrangements(pool: List[Int]) -> List[Optional[List[Int]]]
  all: List[Optional[List[Int]]] = [none]
  for axes in ordered(pool)
    all = all.push(some(axes))
  end

  all
end

def ordered(pool: List[Int]) -> List[List[Int]]
  out: List[List[Int]] = [[]]
  for x in pool
    rest = pool.select do |y: Int| y != x end
    for tail in ordered(rest)
      out = out.push([x] + tail)
    end
  end

  out
end

def ascending(axes: Optional[List[Int]], rank: Int) -> List[Int]
  case axes
    some(given) -> given.sort
    none -> (0...rank).to_a
  end
end

## The reduction written the other way round from the reference: for each kept
## position, walk its reduced block in row-major order of the ascending axes and
## keep the first maximum.
def naive(op: String, shape: List[Int], data: List[Float], axes: List[Int], keep: Bool) -> Reduced
  kept = (0...shape.size).to_a.select do |i: Int| !axes.include?(i) end
  kept_dims = kept.map do |i: Int| shape[i] end
  block_dims = axes.map do |ax: Int| shape[ax] end
  block = Support.volume(block_dims)
  values: List[Float] = []
  for o in 0...Support.volume(kept_dims)
    at = Support.coords(o, kept_dims)
    total = 0.0
    best = 0.0
    best_at = 0
    for s in 0...block
      inside = Support.coords(s, block_dims)
      full = List.repeat(0, shape.size)
      for j in 0...kept.size
        full = full.set(kept[j], at[j])
      end

      for j in 0...axes.size
        full = full.set(axes[j], inside[j])
      end

      v = data[Support.offset(full, shape)]
      total += v
      if s == 0 or v > best
        best = v
        best_at = s
      end
    end

    values = values.push(answer(op, total, best, best_at, block))
  end

  Reduced(shape: reduced_shape(shape, axes, keep), values: values)
end

def answer(op: String, total: Float, best: Float, best_at: Int, block: Int) -> Float
  case op
    "sum" -> total
    "mean" -> total / block.to_f
    "max" -> best
    _ -> best_at.to_f
  end
end

def reduced_shape(shape: List[Int], axes: List[Int], keep: Bool) -> List[Int]
  out: List[Int] = []
  for i in 0...shape.size
    if !axes.include?(i)
      out = out.push(shape[i])
    elsif keep
      out = out.push(1)
    end
  end

  out
end
