## What a compiled function keeps between its runs on ggml (`native/ggml_graph.c`, "Kept leaves"): its constants stay
## bound to the pinned graph, and the graph that ran last keeps its allocation. A run must never read a weight that was
## closed, the weight of another function, or a tensor of an earlier run.
##
## Without the ggml module the checks pass by asserting nothing, unless `TENSOR_REQUIRE_GGML` is set.

import Ggml from "../src/ggml"
import GgmlRaw from "../src/ggml_native"
import Lower from "../src/lower"
import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)

private

def floats(xs: List[Float], shape: List[Int]) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, CPU)
end

## `count` values that start at `first` and rise by one.
def ramp(count: Int, first: Float) -> List[Float] = (0...count).map { |i| first + i.to_f }

def products(xs: List[Float], ws: List[Float]) -> List[Float] = (0...xs.size).map { |i| xs[i] * ws[i] }

## `x * w` over `[k]`, with `w` read from outside the inputs: each test takes its own `k`, so its graph is its own.
def scaled(k: Int, w: Tensor.Tensor) -> Result[Trace.Compiled]
  Trace.compile(
    [Trace.Spec(shape: [k], dtype: Tensor.Dtype.F32, backend: CPU)],
    ->(a: List[Tensor.Tensor]) { ok([try Tensor.mul(a[0], Trace.constant(w))]) },
  )
end

## The values of one run of `compiled` over `x`.
def run(compiled: Trace.Compiled, x: Tensor.Tensor) -> Result[List[Float]]
  out = try Trace.run(compiled, [x])
  Support.floats_of(out[0])
end

def refused(r: Result[List[Float]], text: String) -> Result[Void]
  case r
    ok(_) -> error("ran where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

def close(t: Tensor.Tensor) -> Result[Void]
  Native.close(try Ggml.handle_of(t))
  ok()
end

## Whether the graph of `compiled` still holds the constants of `compiled` from an earlier run.
def kept?(compiled: Trace.Compiled) -> Result[Bool]
  p = compiled.program.unwrap!
  Native.arena do
    lease = try GgmlRaw.acquire(try Ggml.code_of(p.device, "test"), p.key, p.nodes.size + 1, p.keep)
    kept = GgmlRaw.resume(lease, p.pin)
    Native.close(lease)
    ok(kept)
  end
end

## 40 graphs of other shapes, which take the slots of every graph that is not pinned.
def crowd(first: Int) -> Result[Void]
  for k in first...(first + 40)
    x = try floats(ramp(k, 1.0), [1, k])
    _ = try Trace.eval_many(
      [x],
      ->(a: List[Tensor.Tensor]) { ok([try Tensor.matmul(a[0], Trace.constant(try floats(ramp(k, 2.0), [k, 1])))]) },
    )
  end

  ok()
end

## Two functions with the same graph and other weights share one compiled graph: each run reads its own weights,
## whichever ran before.
def test_two_functions_of_one_graph_read_their_own_weights() -> Result[Void]
  return ok() if !try Support.ggml_present()

  k = 41
  x = try floats(ramp(k, 1.0), [k])
  before = try Lower.compiles()
  first = try scaled(k, try floats(ramp(k, 10.0), [k]))
  second = try scaled(k, try floats(ramp(k, -30.0), [k]))
  try Test.eq(try Lower.compiles(), before + 1)
  for _ in 0...3
    try Test.eq(try run(first, x), products(ramp(k, 1.0), ramp(k, 10.0)))
    try Test.eq(try run(first, x), products(ramp(k, 1.0), ramp(k, 10.0)))
    try Test.eq(try run(second, x), products(ramp(k, 1.0), ramp(k, -30.0)))
  end

  try Test.eq(try Lower.compiles(), before + 1)
  Trace.close(first)
  Trace.close(second)
  ok()
end

## Two functions of other graphs read one weight. Each keeps it for itself: the end of one changes nothing for the
## other, and once the weight is closed a run of either is an error, though the graph had it bound.
def test_a_weight_closed_between_runs_is_an_error_for_every_function_that_kept_it() -> Result[Void]
  return ok() if !try Support.ggml_present()

  w = try floats(ramp(42, 3.0), [42])
  x = try floats(ramp(42, 1.0), [42])
  doubled = try scaled(42, w)
  shifted = try Trace.compile(
    [Trace.spec_of(x)],
    ->(a: List[Tensor.Tensor]) { ok([try Tensor.add(a[0], Trace.constant(w))]) },
  )

  sums = (0...42).map { |i| 4.0 + 2.0 * i.to_f }
  try Test.eq(try run(doubled, x), products(ramp(42, 1.0), ramp(42, 3.0)))
  try Test.eq(try run(shifted, x), sums)
  try Test.assert(try kept?(doubled), "a run left nothing bound")
  Trace.close(doubled)
  try Test.eq(try run(shifted, x), sums)
  try Test.assert(try kept?(shifted), "the end of another function took the weight")
  try close(w)
  try Test.assert(!try kept?(shifted), "a closed weight stayed bound")
  try refused(run(shifted, x), "closed")
  refused(run(doubled, x), "closed")
end

## A kept leaf that is bound to another tensor reads that tensor in this run, and the next run of the function has
## its own weight again.
def test_a_kept_leaf_bound_again_reads_the_new_tensor() -> Result[Void]
  return ok() if !try Support.ggml_present()

  k = 43
  x = try floats(ramp(k, 1.0), [k])
  other = try floats(ramp(k, -7.0), [k])
  compiled = try scaled(k, try floats(ramp(k, 5.0), [k]))
  try Test.eq(try run(compiled, x), products(ramp(k, 1.0), ramp(k, 5.0)))
  p = compiled.program.unwrap!
  got: Result[List[Float]] = Native.arena do
    lease = try GgmlRaw.acquire(try Ggml.code_of(p.device, "test"), p.key, p.nodes.size + 1, p.keep)
    try Test.assert(GgmlRaw.resume(lease, p.pin), "a run left nothing bound")
    try GgmlRaw.bind(lease, p.kept[0], try Ggml.handle_of(other))
    try GgmlRaw.bind(lease, p.fed[0], try Ggml.handle_of(x))
    out = try GgmlRaw.output(lease, 0, false)
    try GgmlRaw.run(lease, p.fills)
    Native.close(lease)
    Support.floats_of(Ggml.tensor_of(p.device, p.shapes[0], p.dtypes[0], out))
  end

  try Test.eq(try got, products(ramp(k, 1.0), ramp(k, -7.0)))
  try Test.assert(!try kept?(compiled), "a leaf bound again stayed kept")
  try Test.eq(try run(compiled, x), products(ramp(k, 1.0), ramp(k, 5.0)))
  Trace.close(compiled)
  ok()
end

## A function whose pin ended keeps nothing, so its graph is evicted as any other and built again by the next run,
## which reads the same weight; a weight closed after that is an error.
def test_an_evicted_graph_keeps_nothing() -> Result[Void]
  return ok() if !try Support.ggml_present()

  k = 44
  x = try floats(ramp(k, 1.0), [k])
  w = try floats(ramp(k, 2.0), [k])
  compiled = try scaled(k, w)
  try Test.eq(try run(compiled, x), products(ramp(k, 1.0), ramp(k, 2.0)))
  Trace.close(compiled)
  try Test.assert(!try kept?(compiled), "a function without a pin kept its weights")
  try Test.eq(try run(compiled, x), products(ramp(k, 1.0), ramp(k, 2.0)))
  try Test.assert(!try kept?(compiled), "a run without a pin kept its weights")
  try crowd(500)
  before = try Lower.compiles()
  try Test.eq(try run(compiled, x), products(ramp(k, 1.0), ramp(k, 2.0)))
  try Test.eq(try Lower.compiles(), before + 1)
  try close(w)
  refused(run(compiled, x), "closed")
end

## The cache a `set_rows` writes in place is bound by every run, beside a constant that is kept: a second cache is
## written by its own run and the first is left alone. A cache that is also the kept constant is refused.
def test_a_cache_written_in_place_is_bound_by_every_run() -> Result[Void]
  return ok() if !try Support.ggml_present()

  bias = try floats(ramp(3, 100.0), [1, 1, 3])
  rows = try floats(ramp(3, 1.0), [1, 1, 3])
  compiled = try Trace.compile(
    [
      Trace.Spec(shape: [1, 4, 3], dtype: Tensor.Dtype.F32, backend: CPU),
      Trace.spec_of(rows),
      Trace.Spec(shape: [1], dtype: Tensor.Dtype.I32, backend: CPU),
    ],
    ->(xs: List[Tensor.Tensor]) {
      ok([try Tensor.set_rows(xs[0], try Tensor.add(xs[1], Trace.constant(bias)), xs[2])])
    },
  )

  first = try floats(ramp(12, 0.0), [1, 4, 3])
  second = try floats(ramp(12, 50.0), [1, 4, 3])
  _ = try Trace.run(compiled, [first, rows, try Tensor.from_list([Tensor.Values.Int(2)], [1], Tensor.Dtype.I32, CPU)])
  try Test.assert(try kept?(compiled), "a run left nothing bound")
  _ = try Trace.run(compiled, [second, rows, try Tensor.from_list([Tensor.Values.Int(0)], [1], Tensor.Dtype.I32, CPU)])
  try Test.eq(try Support.floats_of(first), ramp(6, 0.0) + [101.0, 103.0, 105.0] + ramp(3, 9.0))
  try Test.eq(try Support.floats_of(second), [101.0, 103.0, 105.0] + ramp(9, 53.0))

  same = try floats(ramp(12, 0.0), [1, 4, 3])
  aliased = try Trace.compile(
    [Trace.spec_of(same), Trace.spec_of(rows), Trace.Spec(shape: [1], dtype: Tensor.Dtype.I32, backend: CPU)],
    ->(xs: List[Tensor.Tensor]) {
      ok([try Tensor.set_rows(xs[0], xs[1], xs[2]), try Tensor.neg(Trace.constant(same))])
    },
  )

  at = try Tensor.from_list([Tensor.Values.Int(1)], [1], Tensor.Dtype.I32, CPU)
  other = try floats(ramp(12, 0.0), [1, 4, 3])
  _ = try Trace.run(aliased, [other, rows, at])
  case Trace.run(aliased, [same, rows, at])
    ok(_) -> error("a cache that is a kept constant was written in place")
    error(e) -> Test.assert(e.message.include?("another input"), "the error said \"#{e.message}\"")
  end
end

## The graph that ran last runs again without a new plan of its memory: each run still reads the input it was given
## and writes its own output, while the tensors of the run before are open (a stale pointer would read and write
## those), and after other graphs took the plan.
def test_a_graph_run_again_reads_the_tensors_of_its_own_run() -> Result[Void]
  return ok() if !try Support.ggml_present()

  k = 45
  compiled = try Trace.compile(
    [Trace.Spec(shape: [k, 2], dtype: Tensor.Dtype.F32, backend: CPU)],
    ->(a: List[Tensor.Tensor]) {
      turned = try Tensor.reshape(a[0], [2, k])
      ok([try Tensor.mul(turned, Trace.constant(try floats(ramp(2 * k, 1.0), [2, k]))), try Tensor.transpose(a[0])])
    },
  )

  earlier: List[Tensor.Tensor] = []
  for round in 0...4
    if round == 2
      try crowd(600)
    end

    first = 10.0 * (round + 1).to_f
    x = try floats(ramp(2 * k, first), [k, 2])
    out = try Trace.run(compiled, [x])
    for t in earlier
      try close(t)
    end

    try Test.eq(try Support.floats_of(out[0]), products(ramp(2 * k, first), ramp(2 * k, 1.0)))
    turned = (0...(2 * k)).map { |i| first + (2 * (i % k) + i / k).to_f }
    try Test.eq(try Support.floats_of(out[1]), turned)
    earlier = [x] + out
  end

  Trace.close(compiled)
  ok()
end
