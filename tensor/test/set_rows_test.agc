## `Tensor.set_rows`: rows of a cache replaced at positions. On every backend but ggml the answer is a new tensor; in a
## function run on ggml (`Trace.run`) the cache is written in place and the result is that very tensor. The refusals
## of a function that cannot write in place soundly name the node.
##
## Without the ggml module the ggml checks pass by asserting nothing, unless `TENSOR_REQUIRE_GGML` is set.

import Ggml from "../src/ggml"
import Support from "support"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
REF = Tensor.Backend.Reference

private

## The row-major values of a `[lead..., rows, width]` cache once `fresh` (`[lead..., n, width]`) is written at `positions`.
def expected(
  cache: List[Float],
  lead: Int,
  rows: Int,
  width: Int,
  fresh: List[Float],
  positions: List[Int],
) -> List[Float]
  out = cache
  n = positions.size
  for l in 0...lead
    for i in 0...n
      for d in 0...width
        out = out.set((l * rows + positions[i]) * width + d, fresh[(l * n + i) * width + d])
      end
    end
  end

  out
end

def counting(count: Int) -> List[Float] = (0...count).map { |i| i.to_f + 1.0 }

def floats(xs: List[Float], shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, backend)
end

def at(positions: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(positions.map { |i| Tensor.Values.Int(i) }, [positions.size], Tensor.Dtype.I32, backend)
end

def refused(r: Result[Tensor.Tensor], text: String) -> Result[Void]
  case r
    ok(_) -> error("ran where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

## A compiled function error, or a run error, that says `text`.
def failed(r: Result[List[Tensor.Tensor]], text: String) -> Result[Void]
  case r
    ok(_) -> error("ran where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

## A function that does not compile, and says `text`.
def dead(r: Result[Trace.Compiled], text: String) -> Result[Void]
  case r
    ok(_) -> error("compiled where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

def specs_of(ts: List[Tensor.Tensor]) -> List[Trace.Spec] = ts.map { |t| Trace.spec_of(t) }

def trio(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([
    try floats(counting(24), [2, 4, 3], backend),
    try floats([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], [2, 1, 3], backend),
    try at([2], backend),
  ])
end

def test_the_rows_are_replaced_on_the_reference_in_ranks_two_to_four() -> Result[Void]
  for shape in [[4, 3], [2, 4, 3], [1, 2, 4, 3]]
    lead = shape.take(shape.size - 2).reduce(1) do |acc, d| acc * d end
    cache = counting(lead * 12)
    fresh = (0...(lead * 2 * 3)).map { |i| -1.0 - i.to_f }
    got = try Tensor.set_rows(
      try floats(cache, shape, REF),
      try floats(fresh, shape.take(shape.size - 2) + [2, 3], REF),
      try at([3, 0], REF),
    )

    try Test.eq(try Support.floats_of(got), expected(cache, lead, 4, 3, fresh, [3, 0]))
  end

  ok()
end

## The cache a reference run is given is not written: a copy is the answer there.
def test_a_reference_run_leaves_the_cache_alone() -> Result[Void]
  args = try trio(REF)
  compiled = try Trace.compile(
    specs_of(args),
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[1], xs[2])]) },
  )

  out = try Trace.run(compiled, args)
  try Test.eq(
    try Support.floats_of(out[0]),
    expected(counting(24), 2, 4, 3, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0], [2]),
  )

  Test.eq(try Support.floats_of(args[0]), counting(24))
end

def test_set_rows_refuses_what_it_cannot_write() -> Result[Void]
  cache = try floats(counting(12), [4, 3], REF)
  rows = try floats([1.0, 2.0, 3.0], [1, 3], REF)
  try refused(Tensor.set_rows(try floats(counting(4), [4], REF), rows, try at([0], REF)), "rank 2 to 4")
  try refused(Tensor.set_rows(cache, try floats([1.0, 2.0], [1, 2], REF), try at([0], REF)), "rows of shape")
  try refused(Tensor.set_rows(cache, rows, try at([0, 1], REF)), "rows of shape")
  try refused(Tensor.set_rows(cache, rows, try floats([0.0], [1], REF)), "I32")
  try refused(Tensor.set_rows(cache, rows, try at([4], REF)), "indices in")
  refused(Tensor.set_rows(cache, rows, try at([-1], REF)), "indices in")
end

def test_set_rows_has_no_gradient() -> Result[Void]
  cache = try floats(counting(12), [4, 3], REF)
  rows = try floats([1.0, 2.0, 3.0], [1, 3], REF)
  where = try at([1], REF)
  Test.error?(
    Trace.value_and_grad([cache]) do |xs|
      Tensor.sum(try Tensor.set_rows(xs[0], Trace.constant(rows), Trace.constant(where)))
    end,
  )
end

## The cache is written where it lies: the tensor that comes back is the one given, whose values change, step after
## step, and a node may read the result.
def test_a_ggml_run_writes_the_cache_in_place() -> Result[Void]
  return ok() if !try Support.ggml_present()

  args = try trio(CPU)
  compiled = try Trace.compile(
    specs_of(args),
    ->(xs: List[Tensor.Tensor]) {
      y = try Tensor.set_rows(xs[0], xs[1], xs[2])
      ok([y, try Tensor.sum(y)])
    },
  )

  fresh = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
  want = counting(24)
  for position in [2, 0, 3]
    args = [args[0], args[1], try at([position], CPU)]
    out = try Trace.run(compiled, args)
    want = expected(want, 2, 4, 3, fresh, [position])
    try Test.eq(try Support.floats_of(args[0]), want)
    try Test.eq(try Support.floats_of(out[0]), want)
    try Test.assert(
      (try Support.floats_of(out[1]))[0] - want.reduce(0.0) do |acc, x| acc + x end < 0.001,
      "the sum of the result",
    )
  end

  # The result is the tensor given: closing one closes the other.
  out = try Trace.run(compiled, args)
  Native.close(try Ggml.handle_of(args[0]))
  Test.assert(Native.closed?(try Ggml.handle_of(out[0])), "the result is not the cache given")
end

## The same values on every backend that computes: ggml writes in place what the reference answers as a new tensor.
def test_ggml_and_the_reference_agree_on_several_rows() -> Result[Void]
  return ok() if !try Support.ggml_present()

  fresh = (0...12).map { |i| 10.0 + i.to_f }
  cache = counting(24)
  args = [try floats(cache, [2, 4, 3], CPU), try floats(fresh, [2, 2, 3], CPU), try at([1, 3], CPU)]
  compiled = try Trace.compile(
    specs_of(args),
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[1], xs[2])]) },
  )

  _ = try Trace.run(compiled, args)
  reference = try Tensor.set_rows(
    try floats(cache, [2, 4, 3], REF),
    try floats(fresh, [2, 2, 3], REF),
    try at([1, 3], REF),
  )

  Test.eq(try Support.floats_of(args[0]), try Support.floats_of(reference))
end

def test_a_position_outside_the_cache_is_an_error_of_the_run() -> Result[Void]
  return ok() if !try Support.ggml_present()

  args = try trio(CPU)
  compiled = try Trace.compile(
    specs_of(args),
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[1], xs[2])]) },
  )

  try failed(Trace.run(compiled, [args[0], args[1], try at([4], CPU)]), "outside the rows")
  failed(Trace.run(compiled, [args[0], args[1], try at([-1], CPU)]), "outside the rows")
end

## What a function that would write in place unsoundly is refused for, at compile time, naming the nodes.
def test_a_function_that_cannot_write_in_place_is_refused() -> Result[Void]
  return ok() if !try Support.ggml_present()

  args = try trio(CPU)
  specs = specs_of(args)
  # the cache is read by another node
  try dead(
    Trace.compile(
      specs,
      ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[1], xs[2]), try Tensor.add(xs[0], xs[0])]) },
    ),
    "reads node",
  )

  # the cache is not an input
  try dead(
    Trace.compile(
      specs,
      ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(try Tensor.add(xs[0], xs[0]), xs[1], xs[2])]) },
    ),
    "must be an input",
  )

  try dead(
    Trace.compile(
      specs,
      ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(Trace.constant(args[0]), xs[1], xs[2])]) },
    ),
    "must be an input",
  )

  # the result is not an output
  try dead(
    Trace.compile(
      specs,
      ->(xs: List[Tensor.Tensor]) { ok([try Tensor.sum(try Tensor.set_rows(xs[0], xs[1], xs[2]))]) },
    ),
    "must be an output",
  )

  # the cache is also an output
  try dead(
    Trace.compile(specs, ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[1], xs[2]), xs[0]]) }),
    "also an output",
  )

  # the cache is written twice: the second set_rows reads it
  try dead(
    Trace.compile(
      specs,
      ->(xs: List[Tensor.Tensor]) {
        ok([try Tensor.set_rows(xs[0], xs[1], xs[2]), try Tensor.set_rows(xs[0], xs[1], xs[2])])
      },
    ),
    "reads node",
  )

  # the rows are the cache itself: the kernel would read what it writes
  full = [args[0], args[1], try at([0, 1, 2, 3], CPU)]
  try dead(
    Trace.compile(specs_of(full), ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[0], xs[2])]) }),
    "reads node",
  )

  # one result listed twice
  dead(
    Trace.compile(
      specs,
      ->(xs: List[Tensor.Tensor]) {
        y = try Tensor.set_rows(xs[0], xs[1], xs[2])
        ok([y, y])
      },
    ),
    "listed twice",
  )
end

## `into` cannot name a tensor for the output that is the cache, and the cache cannot be bound to two inputs.
def test_the_run_refuses_a_cache_that_is_given_twice_or_an_into_for_it() -> Result[Void]
  return ok() if !try Support.ggml_present()

  args = try trio(CPU)
  compiled = try Trace.compile(
    specs_of(args),
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[1], xs[2])]) },
  )

  try failed(Trace.run(compiled, args, [some(try floats(counting(24), [2, 4, 3], CPU))]), "in place")
  other = try Trace.compile(
    specs_of([args[0], args[0], args[1], args[2]]),
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[2], xs[3]), try Tensor.add(xs[1], xs[1])]) },
  )

  failed(Trace.run(other, [args[0], args[0], args[1], args[2]]), "another input")
end

## `Trace.eval` is functional: it writes a copy, and the cache given is left as it was.
def test_eval_leaves_the_cache_alone() -> Result[Void]
  return ok() if !try Support.ggml_present()

  cache = counting(24)
  fresh = (0...6).map { |i| 10.0 + i.to_f }
  on_ggml = [try floats(cache, [2, 4, 3], CPU), try floats(fresh, [2, 1, 3], CPU), try at([2], CPU)]
  got = try Trace.eval(on_ggml) do |xs| Tensor.set_rows(xs[0], xs[1], xs[2]) end

  want = try Tensor.set_rows(try floats(cache, [2, 4, 3], REF), try floats(fresh, [2, 1, 3], REF), try at([2], REF))
  try Test.eq(try Support.floats_of(got), try Support.floats_of(want))
  try Test.eq(try Support.floats_of(on_ggml[0]), cache)
  Native.close(try Ggml.handle_of(got))
  Test.assert(!Native.closed?(try Ggml.handle_of(on_ggml[0])), "eval gave the cache back")
end

## One plan runs on a second cache: each is written by its own run and the first is not touched by the second.
def test_a_plan_runs_on_a_second_cache() -> Result[Void]
  return ok() if !try Support.ggml_present()

  first = try trio(CPU)
  compiled = try Trace.compile(
    specs_of(first),
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[1], xs[2])]) },
  )

  _ = try Trace.run(compiled, first)
  written = try Support.floats_of(first[0])
  second = [try floats(counting(24).map { |x| -x }, [2, 4, 3], CPU), first[1], try at([1], CPU)]
  _ = try Trace.run(compiled, second)
  try Test.eq(try Support.floats_of(first[0]), written)
  Test.assert(try Support.floats_of(second[0]) != written, "the second cache was not written")
end

## The cache is consumed by a run that fails: the guard of the positions comes after the write.
def test_a_failed_run_still_consumes_the_cache() -> Result[Void]
  return ok() if !try Support.ggml_present()

  args = try trio(CPU)
  compiled = try Trace.compile(
    specs_of(args),
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.set_rows(xs[0], xs[1], xs[2])]) },
  )

  try failed(Trace.run(compiled, [args[0], args[1], try at([7], CPU)]), "outside the rows")
  Test.assert(try Support.floats_of(args[0]) != counting(24), "the cache was left alone by a failed run")
end
