## `Trace.compile` and `Trace.run` (`trace.agc`, `src/lower.agc`): a function traced and lowered once and run many times,
## on ggml, on the reference backend and from several tasks; the refusals of inputs that do not match; and a training
## step (forward, gradient and update) as one compiled function.
##
## Without the ggml module the ggml checks pass by asserting nothing, unless `TENSOR_REQUIRE_GGML` is set (CI sets it),
## and then they fail.

import Lower from "../src/lower"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
REF = Tensor.Backend.Reference
TOL = 0.000001

## How many times `f` of a test ran: a compiled function traces once.
CALLS: ATS[String, Int] = ATS.new()

private

def present() -> Result[Bool]
  return ok(true) if Tensor.available(CPU)
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok(false) if !flag.ok?
  return ok(false) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end

def uniform(count: Int, seed: Int) -> List[Float]
  s = seed
  out: List[Float] = []
  for _ in 0...count
    s = (s * 1103515245 + 12345) % 2147483648
    out = out.push((s.to_f / 2147483648.0) * 2.0 - 1.0)
  end

  out
end

def floats(xs: List[Float], shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, backend)
end

def values(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(x) -> out = out.push(x)
      _ -> return error("a float tensor holds #{v}")
    end
  end

  ok(out)
end

## The largest gap between two tensors' values.
def gap(a: Tensor.Tensor, b: Tensor.Tensor) -> Result[Float]
  xs = try values(a)
  ys = try values(b)
  return error("sizes #{xs.size} and #{ys.size}") if xs.size != ys.size
  worst = 0.0
  for i in 0...xs.size
    if (xs[i] - ys[i]).abs > worst
      worst = (xs[i] - ys[i]).abs
    end
  end

  ok(worst)
end

## `tanh(x w) + x w`, counting its traces under `name`.
def counted(name: String) -> Fn[List[Tensor.Tensor], Result[List[Tensor.Tensor]]]
  ->(a: List[Tensor.Tensor]) {
    CALLS.add(name, 1)
    y = try Tensor.matmul(a[0], a[1])
    ok([try Tensor.add(try Tensor.tanh(y), y)])
  }
end

def specs_of(ts: List[Tensor.Tensor]) -> List[Trace.Spec] = ts.map { |t| Trace.spec_of(t) }

## A `[3, k]` and a `[k, 5]` tensor: each test takes its own `k`, so its graph is its own in the cache.
def pair(backend: Tensor.Backend, seed: Int, k: Int = 4) -> Result[List[Tensor.Tensor]]
  ok([try floats(uniform(3 * k, seed), [3, k], backend), try floats(uniform(5 * k, seed + 1), [k, 5], backend)])
end

def refused(r: Result[List[Tensor.Tensor]], text: String) -> Result[Void]
  case r
    ok(_) -> error("ran where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

def test_a_compiled_function_traces_and_compiles_once() -> Result[Void]
  return ok() if !try present()

  first = try pair(CPU, 1, 6)
  before = try Lower.compiles()
  compiled = try Trace.compile(specs_of(first), counted("once"))
  for seed in [1, 5, 9, 13]
    got = try Trace.run(compiled, try pair(CPU, seed, 6))
    want = try Trace.eval_many(try pair(REF, seed, 6), counted("reference"))
    try Test.assert(try gap(got[0], want[0]) < TOL, "run #{seed} differs from the reference")
  end

  try Test.eq(CALLS.get("once").unwrap!, 1)
  Test.eq(try Lower.compiles(), before + 1)
end

def test_the_reference_backend_runs_a_compiled_function_too() -> Result[Void]
  first = try pair(REF, 1)
  compiled = try Trace.compile(specs_of(first), counted("ref"))
  for seed in [2, 3]
    got = try Trace.run(compiled, try pair(REF, seed))
    want = try Trace.eval_many(try pair(REF, seed), counted("ref-eval"))
    try Test.eq(try values(got[0]), try values(want[0]))
  end

  Test.eq(CALLS.get("ref").unwrap!, 1)
end

def test_inputs_that_do_not_match_the_specs_are_errors() -> Result[Void]
  return ok() if !try present()

  compiled = try Trace.compile(specs_of(try pair(CPU, 1)), counted("mismatch"))
  other = try pair(CPU, 2)
  try refused(Trace.run(compiled, [other[0]]), "takes 2 inputs")
  try refused(Trace.run(compiled, [other[1], other[0]]), "input 0")
  try refused(Trace.run(compiled, [other[0], other[0]]), "input 1")
  wide = [try floats(uniform(15, 3), [3, 5], CPU), try floats(uniform(25, 4), [5, 5], CPU)]
  try refused(Trace.run(compiled, wide), "[3, 4]")
  ints = Tensor.from_list([Tensor.Values.Int(1), Tensor.Values.Int(2)], [2], Tensor.Dtype.I32, CPU)
  try refused(Trace.run(compiled, [try ints, other[1]]), "F32")
  try refused(Trace.run(compiled, try pair(REF, 1)), "Reference")
  try Test.eq(CALLS.get("mismatch").unwrap!, 1)
  got = try Trace.run(compiled, other)
  Test.eq(Tensor.shape(got[0]), [3, 5])
end

## `x w` with `w` read from outside the inputs (a constant of the trace) and with `w` as an input.
def test_a_constant_stays_and_an_input_changes() -> Result[Void]
  return ok() if !try present()

  x = try floats(uniform(12, 1), [3, 4], CPU)
  w1 = try floats(uniform(20, 2), [4, 5], CPU)
  w2 = try floats(uniform(20, 3), [4, 5], CPU)
  fixed = try Trace.compile(
    specs_of([x]),
    ->(a: List[Tensor.Tensor]) { ok([try Tensor.matmul(a[0], Trace.constant(w1))]) },
  )

  open = try Trace.compile(specs_of([x, w1]), ->(a: List[Tensor.Tensor]) { ok([try Tensor.matmul(a[0], a[1])]) })
  want1 = try Trace.run(open, [x, w1])
  want2 = try Trace.run(open, [x, w2])
  try Test.assert(try gap(want1[0], want2[0]) > 0.01, "the two weights give the same product")
  try Test.assert(try gap((try Trace.run(fixed, [x]))[0], want1[0]) < TOL, "the constant is not w1")
  Test.assert(try gap((try Trace.run(fixed, [x]))[0], want1[0]) < TOL, "the constant changed")
end

## A step of gradient descent on `mean((x w + b - y)^2)`: the parameters are inputs, and the step answers the loss and
## the updated parameters.
def descent(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  w = a[0]
  b = a[1]
  diff = try Tensor.sub(try Tensor.linear(a[2], w, some(b)), a[3])
  loss = try Tensor.mean(try Tensor.mul(diff, diff))
  grads = try Trace.gradients(loss, [w, b])
  rate = Tensor.Values.Float(0.1)
  ok([
    loss,
    try Tensor.sub(
      w,
      try Tensor.mul(grads[0], try Tensor.full(Tensor.shape(w), rate, Tensor.Dtype.F32, Tensor.backend(w))),
    ),
    try Tensor.sub(
      b,
      try Tensor.mul(grads[1], try Tensor.full(Tensor.shape(b), rate, Tensor.Dtype.F32, Tensor.backend(b))),
    ),
  ])
end

def training_inputs(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([
    try floats(uniform(12, 21), [4, 3], backend),
    try floats(uniform(3, 22), [3], backend),
    try floats(uniform(32, 23), [8, 4], backend),
    try floats(uniform(24, 24), [8, 3], backend),
  ])
end

## `rounds` steps of `descent` on `backend`: the loss of each, then the parameters.
def train(backend: Tensor.Backend, rounds: Int) -> Result[List[Tensor.Tensor]]
  args = try training_inputs(backend)
  step = try Trace.compile(specs_of(args), &descent)
  losses: List[Float] = []
  for _ in 0...rounds
    outs = try Trace.run(step, args)
    args = [outs[1], outs[2], args[2], args[3]]
    losses = losses.push((try values(outs[0]))[0])
  end

  ok([args[0], args[1], try floats(losses, [rounds], backend)])
end

def test_a_training_step_is_one_compiled_function() -> Result[Void]
  return ok() if !try present()

  got = try train(CPU, 6)
  want = try train(REF, 6)
  for i in 0...3
    try Test.assert(try gap(got[i], want[i]) < TOL, "output #{i} differs from the reference")
  end

  losses = try values(got[2])
  Test.assert(losses[5] < losses[0], "the loss did not fall: #{losses}")
end

## The values of the last of `rounds` runs of `compiled`.
def runs_of(compiled: Trace.Compiled, rounds: Int) -> Result[List[Float]]
  last: List[Float] = []
  for _ in 0...rounds
    got = try Trace.run(compiled, try pair(CPU, 1, 7))
    last = try values(got[0])
  end

  ok(last)
end

## Four tasks run one compiled function at once: each gets the values it gets alone, nothing is traced again, and a
## graph that another task holds is built again (at most once per task: `Lower.compiles()` rises by 1 to 4).
def test_tasks_run_one_compiled_function() -> Result[Void]
  return ok() if !try present()

  start = try Lower.compiles()
  compiled = try Trace.compile(specs_of(try pair(CPU, 1, 7)), counted("tasks"))
  alone = try runs_of(compiled, 1)
  tasks = (0...4).to_a.map { |_| Task.spawn do runs_of(compiled, 8) end }
  for t in tasks
    got = try Task.await(t)
    try Test.eq(got.size, alone.size)
    for i in 0...got.size
      try Test.assert((got[i] - alone[i]).abs < TOL, "a task got #{got[i]} at #{i}, alone it is #{alone[i]}")
    end
  end

  built = try Lower.compiles() - start
  try Test.assert(built >= 1 and built <= 4, "the graph was compiled #{built} times")
  Test.eq(CALLS.get("tasks").unwrap!, 1)
end

## A constant the function was traced with is closed: the run is an error naming it, not a halt.
def test_a_closed_constant_is_an_error() -> Result[Void]
  return ok() if !try present()

  x = try floats(uniform(12, 1), [3, 4], CPU)
  w = try floats(uniform(20, 2), [4, 5], CPU)
  compiled = try Trace.compile(
    specs_of([x]),
    ->(a: List[Tensor.Tensor]) { ok([try Tensor.matmul(a[0], Trace.constant(w))]) },
  )

  _ = try Trace.run(compiled, [x])
  case w.handle
    Tensor.Handle.Ggml(h) -> Native.close(h)
    _ -> nil
  end

  refused(Trace.run(compiled, [x]), "closed")
end

## `x` times a `[k, 1]` ramp: a graph of its own for each `k`, to fill the cache of 32 graphs.
def ramp(k: Int) -> Result[Void]
  x = try floats(uniform(k, k), [1, k], CPU)
  _ = try Trace.eval_many(
    [x],
    ->(a: List[Tensor.Tensor]) {
      w = Trace.constant(try floats(uniform(k, 1), [k, 1], CPU))
      ok([try Tensor.matmul(a[0], w)])
    },
  )

  ok()
end

## 40 graphs of other shapes take 40 slots of a cache of 32: a pinned graph stays, an unpinned one is evicted and
## compiled again.
def test_a_pinned_graph_is_never_evicted() -> Result[Void]
  return ok() if !try present()

  pinned = try Trace.compile(specs_of(try pair(CPU, 1, 9)), counted("pinned"))
  loose = try Trace.compile(specs_of(try pair(CPU, 1, 10)), counted("loose"))
  Trace.close(loose)
  for k in 100...140
    try ramp(k)
  end

  before = try Lower.compiles()
  _ = try Trace.run(pinned, try pair(CPU, 2, 9))
  try Test.eq(try Lower.compiles(), before)
  _ = try Trace.run(loose, try pair(CPU, 2, 10))
  try Test.eq(try Lower.compiles(), before + 1)
  Trace.close(pinned)
  ok()
end

## Outputs written into tensors that exist: the same tensors come back with the values a fresh run gives, a step
## after the other, with two sets taking turns.
def test_outputs_are_written_into_the_tensors_given() -> Result[Void]
  return ok() if !try present()

  args = try pair(CPU, 3, 8)
  compiled = try Trace.compile(specs_of(args), counted("into"))
  _ = try Trace.run(compiled, args)
  spare = [some(try Tensor.zeros([3, 5], Tensor.Dtype.F32, CPU))]
  for seed in [4, 5, 6]
    more = try pair(CPU, seed, 8)
    want = try Trace.run(compiled, more)
    got = try Trace.run(compiled, more, spare)
    try Test.assert(try gap(got[0], want[0]) < TOL, "run #{seed} into a tensor differs from a fresh run")
    try Test.assert(try gap(got[0], spare[0].unwrap!) == 0.0, "the tensor given is not the output")
  end

  refused(Trace.run(compiled, args, [some(args[0])]), "output 0")
end
