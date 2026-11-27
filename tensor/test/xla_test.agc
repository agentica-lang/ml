## `Backend.Xla` (`src/xla.agc`, `native/xla.c`): a traced MLP and a training step compiled by XLA against the
## reference, the eager refusals, and the device check.
##
## Without the plugin (`tools/install_xla.agc`) the checks pass by asserting nothing, unless `TENSOR_REQUIRE_XLA` is
## set (the `xla` job of CI sets it), and then they fail.

import Tensor from "../tensor"
import Trace from "../trace"
import Xla from "../src/xla"
import Support from "xla_support"

private

def test_an_mlp_forward_compiled_by_xla_matches_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  got = try run_forward(Support.XLA)
  want = try run_forward(Support.REF)
  Support.close(got, want)
end

## Three chained SGD steps (the loss, then the new parameters, fed back in): the step of `Train.fit`, without the
## optimizer's fused node.
def test_chained_training_steps_compiled_by_xla_match_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  got = try train(Support.XLA)
  want = try train(Support.REF)
  Support.close(got, want)
end

## `floats32` uploads a packed list: the tensor `floats` makes of the same values, widened.
def test_floats32_uploads_a_packed_list_to_xla() -> Result[Void]
  return ok() if !try Support.present()
  packed: List[Float32] = [1.5, -2.0, 3.0, 0.1]
  got = try Tensor.floats32(packed, [2, 2], Support.XLA)
  try Test.eq(
    try Tensor.to_list(got),
    try Tensor.to_list(try Tensor.floats(packed.map { |x| x.to_f }, [2, 2], Support.REF)),
  )

  Test.error?(Tensor.floats32(packed, [3], Support.XLA))
end

## `value_and_grad` on XLA compiles the graph for the call (`Trace.evaluate`).
def test_value_and_grad_on_xla_matches_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  ys = [0, 3, 1, 2, 3, 0]
  got = try Trace.value_and_grad(try mlp_args(Support.XLA), mlp_loss(try Support.labels(ys, Support.XLA)))
  want = try Trace.value_and_grad(try mlp_args(Support.REF), mlp_loss(try Support.labels(ys, Support.REF)))
  Support.close([got.value] + got.grads, [want.value] + want.grads)
end

## Steps that close what they replace keep the number of live buffers flat: a run keeps nothing of its inputs.
def test_steps_that_close_old_tensors_keep_no_buffer_alive() -> Result[Void]
  return ok() if !try Support.present()
  a = try mlp_args(Support.XLA)
  params = a.drop(1)
  fixed = [try Support.floats([0.5], [], Support.XLA), a[0], try Support.labels([0, 3, 1, 2, 3, 0], Support.XLA)]
  step = try Trace.compile((params + fixed).map { |t| Trace.spec_of(t) }, &sgd_step)
  first = 0
  for i in 0...20
    outs = try Trace.run(step, params + fixed)
    for t in [outs[0]] + params
      free(t)
    end

    params = outs.drop(1)
    if i == 0
      first = Xla.buffers()
    end
  end

  # The counter is the process's: a collection may free a tensor of another test meanwhile, never add one.
  Test.assert(Xla.buffers() <= first, "#{Xla.buffers()} buffers are alive after the steps, #{first} after the first")
end

## A program compiled once serves a retrace of the same function and a run after `Trace.close`, as on ggml: both
## find it in the cache of the process, and `eval` per step compiles once.
def test_a_retrace_and_a_closed_function_compile_nothing_again() -> Result[Void]
  return ok() if !try Support.present()
  args = try mlp_args(Support.XLA)
  specs = args.map { |t| Trace.spec_of(t) }
  f = try Trace.compile(specs, ->(xs: List[Tensor.Tensor]) { ok([try forward(xs)]) })
  want = try Trace.run(f, args)
  before = Xla.compiles()
  Trace.close(f)
  try Support.close(try Trace.run(f, args), want)
  g = try Trace.compile(specs, ->(xs: List[Tensor.Tensor]) { ok([try forward(xs)]) })
  try Support.close(try Trace.run(g, args), want)
  for _ in 0...3
    _ = try Trace.eval(args, &forward)
  end

  Test.assert(Xla.compiles() - before <= 1, "#{Xla.compiles() - before} compiles, at most the one of eval")
end

## `gather` and `adam_step` (one node in a trace) as a compiled program, against the reference: the rows of a
## dataset at the ids of a batch, and an Adam update whose moments the run takes over (they are consumed).
def test_gather_and_adam_step_compiled_by_xla_match_the_reference() -> Result[Void]
  return ok() if !try Support.present()
  got = try gather_and_adam(Support.XLA)
  want = try gather_and_adam(Support.REF)
  Support.close(got, want)
end

## An index outside the rows is an error, as on the reference, not a clamped row.
def test_a_gather_index_outside_the_rows_is_an_error() -> Result[Void]
  return ok() if !try Support.present()
  table = try Support.floats(Support.uniform(15, 31, 1.0), [5, 3], Support.XLA)
  for bad in [[1, 5], [-1, 0]]
    r = Trace.eval(
      [table, try Support.labels(bad, Support.XLA)],
      ->(xs: List[Tensor.Tensor]) { Tensor.gather(xs[0], xs[1]) },
    )

    try Test.assert(!r.ok?, "gather at #{bad} of 5 rows ran on XLA")
    try Test.assert(r.message.include?("outside the rows"), r.message)
  end

  ok()
end

## A dtype that cannot hold the row count of the table needs no upper bound: U8 ids on 300 rows match the reference,
## a negative I8 id on 200 rows and a U8 id of 200 on 100 rows are still errors.
def test_a_narrow_index_dtype_gathers_from_a_larger_table_and_checks_its_bounds() -> Result[Void]
  return ok() if !try Support.present()
  rows = ->(xs: List[Tensor.Tensor]) { Tensor.gather(xs[0], xs[1]) }
  values = Support.uniform(600, 41, 1.0)
  ids = [0, 255, 7].map { |i| Tensor.Values.Int(i) }
  got = try Trace.eval(
    [try Support.floats(values, [300, 2], Support.XLA), try Tensor.from_list(ids, [3], Tensor.Dtype.U8, Support.XLA)],
    rows,
  )

  want = try Trace.eval(
    [try Support.floats(values, [300, 2], Support.REF), try Tensor.from_list(ids, [3], Tensor.Dtype.U8, Support.REF)],
    rows,
  )

  try Support.close([got], [want])
  bad = try Tensor.from_list([Tensor.Values.Int(-1)], [1], Tensor.Dtype.I8, Support.XLA)
  r = Trace.eval([try Support.floats(Support.uniform(400, 42, 1.0), [200, 2], Support.XLA), bad], rows)
  try Test.assert(!r.ok?, "gather at -1 of 200 rows ran on XLA")
  past = try Tensor.from_list([Tensor.Values.Int(200)], [1], Tensor.Dtype.U8, Support.XLA)
  r = Trace.eval([try Support.floats(Support.uniform(200, 43, 1.0), [100, 2], Support.XLA), past], rows)
  Test.assert(!r.ok?, "gather at 200 of 100 rows ran on XLA")
end

def test_an_adam_step_consumes_the_moments_it_updates() -> Result[Void]
  return ok() if !try Support.present()
  args = try adam_args(Support.XLA)
  f = try Trace.compile(args.map { |t| Trace.spec_of(t) }, &adam)
  _ = try Trace.run(f, args)
  try Test.assert(Tensor.to_list(args[0]).ok?, "the parameter was consumed")
  try Test.assert(Tensor.to_list(args[1]).ok?, "the gradient was consumed")
  try Test.assert(!Tensor.to_list(args[2]).ok?, "the first moment was not consumed")
  Test.assert(!Tensor.to_list(args[3]).ok?, "the second moment was not consumed")
end

def test_adam_steps_that_consume_their_moments_keep_no_buffer_alive() -> Result[Void]
  return ok() if !try Support.present()
  args = try adam_args(Support.XLA)
  f = try Trace.compile(args.map { |t| Trace.spec_of(t) }, &adam)
  first = 0
  for i in 0...20
    outs = try Trace.run(f, args)
    for t in [args[0], args[2], args[3]]
      free(t)
    end

    args = [outs[0], args[1], outs[1], outs[2], args[4]]
    if i == 0
      first = Xla.buffers()
    end
  end

  Test.assert(Xla.buffers() <= first, "#{Xla.buffers()} buffers are alive after the steps, #{first} after the first")
end

def test_an_eager_op_on_xla_is_an_error_naming_it() -> Result[Void]
  return ok() if !try Support.present()
  a = try Support.floats([1.0, 2.0], [2], Support.XLA)
  r = Tensor.add(a, a)
  try Test.assert(!r.ok?, "an eager add on XLA ran")
  Test.assert(r.message.start_with?("add does not run eagerly on the XLA backend"), r.message)
end

def test_the_creation_ops_round_trip() -> Result[Void]
  return ok() if !try Support.present()
  ints = [Tensor.Values.Int(-3), Tensor.Values.Int(7)]
  try Test.eq(try Tensor.to_list(try Tensor.from_list(ints, [2], Tensor.Dtype.I64, Support.XLA)), ints)
  try Test.eq(
    try Tensor.to_list(try Tensor.arange(0, 3, 1, Tensor.Dtype.I32, Support.XLA)),
    [Tensor.Values.Int(0), Tensor.Values.Int(1), Tensor.Values.Int(2)],
  )

  Test.eq(
    try Tensor.to_list(try Tensor.ones([2], Tensor.Dtype.Bool, Support.XLA)),
    [Tensor.Values.Bool(true), Tensor.Values.Bool(true)],
  )
end

def test_a_device_other_than_the_cpu_is_an_error() -> Result[Void]
  r = Tensor.zeros([2], Tensor.Dtype.F32, Tensor.Backend.Xla(Tensor.Device.Vulkan(0)))
  try Test.assert(!r.ok?, "XLA made a tensor on Vulkan")
  try Test.assert(r.message.include?("the XLA backend runs on the CPU only"), r.message)
  Test.assert(!Tensor.available(Tensor.Backend.Xla(Tensor.Device.Vulkan(0))), "XLA says it has a Vulkan device")
end

def free(t: Tensor.Tensor) -> Void
  case t.handle
    Tensor.Handle.Xla(h) -> Native.close(h)
    _ -> nil
  end
end

## `p, g, m, v` of shape `[3, 4]` and Adam's 7 numbers.
def adam_args(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([
    try Support.floats(Support.uniform(12, 21, 1.0), [3, 4], backend),
    try Support.floats(Support.uniform(12, 22, 0.5), [3, 4], backend),
    try Support.floats(Support.uniform(12, 23, 0.1), [3, 4], backend),
    try Support.floats(Support.uniform(12, 24, 0.1).map { |x| x.abs }, [3, 4], backend),
    try Support.floats([0.01, 0.9, 0.999, 0.00000001, 0.1, 1.5, 1.2], [7], backend),
  ])
end

def adam(xs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]] = Tensor.adam_step(xs[0], xs[1], xs[2], xs[3], xs[4])

## The rows of a `[5, 3]` table at `[4, 0, 4, 2]`, then an Adam step: one compiled program each.
def gather_and_adam(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  table = try Support.floats(Support.uniform(15, 31, 1.0), [5, 3], backend)
  ids = try Support.labels([4, 0, 4, 2], backend)
  rows = try Trace.compile(
    [Trace.spec_of(table), Trace.spec_of(ids)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.gather(xs[0], xs[1])]) },
  )

  args = try adam_args(backend)
  step = try Trace.compile(args.map { |t| Trace.spec_of(t) }, &adam)
  ok(try Trace.run(rows, [table, ids]) + try Trace.run(step, args))
end

## x `[6, 8]`, then w1, b1, w2, b2 of an 8 → 16 → 4 network.
def mlp_args(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([
    try Support.floats(Support.uniform(48, 7, 1.0), [6, 8], backend),
    try Support.floats(Support.uniform(128, 8, 0.35), [8, 16], backend),
    try Support.floats(Support.uniform(16, 9, 0.1), [16], backend),
    try Support.floats(Support.uniform(64, 10, 0.25), [16, 4], backend),
    try Support.floats(Support.uniform(4, 11, 0.1), [4], backend),
  ])
end

def forward(a: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  h = try Tensor.relu(try Tensor.linear(a[0], a[1], some(a[2])))
  Tensor.linear(h, a[3], some(a[4]))
end

## The mean softmax cross-entropy of `logits` `[batch, classes]` against `labels` `[batch]`.
def cross_entropy(logits: Tensor.Tensor, labels: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(logits)
  top = try Tensor.stop_grad(try Tensor.max(logits, some([1]), true))
  shifted = try Tensor.sub(logits, try Tensor.broadcast_to(top, shape))
  total = try Tensor.sum(try Tensor.exp(shifted), some([1]), true)
  log_probs = try Tensor.sub(shifted, try Tensor.broadcast_to(try Tensor.log(total), shape))
  classes = try Tensor.arange(0, shape[1], 1, Tensor.dtype(labels), Tensor.backend(logits))
  hit = try Tensor.eq(
    try Tensor.broadcast_to(try Tensor.reshape(classes, [1, shape[1]]), shape),
    try Tensor.broadcast_to(try Tensor.reshape(labels, [shape[0], 1]), shape),
  )

  zeros = try Tensor.full(shape, Tensor.Values.Float(0.0), Tensor.dtype(logits), Tensor.backend(logits))
  Tensor.neg(try Tensor.mean(try Tensor.sum(try Tensor.where(hit, log_probs, zeros), some([1]))))
end

def mlp_loss(ys: Tensor.Tensor) -> Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]
  ->(a: List[Tensor.Tensor]) { cross_entropy(try forward(a), Trace.constant(ys)) }
end

## The MLP's logits, through `Trace.compile` and `Trace.run`.
def run_forward(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  args = try mlp_args(backend)
  f = try Trace.compile(args.map { |t| Trace.spec_of(t) }, ->(xs: List[Tensor.Tensor]) { ok([try forward(xs)]) })
  Trace.run(f, args)
end

## One SGD step over `xs`: the parameters, the rate (rank 0), the inputs and the labels; the loss, then the new
## parameters.
def sgd_step(xs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  params = xs.take(4)
  rate = xs[4]
  loss = try cross_entropy(try forward([xs[5]] + params), xs[6])
  grads = try Trace.gradients(loss, params)
  out = [loss]
  for i in 0...params.size
    scaled = try Tensor.mul(grads[i], try Tensor.broadcast_to(rate, Tensor.shape(grads[i])))
    out = out.push(try Tensor.sub(params[i], scaled))
  end

  ok(out)
end

## The losses of three chained steps, then the parameters they end with.
def train(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  a = try mlp_args(backend)
  params = a.drop(1)
  rate = try Support.floats([0.5], [], backend)
  x = a[0]
  y = try Support.labels([0, 3, 1, 2, 3, 0], backend)
  step = try Trace.compile((params + [rate, x, y]).map { |t| Trace.spec_of(t) }, &sgd_step)
  losses: List[Tensor.Tensor] = []
  for _ in 0...3
    outs = try Trace.run(step, params + [rate, x, y])
    losses = losses.push(outs[0])
    params = outs.drop(1)
  end

  ok(losses + params)
end
