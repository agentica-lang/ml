## `Backend.Xla` (`src/xla.agc`, `native/xla.c`): a traced MLP and a training step compiled by XLA against the
## reference, the eager refusals, and the device check.
##
## Without the plugin (`tools/install_xla.agc`) the checks pass by asserting nothing, unless `TENSOR_REQUIRE_XLA` is
## set (the `xla` job of CI sets it), and then they fail.

import Tensor from "../tensor"
import Trace from "../trace"
import Xla from "../src/xla"

XLA = Tensor.Backend.Xla(Tensor.Device.Cpu)
REF = Tensor.Backend.Reference
## The largest gap allowed, relative to the largest reference value of a tensor.
TOL = 0.000001

private

def test_an_mlp_forward_compiled_by_xla_matches_the_reference() -> Result[Void]
  return ok() if !try present()
  got = try run_forward(XLA)
  want = try run_forward(REF)
  close(got, want)
end

## Three chained SGD steps (the loss, then the new parameters, fed back in): the step of `Train.fit`, without the
## optimizer's fused node.
def test_chained_training_steps_compiled_by_xla_match_the_reference() -> Result[Void]
  return ok() if !try present()
  got = try train(XLA)
  want = try train(REF)
  close(got, want)
end

## `value_and_grad` on XLA compiles the graph for the call (`Trace.evaluate`).
def test_value_and_grad_on_xla_matches_the_reference() -> Result[Void]
  return ok() if !try present()
  ys = [0, 3, 1, 2, 3, 0]
  got = try Trace.value_and_grad(try mlp_args(XLA), mlp_loss(try labels(ys, XLA)))
  want = try Trace.value_and_grad(try mlp_args(REF), mlp_loss(try labels(ys, REF)))
  close([got.value] + got.grads, [want.value] + want.grads)
end

## Steps that close what they replace keep the number of live buffers flat: a run keeps nothing of its inputs.
def test_steps_that_close_old_tensors_keep_no_buffer_alive() -> Result[Void]
  return ok() if !try present()
  a = try mlp_args(XLA)
  params = a.drop(1)
  fixed = [try floats([0.5], [], XLA), a[0], try labels([0, 3, 1, 2, 3, 0], XLA)]
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

  Test.eq(Xla.buffers(), first)
end

## `Trace.close` frees the compiled program: a run afterwards is an error.
def test_a_closed_xla_function_does_not_run() -> Result[Void]
  return ok() if !try present()
  args = try mlp_args(XLA)
  f = try Trace.compile(args.map { |t| Trace.spec_of(t) }, ->(xs: List[Tensor.Tensor]) { ok([try forward(xs)]) })
  try Trace.run(f, args)
  Trace.close(f)
  Test.assert(!Trace.run(f, args).ok?, "a closed XLA function ran")
end

def test_an_eager_op_on_xla_is_an_error_naming_it() -> Result[Void]
  return ok() if !try present()
  a = try floats([1.0, 2.0], [2], XLA)
  r = Tensor.add(a, a)
  try Test.assert(!r.ok?, "an eager add on XLA ran")
  Test.assert(r.message.start_with?("add does not run eagerly on the XLA backend"), r.message)
end

def test_the_creation_ops_round_trip() -> Result[Void]
  return ok() if !try present()
  ints = [Tensor.Values.Int(-3), Tensor.Values.Int(7)]
  try Test.eq(try Tensor.to_list(try Tensor.from_list(ints, [2], Tensor.Dtype.I64, XLA)), ints)
  try Test.eq(
    try Tensor.to_list(try Tensor.arange(0, 3, 1, Tensor.Dtype.I32, XLA)),
    [Tensor.Values.Int(0), Tensor.Values.Int(1), Tensor.Values.Int(2)],
  )

  Test.eq(
    try Tensor.to_list(try Tensor.ones([2], Tensor.Dtype.Bool, XLA)),
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

## True when the plugin is there; with `TENSOR_REQUIRE_XLA` set, a missing one is an error, never a pass.
def present() -> Result[Bool]
  return ok(true) if Tensor.available(XLA)
  flag = System.env("TENSOR_REQUIRE_XLA")
  return ok(false) if !flag.ok?
  return ok(false) if try flag == ""
  error("the XLA backend is required but not available (TENSOR_REQUIRE_XLA is set)")
end

## `count` floats in `[-scale, scale)` from `seed`.
def uniform(count: Int, seed: Int, scale: Float) -> List[Float]
  s = seed
  out: List[Float] = []
  for _ in 0...count
    s = (s * 1103515245 + 12345) % 2147483648
    out = out.push(((s.to_f / 2147483648.0) * 2.0 - 1.0) * scale)
  end

  out
end

def floats(xs: List[Float], shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, backend)
end

def labels(xs: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Int(x) }, [xs.size], Tensor.Dtype.I32, backend)
end

## x `[6, 8]`, then w1, b1, w2, b2 of an 8 → 16 → 4 network.
def mlp_args(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  ok([
    try floats(uniform(48, 7, 1.0), [6, 8], backend),
    try floats(uniform(128, 8, 0.35), [8, 16], backend),
    try floats(uniform(16, 9, 0.1), [16], backend),
    try floats(uniform(64, 10, 0.25), [16, 4], backend),
    try floats(uniform(4, 11, 0.1), [4], backend),
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
  rate = try floats([0.5], [], backend)
  x = a[0]
  y = try labels([0, 3, 1, 2, 3, 0], backend)
  step = try Trace.compile((params + [rate, x, y]).map { |t| Trace.spec_of(t) }, &sgd_step)
  losses: List[Tensor.Tensor] = []
  for _ in 0...3
    outs = try Trace.run(step, params + [rate, x, y])
    losses = losses.push(outs[0])
    params = outs.drop(1)
  end

  ok(losses + params)
end

## Every tensor of `got` within `TOL` of the one of `want`, relative to its largest value.
def close(got: List[Tensor.Tensor], want: List[Tensor.Tensor]) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...got.size
    try Test.eq(Tensor.shape(got[i]), Tensor.shape(want[i]))
    a = try values(got[i])
    b = try values(want[i])
    worst = 0.0
    scale = 0.000001
    for k in 0...b.size
      worst = [worst, (a[k] - b[k]).abs].max.unwrap!
      scale = [scale, b[k].abs].max.unwrap!
    end

    return error("output #{i}: gap #{worst / scale} is above #{TOL}") if worst / scale > TOL
  end

  ok()
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
