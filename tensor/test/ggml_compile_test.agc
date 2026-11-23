## `Trace.evaluate` on `Backend.Ggml` (`src/lower.agc`, `native/ggml_graph.c`):
## a small MLP and a cross-entropy, forward and gradients, against the
## reference; the structural cache; the refusals; and the same parity on
## Vulkan device 0 when there is one.
##
## Without the module the checks pass by asserting nothing, unless
## `TENSOR_REQUIRE_GGML` is set (CI sets it), and then they fail. Without a
## Vulkan device the Vulkan checks pass the same way (note.md §5).

import Lower from "../src/lower"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
REF = Tensor.Backend.Reference
## The largest gap allowed, relative to the largest reference value of a tensor.
TOL = 0.000001

private

## True when `backend` is there; with `TENSOR_REQUIRE_GGML` set, a missing
## ggml CPU backend is an error, never a pass.
def present(backend: Tensor.Backend) -> Result[Bool]
  return ok(true) if Tensor.available(backend)
  return ok(false) if backend != CPU
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok(false) if !flag.ok?
  return ok(false) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
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

## The mean softmax cross-entropy of `logits` `[batch, classes]` against
## `labels` `[batch]`, as `train/`'s `Loss.cross_entropy` writes it.
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
  picked = try Tensor.where(hit, log_probs, zeros)
  Tensor.neg(try Tensor.mean(try Tensor.sum(picked, some([1]))))
end

## The arguments of the small MLP on `backend`: x `[6, 8]`, then w1, b1, w2,
## b2 of an 8 → 16 → 4 network, from `seed`.
def mlp_args(backend: Tensor.Backend, seed: Int) -> Result[List[Tensor.Tensor]]
  ok([
    try floats(uniform(48, seed, 1.0), [6, 8], backend),
    try floats(uniform(128, seed + 1, 0.35), [8, 16], backend),
    try floats(uniform(16, seed + 2, 0.1), [16], backend),
    try floats(uniform(64, seed + 3, 0.25), [16, 4], backend),
    try floats(uniform(4, seed + 4, 0.1), [4], backend),
  ])
end

## The MLP's loss as a function of its arguments, against `ys`.
def mlp_loss(ys: Tensor.Tensor) -> Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]
  ->(a: List[Tensor.Tensor]) {
    h = try Tensor.relu(try Tensor.linear(a[0], a[1], some(a[2])))
    cross_entropy(try Tensor.linear(h, a[3], some(a[4])), Trace.constant(ys))
  }
end

## The sum of `k` times the argument: `k` is a fill of the trace.
def scaled(k: Float) -> Fn[List[Tensor.Tensor], Result[Tensor.Tensor]]
  ->(a: List[Tensor.Tensor]) {
    Tensor.sum(
      try Tensor.mul(a[0], try Tensor.full([2, 3], Tensor.Values.Float(k), Tensor.Dtype.F32, Tensor.backend(a[0]))),
    )
  }
end

## `max |got - want| / max |want|` (floored at 1e-6) of two tensors' values.
def gap(got: Tensor.Tensor, want: Tensor.Tensor) -> Result[Float]
  return error("shape #{Tensor.shape(got)} against #{Tensor.shape(want)}") if Tensor.shape(got) != Tensor.shape(want)
  a = try Tensor.to_list(got)
  b = try Tensor.to_list(want)
  worst = 0.0
  scale = 0.000001
  for i in 0...b.size
    x = try float_of(a[i])
    y = try float_of(b[i])
    if (x - y).abs > worst
      worst = (x - y).abs
    end

    if y.abs > scale
      scale = y.abs
    end
  end

  ok(worst / scale)
end

def float_of(v: Tensor.Values) -> Result[Float]
  case v
    Tensor.Values.Float(x) -> ok(x)
    _ -> error("a float tensor holds #{v}")
  end
end

## The value and every gradient of `f` agree between `backend` and the
## reference, within `TOL`.
def parity(
  backend: Tensor.Backend,
  args: List[Tensor.Tensor],
  want_args: List[Tensor.Tensor],
  f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
  want_f: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]],
) -> Result[Void]
  got = try Trace.value_and_grad(args, f)
  want = try Trace.value_and_grad(want_args, want_f)
  outs = [got.value] + got.grads
  wants = [want.value] + want.grads
  for i in 0...outs.size
    g = try gap(outs[i], wants[i])
    return error("output #{i}: gap #{g} is above #{TOL}") if g > TOL
  end

  ok()
end

def mlp_parity(backend: Tensor.Backend) -> Result[Void]
  ys = [0, 3, 1, 2, 3, 0]
  parity(
    backend,
    try mlp_args(backend, 7),
    try mlp_args(REF, 7),
    mlp_loss(try labels(ys, backend)),
    mlp_loss(try labels(ys, REF)),
  )
end

## The cross-entropy alone, differentiated for its logits.
def ce_parity(backend: Tensor.Backend) -> Result[Void]
  xs = uniform(15, 3, 4.0)
  ys = [4, 0, 2]
  got_ys = try labels(ys, backend)
  want_ys = try labels(ys, REF)
  parity(
    backend,
    [try floats(xs, [3, 5], backend)],
    [try floats(xs, [3, 5], REF)],
    ->(a: List[Tensor.Tensor]) { cross_entropy(a[0], Trace.constant(got_ys)) },
    ->(a: List[Tensor.Tensor]) { cross_entropy(a[0], Trace.constant(want_ys)) },
  )
end

## The error `r` answers names `text`.
def refused(r: Result[Tensor.Tensor], text: String) -> Result[Void]
  case r
    ok(_) -> error("computed a tensor where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

def test_an_mlp_and_its_gradients_match_the_reference_on_the_cpu() -> Result[Void]
  return ok() if !try present(CPU)

  try mlp_parity(CPU)
  ce_parity(CPU)
end

def test_a_retrace_with_new_values_compiles_nothing() -> Result[Void]
  return ok() if !try present(CPU)

  x = try floats(uniform(6, 1, 1.0), [2, 3], CPU)
  _ = try Trace.eval([x], scaled(2.0))
  before = try Lower.compiles()
  y = try floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], CPU)
  got = try Trace.eval([y], scaled(0.5))
  try Test.eq(try Lower.compiles(), before)
  try Test.eq(try Tensor.to_list(got), [Tensor.Values.Float(10.5)])
  mlp = try mlp_args(CPU, 11)
  loss = mlp_loss(try labels([1, 1, 0, 2, 3, 0], CPU))
  _ = try Trace.value_and_grad(mlp, loss)
  after_one = try Lower.compiles()
  _ = try Trace.value_and_grad(try mlp_args(CPU, 12), mlp_loss(try labels([0, 1, 2, 3, 0, 1], CPU)))
  Test.eq(try Lower.compiles(), after_one)
end

def test_what_ggml_does_not_lower_is_an_error_naming_it() -> Result[Void]
  return ok() if !try present(CPU)

  x = try floats([1.0, 2.0, 3.0, 4.0], [2, 2], CPU)
  try refused(Trace.eval([x], ->(a: List[Tensor.Tensor]) { Tensor.gelu(a[0]) }), "gelu_tanh_approx is not lowered")
  to_int = ->(a: List[Tensor.Tensor]) { Tensor.to_dtype(a[0], Tensor.Dtype.I32) }
  try refused(Trace.eval([x], to_int), "to_dtype from F32 to I32 is not lowered")
  cube = try floats(uniform(8, 2, 1.0), [2, 2, 2], CPU)
  middle = ->(a: List[Tensor.Tensor]) { Tensor.sum(a[0], some([1])) }
  try refused(Trace.eval([cube], middle), "not a leading or trailing block")
  plain = try floats([1.0, 2.0, 3.0, 4.0], [2, 2], REF)
  mixed = ->(a: List[Tensor.Tensor]) { Tensor.add(a[0], Trace.constant(plain)) }
  try refused(Trace.eval([x], mixed), "a constant on Reference in a trace run on Ggml(Cpu)")
  zero = try floats([1.0, 0.0, 3.0, 4.0], [2, 2], CPU)
  refused(Trace.eval([zero], ->(a: List[Tensor.Tensor]) { Tensor.log(a[0]) }), "not finite")
end

def test_vulkan_matches_the_reference_or_says_why_not() -> Result[Void]
  if !Tensor.available(VULKAN)
    IO.warn("SKIP: no Vulkan device 0; the compiled Vulkan parity runs where there is one")
    return ok()
  end

  try mlp_parity(VULKAN)
  try ce_parity(VULKAN)
  try Test.eq(try System.env("GGML_VK_DISABLE_F16"), "1")
  case Tensor.vulkan_fp16(true)
    ok(_) -> error("fp16 was chosen after a Vulkan device opened with F32")
    error(e) -> Test.assert(e.message.include?("F32 mul_mat already"), "the error said \"#{e.message}\"")
  end
end
