## The benchmark's model, 784 → 512 → 512 → 10 at batch 256 with softmax
## cross-entropy, run on ggml's CPU backend and on a Vulkan device, for the
## Vulkan tests (not a test file itself). The reference is too slow at these
## shapes; ggml CPU is what `ggml_compile_test.agc` ties to it.

import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
SIZES = [784, 512, 512, 10]
BATCH = 256

## The gaps of one comparison: `max |vulkan - cpu| / max |cpu|` for the loss,
## then for the gradient of x and of every parameter, and the CPU loss.
struct Compared
  gaps: List[Float]
  loss: Float
end

## The loss and every gradient on `device` against ggml's CPU; an error when
## the CPU loss is not finite or a CPU gradient is all zeros, since a gap
## against nothing proves nothing.
def compare(device: Tensor.Device) -> Result[Compared]
  on_cpu = try inputs()
  on_device: List[Tensor.Tensor] = []
  for t in on_cpu
    on_device = on_device.push(try Tensor.to_device(t, device))
  end

  want = try Trace.value_and_grad(on_cpu, &loss)
  got = try Trace.value_and_grad(on_device, &loss)
  wants = [want.value] + want.grads.take(want.grads.size - 1)
  gots = [got.value] + got.grads.take(got.grads.size - 1)
  value = try scalar(want.value)
  return error("the CPU loss #{value} is not finite") if !(value.abs < 1.0e30)
  gaps: List[Float] = []
  for i in 0...wants.size
    top = try scalar(try Trace.eval([wants[i]], &largest_first))
    return error("output #{i} is all zeros on the CPU") if !(top > 0.0)
    gaps = gaps.push(try gap(try Tensor.to_device(gots[i], Tensor.Device.Cpu), wants[i]))
  end

  ok(Compared(gaps: gaps, loss: value))
end

private

def floats(xs: List[Float]) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, [xs.size], Tensor.Dtype.F32, CPU)
end

## `n` multiples of 1/32 in `[-1, 1]`, a different walk per `seed`.
def steps(n: Int, seed: Int) -> List[Float]
  (0...n).to_a.map { |i| ((i * (2 * seed + 7) + seed) % 65 - 32).to_f / 32.0 }
end

## `[rows, cols]` as `r1 ⊗ c1 + r2 ⊗ c2` on ggml's CPU, computed by a traced
## graph: every product and sum of these multiples of 1/32 is exact in F32,
## so a copy to the iGPU starts from the same bits.
def matrix(rows: Int, cols: Int, seed: Int) -> Result[Tensor.Tensor]
  args = [
    try floats(steps(rows, seed)),
    try floats(steps(cols, seed + 1)),
    try floats(steps(rows, seed + 2)),
    try floats(steps(cols, seed + 3)),
  ]

  build: Fn[List[Tensor.Tensor], Result[Tensor.Tensor]] = ->(a: List[Tensor.Tensor]) {
    one = try Tensor.mul(
      try Tensor.broadcast_to(try Tensor.reshape(a[0], [rows, 1]), [rows, cols]),
      try Tensor.broadcast_to(try Tensor.reshape(a[1], [1, cols]), [rows, cols]),
    )

    two = try Tensor.mul(
      try Tensor.broadcast_to(try Tensor.reshape(a[2], [rows, 1]), [rows, cols]),
      try Tensor.broadcast_to(try Tensor.reshape(a[3], [1, cols]), [rows, cols]),
    )

    eighth = try Tensor.full([rows, cols], Tensor.Values.Float(0.0625), Tensor.Dtype.F32, Tensor.backend(one))
    Tensor.mul(try Tensor.add(one, two), eighth)
  }

  Trace.eval(args, build)
end

## x, then w and b of each layer, and the labels, on ggml's CPU.
def inputs() -> Result[List[Tensor.Tensor]]
  out = [try matrix(BATCH, SIZES[0], 1)]
  for i in 0...(SIZES.size - 1)
    out = out.push(try matrix(SIZES[i], SIZES[i + 1], 10 + i))
    out = out.push(try floats(steps(SIZES[i + 1], 20 + i).map { |x| x * 0.125 }))
  end

  ys = (0...BATCH).to_a.map { |i| Tensor.Values.Int((i * 7) % 10) }
  ok(out.push(try Tensor.from_list(ys, [BATCH], Tensor.Dtype.I32, CPU)))
end

## The loss of the MLP; the last argument is the labels.
def loss(a: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  h = a[0]
  layers = SIZES.size - 1
  for i in 0...layers
    h = try Tensor.linear(h, a[1 + 2 * i], some(a[2 + 2 * i]))
    if i < layers - 1
      h = try Tensor.relu(h)
    end
  end

  ys = a[a.size - 1]
  shape = Tensor.shape(h)
  top = try Tensor.stop_grad(try Tensor.max(h, some([1]), true))
  shifted = try Tensor.sub(h, try Tensor.broadcast_to(top, shape))
  total = try Tensor.sum(try Tensor.exp(shifted), some([1]), true)
  log_probs = try Tensor.sub(shifted, try Tensor.broadcast_to(try Tensor.log(total), shape))
  classes = try Tensor.arange(0, shape[1], 1, Tensor.Dtype.I32, Tensor.backend(h))
  hit = try Tensor.eq(
    try Tensor.broadcast_to(try Tensor.reshape(classes, [1, shape[1]]), shape),
    try Tensor.broadcast_to(try Tensor.reshape(ys, [shape[0], 1]), shape),
  )

  zeros = try Tensor.full(shape, Tensor.Values.Float(0.0), Tensor.Dtype.F32, Tensor.backend(h))
  Tensor.neg(try Tensor.mean(try Tensor.sum(try Tensor.where(hit, log_probs, zeros), some([1]))))
end

## `max |got - want| / max |want|`, computed on ggml's CPU.
def gap(got: Tensor.Tensor, want: Tensor.Tensor) -> Result[Float]
  num = try scalar(try Trace.eval([got, want], &largest_difference))
  den = try scalar(try Trace.eval([want], &largest_first))
  return ok(num / 0.000001) if den < 0.000001
  ok(num / den)
end

def largest_difference(x: List[Tensor.Tensor]) -> Result[Tensor.Tensor] = largest(try Tensor.sub(x[0], x[1]))

def largest_first(x: List[Tensor.Tensor]) -> Result[Tensor.Tensor] = largest(x[0])

## `max |t|` over every axis.
def largest(t: Tensor.Tensor) -> Result[Tensor.Tensor] = Tensor.max(try Tensor.sqrt(try Tensor.mul(t, t)))

def scalar(t: Tensor.Tensor) -> Result[Float]
  case (try Tensor.to_list(t))[0]
    Tensor.Values.Float(v) -> ok(v)
    _ -> error("a scalar that is not a float")
  end
end
