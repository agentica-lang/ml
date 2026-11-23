## The SGD step of an MLP classifier (784 → 512 → 512 → 10, synthetic data,
## softmax cross-entropy) through the public API on the ggml backend:
## `Trace.value_and_grad` for the loss and gradients, then one `Trace.eval`
## per parameter for the update (`Trace.eval` has one output). Each is traced
## again every step and runs as one compiled graph from the cache, so a step
## is seven traces in Agentica, seven cache hits, the copies in and out, and
## seven graphs on the device.
##
## ```sh
## eval "$(sh tools/build-engine.sh vulkan | grep '^export ')"
## agentica tools/bench-ggml.agc vulkan 64 256 1024 4096
## agentica tools/bench-ggml.agc cpu 256
## ```
##
## Times are wall clock (`Time.monotonic_ns`), the median of the steps after
## two warm ones; the first step, which compiles the graphs, is printed apart.

import Lower from "../src/lower"
import Types from "../src/types"
import Tensor from "../tensor"
import Trace from "../trace"

SIZES = [784, 512, 512, 10]
CLASSES = 10
LR = 0.01
STEPS = 20

struct Stepped
  params: List[Tensor.Tensor]
  loss: Float
  ## Nanoseconds of `value_and_grad`, the rest of the step being the update.
  grad_ns: Int
end

def main() -> Void
  case run(System.args)
    ok(_) -> nil
    error(e) -> IO.warn("bench-ggml: #{e.message}")
  end
end

def run(args: List[String]) -> Result[Void]
  return error("usage: bench-ggml.agc cpu|vulkan BATCH...") if args.size < 2
  device = Tensor.Device.Cpu
  if args[0] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  return error("ggml #{args[0]} is not available") if !Tensor.available(Tensor.Backend.Ggml(device))
  for b in args.drop(1)
    try bench(device, try b.to_i)
  end

  ok()
end

def bench(device: Tensor.Device, batch: Int) -> Result[Void]
  params = try init(device)
  x = try floats(values(batch * SIZES[0], 1), [batch, SIZES[0]], device)
  ys = (0...batch).to_a.map { |i| Tensor.Values.Int((i * 7) % CLASSES) }
  y = try Tensor.to_device(
    try Tensor.from_list(ys, [batch], Tensor.Dtype.I32, Tensor.Backend.Ggml(Tensor.Device.Cpu)),
    device,
  )

  compiled = try Lower.compiles()
  t0 = Time.monotonic_ns()
  first = try step(params, x, y)
  t1 = Time.monotonic_ns()
  params = first.params
  times: List[Int] = []
  grads: List[Int] = []
  loss = first.loss
  for _ in 0...STEPS
    s0 = Time.monotonic_ns()
    s = try step(params, x, y)
    times = times.push(Time.monotonic_ns() - s0)
    grads = grads.push(s.grad_ns)
    params = s.params
    loss = s.loss
  end

  puts(
    "#{Types.device_name(device)} batch #{batch}: first step #{format("%.1f", (t1 - t0).to_f / 1000000.0)} ms " +
    "(#{(try Lower.compiles()) - compiled} graphs compiled), then #{format("%.2f", median_ms(times))} ms per step, " +
    "#{format("%.2f", median_ms(grads))} of it value_and_grad; " +
    "loss #{format("%.4f", first.loss)} -> #{format("%.4f", loss)}",
  )

  ok()
end

## The median of `ns` after the two warm-up steps, in milliseconds.
def median_ms(ns: List[Int]) -> Float
  sorted = ns.drop(2).sort
  sorted[sorted.size / 2].to_f / 1000000.0
end

def step(params: List[Tensor.Tensor], x: Tensor.Tensor, y: Tensor.Tensor) -> Result[Stepped]
  t0 = Time.monotonic_ns()
  vg = try Trace.value_and_grad(params + [x, y], &loss_of)
  grad_ns = Time.monotonic_ns() - t0
  updated: List[Tensor.Tensor] = []
  for i in 0...params.size
    updated = updated.push(try Trace.eval([params[i], vg.grads[i]], &sgd))
  end

  v = try Tensor.to_list(try Tensor.to_device(vg.value, Tensor.Device.Cpu))
  case v[0]
    Tensor.Values.Float(f) -> ok(Stepped(params: updated, loss: f, grad_ns: grad_ns))
    _ -> error("the loss is not a float")
  end
end

## `p - LR * g` over `[p, g]`.
def sgd(a: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  rate = try Tensor.full(Tensor.shape(a[0]), Tensor.Values.Float(LR), Tensor.Dtype.F32, Tensor.backend(a[0]))
  Tensor.sub(a[0], try Tensor.mul(a[1], rate))
end

## The loss of the MLP over `[w1, b1, w2, b2, w3, b3, x, y]`.
def loss_of(a: List[Tensor.Tensor]) -> Result[Tensor.Tensor]
  layers = SIZES.size - 1
  h = a[2 * layers]
  for i in 0...layers
    h = try Tensor.linear(h, a[2 * i], some(a[2 * i + 1]))
    if i < layers - 1
      h = try Tensor.relu(h)
    end
  end

  ys = a[2 * layers + 1]
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

def init(device: Tensor.Device) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for i in 0...(SIZES.size - 1)
    scale = 1.0 / Math.sqrt(SIZES[i].to_f)
    out = out.push(
      try floats(values(SIZES[i] * SIZES[i + 1], 10 + i).map { |v| v * scale }, [SIZES[i], SIZES[i + 1]], device),
    )

    out = out.push(try floats(values(SIZES[i + 1], 20 + i).map { |v| v * 0.01 }, [SIZES[i + 1]], device))
  end

  ok(out)
end

## `n` values in `[-1, 1)`, a different walk per `seed`.
def values(n: Int, seed: Int) -> List[Float]
  out: List[Float] = []
  for i in 0...n
    out = out.push(((i * (2 * seed + 7) + seed) % 1009).to_f / 504.5 - 1.0)
  end

  out
end

def floats(xs: List[Float], shape: List[Int], device: Tensor.Device) -> Result[Tensor.Tensor]
  on_cpu = try Tensor.from_list(
    xs.map { |x| Tensor.Values.Float(x) },
    shape,
    Tensor.Dtype.F32,
    Tensor.Backend.Ggml(Tensor.Device.Cpu),
  )

  Tensor.to_device(on_cpu, device)
end
