## The SGD step of an MLP classifier (784 → 512 → 512 → 10, synthetic data,
## softmax cross-entropy) through the public API on the ggml backend, as
## `Train.fit` takes it: one `Trace.eval_many` traces the loss, its gradients
## and the update, and runs them as one compiled graph from the cache. So a
## step is one trace in Agentica, one cache hit and one graph on the device.
##
## ```sh
## eval "$(sh tools/build-engine.sh vulkan | grep '^export ')"
## agentica tools/bench-ggml.agc vulkan 64 256 1024 4096
## agentica tools/bench-ggml.agc cpu 256
## ```
##
## Times are wall clock (`Time.monotonic_ns`), the median of the steps after
## two warm ones; the first step, which compiles the graph, is printed apart.

import Lower from "../src/lower"
import Types from "../src/types"
import Tensor from "../tensor"
import Trace from "../trace"

SIZES = [784, 512, 512, 10]
CLASSES = 10
LR = 0.01
## Steps timed; the median drops the first two, so it is of 18.
STEPS = 20

struct Stepped
  params: List[Tensor.Tensor]
  loss: Float
end

def main() -> Void
  case run(System.args)
    ok(_) -> nil
    error(e) ->
      IO.warn("bench-ggml: #{e.message}")
      System.exit(1)
  end
end

def run(args: List[String]) -> Result[Void]
  return error("usage: bench-ggml.agc cpu|vulkan BATCH...") if args.size < 2
  device = Tensor.Device.Cpu
  case args[0]
    "cpu" -> nil
    "vulkan" -> device = Tensor.Device.Vulkan(0)
    _ -> return error("the device is cpu or vulkan, got #{args[0]}")
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
  loss = first.loss
  for _ in 0...STEPS
    s0 = Time.monotonic_ns()
    s = try step(params, x, y)
    times = times.push(Time.monotonic_ns() - s0)
    params = s.params
    loss = s.loss
  end

  puts(
    "#{Types.device_name(device)} batch #{batch}: first step #{format("%.1f", (t1 - t0).to_f / 1000000.0)} ms " +
    "(#{(try Lower.compiles()) - compiled} graphs compiled), then #{format("%.2f", median_ms(times))} ms per step; " +
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
  outs = try Trace.eval_many(params + [x, y], &sgd_step)
  v = try Tensor.to_list(try Tensor.to_device(outs[0], Tensor.Device.Cpu))
  case v[0]
    Tensor.Values.Float(f) -> ok(Stepped(params: outs.drop(1), loss: f))
    _ -> error("the loss is not a float")
  end
end

## The loss over `[w1, b1, w2, b2, w3, b3, x, y]`, then each parameter
## `p - LR * g`.
def sgd_step(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  ps = a.take(a.size - 2)
  loss = try loss_of(a)
  grads = try Trace.gradients(loss, ps)
  out = [loss]
  for i in 0...ps.size
    rate = try Tensor.full(Tensor.shape(ps[i]), Tensor.Values.Float(LR), Tensor.Dtype.F32, Tensor.backend(ps[i]))
    out = out.push(try Tensor.sub(ps[i], try Tensor.mul(grads[i], rate)))
  end

  ok(out)
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
