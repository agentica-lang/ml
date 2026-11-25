## One training step of the MLP of `tensor/tools/bench-ggml.agc` (784-512-512-10, relu, softmax cross-entropy), timed
## the way `benchmarks/training/README.md` describes, against PyTorch and JAX. Manual, never in CI. From `train/`:
##
## ```sh
## agentica tools/bench_train.agc BACKEND MODE OPTIMIZER BATCH [STEPS]
## ```
##
## BACKEND is `ggml-cpu`, `ggml-vulkan` (matmul in fp32) or `torch-cpu`. MODE is `fit`, the steps of `Train.fit` (the
## loader's batch included, traced again each step), or `compiled`, one `Trace.compile` of the whole step and a `run`
## per step. OPTIMIZER is `sgd` (lr 0.01) or `adam` (lr 0.001). STEPS (30 by default) are timed after the first step,
## which holds the compile, and three warm ones. The last line is one JSON object: `first_ms`, `median_ms`, `min_ms`,
## `max_ms` and `losses`, the loss of each of the first ten steps.
##
## The init and the batch are the closed forms of `benchmarks/training/` (`values`), the same in every stack.

import Data from "../data"
import Loss from "../loss"
import Optim from "../optim"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Train from "../train"

SIZES = [784, 512, 512, 10]
CLASSES = 10
WARM = 3
LOSSES = 10

struct Measured
  first_ns: Int
  steps_ns: List[Int]
  losses: List[Float]
end

struct Setup
  backend: Tensor.Backend
  optimizer: Optim.Optimizer
  params: List[Tensor.Tensor]
  x: Tensor.Tensor
  y: Tensor.Tensor
  steps: Int
end

struct Counts
  params: Int
  first: Int
  second: Int
end

struct Stepped
  state: Optim.State
  params: List[Tensor.Tensor]
  loss: Float
end

def main() -> Void
  case run(System.args())
    ok(_) -> nil
    error(e) ->
      puts("bench_train: #{e.message}")
      System.exit(2)
  end
end

def run(args: List[String]) -> Result[Void]
  return error("usage: BACKEND MODE OPTIMIZER BATCH [STEPS]") if args.size < 4
  backend = try backend_of(args[0])
  if args[0] == "ggml-vulkan"
    try Tensor.vulkan_fp16(false)
  end

  return error("#{args[0]} is not available") if not Tensor.available(backend)
  optimizer = try optimizer_of(args[2])
  batch = try args[3].to_i
  steps = 30
  if args.size > 4
    steps = try args[4].to_i
  end

  return error("STEPS must be at least 1") if steps < 1
  setup = Setup(
    backend: backend,
    optimizer: optimizer,
    params: try params_of(backend),
    x: try floats(values(batch * SIZES[0], 1), [batch, SIZES[0]], backend),
    y: try labels(batch, backend),
    steps: steps,
  )

  measured = Measured(first_ns: 0, steps_ns: [], losses: [])
  case args[1]
    "fit" -> measured = try fitted(setup)
    "compiled" -> measured = try compiled(setup)
    _ -> return error("the mode is fit or compiled, got #{args[1]}")
  end

  puts(report(measured))
  ok()
end

def backend_of(name: String) -> Result[Tensor.Backend]
  case name
    "ggml-cpu" -> ok(Tensor.Backend.Ggml(Tensor.Device.Cpu))
    "ggml-vulkan" -> ok(Tensor.Backend.Ggml(Tensor.Device.Vulkan(0)))
    "torch-cpu" -> ok(Tensor.Backend.Torch(Tensor.Device.Cpu))
    _ -> error("the backend is ggml-cpu, ggml-vulkan or torch-cpu, got #{name}")
  end
end

def optimizer_of(name: String) -> Result[Optim.Optimizer]
  case name
    "sgd" -> ok(Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.01)))
    "adam" -> ok(Optim.Optimizer.Adam(Optim.Adam(lr: 0.001)))
    _ -> error("the optimizer is sgd or adam, got #{name}")
  end
end

def report(m: Measured) -> String
  sorted = m.steps_ns.sort
  losses = m.losses.take(LOSSES).map { |l| format("%.7f", l) }.join(", ")
  head = "{\"first_ms\": #{ms(m.first_ns)}, \"median_ms\": #{ms(sorted[sorted.size / 2])}"
  tail = "\"min_ms\": #{ms(sorted[0])}, \"max_ms\": #{ms(sorted[sorted.size - 1])}, \"losses\": [#{losses}]}"
  "#{head}, #{tail}"
end

def ms(ns: Int) -> String = format("%.3f", ns.to_f / 1000000.0)

## The steps of `Train.fit` over a dataset of one batch: an epoch is a step, and `on_epoch` stamps its end.
def fitted(s: Setup) -> Result[Measured]
  epochs = 1 + WARM + s.steps
  dataset = try Data.dataset(s.x, s.y)
  loader = Data.Loader(dataset: dataset, batch_size: Tensor.shape(s.x)[0], shuffle: false, prefetch: 0)
  stamps: Channel[Int] = Channel.new(epochs + 1)
  started = Time.monotonic_ns()
  fit = try Train.fit(&loss, s.params, loader, s.optimizer, epochs, some(->(_: Train.Epoch) { stamp(stamps) }))

  ns: List[Int] = []
  last = started
  for _ in 0...epochs
    now = try stamps.recv(0)
    ns = ns.push(now - last)
    last = now
  end

  ok(Measured(first_ns: ns[0], steps_ns: ns.drop(1 + WARM), losses: fit.history))
end

## One `Trace.compile` of the loss, its gradients and the update, with Adam's two step scalars as inputs, then a `run`
## per step.
def compiled(s: Setup) -> Result[Measured]
  started = Time.monotonic_ns()
  state = try Optim.init(s.optimizer, s.params)
  counts = Counts(params: s.params.size, first: state.first.size, second: state.second.size)
  scalar = Trace.Spec(shape: [], dtype: Tensor.dtype(s.params[0]), backend: s.backend)
  scalars = Optim.step_scalars(state).map { |_| scalar }
  moments = (state.first + state.second).map { |t| Trace.spec_of(t) }
  specs = s.params.map { |t| Trace.spec_of(t) } + moments + scalars + [Trace.spec_of(s.x), Trace.spec_of(s.y)]
  program = try Trace.compile(specs, ->(xs: List[Tensor.Tensor]) { traced_step(state, counts, xs) })
  ns: List[Int] = []
  losses: List[Float] = []
  current = s.params
  first = 0
  for i in 0...(1 + WARM + s.steps)
    t0 = Time.monotonic_ns()
    stepped = try compiled_step(program, state, current, s)
    t1 = Time.monotonic_ns()
    state = stepped.state
    current = stepped.params
    losses = losses.push(stepped.loss)
    if i == 0
      first = t1 - started
    elsif i > WARM
      ns = ns.push(t1 - t0)
    end
  end

  Trace.close(program)
  ok(Measured(first_ns: first, steps_ns: ns, losses: losses))
end

def compiled_step(
  program: Trace.Compiled,
  state: Optim.State,
  params: List[Tensor.Tensor],
  s: Setup,
) -> Result[Stepped]
  Native.arena do
    scalars: List[Tensor.Tensor] = []
    for v in Optim.step_scalars(state)
      scalars = scalars.push(try Tensor.from_list([Tensor.Values.Float(v)], [], Tensor.Dtype.F32, s.backend))
    end

    n = params.size
    k = state.first.size
    outs = try Trace.run(program, params + state.first + state.second + scalars + [s.x, s.y])
    moved = state.with(steps: state.steps + 1, first: outs.drop(1 + n).take(k), second: outs.drop(1 + n + k))
    ok(Stepped(state: moved, params: outs.drop(1).take(n), loss: try scalar(outs[0])))
  end
end

def traced_step(state: Optim.State, c: Counts, xs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  ps = xs.take(c.params)
  moments = c.params + c.first + c.second
  held = state.with(first: xs.drop(c.params).take(c.first), second: xs.drop(c.params + c.first).take(c.second))
  rest = xs.drop(moments)
  data = rest.drop(rest.size - 2)
  value = try loss_of(ps, data[0], data[1])
  grads = try Trace.gradients(value, ps)
  stepped = try Optim.step(held, ps, grads, rest.take(rest.size - 2))
  ok([value] + stepped.params + stepped.state.first + stepped.state.second)
end

def scalar(t: Tensor.Tensor) -> Result[Float]
  values = try Tensor.to_list(t)
  case values[0]
    Tensor.Values.Float(f) -> ok(f)
    _ -> error("the loss is not a float")
  end
end

def stamp(stamps: Channel[Int]) -> Void
  _ = stamps.send(Time.monotonic_ns(), 0)
end

def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor] = loss_of(ps, b.inputs, b.targets)

def loss_of(ps: List[Tensor.Tensor], x: Tensor.Tensor, y: Tensor.Tensor) -> Result[Tensor.Tensor]
  layers = SIZES.size - 1
  h = x
  for i in 0...layers
    h = try Tensor.linear(h, ps[2 * i], some(ps[2 * i + 1]))
    if i < layers - 1
      h = try Tensor.relu(h)
    end
  end

  Loss.cross_entropy(h, y)
end

## `n` numbers in `[-1, 1)`, a different walk per `seed`: the closed form of `tensor/tools/bench-ggml.agc`.
def values(n: Int, seed: Int) -> List[Float]
  (0...n).map { |i| ((i * (2 * seed + 7) + seed) % 1009).to_f / 504.5 - 1.0 }
end

def params_of(backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  out: List[Tensor.Tensor] = []
  for i in 0...(SIZES.size - 1)
    scale = 1.0 / Math.sqrt(SIZES[i].to_f)
    weights = values(SIZES[i] * SIZES[i + 1], 10 + i).map { |v| v * scale }
    out = out.push(try floats(weights, [SIZES[i], SIZES[i + 1]], backend))
    out = out.push(try floats(values(SIZES[i + 1], 20 + i).map { |v| v * 0.01 }, [SIZES[i + 1]], backend))
  end

  ok(out)
end

def floats(xs: List[Float], shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, backend)
end

def labels(batch: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  ys = (0...batch).map { |i| Tensor.Values.Int((i * 7) % CLASSES) }
  Tensor.from_list(ys, [batch], Tensor.Dtype.I32, backend)
end
