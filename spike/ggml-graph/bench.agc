## Training-step time of an MLP classifier (784 → 512 → 512 → 10, synthetic
## data, softmax cross-entropy) at large batches, on one backend per run:
##
## - `cpu` (8 threads, `cpuN` for N), `vulkan`: the step traced once, lowered to ONE ggml graph, built
##   and allocated once, then run once per step (the batch is uploaded each
##   step from host memory, the parameters never leave the backend);
## - `torch`, `reference`: the step `Train.fit` runs today, traced again every
##   step and evaluated op by op through `tensor/`.
##
## ```sh
## eval "$(sh ../ggml-vulkan/tools/build-ggml.sh | grep '^export ')"
## eval "$(sh ../../tensor/tools/build-torch.sh)"
## export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH:$CPLUS_INCLUDE_PATH"
## agentica bench.agc vulkan sgd 64 256 1024 4096
## agentica bench.agc cpu adam 1024           # also: two (SGD as two graphs)
## agentica bench.agc torch sgd 64 256
## ```
##
## Times are wall clock in Agentica (`Time.monotonic_ns`), so the FFI and the
## VM are inside every step; the ggml rows also say how much of a step the C
## side spent uploading the batch and computing.

import Lower from "lower"
import Optim from "pkg/train/optim"
import Raw from "ggml_graph"
import Step from "step"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

SIZES = [784, 512, 512, 10]
CLASSES = 10
LR = 0.01
## A run takes steps until this much time has passed, between the bounds.
BUDGET_NS = 3000000000
MIN_STEPS = 5
MAX_STEPS = 200

struct Times
  median_ns: Int
  min_ns: Int
  steps: Int
end

struct Runner
  programs: List[Raw.Program]
  inputs: List[Int]
  adam: Optim.Adam
  with_adam: Bool
end

struct EagerStep
  params: List[Tensor.Tensor]
  state: Optim.State
  trace_ns: Int
  eval_ns: Int
  update_ns: Int
end

def main() -> Void
  case run(System.args())
    ok(_) -> nil
    error(e) -> puts("error: #{e.message}")
  end
end

def run(args: List[String]) -> Result[Void]
  return error("usage: bench.agc cpu|cpuN|vulkan|torch|reference sgd|two|adam BATCH...") if args.size < 3
  which = args[0]
  mode = args[1]
  batches: List[Int] = []
  for a in args.drop(2)
    batches = batches.push(try a.to_i)
  end

  init = Step.init(SIZES, 1)
  puts("# #{which}, #{mode}, MLP #{SIZES}, #{CLASSES} classes")
  if which.start_with?("cpu") or which == "vulkan"
    return ok() if !loaded()
    backend = try Raw.backend_cpu(try threads(which))
    if which == "vulkan"
      backend = try Raw.backend_vulkan()
    end

    puts("# #{try Raw.describe(backend)}")
    for b in batches
      try ggml(backend, init, mode, b)
    end
  else
    device = Tensor.Backend.Reference
    if which == "torch"
      device = Tensor.Backend.Torch(Tensor.Device.Cpu)
    end

    return error("#{which} is not available") if !Tensor.available(device)
    for b in batches
      try eager(device, init, mode, b)
    end
  end

  ok()
end

## `cpu` is 8 threads, `cpuN` is N.
def threads(which: String) -> Result[Int]
  return ok(8) if which == "cpu" or which == "vulkan"
  String.slice(which, 3, which.length - 3).to_i
end

def loaded() -> Bool
  case Native.load("ggmlgraph")
    ok(_) -> true
    error(e) ->
      puts("the ggmlgraph module is not built or loadable: SKIPPED (#{e.message})")
      false
  end
end

def ggml(backend: Raw.Backend, init: Step.Init, mode: String, batch: Int) -> Result[Void]
  Native.arena do
    shapes = init.shapes.map { |s| Step.placeholder(s, Tensor.Dtype.F32) }
    x = Step.placeholder([batch, SIZES[0]], Tensor.Dtype.F32)
    y = Step.placeholder([batch], Tensor.Dtype.I64)
    o = Optim.Adam(lr: 0.001)
    t0 = Time.monotonic_ns()
    steps: List[Lower.Step] = []
    case mode
      "sgd" -> steps = [try Step.sgd_step(shapes, x, y, LR, false)]
      "adam" -> steps = [try Step.adam_step(shapes, x, y, o, false)]
      "two" ->
        two = try Step.sgd_two(shapes, x, y, LR)
        steps = [two.grad, two.update]
      _ -> return error("mode is sgd, two or adam")
    end

    t1 = Time.monotonic_ns()
    built: List[Lower.Built] = []
    for s in steps
      built = built.push(try Lower.build(backend, s))
    end

    t2 = Time.monotonic_ns()
    main = try Lower.compile(built[0])
    programs = [main.program]
    if built.size == 2
      update = built[1]
      for k in 0...update.states.size
        try Raw.share(update.program, update.states[k], main.program, main.states[k])
      end

      programs = programs.push((try Lower.compile(update)).program)
    end

    t3 = Time.monotonic_ns()
    for k in 0...init.values.size
      try Raw.set(main.program, main.states[k], init.values[k])
    end

    try Raw.batches(main.program, main.inputs[0], main.inputs[1], CLASSES, 4, 42)
    t4 = Time.monotonic_ns()
    stats = try Raw.stats(main.program)
    run_step = Runner(programs: programs, inputs: main.inputs, adam: o, with_adam: mode == "adam")
    first = try timed_once(run_step, 0)
    for i in 1..2
      _ = try step_once(run_step, i)
    end

    before = try Raw.stats(main.program)
    times = try timed(run_step, 3)
    after = try Raw.stats(main.program)
    runs = (after[4] - before[4]).to_f
    puts(
      "batch #{batch}: #{ms(times.median_ns)} ms/step (min #{ms(times.min_ns)}, #{times.steps} steps), #{rate(batch, times.median_ns)} samples/s",
    )

    puts(
      "  per step in C: upload #{ms(((after[5] - before[5]).to_f / runs).to_i!)} ms, compute+wait #{ms(((after[6] - before[6]).to_f / runs).to_i!)} ms; first step #{ms(first)} ms",
    )

    puts(
      "  once: trace+grad+optimizer #{ms(t1 - t0)} ms, lower #{ms(t2 - t1)} ms, compile #{ms(t3 - t2)} ms (C #{ms(stats[3])}), upload params #{ms(t4 - t3)} ms",
    )

    puts("  graph: #{stats[0]} ggml nodes, leaves #{mib(stats[1])} MiB, compute buffer #{mib(stats[2])} MiB")
    ok()
  end
end

def eager(backend: Tensor.Backend, init: Step.Init, mode: String, batch: Int) -> Result[Void]
  Native.arena do
    params = try Step.params_on(init, backend)
    x = try Step.floats(Step.uniform(batch * SIZES[0], 5, 1.0), [batch, SIZES[0]], backend)
    y = try Step.ints(Step.labels(batch, CLASSES, 6), backend)
    optimizer = Optim.Optimizer.Sgd(Optim.Sgd(lr: LR))
    if mode == "adam"
      optimizer = Optim.Optimizer.Adam(Optim.Adam(lr: 0.001))
    end

    state = try Optim.init(optimizer, params)
    trace_ns = 0
    eval_ns = 0
    update_ns = 0
    times: List[Int] = []
    start = Time.monotonic_ns()
    i = 0
    while i < MIN_STEPS or (Time.monotonic_ns() - start < BUDGET_NS and i < MAX_STEPS)
      t0 = Time.monotonic_ns()
      done = try one_eager(params, x, y, state)
      t4 = Time.monotonic_ns()
      # the first step warms up, unless one step is all the budget allows
      if i > 0 or t4 - t0 > BUDGET_NS
        times = times.push(t4 - t0)
        trace_ns = trace_ns + done.trace_ns
        eval_ns = eval_ns + done.eval_ns
        update_ns = update_ns + done.update_ns
      end

      params = done.params
      state = done.state
      i += 1
      break if i == 1 and t4 - t0 > BUDGET_NS
    end

    n = [times.size, 1].max.unwrap_or(1)
    t = summary(times)
    puts(
      "batch #{batch}: #{ms(t.median_ns)} ms/step (min #{ms(t.min_ns)}, #{t.steps} steps), #{rate(batch, t.median_ns)} samples/s",
    )

    puts(
      "  per step: trace+grad #{ms(trace_ns / n)} ms, evaluate #{ms(eval_ns / n)} ms, optimizer #{ms(update_ns / n)} ms",
    )

    ok()
  end
end

## One step of `Train.fit`, split in its three parts, in its own arena.
def one_eager(params: List[Tensor.Tensor], x: Tensor.Tensor, y: Tensor.Tensor, state: Optim.State) -> Result[EagerStep]
  Native.arena do
    t0 = Time.monotonic_ns()
    ps = params.map { |p| Trace.constant(p) }
    l = try Step.loss(ps, Trace.constant(x), Trace.constant(y))
    grads = try Trace.gradients(l, ps)
    t1 = Time.monotonic_ns()
    values = try Trace.evaluate([l] + grads, {}, Tensor.backend(params[0]))
    _ = try Tensor.to_list(values[0])
    t2 = Time.monotonic_ns()
    next = try Optim.step(state, params, values.drop(1))
    t3 = Time.monotonic_ns()
    ok(EagerStep(params: next.params, state: next.state, trace_ns: t1 - t0, eval_ns: t2 - t1, update_ns: t3 - t2))
  end
end

## One step of a compiled program: Adam's scalars, the run, and the second
## graph when there is one.
def step_once(r: Runner, i: Int) -> Result[Float]
  if r.with_adam
    scalars = Step.adam_scalars(r.adam, i + 1)
    try Raw.set(r.programs[0], r.inputs[2], [scalars[0]])
    try Raw.set(r.programs[0], r.inputs[3], [scalars[1]])
  end

  loss = try Raw.run(r.programs[0], i)
  for p in r.programs.drop(1)
    _ = try Raw.run(p, i)
  end

  ok(loss)
end

def timed_once(f: Runner, i: Int) -> Result[Int]
  t0 = Time.monotonic_ns()
  _ = try step_once(f, i)
  ok(Time.monotonic_ns() - t0)
end

def timed(f: Runner, from: Int) -> Result[Times]
  times: List[Int] = []
  start = Time.monotonic_ns()
  i = from
  while times.size < MIN_STEPS or (Time.monotonic_ns() - start < BUDGET_NS and times.size < MAX_STEPS)
    times = times.push(try timed_once(f, i))
    i += 1
  end

  ok(summary(times))
end

def summary(times: List[Int]) -> Times
  sorted = times.sort
  return Times(median_ns: 0, min_ns: 0, steps: 0) if sorted.empty?
  Times(median_ns: sorted[sorted.size / 2], min_ns: sorted[0], steps: sorted.size)
end

def ms(ns: Int) -> String = format("%.2f", ns.to_f / 1000000.0)

def mib(bytes: Int) -> String = format("%.1f", bytes.to_f / 1048576.0)

def rate(batch: Int, ns: Int) -> String
  return "-" if ns == 0
  format("%.0f", batch.to_f / (ns.to_f / 1000000000.0))
end
