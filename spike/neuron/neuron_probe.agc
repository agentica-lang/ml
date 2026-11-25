## Spike ml#80: plugin open, compile seconds per shape, overhead per run, MLP forward ms per batch. From tensor/:
## agentica tools/neuron_probe.agc [BATCHES...]   (AGENTICA_ML_XLA_PLUGIN picks the plugin)

import Tensor from "../tensor"
import Trace from "../trace"

XLA = Tensor.Backend.Xla(Tensor.Device.Cpu)
RUNS = 200

def main() -> Void
  case probe(System.args())
    ok(_) -> nil
    error(e) ->
      puts("probe: #{e.message}")
      System.exit(2)
  end
end

def probe(args: List[String]) -> Result[Void]
  t0 = Time.monotonic_ns()
  return error("xla not available") if !Tensor.available(XLA)
  puts("{\"what\": \"open\", \"ms\": #{ms(Time.monotonic_ns() - t0)}}")

  a = try floats(ramp(4), [4])
  b = try floats(ramp(4), [4])
  t1 = Time.monotonic_ns()
  add = try Trace.compile(
    [Trace.spec_of(a), Trace.spec_of(b)],
    ->(xs: List[Tensor.Tensor]) { ok([try Tensor.add(xs[0], xs[1])]) },
  )

  puts("{\"what\": \"compile add[4]\", \"ms\": #{ms(Time.monotonic_ns() - t1)}}")
  puts("{\"what\": \"run add[4]\", \"median_ms\": #{timed(add, [a, b])}}")
  t2 = Time.monotonic_ns()
  for _ in 0...RUNS
    _ = try Tensor.to_list((try Trace.run(add, [a, b]))[0])
  end

  puts("{\"what\": \"run+read add[4]\", \"mean_ms\": #{ms((Time.monotonic_ns() - t2) / RUNS)}}")

  batches = [64, 256, 1024]
  if args.size > 0
    batches = args.map { |s| s.to_i.unwrap! }
  end

  params = [
    try floats(ramp(784 * 512), [784, 512]),
    try floats(ramp(512), [512]),
    try floats(ramp(512 * 512), [512, 512]),
    try floats(ramp(512), [512]),
    try floats(ramp(512 * 10), [512, 10]),
    try floats(ramp(10), [10]),
  ]

  for n in batches
    x = try floats(ramp(n * 784), [n, 784])
    args2 = [x] + params
    t3 = Time.monotonic_ns()
    f = try Trace.compile(args2.map { |t| Trace.spec_of(t) }, &forward)
    c = ms(Time.monotonic_ns() - t3)
    puts("{\"what\": \"mlp forward\", \"batch\": #{n}, \"compile_ms\": #{c}, \"median_ms\": #{timed(f, args2)}}")
  end

  ok()
end

def forward(a: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  h = try Tensor.relu(try Tensor.linear(a[0], a[1], some(a[2])))
  h = try Tensor.relu(try Tensor.linear(h, a[3], some(a[4])))
  ok([try Tensor.linear(h, a[5], some(a[6]))])
end

def timed(f: Trace.Compiled, args: List[Tensor.Tensor]) -> String
  ns: List[Int] = []
  for _ in 0...RUNS
    t = Time.monotonic_ns()
    _ = Trace.run(f, args)
    ns = ns.push(Time.monotonic_ns() - t)
  end

  ms(ns.sort[ns.size / 2])
end

def ramp(count: Int) -> List[Float]
  out: List[Float] = []
  for i in 0...count
    out = out.push(((i % 97).to_f - 48.0) / 4800.0)
  end

  out
end

def floats(xs: List[Float], shape: List[Int]) -> Result[Tensor.Tensor]
  Tensor.from_list(xs.map { |x| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, XLA)
end

def ms(ns: Int) -> String = format("%.3f", ns.to_f / 1000000.0)
