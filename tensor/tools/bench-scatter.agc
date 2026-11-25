## `scatter_add` (the gradient of an embedding lookup) on the ggml backend: the
## op alone for three table sizes, the gradient step of a lookup, and, on
## Vulkan, the round trip through the CPU that would be the other way to do it
## (the updates down, ggml's CPU `get_rows_back`, the answer back up).
##
## ```sh
## eval "$(sh tools/build-engine.sh vulkan | grep '^export ')"
## agentica tools/bench-scatter.agc vulkan
## TENSOR_SCATTER_ADD=get_rows_back agentica tools/bench-scatter.agc vulkan
## agentica tools/bench-scatter.agc cpu
## ```
##
## Vulkan runs in F32 (`vulkan_fp16(false)`, as training does), the only mode the masked matmul is used in.
## `TENSOR_SCATTER_ADD=get_rows_back` makes the Vulkan lowering keep ggml's
## kernel (one process, one choice). Times are wall clock, the median of the
## runs after two warm ones, with the first (which compiles the graph) apart.
## Every index appears twice or more, and the gradient column includes tracing the loss.

import Lower from "../src/lower"
import Types from "../src/types"
import Tensor from "../tensor"
import Trace from "../trace"

## The rows of the table, the indices and the width of a row: the three cases of ml#102, and one with k near the rows.
CASES: List[List[Int]] = [[1000, 256, 64], [10000, 1024, 64], [50000, 4096, 64], [10000, 8192, 64]]
## Runs timed; the median drops the first two, so it is of 18.
RUNS = 20

def main() -> Void
  case run(System.args)
    ok(_) -> nil
    error(e) ->
      IO.warn("bench-scatter: #{e.message}")
      System.exit(1)
  end
end

def run(args: List[String]) -> Result[Void]
  return error("usage: bench-scatter.agc cpu|vulkan") if args.size != 1
  device = Tensor.Device.Cpu
  case args[0]
    "cpu" -> nil
    "vulkan" -> device = Tensor.Device.Vulkan(0)
    _ -> return error("the device is cpu or vulkan, got #{args[0]}")
  end

  try Tensor.vulkan_fp16(false)
  return error("ggml #{args[0]} is not available") if !Tensor.available(Tensor.Backend.Ggml(device))
  puts("Vulkan precision: F32")
  for c in CASES
    try bench(device, c[0], c[1], c[2])
  end

  ok()
end

def bench(device: Tensor.Device, rows: Int, k: Int, width: Int) -> Result[Void]
  cpu = Tensor.Device.Cpu
  ids = try labels((0...k).to_a.map { |i| (i * 7919) % [rows, k / 2].min.unwrap! }, device)
  updates = try floats(values(k * width, 3), [k, width], device)
  table = try floats(values(rows * width, 5), [rows, width], device)
  scatter = try timed(-> { Trace.eval([ids, updates]) do |a| Tensor.scatter_add(a[0], a[1], rows) end })
  step = try timed(-> { lookup_gradient(table, ids) })
  line = "#{label(device)} #{rows} x #{width}, #{k} indices: scatter_add #{scatter}; lookup gradient #{step}"
  if device != cpu
    line = "#{line}; CPU round trip #{try round_trip(device, ids, updates, rows)}"
  end

  puts(line)
  ok()
end

## The gradient of the sum of squares of the rows of `table` that `ids` name: a `scatter_add` of the rows' gradient.
def lookup_gradient(table: Tensor.Tensor, ids: Tensor.Tensor) -> Result[Tensor.Tensor]
  both = try Trace.value_and_grad([table]) do |a|
    rows = try Tensor.gather(a[0], Trace.constant(ids))
    Tensor.sum(try Tensor.mul(rows, rows))
  end

  ok(both.grads[0])
end

## The updates and the indices down to the CPU, the scatter there and the answer back up: what a lowering that
## placed the op on the CPU would pay, as a bound (the answer's copy to the device is part of it).
def round_trip(device: Tensor.Device, ids: Tensor.Tensor, updates: Tensor.Tensor, rows: Int) -> Result[String]
  cpu = Tensor.Device.Cpu
  timed(
    -> {
      up = try Tensor.to_device(updates, cpu)
      at = try Tensor.to_device(ids, cpu)
      answer = try Trace.eval([at, up]) do |a| Tensor.scatter_add(a[0], a[1], rows) end
      Tensor.to_device(answer, device)
    },
  )
end

## `f` run `RUNS` times: "first X ms (n graphs), median Y ms".
def timed(f: Fn[Result[Tensor.Tensor]]) -> Result[String]
  compiled = try Lower.compiles()
  t0 = Time.monotonic_ns()
  try f.call()
  first = Time.monotonic_ns() - t0
  graphs = try Lower.compiles() - compiled
  times: List[Int] = []
  for _ in 0...RUNS
    s0 = Time.monotonic_ns()
    try f.call()
    times = times.push(Time.monotonic_ns() - s0)
  end

  ok("#{format("%.2f", median_ms(times))} ms (first #{format("%.1f", first.to_f / 1000000.0)} ms, #{graphs} graphs)")
end

def label(device: Tensor.Device) -> String
  name = Types.device_name(device)
  return name if device == Tensor.Device.Cpu
  case System.env("TENSOR_SCATTER_ADD")
    ok(mode) -> "#{name} (#{mode})"
    error(_) -> name
  end
end

## The median of `ns` after the two warm-up runs, in milliseconds.
def median_ms(ns: List[Int]) -> Float
  sorted = ns.drop(2).sort
  sorted[sorted.size / 2].to_f / 1000000.0
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

def labels(xs: List[Int], device: Tensor.Device) -> Result[Tensor.Tensor]
  on_cpu = try Tensor.from_list(
    xs.map { |x| Tensor.Values.Int(x) },
    [xs.size],
    Tensor.Dtype.I32,
    Tensor.Backend.Ggml(Tensor.Device.Cpu),
  )

  Tensor.to_device(on_cpu, device)
end
