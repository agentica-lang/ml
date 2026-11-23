## What ggml costs per op, on the backends this machine has: the CPU with AVX2
## at one thread and at eight, and the Vega iGPU through Vulkan.
##
## Every number here is a wall-clock measurement of one module call plus the
## close of the tensor it made, taken in Agentica with `Time.monotonic_ns`.
## That means the FFI dispatch (M0 measured 8-15 µs for a `compute def`) and
## the tensor allocation are inside every row, which is the point: this is
## what an op-by-op backend over ggml would cost. The fixed-cost section says
## how much of each row is not arithmetic.
##
## One backend per invocation, because this is a laptop APU: the CPU and the
## iGPU share a power and thermal budget, and measuring them in one process
## makes whichever runs second look slow. From `spike/ggml-vulkan/`, with ggml
## built by `tools/build-ggml.sh`:
##
## ```sh
## eval "$(sh tools/build-ggml.sh)"
## agentica bench.agc cpu1
## agentica bench.agc cpu8
## agentica bench.agc vulkan                     # fp16 mul_mat (ggml's default here)
## GGML_VK_DISABLE_F16=1 agentica bench.agc vulkan  # the same, fp32 mul_mat
## ```

import Ggml from "ggml_spike"

## The sizes that matter: the small square a block of a model multiplies, the
## middle one, the largest this spike runs, and one transformer-shaped product
## (`512 × 512` by `512 × 2048`).
SHAPES: List[Shape] = [
  Shape(name: "256^3", m: 256, n: 256, k: 256),
  Shape(name: "512^3", m: 512, n: 512, k: 512),
  Shape(name: "1024^3", m: 1024, n: 1024, k: 1024),
  Shape(name: "512x2048, k512", m: 512, n: 2048, k: 512),
]

## One `mul_mat` shape: `a` is `[m, k]` and `b` is `[n, k]`, both row-major, so
## the product ggml computes (`a · bᵀ`) is `[m, n]` and the work is
## `2 * m * n * k` floating point operations.
struct Shape
  name: String
  m: Int
  n: Int
  k: Int
end

def main() -> Void
  case Native.load("ggmlspike")
    ok(_) ->
      case run()
        ok(_) -> puts("")
        error(e) -> fail(e)
      end

    error(e) ->
      puts("the ggmlspike module is not built or loadable: SKIPPED")
      puts("  #{e.message}")
      puts("  (ggml is a system dependency of this spike: sh tools/build-ggml.sh)")
  end
end

## `cpu1`, `cpu8` or `vulkan`: the backend this process measures. One backend
## per run, so nothing measured earlier is still heating the machine.
def run() -> Result[Void]
  args = System.args()
  which = "all"
  if args.size > 0
    which = args[0]
  end

  puts("# bench: one module call per row, plus the close of the tensor it made")
  if which == "cpu1" || which == "all"
    cpu = try Ggml.backend_cpu(1)
    try backend("cpu, 1 thread", cpu, false)
  end

  if which == "cpu8" || which == "all"
    cpu = try Ggml.backend_cpu(8)
    try backend("cpu, 8 threads", cpu, false)
  end

  if which == "vulkan" || which == "all"
    gpu = try Ggml.backend_vulkan()
    try backend("vulkan", gpu, true)
  end

  ok()
end

## The fixed costs, the four `mul_mat` rows, and — on the iGPU — the warm-up
## sequence and the memory the largest case allocates.
def backend(label: String, backend: Ggml.Backend, gpu: Bool) -> Result[Void]
  puts("")
  puts("# #{label}: #{try Ggml.describe(backend)}")
  puts("# fixed costs: the part of every row below that is not arithmetic")
  puts(format("%-16s %-22s %10s %10s %10s", "backend", "what", "median", "min", "max"))
  try fixed_costs(label, backend)

  puts("")
  puts("# mul_mat, a[m, k] . b[n, k]^T, one graph and one submit per op")
  puts(format("%-22s %-14s %9s %9s %8s %8s", "shape", "backend", "median ms", "min ms", "GFLOPS", "GFLOPS"))
  puts(format("%-22s %-14s %9s %9s %8s %8s", "", "", "", "", "(median)", "(min)"))
  for shape in SHAPES
    try measure(backend, label, shape, rounds_for(shape))
  end

  if gpu
    puts("")
    try warmup(backend)
    puts("")
    try memory_report(backend)
  end

  ok()
end

## Rounds per row: the smallest shape is cheap enough to repeat, the largest
## is not. Same counts as `tools/bench-torch.cpp`.
def rounds_for(shape: Shape) -> Int
  flops = 2 * shape.m * shape.n * shape.k
  if flops >= 2_147_483_648
    4
  else
    if flops >= 268_435_456
      7
    else
      11
    end
  end
end

## One dispatch with no op in it, one 4 MB tensor made and freed, and reading
## it back (as a checksum, and as a `List[Float]`, which pays the per-value
## decode M0 measured at ~20 ns per value).
def fixed_costs(label: String, backend: Ggml.Backend) -> Result[Void]
  Native.arena do
    rounds = 200
    empty: List[Int] = []
    i = 0
    while i < rounds
      start = Time.monotonic_ns()
      try Ggml.empty_graph(backend)
      empty = empty.push(Time.monotonic_ns() - start)
      i += 1
    end

    row(label, "empty graph", empty)

    big = try Ggml.create(backend, [1024, 1024])
    created: List[Int] = []
    i = 0
    while i < 20
      start = Time.monotonic_ns()
      t = try Ggml.create(backend, [1024, 1024])
      Native.close(t)
      created = created.push(Time.monotonic_ns() - start)
      i += 1
    end

    row(label, "create 1024x1024", created)

    read: List[Int] = []
    i = 0
    while i < 20
      start = Time.monotonic_ns()
      _ = try Ggml.checksum(big)
      read = read.push(Time.monotonic_ns() - start)
      i += 1
    end

    row(label, "read back 4 MiB", read)

    small = try Ggml.create(backend, [64, 64])
    listed: List[Int] = []
    i = 0
    while i < 20
      start = Time.monotonic_ns()
      _ = try Ggml.readback(small)
      listed = listed.push(Time.monotonic_ns() - start)
      i += 1
    end

    row(label, "read back 4096 as List", listed)
    ok()
  end
end

## One row of the fixed-cost table: median, min and max in milliseconds.
def row(label: String, what: String, samples: List[Int]) -> Void
  ordered = samples.sort_by do |ns: Int| ns end
  median = ordered[ordered.size / 2]
  puts(
    format(
      "%-16s %-22s %10.3f %10.3f %10.3f",
      label,
      what,
      ns_ms(median),
      ns_ms(ordered[0]),
      ns_ms(ordered[ordered.size - 1]),
    ),
  )
end

def ns_ms(ns: Int) -> Float = ns.to_f / 1_000_000.0

## One row of the mul_mat table: a fresh `a` and `b` on `backend`, one warm-up
## op, then `rounds` timed calls.
def measure(backend: Ggml.Backend, label: String, shape: Shape, rounds: Int) -> Result[Void]
  Native.arena do
    a = try Ggml.create(backend, [shape.m, shape.k])
    b = try Ggml.create(backend, [shape.n, shape.k])
    warm = try Ggml.mul_mat(a, b)
    Native.close(warm)
    samples: List[Int] = []
    i = 0
    while i < rounds
      start = Time.monotonic_ns()
      c = try Ggml.mul_mat(a, b)
      Native.close(c)
      samples = samples.push(Time.monotonic_ns() - start)
      i += 1
    end

    result_row(label, shape, samples)
    ok()
  end
end

## `GFLOPS` is `flops / ns`: `2 * m * n * k` operations in `ns` nanoseconds is
## that many flops per nanosecond, which is gigaflops per second.
def result_row(label: String, shape: Shape, samples: List[Int]) -> Void
  ordered = samples.sort_by do |ns: Int| ns end
  median = ordered[ordered.size / 2]
  flops = 2.0 * shape.m.to_f * shape.n.to_f * shape.k.to_f
  puts(
    format(
      "%-22s %-14s %9.3f %9.3f %8.1f %8.1f",
      shape.name,
      label,
      ns_ms(median),
      ns_ms(ordered[0]),
      flops / median.to_f,
      flops / ordered[0].to_f,
    ),
  )
end

## The iGPU's first calls, one at a time: a laptop APU ramps its clocks, so the
## first op is not what the steady state pays, and one op every few
## milliseconds keeps it slow. This is the warm-up the table above hides.
def warmup(backend: Ggml.Backend) -> Result[Void]
  Native.arena do
    puts("# vulkan warm-up: 512^3 mul_mat, one submit per call, in order")
    a = try Ggml.create(backend, [512, 512])
    b = try Ggml.create(backend, [512, 512])
    i = 0
    while i < 10
      start = Time.monotonic_ns()
      c = try Ggml.mul_mat(a, b)
      Native.close(c)
      puts(format("  call %2d  %8.3f ms", i + 1, ns_ms(Time.monotonic_ns() - start)))
      i += 1
    end

    ok()
  end
end

## What the backend allocates for the largest case, and what the device says it
## has: on a Vega iGPU both of those are host memory.
def memory_report(backend: Ggml.Backend) -> Result[Void]
  Native.arena do
    a = try Ggml.create(backend, [1024, 1024])
    b = try Ggml.create(backend, [1024, 1024])
    c = try Ggml.mul_mat(a, b)
    puts("# memory, largest case (1024^3)")
    puts("  a 4 MiB of values, backend buffer #{try Ggml.allocated(a)} bytes")
    puts("  b 4 MiB of values, backend buffer #{try Ggml.allocated(b)} bytes")
    puts("  c 4 MiB of values, backend buffer #{try Ggml.allocated(c)} bytes")
    memory = try Ggml.memory(backend)
    puts("  device memory: free #{memory[0]}, total #{memory[1]}")
    ok()
  end
end

def fail(e: Error) -> Void
  puts("bench: #{e.message}")
  System.exit(1)
end
