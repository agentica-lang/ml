## The ggml-vulkan spike, as a program: the boundary doing something real on
## both of ggml's backends, and the assertion that the Vulkan half really ran
## on the iGPU.
##
## It enumerates ggml's devices, opens the CPU backend (8 threads) and the
## Vulkan backend, and on each one: makes two matrices, adds them, relus the
## sum, multiplies them, and reads the results back. The checksums of the two
## backends are compared, so a Vulkan backend that silently ran on the CPU
## would be a number this program still has to explain — and the device type
## printed for each backend is the thing that says which one ran.
##
## From `spike/ggml-vulkan/`, with ggml built by `tools/build-ggml.sh` (without
## it the whole program says the module is unbuilt):
##
## ```sh
## eval "$(sh tools/build-ggml.sh)"
## agentica demo.agc
## ```

import Ggml from "ggml_spike"

## The matrix the demo works on: 64 by 64, the shape `tensor/tools/bench.agc`
## calls wide. Small enough to read back every value, large enough that the
## ops do arithmetic.
SIDE = 64

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

def run() -> Result[Void]
  puts("# ggml devices, as ggml_backend_dev_count/ggml_backend_dev_get report them")
  for device in try Ggml.devices()
    puts("  #{device}")
  end

  cpu = try Ggml.backend_cpu(8)
  gpu = try Ggml.backend_vulkan()
  puts("")
  puts("# describe: what the backend in use really is, read back from ggml")
  puts("  cpu:    #{try Ggml.describe(cpu)}")
  puts("  vulkan: #{try Ggml.describe(gpu)}")
  puts("")
  puts("# device memory (free, total) in bytes")
  try print_memory("cpu", cpu)
  try print_memory("vulkan", gpu)

  puts("")
  puts("# the same five calls on both backends, [64, 64] F32")
  cpu_sum = try ops("cpu", cpu)
  gpu_sum = try ops("vulkan", gpu)
  gap = (cpu_sum - gpu_sum).abs
  relative = gap / cpu_sum.abs
  puts("")
  puts(format("checksum of mul_mat(add, add): cpu %.6f, vulkan %.6f", cpu_sum, gpu_sum))
  puts(format("gap %.6f, %.6f of the cpu sum (the Vulkan default computes mul_mat in fp16 here)", gap, relative))
  if relative <= 0.002
    puts("the two backends agree: the Vulkan run computed the same tensor")
  else
    puts("MISMATCH: the Vulkan run did not compute the same tensor")
    System.exit(1)
  end

  puts("")
  puts("# a shape error is a Result error, not a halt, and it names the shapes")
  wide = try Ggml.create(cpu, [2, 3])
  tall = try Ggml.create(cpu, [3, 2])
  case Ggml.add(wide, tall)
    ok(_) -> puts("  add [2, 3] + [3, 2] returned a tensor: ggml broadcast, the module did not refuse")
    error(e) -> puts("  add [2, 3] + [3, 2]: #{e.message}")
  end

  puts("")
  puts("# the cost of one dispatch round trip with no op in it, 200 rounds")
  try print_dispatch_cost("cpu, 8 threads", cpu)
  try print_dispatch_cost("vulkan", gpu)
  ok()
end

## add, relu and mul_mat on `backend`, over the deterministic pattern the
## module fills every tensor with, and the checksum of the product. Each
## intermediate is closed as a step would close it.
def ops(label: String, backend: Ggml.Backend) -> Result[Float]
  a = try Ggml.create(backend, [SIDE, SIDE])
  b = try Ggml.create(backend, [SIDE, SIDE])
  puts(format("  %-7s a %s, %d bytes on the backend", label, shape_text(SIDE), try Ggml.allocated(a)))
  sum = try Ggml.add(a, b)
  positive = try Ggml.relu(sum)
  product = try Ggml.mul_mat(positive, b)
  puts(format("  %-7s relu(add) checksum %.6f", label, try Ggml.checksum(positive)))
  puts(
    format("  %-7s mul_mat checksum  %.6f (%d bytes)", label, try Ggml.checksum(product), try Ggml.allocated(product)),
  )

  puts(format("  %-7s first values of mul_mat: %s", label, try first_values(product)))
  result = try Ggml.checksum(product)
  Native.close(a)
  Native.close(b)
  Native.close(sum)
  Native.close(positive)
  Native.close(product)
  ok(result)
end

## The first four values of a tensor, read back as a `List[Float]`: the slow
## path (M0 measured ~20 ns per value), used here for two numbers, not timing.
def first_values(t: Ggml.Tensor) -> Result[List[Float]]
  values = try Ggml.readback(t)
  first: List[Float] = []
  i = 0
  while i < 4 && i < values.size
    first = first.push(values[i])
    i += 1
  end

  ok(first)
end

def print_memory(label: String, backend: Ggml.Backend) -> Result[Void]
  memory = try Ggml.memory(backend)
  puts(format("  %-7s free %d, total %d", label, memory[0], memory[1]))
  ok()
end

## `rounds` empty graphs on `backend` and the median: the fixed cost of one
## dispatch, which every op in the benchmark pays on top of its arithmetic.
def print_dispatch_cost(label: String, backend: Ggml.Backend) -> Result[Void]
  rounds = 200
  samples: List[Int] = []
  i = 0
  while i < rounds
    start = Time.monotonic_ns()
    try Ggml.empty_graph(backend)
    samples = samples.push(Time.monotonic_ns() - start)
    i += 1
  end

  ordered = samples.sort_by do |ns: Int| ns end
  puts(
    format(
      "  %-16s median %6d ns, min %6d ns, max %6d ns",
      label,
      ordered[rounds / 2],
      ordered[0],
      ordered[rounds - 1],
    ),
  )

  ok()
end

def shape_text(side: Int) -> String = "[#{side}, #{side}] F32"

def fail(e: Error) -> Void
  puts("demo: #{e.message}")
  System.exit(1)
end
