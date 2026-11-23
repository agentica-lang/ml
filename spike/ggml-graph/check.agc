## Prints the correctness gaps of one SGD step (one graph and two), and of
## two Adam steps, on ggml's CPU backend and on the iGPU through Vulkan,
## against the reference backend (`compare.agc`).
##
## ```sh
## eval "$(sh ../ggml-vulkan/tools/build-ggml.sh | grep '^export ')"
## agentica check.agc
## GGML_VK_DISABLE_F16=1 agentica check.agc      # fp32 matmul on the iGPU
## ```

import Compare from "compare"
import Raw from "ggml_graph"

def main() -> Void
  case Native.load("ggmlgraph")
    ok(_) ->
      case run()
        ok(_) -> nil
        error(e) -> puts("error: #{e.message}")
      end

    error(e) -> puts("the ggmlgraph module is not built or loadable: SKIPPED (#{e.message})")
  end
end

def run() -> Result[Void]
  cpu = try Raw.backend_cpu(4)
  try report("ggml CPU", cpu)
  case Raw.backend_vulkan()
    ok(gpu) -> try report("ggml Vulkan", gpu)
    error(e) -> puts("ggml Vulkan: no device (#{e.message})")
  end

  ok()
end

def report(name: String, backend: Raw.Backend) -> Result[Void]
  puts("#{name}: #{try Raw.describe(backend)}")
  line("sgd, one graph", try Compare.sgd(backend))
  line("sgd, two graphs", try Compare.sgd_two(backend))
  line("adam, one graph, 2 steps", try Compare.adam(backend))
  ok()
end

def line(name: String, g: Compare.Gaps) -> Void
  puts(format("  %-26s loss %s  grads %s  params %s", name, sci(g.loss), sci(g.grads), sci(g.params)))
end

## `x` as `d.dde-N`.
def sci(x: Float) -> String
  return "0" if x == 0.0
  m = x
  e = 0
  while m < 1.0
    m = m * 10.0
    e -= 1
  end

  while m >= 10.0
    m = m / 10.0
    e += 1
  end

  format("%.2fe%d", m, e)
end
