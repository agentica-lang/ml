## Prints the correctness gaps against the reference backend
## (`compare.agc`): one SGD step (one graph and two) and two Adam steps of a
## small MLP, and two SGD steps of the benchmark's MLP (784 → 512 → 512 → 10,
## batch 256, about 35 s of reference time), on ggml's CPU backend and on the
## iGPU. The iGPU's `mul_mat` is F32 unless `fp16` is given; ggml fixes that
## choice per process (`backend_vulkan`).
##
## ```sh
## agentica check.agc                # everything, F32 on the iGPU
## agentica check.agc fp16           # fp16 mul_mat on the iGPU (opt-in)
## agentica check.agc small          # without the benchmark's model
## ```

import Compare from "compare"
import Raw from "ggml_graph"

def main() -> Void
  args = System.args()
  case Native.load("ggmlgraph")
    ok(_) ->
      case run(!args.include?("small"), args.include?("fp16"))
        ok(_) -> nil
        error(e) -> puts("error: #{e.message}")
      end

    error(e) -> puts("the ggmlgraph module is not built or loadable: SKIPPED (#{e.message})")
  end
end

def run(with_big: Bool, fp16: Bool) -> Result[Void]
  backends = [try Raw.backend_cpu(4)]
  names = ["ggml CPU"]
  case Raw.backend_vulkan(fp16)
    ok(gpu) ->
      backends = backends.push(gpu)
      if fp16
        names = names.push("ggml Vulkan, fp16 mul_mat")
      else
        names = names.push("ggml Vulkan, F32 mul_mat")
      end

    error(e) -> puts("ggml Vulkan: no device (#{e.message})")
  end

  for i in 0...backends.size
    puts("#{names[i]}: #{try Raw.describe(backends[i])}")
    line("sgd, one graph", try Compare.sgd(backends[i]))
    line("sgd, two graphs", try Compare.sgd_two(backends[i]))
    line("adam, 2 steps", try Compare.adam(backends[i]))
  end

  if backends.size > 1
    puts("benchmark model, #{Compare.BIG_STEPS} SGD steps at batch #{Compare.BIG_BATCH}, data made in C:")
    line("#{names[1]} against ggml CPU", try Compare.cross(backends[1], backends[0]))
  end

  if with_big
    puts("benchmark model, #{Compare.BIG_STEPS} SGD steps at batch #{Compare.BIG_BATCH}:")
    ref = try Compare.big_reference()
    for i in 0...backends.size
      line(names[i], try Compare.big(backends[i], ref))
    end
  end

  ok()
end

def line(name: String, g: Compare.Gaps) -> Void
  puts(format("  %-30s loss %s  grads %s  params %s", name, sci(g.loss), sci(g.grads), sci(g.params)))
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
