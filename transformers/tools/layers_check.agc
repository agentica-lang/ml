## Checks that `layers_on_device` changes no token: greedy generation from one prompt gives the same ids with every block
## on the device, with half of them and with none (the rest run on the CPU, with their KV cache in host memory). On a
## CPU backend the three runs are the same model. Manual (it needs the checkpoint); run from `transformers/`:
##
##   agentica tools/layers_check.agc MODEL_DIR_OR_GGUF [cpu|vulkan]
##
## `WEIGHTS` (`q4_k` for a Q4_K_M file) says how the matrices are held, and `AGENTICA_ML_KV=host` keeps the KV cache in host
## memory (`Placement.Kv.Host`).

import Generate from "../generate"
import Placement from "../placement"
import Qwen3 from "../qwen3"
import Load from "pkg/hub/load"
import Tensor from "pkg/tensor/tensor"

## "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n" as Qwen3 reads it.
PROMPT = [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198, 151644, 77091, 198]
STOP = [151645, 151643]
NEW = 24

def main() -> Void
  args = System.args
  if args.size < 1
    puts("usage: layers_check.agc MODEL_DIR_OR_GGUF [cpu|vulkan]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 1 and args[1] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  case run(args[0], Tensor.Backend.Ggml(device))
    ok(_) -> puts("the same ids with every block on the device, half of them and none")
    error(e) ->
      puts("layers_check: #{e.message}")
      System.exit(1)
  end
end

private

def run(dir: String, backend: Tensor.Backend) -> Result[Void]
  try Load.vulkan_fp16_from_env()
  weights = try Load.weights_from_env()
  blocks = (try opened(dir, backend, weights, none)).config.num_hidden_layers
  want = try generated(dir, backend, weights, none)
  puts("all #{blocks}: #{want}")
  for n in [blocks / 2, 0]
    got = try generated(dir, backend, weights, some(n))
    puts("layers_on_device #{n}: #{got}")
    return error("layers_on_device #{n} gave #{got}, every block on the device #{want}") if got != want
  end

  ok()
end

def opened(dir: String, backend: Tensor.Backend, weights: Tensor.Dtype, n: Optional[Int]) -> Result[Qwen3.Qwen3]
  m = try Qwen3.open(dir, backend, weights, n)
  Qwen3.close_model(m)
  ok(m)
end

def generated(dir: String, backend: Tensor.Backend, weights: Tensor.Dtype, n: Optional[Int]) -> Result[List[Int]]
  m = try Qwen3.open(dir, backend, weights, n)
  g = try Generate.open(m, kv: try Placement.kv_from_env())
  ids = Generate.generate(g, PROMPT, NEW, STOP)
  Generate.close(g)
  Qwen3.close_model(m)
  ids
end
