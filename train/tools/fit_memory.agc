## Shows that `Train.fit` frees what a step replaces: a fit of one parameter of 16,000,000 `F32` values (64 MB, with
## Adam's two moments) on ggml CPU, for STEPS steps. Its peak memory does not grow with STEPS (0.9 to 1.0 GB at 6 and
## at 30 steps when it was written; 1.91 GB at 6 steps before `fit` freed anything). Manual; from `train/`:
##
## ```sh
## /usr/bin/time -v agentica tools/fit_memory.agc STEPS 2>&1 | grep -e "^steps" -e "Maximum resident"
## ```

import Data from "../data"
import Optim from "../optim"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"

VALUES = 16000000

def main() -> Void
  args = System.args
  if args.size != 1 or !args[0].to_i.ok?
    puts("usage: fit_memory.agc STEPS")
    System.exit(2)
  end

  steps = args[0].to_i!
  case run(steps)
    ok(loss) -> puts("steps #{steps} loss #{loss}")
    error(e) ->
      puts("fit_memory: #{e.message}")
      System.exit(1)
  end
end

private

def loss(ps: List[Tensor.Tensor], _b: Data.Batch) -> Result[Tensor.Tensor] = Tensor.mean(try Tensor.mul(ps[0], ps[0]))

## One step per epoch: the dataset is one row, which the loss does not read.
def run(steps: Int) -> Result[Float]
  return error("STEPS must be at least 1, got #{steps}") if steps < 1
  backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  p = try Tensor.full([VALUES], Tensor.Values.Float(0.5), Tensor.Dtype.F32, backend)
  one = try Tensor.full([1, 1], Tensor.Values.Float(1.0), Tensor.Dtype.F32, backend)
  loader = Data.Loader(dataset: try Data.dataset(one, one), batch_size: 1, shuffle: false, prefetch: 0)
  fit = try Train.fit(&loss, [p], loader, Optim.Optimizer.Adam(Optim.Adam(lr: 0.01)), steps)
  ok(fit.history[steps - 1])
end
