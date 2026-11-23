## Steps per second of `fit` with the batches built on demand (`prefetch: 0`)
## and by the prefetch task (`prefetch: 2`), with a step made slow on purpose
## so the overlap shows. Not a test: run from `train/`,
##
## ```sh
## agentica tools/prefetch_bench.agc [ref|torch] [delay_ms] [rows] [width] [batch]
## ```

import Data from "../data"
import Loss from "../loss"
import Optim from "../optim"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"

EPOCHS = 2

struct Setup
  backend: Tensor.Backend
  delay: Int
  rows: Int
  width: Int
  batch: Int
end

def main() -> Void
  case run()
    ok(_) -> nil
    error(e) ->
      puts("bench: #{e.message}")
      System.exit(1)
  end
end

def run() -> Result[Void]
  args = System.args
  setup = Setup(
    backend: backend(arg(args, 0, "ref")),
    delay: try arg(args, 1, "20").to_i,
    rows: try arg(args, 2, "1024").to_i,
    width: try arg(args, 3, "32").to_i,
    batch: try arg(args, 4, "128").to_i,
  )

  d = try dataset(setup)
  try build_only(Data.Loader(dataset: d, batch_size: setup.batch, seed: 1))
  for prefetch in [0, 2]
    loader = Data.Loader(dataset: d, batch_size: setup.batch, seed: 1, prefetch: prefetch)
    steps = Data.count(loader) * EPOCHS
    slow = ->(ps: List[Tensor.Tensor], b: Data.Batch) { slow_loss(setup.delay, ps, b) }
    began = Time.monotonic_ms
    _ = try Train.fit(
      slow,
      [try Tensor.zeros([setup.width, 1], Tensor.Dtype.F32, setup.backend)],
      loader,
      sgd(),
      EPOCHS,
    )

    ms = Time.monotonic_ms - began
    puts("prefetch #{prefetch}: #{steps} steps in #{ms} ms, #{format("%.2f", steps.to_f * 1000.0 / ms.to_f)} steps/s")
  end

  ok()
end

## The cost the prefetch task can hide: building every batch of one epoch.
def build_only(loader: Data.Loader) -> Result[Void]
  began = Time.monotonic_ms
  rows = Data.order(loader, 1)
  for i in 0...Data.count(loader)
    Data.release(try Data.batch(loader, rows, i))
  end

  ms = Time.monotonic_ms - began
  puts("building a batch: #{format("%.2f", ms.to_f / Data.count(loader).to_f)} ms")
  ok()
end

def sgd() -> Optim.Optimizer = Optim.Optimizer.Sgd(Optim.Sgd(lr: 0.0001))

def slow_loss(delay: Int, ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  Task.sleep(delay)
  rows = Tensor.shape(b.inputs)[0]
  Loss.mse(try Tensor.reshape(try Tensor.matmul(b.inputs, ps[0]), [rows]), b.targets)
end

def dataset(s: Setup) -> Result[Data.Dataset]
  xs = (0...(s.rows * s.width)).to_a.map { |i| Tensor.Values.Float((i % 17).to_f / 17.0) }
  ys = (0...s.rows).to_a.map { |i| Tensor.Values.Float((i % 5).to_f) }
  Data.dataset(
    try Tensor.from_list(xs, [s.rows, s.width], Tensor.Dtype.F32, s.backend),
    try Tensor.from_list(ys, [s.rows], Tensor.Dtype.F32, s.backend),
  )
end

def backend(name: String) -> Tensor.Backend
  return Tensor.Backend.Torch(Tensor.Device.Cpu) if name == "torch"
  Tensor.Backend.Reference
end

def arg(args: List[String], i: Int, default: String) -> String
  return args[i] if i < args.size
  default
end
