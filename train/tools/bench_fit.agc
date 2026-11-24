## Times `Train.fit` on a relu MLP classifier and the loader alone, to see how much of a training step the device spends
## waiting for the host. Manual, like asr/tools/bench_step.agc. It prints tab-separated lines: `epoch N MS` (the wall time
## of each epoch of `fit`, the first holds the compile), `fit MS_PER_STEP` (the mean of the epochs after the first) and
## `loader MS_PER_BATCH` (the batches of one epoch taken and released with nothing in between). With
## `AGENTICA_VM_TRACE=profile` the `ml.*` sections of the report split a step (`ml.trace`, `ml.plan`, `ml.acquire`,
## `ml.bind`, `ml.run`, `ml.readback`). From `train/`:
##
## ```sh
## agentica tools/bench_fit.agc BACKEND ROWS BATCH PREFETCH [WIDTH [EPOCHS]]
## ```
##
## BACKEND is `cpu` (ggml CPU) or `vulkan`; the model is 256 inputs, two hidden layers of WIDTH (512 by default) and 10
## classes, trained by Adam on ROWS random rows in batches of BATCH, with PREFETCH batches built ahead (0 builds each one
## when it is taken); EPOCHS defaults to 4.

import Data from "../data"
import Loss from "../loss"
import Optim from "../optim"
import Tensor from "pkg/tensor/tensor"
import Train from "../train"

IN = 256
CLASSES = 10

def main() -> Void
  case run(System.args())
    ok(_) -> nil
    error(e) ->
      puts("bench_fit: #{e.message}")
      System.exit(2)
  end
end

def run(args: List[String]) -> Result[Void]
  return error("usage: BACKEND ROWS BATCH PREFETCH [WIDTH [EPOCHS]]") if args.size < 4
  backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  if args[0] == "vulkan"
    backend = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
  end

  return error("#{args[0]} is not available") if not Tensor.available(backend)
  rows = try args[1].to_i
  batch = try args[2].to_i
  prefetch = try args[3].to_i
  width = 512
  if args.size > 4
    width = try args[4].to_i
  end

  epochs = 4
  if args.size > 5
    epochs = try args[5].to_i
  end

  dataset = try dataset_of(rows, backend)
  loader = Data.Loader(dataset: dataset, batch_size: batch, shuffle: true, seed: 7, prefetch: prefetch)
  try time_loader(loader)
  time_fit(loader, try params_of(width, backend), epochs)
end

def time_loader(loader: Data.Loader) -> Result[Void]
  feed = try Data.feed(loader, 1)
  n = Data.count(loader)
  started = Time.monotonic_ms()
  for i in 0...n
    Data.release(try Data.take(Data.epoch(feed, 1), i))
  end

  elapsed = Time.monotonic_ms() - started
  Data.stop(feed)
  puts("loader\t#{elapsed.to_f / n.to_f}")
  ok()
end

def time_fit(loader: Data.Loader, params: List[Tensor.Tensor], epochs: Int) -> Result[Void]
  stamps: Channel[Int] = Channel.new(epochs + 1)
  started = Time.monotonic_ms()
  try stamps.send(started, 0)
  fit = try Train.fit(
    &loss,
    params,
    loader,
    Optim.Optimizer.Adam(Optim.Adam(lr: 0.001)),
    epochs,
    some(->(_: Train.Epoch) { tick(stamps) }),
  )

  last = try stamps.recv(0)
  after = 0
  for e in 1..epochs
    now = try stamps.recv(0)
    puts("epoch\t#{e}\t#{now - last}\t#{fit.history[e - 1]}")
    if e > 1
      after += now - last
    end

    last = now
  end

  if epochs > 1
    puts("fit\t#{after.to_f / ((epochs - 1) * Data.count(loader)).to_f}")
  end

  ok()
end

def tick(stamps: Channel[Int]) -> Void
  _ = stamps.send(Time.monotonic_ms(), 0)
end

def loss(ps: List[Tensor.Tensor], b: Data.Batch) -> Result[Tensor.Tensor]
  h = try Tensor.relu(
    try Tensor.add(
      try Tensor.matmul(b.inputs, ps[0]),
      try Tensor.broadcast_to(ps[1], [Tensor.shape(b.inputs)[0], Tensor.shape(ps[1])[0]]),
    ),
  )

  h = try Tensor.relu(
    try Tensor.add(
      try Tensor.matmul(h, ps[2]),
      try Tensor.broadcast_to(ps[3], [Tensor.shape(h)[0], Tensor.shape(ps[3])[0]]),
    ),
  )

  logits = try Tensor.add(try Tensor.matmul(h, ps[4]), try Tensor.broadcast_to(ps[5], [Tensor.shape(h)[0], CLASSES]))
  Loss.cross_entropy(logits, b.targets)
end

## Numbers in `-0.5...0.5`, from a linear congruence of `seed`: reproducible, not statistical.
def numbers(n: Int, seed: Int) -> List[Tensor.Values]
  out: List[Tensor.Values] = []
  state = seed
  for _ in 0...n
    state = (state * 1103515245 + 12345) % 2147483648
    out = out.push(Tensor.Values.Float(state.to_f / 2147483648.0 - 0.5))
  end

  out
end

def params_of(width: Int, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  shapes = [[IN, width], [width], [width, width], [width], [width, CLASSES], [CLASSES]]
  out: List[Tensor.Tensor] = []
  for i in 0...shapes.size
    n = shapes[i].reduce(1) do |acc, d| acc * d end
    scale = 0.1
    values = numbers(n, i + 1).map { |v|
      case v
        Tensor.Values.Float(f) -> Tensor.Values.Float(f * scale)
        _ -> v
      end
    }

    out = out.push(try Tensor.from_list(values, shapes[i], Tensor.Dtype.F32, backend))
  end

  ok(out)
end

def dataset_of(rows: Int, backend: Tensor.Backend) -> Result[Data.Dataset]
  labels = (0...rows).map { |i| Tensor.Values.Int(i % CLASSES) }
  Data.dataset(
    try Tensor.from_list(numbers(rows * IN, 99), [rows, IN], Tensor.Dtype.F32, backend),
    try Tensor.from_list(labels, [rows], Tensor.Dtype.I32, backend),
  )
end
