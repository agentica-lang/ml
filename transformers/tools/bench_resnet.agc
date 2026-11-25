## Times ResNet on ggml: the forward of a batch of `BATCH` images of 224 by 224 (one warm-up run first, then the median
## of `REPS` runs), or with `backward` the forward and backward of the folded model: the forward, a loss (the mean
## square of the logits) and the gradient of every weight and of the images, in one graph (no optimizer step). The model is the folded one that
## `ResNet.open` loads, so the batch norms are inference ones. Manual; run from `transformers/`:
##
##   agentica tools/bench_resnet.agc MODEL_DIR [cpu|vulkan] [BATCH] [REPS] [forward|backward]

import Lift from "../lift"
import Nn from "pkg/nn/nn"
import ResNet from "../resnet"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

SIDE = 224

def main() -> Void
  args = System.args
  if args.empty?
    puts("usage: bench_resnet.agc MODEL_DIR [cpu|vulkan] [BATCH] [REPS] [forward|backward]")
    System.exit(2)
  end

  device = Tensor.Device.Cpu
  if args.size > 1 and args[1] == "vulkan"
    device = Tensor.Device.Vulkan(0)
  end

  batch = 1
  reps = 5
  if args.size > 2
    batch = args[2].to_i!
  end

  if args.size > 3
    reps = args[3].to_i!
  end

  backward = args.size > 4 and args[4] == "backward"
  case measure(args[0], Tensor.Backend.Ggml(device), batch, reps, backward)
    ok(line) -> puts(line)
    error(e) ->
      puts("bench_resnet: #{e.message}")
      System.exit(1)
  end
end

private

def measure(dir: String, backend: Tensor.Backend, batch: Int, reps: Int, backward: Bool) -> Result[String]
  try Tensor.vulkan_fp16(false)
  started = Time.monotonic_ms
  m = try ResNet.open(dir, backend)
  load = Time.monotonic_ms - started
  images = try Tensor.full([batch, 3, SIDE, SIDE], Tensor.Values.Float(0.5), Tensor.Dtype.F32, backend)
  step = -> { forward(m, images) }
  if backward
    step = -> { train(m, images) }
  end

  try step.call()
  times: List[Int] = []
  for _ in 0...reps
    started = Time.monotonic_ms
    try step.call()
    times = times.push(Time.monotonic_ms - started)
  end

  median = times.sort[times.size / 2]
  what = "forward"
  if backward
    what = "forward and backward"
  end

  ok(
    "load #{load} ms; #{what}, batch #{batch}: median #{median} ms of #{reps} runs, #{tenths(batch * 10000 / [median, 1].max.unwrap!)} images/s",
  )
end

## `n` tenths as a decimal number.
def tenths(n: Int) -> String = "#{n / 10}.#{n % 10}"

def forward(m: ResNet.ResNet, images: Tensor.Tensor) -> Result[Void]
  ResNet.close(try ResNet.run(m, images))
  ok()
end

## One step: the loss and its gradients for the weights of `m` and for `images`, all read back.
def train(m: ResNet.ResNet, images: Tensor.Tensor) -> Result[Void]
  ws = weights(m)
  out = try Trace.value_and_grad(
    ws + [images],
    ->(xs: List[Tensor.Tensor]) {
      logits = try ResNet.forward(with_weights(m, xs.take(ws.size)), xs[ws.size])
      Tensor.mean(try Tensor.mul(logits, logits))
    },
  )

  Lift.close([out.value] + out.grads)
  ok()
end

## The weight and the bias of each convolution, in the order of the model, then the classifier's.
def weights(m: ResNet.ResNet) -> List[Tensor.Tensor]
  out = Lift.conv2d_parts(m.stem)
  for stage in m.stages
    for l in stage
      for c in l.convs
        out = out + Lift.conv2d_parts(c)
      end

      case l.shortcut
        some(c) -> out = out + Lift.conv2d_parts(c)
        none -> nil
      end
    end
  end

  out + Lift.dense_parts(m.classifier)
end

## `m` with the tensors `ws`, in the order of `weights`.
def with_weights(m: ResNet.ResNet, ws: List[Tensor.Tensor]) -> ResNet.ResNet
  at = 2
  stages: List[List[ResNet.Layer]] = []
  for stage in m.stages
    layers: List[ResNet.Layer] = []
    for l in stage
      convs: List[Nn.Conv2d] = []
      for c in l.convs
        convs = convs.push(c.with(weight: ws[at], bias: some(ws[at + 1])))
        at = at + 2
      end

      shortcut = l.shortcut
      case l.shortcut
        some(c) ->
          shortcut = some(c.with(weight: ws[at], bias: some(ws[at + 1])))
          at = at + 2
        none -> nil
      end

      layers = layers.push(ResNet.Layer(convs: convs, shortcut: shortcut))
    end

    stages = stages.push(layers)
  end

  m.with(
    stem: m.stem.with(weight: ws[0], bias: some(ws[1])),
    stages: stages,
    classifier: m.classifier.with(w: ws[at], b: some(ws[at + 1])),
  )
end
