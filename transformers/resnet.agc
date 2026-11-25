## ResNet for image classification, from a Hugging Face checkpoint (`ResNetForImageClassification`, such as
## `microsoft/resnet-50` and `microsoft/resnet-18`; `transformers/DESIGN.md`): a 7x7 stem and a max pool, four stages of
## bottleneck or basic residual layers, a global average pool and a linear head.
##
## Each convolution and the batch norm after it load as one convolution (`Nn.fold_batch_norm`), so the model is for
## inference: the folded weights are not the checkpoint's. The input is `[batch, 3, height, width]` of `F32`, already
## normalized as the checkpoint's image processor does (`ImageNet` mean and std): `Vision` of `vision/` makes it.

import Hub from "pkg/hub/hub"
import Lift from "lift"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Source from "tokenizer/source"
import Tensor from "pkg/tensor/tensor"

## The `eps` of every batch norm: PyTorch's default, which Hugging Face's ResNet does not put in its config.
BATCH_NORM_EPS = 0.00001

## The fields of `config.json` that the model reads.
struct Config
  model_type: String
  num_channels: Int = 3
  embedding_size: Int
  hidden_sizes: List[Int]
  depths: List[Int]
  layer_type: String
  hidden_act: String = "relu"
  downsample_in_first_stage: Bool = false
  downsample_in_bottleneck: Bool = false
  id2label: Map[String, String] = {}
end

## One residual layer: two (basic) or three (bottleneck) convolutions, the last without a ReLU, and a convolution of the
## shortcut when the layer changes the width or the size.
struct Layer
  convs: List[Nn.Conv2d]
  shortcut: Optional[Nn.Conv2d]
end

struct ResNet
  config: Config
  stem: Nn.Conv2d
  stages: List[List[Layer]]
  classifier: Nn.Dense
end

## A class of an image: its index, its name and its logit.
struct Prediction
  id: Int
  label: String
  logit: Float
end

## The model in `dir`, which holds `model.safetensors` and `config.json`.
def open(dir: String, backend: Tensor.Backend) -> Result[ResNet]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  config = try Source.within("#{dir}/config.json", JSON.decode(text, Config))
  load(try Hub.open("#{dir}/model.safetensors"), config, backend)
end

## The model of an open checkpoint, its shapes checked against `config`.
def load(c: Hub.Checkpoint, config: Config, backend: Tensor.Backend) -> Result[ResNet]
  return error("model_type is #{config.model_type}, not resnet") if config.model_type != "resnet"
  return error("hidden_act #{config.hidden_act} is not supported, only relu") if config.hidden_act != "relu"
  if config.layer_type != "bottleneck" and config.layer_type != "basic"
    return error("layer_type #{config.layer_type} is not supported, only bottleneck and basic")
  end

  if config.hidden_sizes.size != config.depths.size or config.depths.empty?
    return error("hidden_sizes #{config.hidden_sizes} and depths #{config.depths} need one entry for each stage")
  end

  stem = try conv_bn(c, "resnet.embedder.embedder", 7, 2, backend)
  try expect_shape("the stem", Tensor.shape(stem.weight), [config.embedding_size, config.num_channels, 7, 7])
  stages: List[List[Layer]] = []
  width = config.embedding_size
  for s in 0...config.depths.size
    layers: List[Layer] = []
    for l in 0...config.depths[s]
      stride = 1
      if l == 0 and (s > 0 or config.downsample_in_first_stage)
        stride = 2
      end

      layers = layers.push(
        try layer(c, config, "resnet.encoder.stages.#{s}.layers.#{l}", width, config.hidden_sizes[s], stride, backend),
      )

      width = config.hidden_sizes[s]
    end

    stages = stages.push(layers)
  end

  head = try Load.dense(c, "classifier.1", backend)
  try expect_shape("classifier.1.weight", Tensor.shape(head.w), [config.id2label.size, width])
  ok(ResNet(config: config, stem: stem, stages: stages, classifier: head))
end

## The logits `[batch, classes]` of `images` (`[batch, channels, height, width]`, normalized): one graph on ggml. The
## caller closes the logits (`close`) and the images.
def run(m: ResNet, images: Tensor.Tensor) -> Result[Tensor.Tensor]
  if !fits?(m, images)
    return error(
      "the model needs images of [batch, #{m.config.num_channels}, height, width], got #{Tensor.shape(images)}",
    )
  end

  outs = try Lift.staged(
    m,
    ->(p: ResNet) { traced(p) },
    [images],
    ->(p: ResNet, xs: List[Tensor.Tensor]) { ok([try forward(p, xs[0])]) },
  )

  ok(outs[0])
end

## The model as plain `Tensor` calls, not one staged graph: inside a trace it is part of it, and outside one it runs
## eagerly, which ggml refuses ("does not run eagerly on the ggml backend"); `run` is the call for inference.
def forward(m: ResNet, images: Tensor.Tensor) -> Result[Tensor.Tensor]
  if !fits?(m, images)
    return error(
      "the model needs images of [batch, #{m.config.num_channels}, height, width], got #{Tensor.shape(images)}",
    )
  end

  x = try Tensor.relu(try Nn.conv2d(m.stem, images))
  x = try Tensor.max_pool2d(x, 3, 2, 1)
  for stage in m.stages
    for l in stage
      x = try residual(l, x)
    end
  end

  Nn.dense(m.classifier, try Nn.global_avg_pool2d(x))
end

## The `k` best classes of each row of `logits` (`[batch, classes]`), best first, with the names of `config.json`.
def top(m: ResNet, logits: Tensor.Tensor, k: Int) -> Result[List[List[Prediction]]]
  shape = Tensor.shape(logits)
  return error("top needs logits of [batch, classes], got #{shape}") if shape.size != 2 or k < 1 or k > shape[1]
  values: List[Float] = []
  for v in try Tensor.to_list(logits)
    case v
      Tensor.Values.Float(f) -> values = values.push(f)
      _ -> return error("the logits are floats")
    end
  end

  out: List[List[Prediction]] = []
  for row in 0...shape[0]
    ids = (0...shape[1]).to_a.sort_by { |i| -values[row * shape[1] + i] }.take(k)
    out = out.push(
      ids.map { |i|
        Prediction(id: i, label: m.config.id2label.get("#{i}").unwrap_or("#{i}"), logit: values[row * shape[1] + i])
      },
    )
  end

  ok(out)
end

## Closes the logits of `run`.
def close(logits: Tensor.Tensor) -> Void
  Lift.close([logits])
end

## Closes the weights of `m` now. Nothing may use `m` afterwards.
def close_model(m: ResNet) -> Void
  tensors = Lift.conv2d_parts(m.stem) + Lift.dense_parts(m.classifier)
  for stage in m.stages
    for l in stage
      for conv in l.convs
        tensors = tensors + Lift.conv2d_parts(conv)
      end

      case l.shortcut
        some(p) -> tensors = tensors + Lift.conv2d_parts(p)
        none -> nil
      end
    end
  end

  Lift.close(tensors)
end

private

def fits?(m: ResNet, images: Tensor.Tensor) -> Bool
  shape = Tensor.shape(images)
  shape.size == 4 and shape[1] == m.config.num_channels
end

def expect_shape(name: String, got: List[Int], want: List[Int]) -> Result[Void]
  return error("#{name}: needs #{want}, got #{got}") if got != want
  ok()
end

## The tensors of `m` as `Trace` constants (`Lift`).
def traced(m: ResNet) -> ResNet
  m.with(
    stem: Lift.conv2d(m.stem),
    stages: m.stages.map { |stage|
      stage.map { |l|
        Layer(convs: l.convs.map { |c| Lift.conv2d(c) }, shortcut: l.shortcut.map { |c| Lift.conv2d(c) })
      }
    },
    classifier: Lift.dense(m.classifier),
  )
end

## `x` through the convolutions of `l` (a ReLU after each but the last), plus the shortcut, then a ReLU.
def residual(l: Layer, x: Tensor.Tensor) -> Result[Tensor.Tensor]
  h = x
  for i in 0...l.convs.size
    h = try Nn.conv2d(l.convs[i], h)
    if i < l.convs.size - 1
      h = try Tensor.relu(h)
    end
  end

  skip = x
  case l.shortcut
    some(p) -> skip = try Nn.conv2d(p, x)
    none -> nil
  end

  Tensor.relu(try Tensor.add(h, skip))
end

## The layer `prefix` from `width` to `out` channels, with the stride of its first convolution that Hugging Face puts
## where `downsample_in_bottleneck` says (the 3x3 one by default).
def layer(
  c: Hub.Checkpoint,
  config: Config,
  prefix: String,
  width: Int,
  out: Int,
  stride: Int,
  backend: Tensor.Backend,
) -> Result[Layer]
  convs: List[Nn.Conv2d] = []
  if config.layer_type == "bottleneck"
    mid = out / 4
    first = 1
    second = stride
    if config.downsample_in_bottleneck
      first = stride
      second = 1
    end

    convs = [
      try conv_bn(c, "#{prefix}.layer.0", 1, first, backend),
      try conv_bn(c, "#{prefix}.layer.1", 3, second, backend),
      try conv_bn(c, "#{prefix}.layer.2", 1, 1, backend),
    ]

    try expect_shape("#{prefix}.layer.0", Tensor.shape(convs[0].weight), [mid, width, 1, 1])
  else
    convs = [
      try conv_bn(c, "#{prefix}.layer.0", 3, stride, backend),
      try conv_bn(c, "#{prefix}.layer.1", 3, 1, backend),
    ]

    try expect_shape("#{prefix}.layer.0", Tensor.shape(convs[0].weight), [out, width, 3, 3])
  end

  shortcut: Optional[Nn.Conv2d] = none
  if width != out or stride != 1
    shortcut = some(try conv_bn(c, "#{prefix}.shortcut", 1, stride, backend))
  end

  ok(Layer(convs: convs, shortcut: shortcut))
end

## The convolution `{prefix}.convolution` of a kernel `kernel` by `kernel` and the batch norm `{prefix}.normalization`
## after it, as one convolution (`Nn.fold_batch_norm`), run as one graph on ggml. The checkpoint's tensors are closed.
def conv_bn(c: Hub.Checkpoint, prefix: String, kernel: Int, stride: Int, backend: Tensor.Backend) -> Result[Nn.Conv2d]
  weight = try Load.weight(c, "#{prefix}.convolution.weight", 4, backend)
  norm = "#{prefix}.normalization"
  parts = [
    weight,
    try Load.weight(c, "#{norm}.weight", 1, backend),
    try Load.weight(c, "#{norm}.bias", 1, backend),
    try Load.weight(c, "#{norm}.running_mean", 1, backend),
    try Load.weight(c, "#{norm}.running_var", 1, backend),
  ]

  conv = Nn.Conv2d(weight: weight, bias: none, stride: stride, padding: kernel / 2)
  folded = Lift.staged(
    BATCH_NORM_EPS,
    ->(eps: Float) { eps },
    parts,
    ->(eps: Float, xs: List[Tensor.Tensor]) { fold(conv, eps, xs) },
  )

  Lift.close(parts)
  out = try folded
  ok(conv.with(weight: out[0], bias: some(out[1])))
end

## The weight and the bias of `conv` folded with the batch norm `xs` holds after the weight: `gamma`, `beta`, `mean`, `var`.
def fold(conv: Nn.Conv2d, eps: Float, xs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  folded = try Nn.fold_batch_norm(conv.with(weight: xs[0]), xs[1], xs[2], xs[3], xs[4], eps)
  ok([folded.weight, try folded.bias.to_result("the folded convolution has a bias")])
end
