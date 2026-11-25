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

## A convolution to load: its name in the checkpoint, its kernel and its stride.
struct Spec
  prefix: String
  kernel: Int
  stride: Int
end

## Where a layer sits: its name in the checkpoint, its width in and out, and the stride of its first convolution.
struct Place
  prefix: String
  width: Int
  out: Int
  stride: Int
end

## The model in `dir`, which holds `model.safetensors` and `config.json`.
def open(dir: String, backend: Tensor.Backend) -> Result[ResNet]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  config = try Source.within("#{dir}/config.json", JSON.decode(text, Config))
  load(try Hub.open("#{dir}/model.safetensors"), config, backend)
end

## The model of an open checkpoint, its shapes checked against `config`. A weight that fails to load, or a shape that is
## wrong, closes the ones loaded before it.
def load(c: Hub.Checkpoint, config: Config, backend: Tensor.Backend) -> Result[ResNet]
  return error("model_type is #{config.model_type}, not resnet") if config.model_type != "resnet"
  return error("hidden_act #{config.hidden_act} is not supported, only relu") if config.hidden_act != "relu"
  if config.layer_type != "bottleneck" and config.layer_type != "basic"
    return error("layer_type #{config.layer_type} is not supported, only bottleneck and basic")
  end

  if config.hidden_sizes.size != config.depths.size or config.depths.empty?
    return error("hidden_sizes #{config.hidden_sizes} and depths #{config.depths} need one entry for each stage")
  end

  places = places_of(config)
  specs = [Spec(prefix: "resnet.embedder.embedder", kernel: 7, stride: 2)]
  for stage in places
    for place in stage
      specs = specs + layer_specs(config, place)
    end
  end

  convs = try conv_bns(c, specs, backend)
  dense = Load.dense(c, "classifier.1", backend)
  case dense
    ok(_) -> nil
    error(e) ->
      close_convs(convs)
      return error(e)
  end

  head = try dense

  case assembled(config, places, convs, head)
    ok(m) -> ok(m)
    error(e) ->
      close_convs(convs)
      Lift.close(Lift.dense_parts(head))
      error(e)
  end
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
  convs = [m.stem]
  for stage in m.stages
    for l in stage
      convs = convs + l.convs
      case l.shortcut
        some(p) -> convs = convs.push(p)
        none -> nil
      end
    end
  end

  close_convs(convs)
  Lift.close(Lift.dense_parts(m.classifier))
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

def places_of(config: Config) -> List[List[Place]]
  stages: List[List[Place]] = []
  width = config.embedding_size
  for s in 0...config.depths.size
    layers: List[Place] = []
    for l in 0...config.depths[s]
      stride = 1
      if l == 0 and (s > 0 or config.downsample_in_first_stage)
        stride = 2
      end

      out = config.hidden_sizes[s]
      layers = layers.push(
        Place(prefix: "resnet.encoder.stages.#{s}.layers.#{l}", width: width, out: out, stride: stride),
      )

      width = out
    end

    stages = stages.push(layers)
  end

  stages
end

## The convolutions of a layer in order, a shortcut last when the layer changes the width or the size. Hugging Face puts
## the stride of a bottleneck where `downsample_in_bottleneck` says (the 3x3 convolution by default).
def layer_specs(config: Config, p: Place) -> List[Spec]
  specs: List[Spec] = []
  if config.layer_type == "bottleneck"
    first = 1
    second = p.stride
    if config.downsample_in_bottleneck
      first = p.stride
      second = 1
    end

    specs = [
      Spec(prefix: "#{p.prefix}.layer.0", kernel: 1, stride: first),
      Spec(prefix: "#{p.prefix}.layer.1", kernel: 3, stride: second),
      Spec(prefix: "#{p.prefix}.layer.2", kernel: 1, stride: 1),
    ]
  else
    specs = [
      Spec(prefix: "#{p.prefix}.layer.0", kernel: 3, stride: p.stride),
      Spec(prefix: "#{p.prefix}.layer.1", kernel: 3, stride: 1),
    ]
  end

  if p.width != p.out or p.stride != 1
    specs = specs.push(Spec(prefix: "#{p.prefix}.shortcut", kernel: 1, stride: p.stride))
  end

  specs
end

## The convolutions of `specs`, in order. When one fails, the ones before it are closed.
def conv_bns(c: Hub.Checkpoint, specs: List[Spec], backend: Tensor.Backend) -> Result[List[Nn.Conv2d]]
  out: List[Nn.Conv2d] = []
  for s in specs
    case conv_bn(c, s.prefix, s.kernel, s.stride, backend)
      ok(conv) -> out = out.push(conv)
      error(e) ->
        close_convs(out)
        return error(e)
    end
  end

  ok(out)
end

def close_convs(convs: List[Nn.Conv2d]) -> Void
  tensors: List[Tensor.Tensor] = []
  for conv in convs
    tensors = tensors + Lift.conv2d_parts(conv)
  end

  Lift.close(tensors)
end

## The model of `convs` (those of `layer_specs` in order, after the stem) and `head`, with the shapes checked.
def assembled(config: Config, places: List[List[Place]], convs: List[Nn.Conv2d], head: Nn.Dense) -> Result[ResNet]
  stem = convs[0]
  try expect_shape("the stem", Tensor.shape(stem.weight), [config.embedding_size, config.num_channels, 7, 7])
  per_layer = 2
  if config.layer_type == "bottleneck"
    per_layer = 3
  end

  at = 1
  stages: List[List[Layer]] = []
  for stage in places
    layers: List[Layer] = []
    for p in stage
      first = convs[at]
      want = [p.out, p.width, 3, 3]
      if config.layer_type == "bottleneck"
        want = [p.out / 4, p.width, 1, 1]
      end

      try expect_shape("#{p.prefix}.layer.0", Tensor.shape(first.weight), want)
      shortcut: Optional[Nn.Conv2d] = none
      used = per_layer
      if p.width != p.out or p.stride != 1
        shortcut = some(convs[at + per_layer])
        used = per_layer + 1
      end

      layers = layers.push(Layer(convs: convs.drop(at).take(per_layer), shortcut: shortcut))
      at = at + used
    end

    stages = stages.push(layers)
  end

  width = config.hidden_sizes.last
  classes = Tensor.shape(head.w)
  try expect_shape("classifier.1.weight", [classes[1]], [width])
  if !config.id2label.empty? and config.id2label.size != classes[0]
    return error("id2label names #{config.id2label.size} classes, classifier.1.weight has #{classes[0]}")
  end

  ok(ResNet(config: config, stem: stem, stages: stages, classifier: head))
end

## The convolution `{prefix}.convolution` of a kernel `kernel` by `kernel` and the batch norm `{prefix}.normalization`
## after it, as one convolution (`Nn.fold_batch_norm`), run as one graph on ggml. The checkpoint's tensors are closed.
def conv_bn(c: Hub.Checkpoint, prefix: String, kernel: Int, stride: Int, backend: Tensor.Backend) -> Result[Nn.Conv2d]
  norm = "#{prefix}.normalization"
  names = [
    "#{prefix}.convolution.weight",
    "#{norm}.weight",
    "#{norm}.bias",
    "#{norm}.running_mean",
    "#{norm}.running_var",
  ]

  parts: List[Tensor.Tensor] = []
  for i in 0...names.size
    rank = 1
    if i == 0
      rank = 4
    end

    case Load.weight(c, names[i], rank, backend)
      ok(t) -> parts = parts.push(t)
      error(e) ->
        Lift.close(parts)
        return error(e)
    end
  end

  conv = Nn.Conv2d(weight: parts[0], bias: none, stride: stride, padding: kernel / 2)
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

## The weight and the bias of `conv` folded with the batch norm that `xs` holds after the weight: `gamma`, `beta`,
## `mean`, `var`.
def fold(conv: Nn.Conv2d, eps: Float, xs: List[Tensor.Tensor]) -> Result[List[Tensor.Tensor]]
  folded = try Nn.fold_batch_norm(conv.with(weight: xs[0]), xs[1], xs[2], xs[3], xs[4], eps)
  ok([folded.weight, try folded.bias.to_result("the folded convolution has a bias")])
end
