## ViT for image classification, from a Hugging Face checkpoint (`ViTForImageClassification`, such as
## `google/vit-base-patch16-224`; `transformers/DESIGN.md`): a patch embedding (a convolution whose stride is its
## kernel), a class token and learned positions, pre-norm encoder blocks with the exact GELU (`Nn.EncoderStack`), a
## final layer norm and a linear head on the class token.
##
## The input is `[batch, 3, size, size]` of `F32`, already normalized as the checkpoint's image processor does (a
## stretch to `size` by `size` and the mean and std of the config): `Vision` of `vision/` makes it.

import Classify from "classify"
import Hub from "pkg/hub/hub"
import Lift from "lift"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Source from "tokenizer/source"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Weights from "weights"

## The fields of `config.json` that the model reads.
struct Config
  model_type: String
  hidden_size: Int
  num_hidden_layers: Int
  num_attention_heads: Int
  intermediate_size: Int
  hidden_act: String
  layer_norm_eps: Float
  image_size: Int
  patch_size: Int
  num_channels: Int = 3
  qkv_bias: Bool = true
  id2label: Map[String, String] = {}
end

struct ViT
  config: Config
  patch: Nn.Conv2d
  ## `[1, 1, hidden_size]`
  cls: Tensor.Tensor
  ## `[1, patches + 1, hidden_size]`, the class token first.
  positions: Tensor.Tensor
  stack: Nn.EncoderStack
  classifier: Nn.Dense
end

## The model in `dir`, which holds `model.safetensors` and `config.json`.
def open(dir: String, backend: Tensor.Backend) -> Result[ViT]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  config = try Source.within("#{dir}/config.json", JSON.decode(text, Config))
  load(try Hub.open("#{dir}/model.safetensors"), config, backend)
end

## The model of an open checkpoint. The names and shapes of its tensors are checked against `config` before any is
## loaded, and a tensor that then fails to load closes the ones loaded before it.
def load(c: Hub.Checkpoint, config: Config, backend: Tensor.Backend) -> Result[ViT]
  return error("model_type is #{config.model_type}, not vit") if config.model_type != "vit"
  return error("hidden_act #{config.hidden_act} is not supported, only gelu") if config.hidden_act != "gelu"
  width = config.hidden_size
  if width % config.num_attention_heads != 0
    return error("hidden_size #{width} is not a multiple of #{config.num_attention_heads} heads")
  end

  if config.patch_size < 1 or config.image_size % config.patch_size != 0
    return error("image_size #{config.image_size} is not a multiple of patch_size #{config.patch_size}")
  end

  head = try Hub.shape(c, "classifier.weight")
  return error("classifier.weight: needs rank 2, got #{head}") if head.size != 2
  if !config.id2label.empty? and config.id2label.size != head[0]
    return error("id2label names #{config.id2label.size} classes, classifier.weight has #{head[0]}")
  end

  specs = specs_of(config, head[0])
  try Weights.check(c, specs)
  t = try Weights.load(c, specs, backend)
  eps = config.layer_norm_eps
  blocks: List[Nn.EncoderBlock] = []
  for i in 0...config.num_hidden_layers
    blocks = blocks.push(
      Weights.block_of(
        t,
        block_names("vit.encoder.layer.#{i}."),
        config.num_attention_heads,
        Nn.Activation.GeluErf,
        eps,
      ),
    )
  end

  patch = "vit.embeddings.patch_embeddings.projection"
  ok(
    ViT(
      config: config,
      patch: Nn.Conv2d(
        weight: Weights.at(t, "#{patch}.weight"),
        bias: some(Weights.at(t, "#{patch}.bias")),
        stride: config.patch_size,
      ),
      cls: Weights.at(t, "vit.embeddings.cls_token"),
      positions: Weights.at(t, "vit.embeddings.position_embeddings"),
      stack: Nn.EncoderStack(blocks: blocks, final_norm: some(Weights.norm_of(t, "vit.layernorm", eps))),
      classifier: Weights.dense_of(t, "classifier"),
    ),
  )
end

## The logits `[batch, classes]` of `images` (`[batch, channels, size, size]`, normalized): one graph on ggml. The
## caller closes the logits (`close`) and the images.
def run(m: ViT, images: Tensor.Tensor) -> Result[Tensor.Tensor]
  try expect_images(m, images)
  outs = try Lift.staged(
    m,
    ->(p: ViT) { traced(p) },
    [images],
    ->(p: ViT, xs: List[Tensor.Tensor]) { ok([try forward(p, xs[0])]) },
  )

  ok(outs[0])
end

## The model as plain `Tensor` calls, not one staged graph: inside a trace it is part of it, and outside one it runs
## eagerly, which ggml refuses ("does not run eagerly on the ggml backend"); `run` is the call for inference.
def forward(m: ViT, images: Tensor.Tensor) -> Result[Tensor.Tensor]
  try expect_images(m, images)
  batch = Tensor.shape(images)[0]
  width = m.config.hidden_size
  grid = m.config.image_size / m.config.patch_size
  patches = try Tensor.reshape(try Nn.conv2d(m.patch, images), [batch, width, grid * grid])
  tokens = try Tensor.concat(
    [try Tensor.broadcast_to(m.cls, [batch, 1, width]), try Tensor.permute(patches, [0, 2, 1])],
    1,
  )

  h = try Tensor.add(tokens, try Tensor.broadcast_to(m.positions, Tensor.shape(tokens)))
  h = try Nn.encoder_stack(m.stack, h)
  Nn.dense(m.classifier, try Tensor.reshape(try Tensor.slice(h, [0, 0, 0], [batch, 1, width]), [batch, width]))
end

## The `k` best classes of each row of `logits` (`[batch, classes]`), best first, with the names of `config.json`.
def top(m: ViT, logits: Tensor.Tensor, k: Int) -> Result[List[List[Classify.Prediction]]]
  Classify.top(m.config.id2label, logits, k)
end

## Closes the logits of `run`.
def close(logits: Tensor.Tensor) -> Void
  Lift.close([logits])
end

## Closes the weights of `m` now. Nothing may use `m` afterwards.
def close_model(m: ViT) -> Void
  tensors = Lift.conv2d_parts(m.patch) + [m.cls, m.positions] + Lift.dense_parts(m.classifier)
  for b in m.stack.blocks
    tensors = tensors + Lift.block_parts(b)
  end

  case m.stack.final_norm
    some(n) -> tensors = tensors + Lift.norm_parts(n)
    none -> nil
  end

  Lift.close(tensors)
end

private

def expect_images(m: ViT, images: Tensor.Tensor) -> Result[Void]
  want = [m.config.num_channels, m.config.image_size, m.config.image_size]
  shape = Tensor.shape(images)
  if shape.size != 4 or shape.drop(1) != want
    return error("the model needs images of [batch, #{want[0]}, #{want[1]}, #{want[2]}], got #{shape}")
  end

  ok()
end

## The tensors of `m` as `Trace` constants (`Lift`).
def traced(m: ViT) -> ViT
  m.with(
    patch: Lift.conv2d(m.patch),
    cls: Trace.constant(m.cls),
    positions: Trace.constant(m.positions),
    stack: Lift.stack(m.stack),
    classifier: Lift.dense(m.classifier),
  )
end

## The names of one block, under `p` (`vit.encoder.layer.N.`).
def block_names(p: String) -> Load.BlockNames
  Load.BlockNames(
    norm1: "#{p}layernorm_before",
    query: "#{p}attention.attention.query",
    key: "#{p}attention.attention.key",
    value: "#{p}attention.attention.value",
    output: "#{p}attention.output.dense",
    norm2: "#{p}layernorm_after",
    up: "#{p}intermediate.dense",
    down: "#{p}output.dense",
  )
end

## Every tensor that `load` reads, with its shape.
def specs_of(config: Config, classes: Int) -> List[Weights.Spec]
  width = config.hidden_size
  grid = config.image_size / config.patch_size
  patch = "vit.embeddings.patch_embeddings.projection"
  specs = [
    Weights.Spec(name: "#{patch}.weight", shape: [width, config.num_channels, config.patch_size, config.patch_size]),
    Weights.Spec(name: "#{patch}.bias", shape: [width]),
    Weights.Spec(name: "vit.embeddings.cls_token", shape: [1, 1, width]),
    Weights.Spec(name: "vit.embeddings.position_embeddings", shape: [1, grid * grid + 1, width]),
    Weights.Spec(name: "vit.layernorm.weight", shape: [width]),
    Weights.Spec(name: "vit.layernorm.bias", shape: [width]),
    Weights.Spec(name: "classifier.weight", shape: [classes, width]),
    Weights.Spec(name: "classifier.bias", shape: [classes]),
  ]

  for i in 0...config.num_hidden_layers
    specs = (
      specs +
      Weights.block_specs(block_names("vit.encoder.layer.#{i}."), width, config.intermediate_size, config.qkv_bias)
    )
  end

  specs
end
