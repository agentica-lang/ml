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

## A tensor of the checkpoint and the shape that `config.json` gives it.
struct Spec
  name: String
  shape: List[Int]
end

## The model in `dir`, which holds `model.safetensors` and `config.json`.
def open(dir: String, backend: Tensor.Backend) -> Result[ViT]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  config = try Source.within("#{dir}/config.json", JSON.decode(text, Config))
  load(try Hub.open("#{dir}/model.safetensors"), config, backend)
end

## The model of an open checkpoint. The names and shapes of its tensors are checked against `config` before any is
## loaded, so a checkpoint that does not fit leaves nothing open.
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

  try expect_specs(c, specs_of(config, head[0]))
  patch_name = "vit.embeddings.patch_embeddings.projection"
  eps = config.layer_norm_eps
  blocks: List[Nn.EncoderBlock] = []
  for i in 0...config.num_hidden_layers
    blocks = blocks.push(
      try Load.encoder_block(
        c,
        block_names("vit.encoder.layer.#{i}."),
        config.num_attention_heads,
        Nn.Activation.GeluErf,
        eps,
        backend,
      ),
    )
  end

  ok(
    ViT(
      config: config,
      patch: Nn.Conv2d(
        weight: try Load.weight(c, "#{patch_name}.weight", 4, backend),
        bias: some(try Load.weight(c, "#{patch_name}.bias", 1, backend)),
        stride: config.patch_size,
      ),
      cls: try Load.weight(c, "vit.embeddings.cls_token", 3, backend),
      positions: try Load.weight(c, "vit.embeddings.position_embeddings", 3, backend),
      stack: Nn.EncoderStack(blocks: blocks, final_norm: some(try Load.layer_norm(c, "vit.layernorm", eps, backend))),
      classifier: try Load.dense(c, "classifier", backend),
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
    stack: Nn.EncoderStack(
      blocks: m.stack.blocks.map { |b| Lift.block(b) },
      final_norm: m.stack.final_norm.map { |n| Lift.norm(n) },
    ),
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
def specs_of(config: Config, classes: Int) -> List[Spec]
  width = config.hidden_size
  grid = config.image_size / config.patch_size
  patch = "vit.embeddings.patch_embeddings.projection"
  specs = [
    Spec(name: "#{patch}.weight", shape: [width, config.num_channels, config.patch_size, config.patch_size]),
    Spec(name: "#{patch}.bias", shape: [width]),
    Spec(name: "vit.embeddings.cls_token", shape: [1, 1, width]),
    Spec(name: "vit.embeddings.position_embeddings", shape: [1, grid * grid + 1, width]),
    Spec(name: "vit.layernorm.weight", shape: [width]),
    Spec(name: "vit.layernorm.bias", shape: [width]),
    Spec(name: "classifier.weight", shape: [classes, width]),
    Spec(name: "classifier.bias", shape: [classes]),
  ]

  for i in 0...config.num_hidden_layers
    n = block_names("vit.encoder.layer.#{i}.")
    for norm in [n.norm1, n.norm2]
      specs = specs + [Spec(name: "#{norm}.weight", shape: [width]), Spec(name: "#{norm}.bias", shape: [width])]
    end

    for p in [n.query, n.key, n.value, n.output]
      specs = specs.push(Spec(name: "#{p}.weight", shape: [width, width]))
      if config.qkv_bias or p == n.output
        specs = specs.push(Spec(name: "#{p}.bias", shape: [width]))
      end
    end

    specs = specs + [
      Spec(name: "#{n.up}.weight", shape: [config.intermediate_size, width]),
      Spec(name: "#{n.up}.bias", shape: [config.intermediate_size]),
      Spec(name: "#{n.down}.weight", shape: [width, config.intermediate_size]),
      Spec(name: "#{n.down}.bias", shape: [width]),
    ]
  end

  specs
end

def expect_specs(c: Hub.Checkpoint, specs: List[Spec]) -> Result[Void]
  for s in specs
    return error("#{s.name}: the checkpoint has no such tensor") if !Hub.has?(c, s.name)
    got = try Hub.shape(c, s.name)
    return error("#{s.name}: needs #{s.shape}, got #{got}") if got != s.shape
  end

  ok()
end
