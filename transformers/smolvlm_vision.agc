## The vision tower and the connector of SmolVLM (`Idefics3ForConditionalGeneration`, such as
## `HuggingFaceTB/SmolVLM-256M-Instruct`; `transformers/DESIGN.md`): a SigLIP tower (a patch embedding that is a
## convolution whose stride is its kernel, learned positions, pre-norm blocks of full attention with an MLP of
## `gelu_pytorch_tanh`, and a final layer norm), then the connector: a pixel shuffle that joins each `scale_factor` by
## `scale_factor` square of patches into one vector, and one linear layer, with no bias, to the width of the text model.
##
## The input is `[tiles, channels, size, size]` in `F32`, the pixels of Hugging Face's image processor (rescaled to 0 to
## 1, then normalized with the mean and std of its config): `Vision` of `vision/` makes it. The output is `[tiles,
## (size / patch / scale_factor)^2, text_width]`: the rows that the text model reads at the image tokens, one tile at a
## time. Every tile is a whole square of the tower's size with no padding (a padded tile has a patch mask, which this
## file does not read).

import Hub from "pkg/hub/hub"
import Lift from "lift"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Qwen3 from "qwen3"
import Source from "tokenizer/source"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"
import Weights from "weights"

## The prefix of the tower's tensors.
PREFIX = "model.vision_model."

## The connector's only tensor: the linear layer `[text_width, scale_factor^2 * width]`.
PROJECTION = "model.connector.modality_projection.proj"

## The `vision_config` fields that the model reads.
struct Config
  hidden_size: Int
  num_hidden_layers: Int
  num_attention_heads: Int
  intermediate_size: Int
  hidden_act: String
  layer_norm_eps: Float
  image_size: Int
  patch_size: Int
  num_channels: Int = 3
end

struct Vision
  config: Config
  scale_factor: Int
  text_width: Int
  patch: Nn.Conv2d
  ## `[patches, hidden_size]`, one row for each position of a tile.
  positions: Tensor.Tensor
  stack: Nn.EncoderStack
  projection: Nn.Dense
end

## The tower and the connector in `dir`, which holds `config.json` and `model.safetensors` (or its `.index.json`). Only
## the tensors `model.vision_model.*` and `model.connector.*` are read.
def open(dir: String, backend: Tensor.Backend) -> Result[Vision]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  file = try Source.within("#{dir}/config.json", JSON.decode(text, Document))
  return error("model_type is #{file.model_type}, not idefics3") if file.model_type != "idefics3"
  load(try Qwen3.checkpoint(dir), file.vision_config, file.scale_factor, file.text_config.hidden_size, backend)
end

## The tower of an open checkpoint, with the connector to `text_width`. The names and shapes of its tensors are checked
## against `config` before any is loaded, and a tensor that then fails to load closes the ones loaded before it.
def load(
  c: Hub.Checkpoint,
  config: Config,
  scale_factor: Int,
  text_width: Int,
  backend: Tensor.Backend,
) -> Result[Vision]
  try check_config(config, scale_factor)
  specs = specs_of(config, scale_factor, text_width)
  try Weights.check(c, specs)
  t = try Weights.load(c, specs, backend)
  blocks: List[Nn.EncoderBlock] = []
  for i in 0...config.num_hidden_layers
    blocks = blocks.push(
      Weights.block_of(
        t,
        block_names("#{PREFIX}encoder.layers.#{i}."),
        config.num_attention_heads,
        Nn.Activation.GeluTanh,
        config.layer_norm_eps,
      ),
    )
  end

  patch = "#{PREFIX}embeddings.patch_embedding"
  ok(
    Vision(
      config: config,
      scale_factor: scale_factor,
      text_width: text_width,
      patch: Nn.Conv2d(
        weight: Weights.at(t, "#{patch}.weight"),
        bias: some(Weights.at(t, "#{patch}.bias")),
        stride: config.patch_size,
      ),
      positions: Weights.at(t, "#{PREFIX}embeddings.position_embedding.weight"),
      stack: Nn.EncoderStack(
        blocks: blocks,
        final_norm: some(Weights.norm_of(t, "#{PREFIX}post_layernorm", config.layer_norm_eps)),
      ),
      projection: Weights.dense_of(t, PROJECTION),
    ),
  )
end

## How many rows one tile gives: the image tokens of a tile in the prompt (`image_seq_len` of the processor).
def tokens(m: Vision) -> Int
  side = m.config.image_size / m.config.patch_size / m.scale_factor
  side * side
end

## The features of `tiles` (`[n, channels, size, size]`, normalized, or `[channels, size, size]` for one tile, which
## `Vision.to_tensor` and `Vision.normalize` make): `[n, tokens(m), text_width]`. One graph on ggml. The caller closes
## the features (`Lift.close`) and the tiles.
def encode(m: Vision, tiles: Tensor.Tensor) -> Result[Tensor.Tensor]
  want = [m.config.num_channels, m.config.image_size, m.config.image_size]
  shape = Tensor.shape(tiles)
  if shape.size < 3 or shape.size > 4 or shape.drop(shape.size - 3) != want or (shape.size == 4 and shape[0] < 1)
    return error(
      "the tiles need [n, #{want[0]}, #{want[1]}, #{want[2]}] or one of [#{want[0]}, #{want[1]}, #{want[2]}], got #{shape}",
    )
  end

  outs = try Lift.staged(
    m,
    ->(p: Vision) { traced(p) },
    [tiles],
    ->(p: Vision, xs: List[Tensor.Tensor]) { ok([try graph(p, xs[0])]) },
  )

  ok(outs[0])
end

## Closes the weights of `m` now. Nothing may use `m` afterwards. The features that `encode` gave belong to the caller.
def close_model(m: Vision) -> Void
  tensors = Lift.conv2d_parts(m.patch) + [m.positions] + Lift.dense_parts(m.projection)
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

## `config.json` of a checkpoint: the tower is in `vision_config`, and the width of the text model in `text_config`.
struct Document
  model_type: String
  scale_factor: Int
  vision_config: Config
  text_config: TextSize
end

struct TextSize
  hidden_size: Int
end

## The tensors of `m` as `Trace` constants (`Lift`).
def traced(m: Vision) -> Vision
  m.with(
    patch: Lift.conv2d(m.patch),
    positions: Trace.constant(m.positions),
    stack: Lift.stack(m.stack),
    projection: Lift.dense(m.projection),
  )
end

def check_config(config: Config, scale_factor: Int) -> Result[Void]
  if config.hidden_act != "gelu_pytorch_tanh"
    return error("hidden_act #{config.hidden_act} is not supported, only gelu_pytorch_tanh")
  end

  width = config.hidden_size
  if config.num_attention_heads < 1 or width % config.num_attention_heads != 0
    return error("hidden_size #{width} is not a multiple of #{config.num_attention_heads} heads")
  end

  if config.patch_size < 1 or config.image_size % config.patch_size != 0
    return error("image_size #{config.image_size} is not a multiple of patch_size #{config.patch_size}")
  end

  side = config.image_size / config.patch_size
  if scale_factor < 1 or side % scale_factor != 0
    return error("scale_factor #{scale_factor} needs 1 or more and has to divide the #{side} patches of a side")
  end

  ok()
end

## The names of one block, under `p` (`model.vision_model.encoder.layers.N.`).
def block_names(p: String) -> Load.BlockNames
  Load.BlockNames(
    norm1: "#{p}layer_norm1",
    query: "#{p}self_attn.q_proj",
    key: "#{p}self_attn.k_proj",
    value: "#{p}self_attn.v_proj",
    output: "#{p}self_attn.out_proj",
    norm2: "#{p}layer_norm2",
    up: "#{p}mlp.fc1",
    down: "#{p}mlp.fc2",
  )
end

## Every tensor that `load` reads, with its shape.
def specs_of(config: Config, scale_factor: Int, text_width: Int) -> List[Weights.Spec]
  width = config.hidden_size
  side = config.image_size / config.patch_size
  patch = "#{PREFIX}embeddings.patch_embedding"
  specs = [
    Weights.Spec(name: "#{patch}.weight", shape: [width, config.num_channels, config.patch_size, config.patch_size]),
    Weights.Spec(name: "#{patch}.bias", shape: [width]),
    Weights.Spec(name: "#{PREFIX}embeddings.position_embedding.weight", shape: [side * side, width]),
    Weights.Spec(name: "#{PREFIX}post_layernorm.weight", shape: [width]),
    Weights.Spec(name: "#{PREFIX}post_layernorm.bias", shape: [width]),
    Weights.Spec(name: "#{PROJECTION}.weight", shape: [text_width, width * scale_factor * scale_factor]),
  ]

  for i in 0...config.num_hidden_layers
    names = block_names("#{PREFIX}encoder.layers.#{i}.")
    specs = specs + Weights.block_specs(names, width, config.intermediate_size)
  end

  specs
end

## The tower and the connector on `tiles` `[n, channels, size, size]`: `[n, tokens, text_width]`.
def graph(m: Vision, tiles: Tensor.Tensor) -> Result[Tensor.Tensor]
  size = m.config.image_size
  n = 1
  if Tensor.shape(tiles).size == 4
    n = Tensor.shape(tiles)[0]
  end

  batched = try Tensor.reshape(tiles, [n, m.config.num_channels, size, size])
  width = m.config.hidden_size
  side = size / m.config.patch_size
  patches = try Tensor.reshape(try Nn.conv2d(m.patch, batched), [n, width, side * side])
  rows = try Tensor.permute(patches, [0, 2, 1])
  positions = try Tensor.reshape(m.positions, [1, side * side, width])
  h = try Tensor.add(rows, try Tensor.broadcast_to(positions, Tensor.shape(rows)))
  Nn.dense(m.projection, try pixel_shuffle(try Nn.encoder_stack(m.stack, h), side, m.scale_factor))
end

## Hugging Face's `Idefics3Connector.pixel_shuffle`: the rows `[n, side * side, width]` of a square of `side` by `side`
## patches become `[n, (side / scale)^2, width * scale^2]`, each row the `scale` by `scale` square it joins, the columns
## of the square's rows running first.
def pixel_shuffle(h: Tensor.Tensor, side: Int, scale: Int) -> Result[Tensor.Tensor]
  n = Tensor.shape(h)[0]
  width = Tensor.shape(h)[2]
  grouped = try Tensor.reshape(h, [n, side, side / scale, width * scale])
  swapped = try Tensor.reshape(
    try Tensor.permute(grouped, [0, 2, 1, 3]),
    [n, side / scale, side / scale, width * scale * scale],
  )

  Tensor.reshape(
    try Tensor.permute(swapped, [0, 2, 1, 3]),
    [n, (side * side) / (scale * scale), width * scale * scale],
  )
end
