## CLIP from a Hugging Face checkpoint (`CLIPModel`, such as `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`;
## `transformers/DESIGN.md`): a vision tower (a convolution of stride `patch_size`, a class embedding, learned
## positions, `pre_layrnorm`, pre-norm encoder blocks, `post_layernorm` on the class token and a projection) and a text
## tower (token and position embeddings, pre-norm blocks under a causal mask, a final layer norm, the vector at the end
## token and a projection), both into one space of `projection_dim`. Their embeddings give the similarity of an image
## and a text, and zero-shot classification: the probability of each text of a list for an image.
##
## The image input is `[batch, 3, size, size]` of `F32`, normalized as the image processor of the checkpoint does
## (`Vision` of `vision/` makes it). The text input is the ids of `Clip.encode` of `tokenizer/clip.agc`.

import Hub from "pkg/hub/hub"
import Lift from "lift"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Sentence from "sentence"
import Source from "tokenizer/source"
import Tensor from "pkg/tensor/tensor"
import Weights from "weights"

## The fields of the `text_config` of `config.json` that the model reads.
struct TextConfig
  hidden_size: Int
  intermediate_size: Int
  num_hidden_layers: Int
  num_attention_heads: Int
  hidden_act: String
  layer_norm_eps: Float
  max_position_embeddings: Int
  vocab_size: Int
end

## The fields of the `vision_config` of `config.json` that the model reads.
struct VisionConfig
  hidden_size: Int
  intermediate_size: Int
  num_hidden_layers: Int
  num_attention_heads: Int
  hidden_act: String
  layer_norm_eps: Float
  image_size: Int
  patch_size: Int
  num_channels: Int = 3
end

struct Config
  model_type: String
  projection_dim: Int
  text_config: TextConfig
  vision_config: VisionConfig
end

struct Vision
  patch: Nn.Conv2d
  ## `[hidden_size]`
  class_embedding: Tensor.Tensor
  ## `[patches + 1, hidden_size]`, the class embedding first.
  positions: Nn.Embedding
  pre_norm: Nn.LayerNorm
  stack: Nn.EncoderStack
  post_norm: Nn.LayerNorm
  projection: Nn.Dense
end

struct Text
  tokens: Nn.Embedding
  positions: Nn.Embedding
  ## Ends with the final layer norm.
  stack: Nn.EncoderStack
  projection: Nn.Dense
end

struct Clip
  config: Config
  vision: Vision
  text: Text
  ## `exp(logit_scale)`: what the cosine of an image and a text is multiplied by.
  scale: Float
end

## The model in `dir`, which holds `model.safetensors` and `config.json`.
def open(dir: String, backend: Tensor.Backend) -> Result[Clip]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  config = try Source.within("#{dir}/config.json", JSON.decode(text, Config))
  load(try Hub.open("#{dir}/model.safetensors"), config, backend)
end

## The model of an open checkpoint. The names and shapes of its tensors are checked against `config` before any is
## loaded, and a tensor that then fails to load closes the ones loaded before it.
def load(c: Hub.Checkpoint, config: Config, backend: Tensor.Backend) -> Result[Clip]
  return error("model_type is #{config.model_type}, not clip") if config.model_type != "clip"
  v = config.vision_config
  x = config.text_config
  for act in [v.hidden_act, x.hidden_act]
    return error("hidden_act #{act} is not supported, only gelu") if act != "gelu"
  end

  for heads in [[v.hidden_size, v.num_attention_heads], [x.hidden_size, x.num_attention_heads]]
    return error("hidden_size #{heads[0]} is not a multiple of #{heads[1]} heads") if heads[0] % heads[1] != 0
  end

  if v.patch_size < 1 or v.image_size % v.patch_size != 0
    return error("image_size #{v.image_size} is not a multiple of patch_size #{v.patch_size}")
  end

  specs = specs_of(config)
  try Weights.check(c, specs)
  t = try Weights.load(c, specs, backend)
  scale = scale_of(t)
  case scale
    ok(_) -> nil
    error(e) ->
      Lift.close(t.values)
      return error(e)
  end

  ok(Clip(config: config, vision: vision_of(t, v), text: text_of(t, x), scale: Math.exp(try scale)))
end

## The embeddings `[batch, projection_dim]` of `images` (`[batch, channels, size, size]`, normalized): one graph on
## ggml, as `get_image_features` of Hugging Face gives them, not normalized. The caller closes them (`close`) and the
## images.
def image_embeds(m: Clip, images: Tensor.Tensor) -> Result[Tensor.Tensor]
  v = m.config.vision_config
  shape = Tensor.shape(images)
  if shape.size != 4 or shape.drop(1) != [v.num_channels, v.image_size, v.image_size]
    return error(
      "the model needs images of [batch, #{v.num_channels}, #{v.image_size}, #{v.image_size}], got #{shape}",
    )
  end

  outs = try Lift.staged(
    m,
    ->(p: Clip) { traced(p) },
    [images],
    ->(p: Clip, xs: List[Tensor.Tensor]) { ok([try vision_forward(p, xs[0])]) },
  )

  ok(outs[0])
end

## The embeddings `[batch, projection_dim]` of `rows`, the ids of `Clip.encode` of each text (start token, tokens, end
## token), not normalized (`get_text_features`). The rows are padded to the longest; the model is causal, so the
## padding changes nothing, and the vector read is the one at each row's last id. No row may be empty or longer than
## `max_position_embeddings`. The caller closes them (`close`).
def text_embeds(m: Clip, rows: List[List[Int]]) -> Result[Tensor.Tensor]
  return error("text_embeds needs at least one row") if rows.empty?
  seq = rows.map { |r| r.size }.max.unwrap!
  limit = m.config.text_config.max_position_embeddings
  return error("a row has #{seq} ids, the model reads #{limit} at most") if seq > limit
  return error("a row has no ids") if rows.any? { |r| r.empty? }
  ids: List[Tensor.Values] = []
  last: List[Tensor.Values] = []
  for r in rows
    for i in 0...seq
      ids = ids.push(Tensor.Values.Int(padded(r, i)))
      last = last.push(Tensor.Values.Float(pick(r.size, i)))
    end
  end

  backend = Tensor.backend(m.text.tokens.table)
  shape = [rows.size, seq]
  tokens = try Tensor.from_list(ids, shape, Tensor.Dtype.I32, backend)
  marks = try Tensor.from_list(last, shape, Tensor.Dtype.F32, backend)
  outs = Lift.staged(
    m,
    ->(p: Clip) { traced(p) },
    [tokens, marks],
    ->(p: Clip, xs: List[Tensor.Tensor]) { ok([try text_forward(p, xs[0], xs[1])]) },
  )

  Lift.close([tokens, marks])
  ok((try outs)[0])
end

## `scale` times the cosine of each image embedding (`[images, projection_dim]`) with each text embedding
## (`[texts, projection_dim]`): `[images, texts]`, the `logits_per_image` of Hugging Face. The caller closes it.
def logits(m: Clip, images: Tensor.Tensor, texts: Tensor.Tensor) -> Result[Tensor.Tensor]
  scored(m, images, texts, false)
end

## The probability of each text for each image: `softmax` of `logits` over the texts, `[images, texts]`. Zero-shot
## classification is this with a prompt for each class (`a photo of a cat`). The caller closes it.
def probabilities(m: Clip, images: Tensor.Tensor, texts: Tensor.Tensor) -> Result[Tensor.Tensor]
  scored(m, images, texts, true)
end

## Closes embeddings, logits or probabilities.
def close(t: Tensor.Tensor) -> Void
  Lift.close([t])
end

## Closes the weights of `m` now. Nothing may use `m` afterwards.
def close_model(m: Clip) -> Void
  v = m.vision
  x = m.text
  tensors = (
    Lift.conv2d_parts(v.patch) +
    [v.class_embedding, v.positions.table] +
    Lift.norm_parts(v.pre_norm) +
    Lift.norm_parts(v.post_norm) +
    Lift.dense_parts(v.projection) +
    [x.tokens.table, x.positions.table] +
    Lift.dense_parts(x.projection)
  )

  for b in v.stack.blocks + x.stack.blocks
    tensors = tensors + Lift.block_parts(b)
  end

  case x.stack.final_norm
    some(n) -> tensors = tensors + Lift.norm_parts(n)
    none -> nil
  end

  Lift.close(tensors)
end

private

VISION = "vision_model."
TEXT = "text_model."

## The tensors of `m` as `Trace` constants (`Lift`).
def traced(m: Clip) -> Clip
  v = m.vision
  x = m.text
  m.with(
    vision: v.with(
      patch: Lift.conv2d(v.patch),
      class_embedding: Lift.constant(v.class_embedding),
      positions: Lift.embedding(v.positions),
      pre_norm: Lift.norm(v.pre_norm),
      stack: Lift.stack(v.stack),
      post_norm: Lift.norm(v.post_norm),
      projection: Lift.dense(v.projection),
    ),
    text: x.with(
      tokens: Lift.embedding(x.tokens),
      positions: Lift.embedding(x.positions),
      stack: Lift.stack(x.stack),
      projection: Lift.dense(x.projection),
    ),
  )
end

## The names of one block, under `p` (`vision_model.encoder.layers.N.`).
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

def specs_of(config: Config) -> List[Weights.Spec]
  v = config.vision_config
  x = config.text_config
  grid = v.image_size / v.patch_size
  specs = [
    Weights.Spec(name: "#{VISION}embeddings.class_embedding", shape: [v.hidden_size]),
    Weights.Spec(
      name: "#{VISION}embeddings.patch_embedding.weight",
      shape: [v.hidden_size, v.num_channels, v.patch_size, v.patch_size],
    ),
    Weights.Spec(name: "#{VISION}embeddings.position_embedding.weight", shape: [grid * grid + 1, v.hidden_size]),
    Weights.Spec(name: "#{VISION}pre_layrnorm.weight", shape: [v.hidden_size]),
    Weights.Spec(name: "#{VISION}pre_layrnorm.bias", shape: [v.hidden_size]),
    Weights.Spec(name: "#{VISION}post_layernorm.weight", shape: [v.hidden_size]),
    Weights.Spec(name: "#{VISION}post_layernorm.bias", shape: [v.hidden_size]),
    Weights.Spec(name: "visual_projection.weight", shape: [config.projection_dim, v.hidden_size]),
    Weights.Spec(name: "#{TEXT}embeddings.token_embedding.weight", shape: [x.vocab_size, x.hidden_size]),
    Weights.Spec(
      name: "#{TEXT}embeddings.position_embedding.weight",
      shape: [x.max_position_embeddings, x.hidden_size],
    ),
    Weights.Spec(name: "#{TEXT}final_layer_norm.weight", shape: [x.hidden_size]),
    Weights.Spec(name: "#{TEXT}final_layer_norm.bias", shape: [x.hidden_size]),
    Weights.Spec(name: "text_projection.weight", shape: [config.projection_dim, x.hidden_size]),
    Weights.Spec(name: "logit_scale", shape: []),
  ]

  for i in 0...v.num_hidden_layers
    specs = (
      specs +
      Weights.block_specs(block_names("#{VISION}encoder.layers.#{i}."), v.hidden_size, v.intermediate_size)
    )
  end

  for i in 0...x.num_hidden_layers
    specs = specs + Weights.block_specs(block_names("#{TEXT}encoder.layers.#{i}."), x.hidden_size, x.intermediate_size)
  end

  specs
end

## `logit_scale` of the tensors `t`, read as a number and closed.
def scale_of(t: Map[String, Tensor.Tensor]) -> Result[Float]
  values = Tensor.to_list(Weights.at(t, "logit_scale"))
  Lift.close([Weights.at(t, "logit_scale")])
  first = (try values).first
  return error("logit_scale is a number") if (try values).size != 1
  case first
    Tensor.Values.Float(f) -> ok(f)
    _ -> error("logit_scale is a float")
  end
end

def vision_of(t: Map[String, Tensor.Tensor], v: VisionConfig) -> Vision
  e = "#{VISION}embeddings."
  blocks: List[Nn.EncoderBlock] = []
  for i in 0...v.num_hidden_layers
    names = block_names("#{VISION}encoder.layers.#{i}.")
    blocks = blocks.push(Weights.block_of(t, names, v.num_attention_heads, Nn.Activation.GeluErf, v.layer_norm_eps))
  end

  Vision(
    patch: Nn.Conv2d(weight: Weights.at(t, "#{e}patch_embedding.weight"), bias: none, stride: v.patch_size),
    class_embedding: Weights.at(t, "#{e}class_embedding"),
    positions: Nn.Embedding(table: Weights.at(t, "#{e}position_embedding.weight")),
    pre_norm: Weights.norm_of(t, "#{VISION}pre_layrnorm", v.layer_norm_eps),
    stack: Nn.EncoderStack(blocks: blocks),
    post_norm: Weights.norm_of(t, "#{VISION}post_layernorm", v.layer_norm_eps),
    projection: Weights.dense_of(t, "visual_projection"),
  )
end

def text_of(t: Map[String, Tensor.Tensor], x: TextConfig) -> Text
  e = "#{TEXT}embeddings."
  blocks: List[Nn.EncoderBlock] = []
  for i in 0...x.num_hidden_layers
    names = block_names("#{TEXT}encoder.layers.#{i}.")
    blocks = blocks.push(Weights.block_of(t, names, x.num_attention_heads, Nn.Activation.GeluErf, x.layer_norm_eps))
  end

  Text(
    tokens: Nn.Embedding(table: Weights.at(t, "#{e}token_embedding.weight")),
    positions: Nn.Embedding(table: Weights.at(t, "#{e}position_embedding.weight")),
    stack: Nn.EncoderStack(
      blocks: blocks,
      final_norm: some(Weights.norm_of(t, "#{TEXT}final_layer_norm", x.layer_norm_eps)),
    ),
    projection: Weights.dense_of(t, "text_projection"),
  )
end

## The vision tower as `Tensor` calls: the patches as tokens after the class embedding, the positions, `pre_layrnorm`,
## the blocks, and `post_layernorm` and the projection of the class token.
def vision_forward(m: Clip, images: Tensor.Tensor) -> Result[Tensor.Tensor]
  v = m.vision
  size = m.config.vision_config
  batch = Tensor.shape(images)[0]
  width = size.hidden_size
  grid = size.image_size / size.patch_size
  patches = try Tensor.reshape(try Nn.conv2d(v.patch, images), [batch, width, grid * grid])
  cls = try Tensor.broadcast_to(try Tensor.reshape(v.class_embedding, [1, 1, width]), [batch, 1, width])
  tokens = try Tensor.concat([cls, try Tensor.permute(patches, [0, 2, 1])], 1)
  h = try Nn.layer_norm(v.pre_norm, try Nn.positional(v.positions, tokens))
  h = try Nn.encoder_stack(v.stack, h)
  first = try Tensor.reshape(try Tensor.slice(h, [0, 0, 0], [batch, 1, width]), [batch, width])
  Nn.dense(v.projection, try Nn.layer_norm(v.post_norm, first))
end

## The text tower as `Tensor` calls: `ids` (`I32`, `[batch, seq]`) and `marks` (`F32`, 1.0 at the last id of each row
## and 0.0 elsewhere), which `Sentence.mean_pooled` reads as the one token to take.
def text_forward(m: Clip, ids: Tensor.Tensor, marks: Tensor.Tensor) -> Result[Tensor.Tensor]
  x = m.text
  shape = Tensor.shape(ids)
  width = m.config.text_config.hidden_size
  words = try Nn.embedding(x.tokens, try Tensor.reshape(ids, [shape[0] * shape[1]]))
  h = try Nn.positional(x.positions, try Tensor.reshape(words, [shape[0], shape[1], width]))
  h = try Nn.encoder_stack(x.stack, h, none, true)
  Nn.dense(x.projection, try Sentence.mean_pooled(h, marks))
end

def scored(m: Clip, images: Tensor.Tensor, texts: Tensor.Tensor, soft: Bool) -> Result[Tensor.Tensor]
  dim = m.config.projection_dim
  a = Tensor.shape(images)
  b = Tensor.shape(texts)
  if a.size != 2 or b.size != 2 or a[1] != dim or b[1] != dim
    return error("the embeddings need [n, #{dim}], got #{a} and #{b}")
  end

  outs = try Lift.staged(
    m.scale,
    ->(s: Float) { s },
    [images, texts],
    ->(s: Float, xs: List[Tensor.Tensor]) { ok([try similarity(s, xs[0], xs[1], soft)]) },
  )

  ok(outs[0])
end

def similarity(scale: Float, images: Tensor.Tensor, texts: Tensor.Tensor, soft: Bool) -> Result[Tensor.Tensor]
  cosines = try Tensor.matmul(try Sentence.normalized(images), try Tensor.transpose(try Sentence.normalized(texts)))
  shape = Tensor.shape(cosines)
  by = try Tensor.full(shape, Tensor.Values.Float(scale), Tensor.dtype(cosines), Tensor.backend(cosines))
  scores = try Tensor.mul(cosines, by)
  return Tensor.softmax(scores, 1) if soft
  ok(scores)
end

def padded(r: List[Int], i: Int) -> Int
  return r[i] if i < r.size
  0
end

## 1.0 at the last of `size` ids, 0.0 elsewhere.
def pick(size: Int, i: Int) -> Float
  return 1.0 if i == size - 1
  0.0
end
