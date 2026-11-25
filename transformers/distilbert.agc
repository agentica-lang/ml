## DistilBERT for inference, from a Hugging Face checkpoint: the embeddings (words and learned positions, a layer
## norm, no token types), the post-norm encoder blocks with the exact GELU, and no pooler
## (`transformers/DESIGN.md`). `embed` adds the mean pooling and the optional normalization of a
## sentence-transformers model (`sentence-transformers/msmarco-distilbert-cos-v5`).
##
## It loads the checkpoint of a `DistilBertModel` (no prefix) and of a `DistilBertFor...` (`distilbert.` prefix,
## whose heads are ignored).

import Hub from "pkg/hub/hub"
import Lift from "lift"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Sentence from "sentence"
import Source from "tokenizer/source"
import Tensor from "pkg/tensor/tensor"
import WordPiece from "tokenizer/wordpiece"

## The `eps` of every layer norm: Hugging Face's DistilBERT has it in the code, not in the config.
LAYER_NORM_EPS = 0.000000000001

## The fields of `config.json` that the model reads.
struct Config
  dim: Int
  n_layers: Int
  n_heads: Int
  hidden_dim: Int
  vocab_size: Int
  max_position_embeddings: Int
  model_type: String
  activation: String = "gelu"
  pad_token_id: Int = 0
  sinusoidal_pos_embds: Bool = false
end

struct DistilBert
  config: Config
  words: Nn.Embedding
  positions: Nn.Embedding
  norm: Nn.LayerNorm
  stack: Nn.EncoderStack
  ## What the names of the checkpoint start with: `distilbert.` in a `DistilBertFor...` file, nothing otherwise.
  prefix: String = ""
end

## The model in `dir`, which holds `model.safetensors` and `config.json`.
def open(dir: String, backend: Tensor.Backend) -> Result[DistilBert]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  config = try Source.within("#{dir}/config.json", JSON.decode(text, Config))
  load(try Hub.open("#{dir}/model.safetensors"), config, backend)
end

## The model of an open checkpoint, its shapes checked against `config`.
def load(c: Hub.Checkpoint, config: Config, backend: Tensor.Backend) -> Result[DistilBert]
  return error("model_type is #{config.model_type}, not distilbert") if config.model_type != "distilbert"
  return error("sinusoidal position embeddings are not supported") if config.sinusoidal_pos_embds
  return error("activation #{config.activation} is not supported, only gelu") if config.activation != "gelu"
  width = config.dim
  if config.n_heads < 1 or width % config.n_heads != 0
    return error("dim #{width} is not a multiple of #{config.n_heads} heads")
  end

  prefix = ""
  if Hub.has?(c, "distilbert.embeddings.word_embeddings.weight")
    prefix = "distilbert."
  end

  e = "#{prefix}embeddings."
  words = try Load.embedding(c, "#{e}word_embeddings.weight", backend)
  try expect_shape("#{e}word_embeddings.weight", words.table, [config.vocab_size, width])
  positions = try Load.embedding(c, "#{e}position_embeddings.weight", backend)
  try expect_shape("#{e}position_embeddings.weight", positions.table, [config.max_position_embeddings, width])
  blocks: List[Nn.EncoderBlock] = []
  for i in 0...config.n_layers
    block = try Load.encoder_block(
      c,
      block_names("#{prefix}transformer.layer.#{i}."),
      config.n_heads,
      Nn.Activation.GeluErf,
      LAYER_NORM_EPS,
      backend,
    )

    try expect_shape("layer #{i} lin1", block.mlp.up.w, [config.hidden_dim, width])
    blocks = blocks.push(block.with(norm_first: false))
  end

  ok(
    DistilBert(
      config: config,
      words: words,
      positions: positions,
      norm: try Load.layer_norm(c, "#{e}LayerNorm", LAYER_NORM_EPS, backend),
      stack: Nn.EncoderStack(blocks: blocks),
      prefix: prefix,
    ),
  )
end

## The last hidden state `[batch, seq, dim]` of the batch of `encodings` (`WordPiece.encode`), padded with
## `pad_token_id` to the longest; the padding rows are in it. Every encoding needs a token. The caller closes the
## result (`Lift.close`).
def run(m: DistilBert, encodings: List[WordPiece.Encoding]) -> Result[Tensor.Tensor]
  batch = try inputs(m, encodings)
  out = through(m, batch, Pooling.Hidden)
  Lift.close([batch.ids, batch.keep])
  out
end

## The sentence embedding of each encoding: the mean of its token vectors (padding left out) as `[batch, dim]`,
## which the caller closes; with `normalize`, each over its length (the `Normalize` module of the checkpoint).
def embed(m: DistilBert, encodings: List[WordPiece.Encoding], normalize: Bool = false) -> Result[Tensor.Tensor]
  batch = try inputs(m, encodings)
  out = through(m, batch, pooling_of(normalize))
  Lift.close([batch.ids, batch.keep])
  out
end

## `ids` (`I32`, `[batch, seq]`) and `mask` (`F32`, `[batch, seq]`: 1.0 for a token, 0.0 for padding), through the
## model: the last hidden state. `seq` may not pass `max_position_embeddings`.
def forward(m: DistilBert, ids: Tensor.Tensor, mask: Tensor.Tensor) -> Result[Tensor.Tensor]
  through(m, Inputs(ids: ids, keep: mask), Pooling.Hidden)
end

## The mean embedding `[batch, dim]` of `ids` and `mask` (as `forward` takes them), with `normalize` over its length:
## the model as plain `Tensor` calls, not one staged graph. Inside a loss, with the traced weights of `with_weights`,
## it is part of the trace, so `Train.fit` takes its gradient. Outside a trace it runs eagerly, which ggml refuses
## ("does not run eagerly on the ggml backend"): `embed` is the call for inference.
def pooled(m: DistilBert, ids: Tensor.Tensor, mask: Tensor.Tensor, normalize: Bool = false) -> Result[Tensor.Tensor]
  try check_inputs(m, ids, mask)
  graph(m, ids, mask, pooling_of(normalize))
end

## The names of the embedding tables among `weights(m)`: what `Lora.save` of `train` must not write as the adapter
## of a linear layer.
def embeddings(m: DistilBert) -> List[String]
  ["word_embeddings", "position_embeddings"].map { |table: String| "#{m.prefix}embeddings.#{table}.weight" }
end

## The weights of `m` by their names in the checkpoint (`transformer.layer.0.attention.q_lin.weight`, under
## `distilbert.` when the file has it), for training the model or part of it (`Params.split` of `train`).
def weights(m: DistilBert) -> Map[String, Tensor.Tensor]
  e = "#{m.prefix}embeddings."
  out = {"#{e}word_embeddings.weight": m.words.table, "#{e}position_embeddings.weight": m.positions.table}
  out = norm_weights(out, "#{e}LayerNorm", m.norm)
  for i in 0...m.stack.blocks.size
    b = m.stack.blocks[i]
    n = block_names("#{m.prefix}transformer.layer.#{i}.")
    out = norm_weights(out, n.norm1, b.norm1)
    out = dense_weights(out, n.query, b.attention.query)
    out = dense_weights(out, n.key, b.attention.key)
    out = dense_weights(out, n.value, b.attention.value)
    out = dense_weights(out, n.output, b.attention.output)
    out = norm_weights(out, n.norm2, b.norm2)
    out = dense_weights(out, n.up, b.mlp.up)
    out = dense_weights(out, n.down, b.mlp.down)
  end

  out
end

## `m` with the tensors of `weights` in place of its own: the names must be those of `weights(m)`, no more and no
## fewer, each with the shape the model holds. Inside a loss the tensors are traced and `pooled` of the result is part
## of the trace. The result shares the tensors of `weights`, with no copy: `close_model` of it closes them.
def with_weights(m: DistilBert, weights: Map[String, Tensor.Tensor]) -> Result[DistilBert]
  own = weights(m)
  for name in weights.keys.sort
    old = try own.get(name).to_result("the model has no weight #{name}")
    if Tensor.shape(weights[name]) != Tensor.shape(old)
      return error("#{name}: the model holds #{Tensor.shape(old)}, got #{Tensor.shape(weights[name])}")
    end
  end

  for name in own.keys.sort
    return error("the weights have no #{name}") if !weights.has_key?(name)
  end

  e = "#{m.prefix}embeddings."
  blocks: List[Nn.EncoderBlock] = []
  for i in 0...m.stack.blocks.size
    b = m.stack.blocks[i]
    n = block_names("#{m.prefix}transformer.layer.#{i}.")
    a = b.attention
    blocks = blocks.push(
      b.with(
        norm1: with_norm(weights, n.norm1, b.norm1),
        attention: a.with(
          query: with_dense(weights, n.query, a.query),
          key: with_dense(weights, n.key, a.key),
          value: with_dense(weights, n.value, a.value),
          output: with_dense(weights, n.output, a.output),
        ),
        norm2: with_norm(weights, n.norm2, b.norm2),
        mlp: b.mlp.with(up: with_dense(weights, n.up, b.mlp.up), down: with_dense(weights, n.down, b.mlp.down)),
      ),
    )
  end

  ok(
    m.with(
      words: Nn.Embedding(table: weights["#{e}word_embeddings.weight"]),
      positions: Nn.Embedding(table: weights["#{e}position_embeddings.weight"]),
      norm: with_norm(weights, "#{e}LayerNorm", m.norm),
      stack: Nn.EncoderStack(blocks: blocks),
    ),
  )
end

## Closes the weights of `m` now. Nothing may use `m` afterwards: a call that does is an error, never a read of freed
## memory.
def close_model(m: DistilBert) -> Void
  tensors = [m.words.table, m.positions.table] + Lift.norm_parts(m.norm)
  for b in m.stack.blocks
    tensors = tensors + Lift.block_parts(b)
  end

  Lift.close(tensors)
end

private

## The input tensors of a batch: the ids (`I32`) and the mask (`F32`).
struct Inputs
  ids: Tensor.Tensor
  keep: Tensor.Tensor
end

## What follows the encoder in one graph.
enum Pooling
  Hidden
  Mean
  Normalized
end

## The tensors of `m` as `Trace` constants (`Lift`).
def traced(m: DistilBert) -> DistilBert
  m.with(
    words: Lift.embedding(m.words),
    positions: Lift.embedding(m.positions),
    norm: Lift.norm(m.norm),
    stack: Nn.EncoderStack(blocks: m.stack.blocks.map { |b| Lift.block(b) }),
  )
end

## The names of one block, under `p` (`transformer.layer.N.`).
def block_names(p: String) -> Load.BlockNames
  Load.BlockNames(
    norm1: "#{p}sa_layer_norm",
    query: "#{p}attention.q_lin",
    key: "#{p}attention.k_lin",
    value: "#{p}attention.v_lin",
    output: "#{p}attention.out_lin",
    norm2: "#{p}output_layer_norm",
    up: "#{p}ffn.lin1",
    down: "#{p}ffn.lin2",
  )
end

def dense_weights(out: Map[String, Tensor.Tensor], name: String, d: Nn.Dense) -> Map[String, Tensor.Tensor]
  named = out.put("#{name}.weight", d.w)
  case d.b
    some(b) -> named.put("#{name}.bias", b)
    none -> named
  end
end

def norm_weights(out: Map[String, Tensor.Tensor], name: String, n: Nn.LayerNorm) -> Map[String, Tensor.Tensor]
  out.put("#{name}.weight", n.gamma).put("#{name}.bias", n.beta)
end

## `d` and `n` with the tensors of `weights`, whose names `with_weights` has checked.
def with_dense(weights: Map[String, Tensor.Tensor], name: String, d: Nn.Dense) -> Nn.Dense
  d.with(w: weights["#{name}.weight"], b: d.b.map { |_| weights["#{name}.bias"] })
end

def with_norm(weights: Map[String, Tensor.Tensor], name: String, n: Nn.LayerNorm) -> Nn.LayerNorm
  n.with(gamma: weights["#{name}.weight"], beta: weights["#{name}.bias"])
end

def expect_shape(name: String, t: Tensor.Tensor, want: List[Int]) -> Result[Void]
  return error("#{name}: needs #{want}, got #{Tensor.shape(t)}") if Tensor.shape(t) != want
  ok()
end

def pooling_of(normalize: Bool) -> Pooling
  return Pooling.Normalized if normalize
  Pooling.Mean
end

def check_inputs(m: DistilBert, ids: Tensor.Tensor, mask: Tensor.Tensor) -> Result[Void]
  shape = Tensor.shape(ids)
  if shape.size != 2 or Tensor.shape(mask) != shape
    return error("the model needs ids and a mask of one [batch, seq] shape, got #{shape}")
  end

  if shape[1] > m.config.max_position_embeddings
    return error("#{shape[1]} tokens pass max_position_embeddings #{m.config.max_position_embeddings}")
  end

  ok()
end

## The model over `batch`, then `pooling`: one graph on ggml.
def through(m: DistilBert, batch: Inputs, pooling: Pooling) -> Result[Tensor.Tensor]
  try check_inputs(m, batch.ids, batch.keep)
  outs = try Lift.staged(
    m,
    ->(p: DistilBert) { traced(p) },
    [batch.ids, batch.keep],
    ->(p: DistilBert, xs: List[Tensor.Tensor]) { ok([try graph(p, xs[0], xs[1], pooling)]) },
  )

  ok(outs[0])
end

## The batch of `encodings` as tensors: padded to the longest, with the mask.
def inputs(m: DistilBert, encodings: List[WordPiece.Encoding]) -> Result[Inputs]
  return error("run needs at least one encoding") if encodings.empty?
  seq = encodings.map { |e| e.ids.size }.max.unwrap!
  ids: List[Tensor.Values] = []
  keep: List[Tensor.Values] = []
  for e in encodings
    return error("an encoding has no tokens") if e.ids.empty?
    for i in 0...seq
      if i < e.ids.size
        ids = ids.push(Tensor.Values.Int(e.ids[i]))
        keep = keep.push(Tensor.Values.Float(1.0))
      else
        ids = ids.push(Tensor.Values.Int(m.config.pad_token_id))
        keep = keep.push(Tensor.Values.Float(0.0))
      end
    end
  end

  backend = Tensor.backend(m.words.table)
  shape = [encodings.size, seq]
  id_tensor = try Tensor.from_list(ids, shape, Tensor.Dtype.I32, backend)
  case Tensor.from_list(keep, shape, Tensor.Dtype.F32, backend)
    ok(mask) -> ok(Inputs(ids: id_tensor, keep: mask))
    error(e) ->
      Lift.close([id_tensor])
      error(e)
  end
end

## The embeddings and the blocks, then the pooling.
def graph(m: DistilBert, ids: Tensor.Tensor, mask: Tensor.Tensor, pooling: Pooling) -> Result[Tensor.Tensor]
  shape = Tensor.shape(ids)
  batch = shape[0]
  seq = shape[1]
  width = m.config.dim
  words = try Tensor.reshape(try Nn.embedding(m.words, try Tensor.reshape(ids, [batch * seq])), [batch, seq, width])
  h = try Nn.layer_norm(m.norm, try Nn.positional(m.positions, words))
  h = try Nn.encoder_stack(m.stack, h, some(try Nn.padding_bias(mask)))
  case pooling
    Pooling.Hidden -> ok(h)
    Pooling.Mean -> Sentence.mean_pooled(h, mask)
    Pooling.Normalized -> Sentence.normalized(try Sentence.mean_pooled(h, mask))
  end
end
