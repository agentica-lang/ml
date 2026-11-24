## BERT for inference, from a Hugging Face checkpoint: the embeddings (words,
## positions, token types, a layer norm), the post-norm encoder blocks with the
## exact GELU, and the pooler (`transformers/DESIGN.md`).
##
## It loads `bert-base-uncased`-style checkpoints, with or without the `bert.`
## prefix and with `gamma`/`beta` or `weight`/`bias` layer norms; the heads of
## `BertFor...` (`cls.*`) are ignored.

import Hub from "pkg/hub/hub"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import WordPiece from "tokenizer/wordpiece"
import Source from "tokenizer/source"
import Lift from "lift"

## The fields of `config.json` that the model reads.
struct Config
  hidden_size: Int
  num_hidden_layers: Int
  num_attention_heads: Int
  intermediate_size: Int
  vocab_size: Int
  max_position_embeddings: Int
  type_vocab_size: Int
  layer_norm_eps: Float
  hidden_act: String
  model_type: String
  pad_token_id: Int = 0
  position_embedding_type: String = "absolute"
end

struct Bert
  config: Config
  words: Nn.Embedding
  positions: Nn.Embedding
  types: Nn.Embedding
  norm: Nn.LayerNorm
  stack: Nn.EncoderStack
  ## `tanh(dense(first token))`; none when the checkpoint has no pooler.
  pooler: Optional[Nn.Dense]
end

## What `forward` computes for a batch.
struct Output
  ## `[batch, seq, hidden_size]`: the last hidden state, padding rows included.
  hidden: Tensor.Tensor
  ## `[batch, hidden_size]`: the pooler of the first token; none without a pooler.
  pooled: Optional[Tensor.Tensor]
end

## The model in `dir`, which holds `model.safetensors` and `config.json`.
def open(dir: String, backend: Tensor.Backend) -> Result[Bert]
  text = try Source.within("#{dir}/config.json", Source.read_text("#{dir}/config.json"))
  config = try Source.within("#{dir}/config.json", JSON.decode(text, Config))
  load(try Hub.open("#{dir}/model.safetensors"), config, backend)
end

## The model of an open checkpoint, its shapes checked against `config`.
def load(c: Hub.Checkpoint, config: Config, backend: Tensor.Backend) -> Result[Bert]
  return error("model_type is #{config.model_type}, not bert") if config.model_type != "bert"
  if config.position_embedding_type != "absolute"
    return error("position_embedding_type #{config.position_embedding_type} is not supported")
  end

  width = config.hidden_size
  if width % config.num_attention_heads != 0
    return error("hidden_size #{width} is not a multiple of #{config.num_attention_heads} heads")
  end

  activation = try activation_of(config.hidden_act)
  prefix = ""
  if Hub.has?(c, "bert.embeddings.word_embeddings.weight")
    prefix = "bert."
  end

  e = "#{prefix}embeddings."
  eps = config.layer_norm_eps
  words = try Load.embedding(c, "#{e}word_embeddings.weight", backend)
  try expect_shape("#{e}word_embeddings.weight", words.table, [config.vocab_size, width])
  positions = try Load.embedding(c, "#{e}position_embeddings.weight", backend)
  try expect_shape("#{e}position_embeddings.weight", positions.table, [config.max_position_embeddings, width])
  types = try Load.embedding(c, "#{e}token_type_embeddings.weight", backend)
  try expect_shape("#{e}token_type_embeddings.weight", types.table, [config.type_vocab_size, width])
  blocks: List[Nn.EncoderBlock] = []
  for i in 0...config.num_hidden_layers
    block = try Load.encoder_block(
      c,
      block_names("#{prefix}encoder.layer.#{i}."),
      config.num_attention_heads,
      activation,
      eps,
      backend,
    )

    try expect_shape("layer #{i} up", block.mlp.up.w, [width, config.intermediate_size])
    blocks = blocks.push(block.with(norm_first: false))
  end

  pooler: Optional[Nn.Dense] = none
  if Hub.has?(c, "#{prefix}pooler.dense.weight")
    pooler = some(try Load.dense(c, "#{prefix}pooler.dense", backend))
  end

  ok(
    Bert(
      config: config,
      words: words,
      positions: positions,
      types: types,
      norm: try Load.layer_norm(c, "#{e}LayerNorm", eps, backend),
      stack: Nn.EncoderStack(blocks: blocks),
      pooler: pooler,
    ),
  )
end

## The batch of `encodings` (`WordPiece.encode`), padded with `pad_token_id`
## to the longest, through the model. Every encoding needs a token. The
## outputs belong to the caller, who closes them (`Bert.close`).
def run(m: Bert, encodings: List[WordPiece.Encoding]) -> Result[Output]
  batch = try inputs(m, encodings)
  outs = through(m, batch, true)
  Lift.close([batch.ids, batch.type_ids, batch.keep])
  to_output(try outs)
end

## The sentence embedding of each encoding: the mean of its token vectors
## (padding left out) as `[batch, hidden_size]`, which the caller closes. The
## pooler does not run.
def embed(m: Bert, encodings: List[WordPiece.Encoding]) -> Result[Tensor.Tensor]
  batch = try inputs(m, encodings)
  outs = through(m, batch, false)
  mean: Result[List[Tensor.Tensor]] = case outs
    ok(xs) ->
      Lift.staged(
        m,
        ->(p: Bert) { p },
        [xs[0], batch.keep],
        ->(_: Bert, ys: List[Tensor.Tensor]) { ok([try mean_pooled(ys[0], ys[1])]) },
      )
    error(e) -> error(e.message)
  end

  Lift.close([batch.ids, batch.type_ids, batch.keep])
  case outs
    ok(xs) -> Lift.close(xs)
    error(_) -> nil
  end

  ok((try mean)[0])
end

## `ids` and `type_ids` (`I32`, `[batch, seq]`) and `mask` (`F32`, `[batch, seq]`:
## 1.0 for a token, 0.0 for padding), through the model. `seq` may not pass
## `max_position_embeddings`.
def forward(m: Bert, ids: Tensor.Tensor, type_ids: Tensor.Tensor, mask: Tensor.Tensor) -> Result[Output]
  shape = Tensor.shape(ids)
  if shape.size != 2 or Tensor.shape(type_ids) != shape or Tensor.shape(mask) != shape
    return error("forward needs ids, type_ids and mask of one [batch, seq] shape, got #{shape}")
  end

  to_output(try through(m, Inputs(ids: ids, type_ids: type_ids, keep: mask), true))
end

## Closes the tensors of an `Output`.
def close(out: Output) -> Void
  Lift.close([out.hidden] + out.pooled.map { |p| [p] }.unwrap_or([]))
end

## Closes the weights of `m` now. Nothing may use `m` afterwards: a call that
## does is an error, never a read of freed memory.
def close_model(m: Bert) -> Void
  tensors = [m.words.table, m.positions.table, m.types.table] + Lift.norm_parts(m.norm)
  for b in m.stack.blocks
    tensors = tensors + Lift.block_parts(b)
  end

  case m.pooler
    some(p) -> tensors = tensors + Lift.dense_parts(p)
    none -> nil
  end

  Lift.close(tensors)
end

private

## The three input tensors of a batch: ids and type ids (`I32`) and the mask (`F32`).
struct Inputs
  ids: Tensor.Tensor
  type_ids: Tensor.Tensor
  keep: Tensor.Tensor
end

## The tensors of `m` as `Trace` constants (`Lift`).
def traced(m: Bert) -> Bert
  m.with(
    words: Lift.embedding(m.words),
    positions: Lift.embedding(m.positions),
    types: Lift.embedding(m.types),
    norm: Lift.norm(m.norm),
    stack: Nn.EncoderStack(blocks: m.stack.blocks.map { |b| Lift.block(b) }),
    pooler: m.pooler.map { |p| Lift.dense(p) },
  )
end

## The names of one block, under `p` (`bert.encoder.layer.N.`).
def block_names(p: String) -> Load.BlockNames
  Load.BlockNames(
    norm1: "#{p}attention.output.LayerNorm",
    query: "#{p}attention.self.query",
    key: "#{p}attention.self.key",
    value: "#{p}attention.self.value",
    output: "#{p}attention.output.dense",
    norm2: "#{p}output.LayerNorm",
    up: "#{p}intermediate.dense",
    down: "#{p}output.dense",
  )
end

def padded(xs: List[Int], i: Int, pad: Int) -> Int
  return xs[i] if i < xs.size
  pad
end

## 1.0 for position `i` of `size` tokens, 0.0 for padding.
def real_token(size: Int, i: Int) -> Float
  return 1.0 if i < size
  0.0
end

def activation_of(name: String) -> Result[Nn.Activation]
  case name
    "gelu" -> ok(Nn.Activation.GeluErf)
    "gelu_new" -> ok(Nn.Activation.Gelu)
    "gelu_pytorch_tanh" -> ok(Nn.Activation.Gelu)
    "relu" -> ok(Nn.Activation.Relu)
    _ -> error("hidden_act #{name} is not supported")
  end
end

def expect_shape(name: String, t: Tensor.Tensor, want: List[Int]) -> Result[Void]
  return error("#{name}: needs #{want}, got #{Tensor.shape(t)}") if Tensor.shape(t) != want
  ok()
end

## The model over `batch`: the hidden state, and the pooler with `pool`.
def through(m: Bert, batch: Inputs, pool: Bool) -> Result[List[Tensor.Tensor]]
  Lift.staged(
    m,
    ->(p: Bert) { traced(p) },
    [batch.ids, batch.type_ids, batch.keep],
    ->(p: Bert, xs: List[Tensor.Tensor]) { graph(p, xs[0], xs[1], xs[2], pool) },
  )
end

def to_output(outs: List[Tensor.Tensor]) -> Result[Output]
  pooled: Optional[Tensor.Tensor] = none
  if outs.size > 1
    pooled = some(outs[1])
  end

  ok(Output(hidden: outs[0], pooled: pooled))
end

## The batch of `encodings` as tensors: padded to the longest, with the mask.
def inputs(m: Bert, encodings: List[WordPiece.Encoding]) -> Result[Inputs]
  return error("run needs at least one encoding") if encodings.empty?
  seq = encodings.map { |e| e.ids.size }.max.unwrap!
  ids: List[Tensor.Values] = []
  types: List[Tensor.Values] = []
  for e in encodings
    return error("an encoding has no tokens") if e.ids.empty?
    return error("an encoding has #{e.ids.size} ids but #{e.type_ids.size} type ids") if e.ids.size != e.type_ids.size
    for i in 0...seq
      ids = ids.push(Tensor.Values.Int(padded(e.ids, i, m.config.pad_token_id)))
      types = types.push(Tensor.Values.Int(padded(e.type_ids, i, 0)))
    end
  end

  backend = Tensor.backend(m.words.table)
  shape = [encodings.size, seq]
  ok(
    Inputs(
      ids: try Tensor.from_list(ids, shape, Tensor.Dtype.I32, backend),
      type_ids: try Tensor.from_list(types, shape, Tensor.Dtype.I32, backend),
      keep: try Tensor.from_list(keep_mask(encodings, seq), shape, Tensor.Dtype.F32, backend),
    ),
  )
end

## 1.0 for each token of each encoding and 0.0 for its padding, row after row.
def keep_mask(encodings: List[WordPiece.Encoding], seq: Int) -> List[Tensor.Values]
  keep: List[Tensor.Values] = []
  for e in encodings
    for i in 0...seq
      keep = keep.push(Tensor.Values.Float(real_token(e.ids.size, i)))
    end
  end

  keep
end

## The embeddings, the blocks and, with `pool` and a pooler, the pooler: the
## hidden state and `tanh(pooler(hidden[:, 0]))`.
def graph(
  m: Bert,
  ids: Tensor.Tensor,
  type_ids: Tensor.Tensor,
  mask: Tensor.Tensor,
  pool: Bool,
) -> Result[List[Tensor.Tensor]]
  shape = Tensor.shape(ids)
  batch = shape[0]
  seq = shape[1]
  width = m.config.hidden_size
  flat = [batch * seq]
  words = try Tensor.reshape(try Nn.embedding(m.words, try Tensor.reshape(ids, flat)), [batch, seq, width])
  kinds = try Tensor.reshape(try Nn.embedding(m.types, try Tensor.reshape(type_ids, flat)), [batch, seq, width])
  h = try Nn.layer_norm(m.norm, try Nn.positional(m.positions, try Tensor.add(words, kinds)))
  h = try Nn.encoder_stack(m.stack, h, some(try padding_bias(mask)))
  return ok([h]) if not pool
  case m.pooler
    some(p) ->
      first = try Tensor.reshape(try Tensor.slice(h, [0, 0, 0], [batch, 1, width]), [batch, width])
      ok([h, try Tensor.tanh(try Nn.dense(p, first))])
    none -> ok([h])
  end
end

## The additive bias `[batch, 1, seq, seq]` of a `[batch, seq]` mask: `MASKED`
## on the keys that are padding, 0.0 elsewhere.
def padding_bias(mask: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(mask)
  ones = try Tensor.full(shape, Tensor.Values.Float(1.0), Tensor.Dtype.F32, Tensor.backend(mask))
  scale = try Tensor.full(shape, Tensor.Values.Float(Nn.MASKED), Tensor.Dtype.F32, Tensor.backend(mask))
  bias = try Tensor.mul(try Tensor.sub(ones, mask), scale)
  Tensor.broadcast_to(try Tensor.reshape(bias, [shape[0], 1, 1, shape[1]]), [shape[0], 1, shape[1], shape[1]])
end

## `hidden` `[batch, seq, width]` averaged over the tokens `mask` keeps. The
## tokens are the last axis while they are summed, as ggml lowers only sums
## over a leading or trailing block of axes.
def mean_pooled(hidden: Tensor.Tensor, mask: Tensor.Tensor) -> Result[Tensor.Tensor]
  shape = Tensor.shape(hidden)
  across = [shape[0], shape[2], shape[1]]
  row = try Tensor.reshape(mask, [shape[0], 1, shape[1]])
  masked = try Tensor.mul(try Tensor.permute(hidden, [0, 2, 1]), try Tensor.broadcast_to(row, across))
  total = try Tensor.sum(masked, some([2]))
  count = try Tensor.sum(mask, some([1]), true)
  Tensor.div(total, try Tensor.broadcast_to(count, [shape[0], shape[2]]))
end
