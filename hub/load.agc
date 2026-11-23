## `nn/` layers from a checkpoint, by tensor name (`hub/DESIGN.md`).
##
## Names follow Hugging Face: a dense layer is `{prefix}.weight` stored
## `[out, in]` (transposed here to `nn/`'s `[in, out]`) and an optional
## `{prefix}.bias`; a layer norm is `{prefix}.weight` and `{prefix}.bias`.
## Float weights become `F32`, the compute dtype (`tensor/DESIGN.md` §6).

import Hub from "hub"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## The prefix of every layer of one encoder block.
struct BlockNames
  norm1: String
  query: String
  key: String
  value: String
  output: String
  norm2: String
  up: String
  down: String
end

## The prefix of every layer of one decoder block: self-attention
## (`query`...`output`), cross-attention over the encoder output
## (`cross_query`...`cross_output`), and the `mlp`.
struct DecoderBlockNames
  norm1: String
  query: String
  key: String
  value: String
  output: String
  norm2: String
  cross_query: String
  cross_key: String
  cross_value: String
  cross_output: String
  norm3: String
  up: String
  down: String
end

## The float tensor `name` as `F32`, of `rank`.
def weight(c: Hub.Checkpoint, name: String, rank: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  t = try Hub.tensor(c, name, backend)
  shape = Tensor.shape(t)
  return error("#{name}: needs rank #{rank}, got #{shape}") if shape.size != rank
  case Tensor.dtype(t)
    Tensor.Dtype.F32 -> ok(t)
    Tensor.Dtype.F16 -> Tensor.to_dtype(t, Tensor.Dtype.F32)
    Tensor.Dtype.F64 -> Tensor.to_dtype(t, Tensor.Dtype.F32)
    _ -> error("#{name}: needs a float dtype, got #{Tensor.dtype(t)}")
  end
end

def dense(c: Hub.Checkpoint, prefix: String, backend: Tensor.Backend) -> Result[Nn.Dense]
  w = try weight(c, "#{prefix}.weight", 2, backend)
  b = try bias(c, prefix, Tensor.shape(w)[0], backend)
  t = transposed(w)
  close(w)
  ok(Nn.Dense(w: try t, b: b))
end

## `t` transposed, through a trace: a backend that computes only whole graphs
## (`Ggml`) loads too.
def transposed(t: Tensor.Tensor) -> Result[Tensor.Tensor]
  Trace.eval([t], ->(xs: List[Tensor.Tensor]) { Tensor.transpose(xs[0]) })
end

## A conv layer: `{prefix}.weight` `[out, in, kernel]`, as stored; `{prefix}.bias` if present.
def conv1d(c: Hub.Checkpoint, prefix: String, stride: Int, padding: Int, backend: Tensor.Backend) -> Result[Nn.Conv1d]
  w = try weight(c, "#{prefix}.weight", 3, backend)
  b = try bias(c, prefix, Tensor.shape(w)[0], backend)
  ok(Nn.Conv1d(weight: w, bias: b, stride: stride, padding: padding))
end

def layer_norm(c: Hub.Checkpoint, prefix: String, eps: Float, backend: Tensor.Backend) -> Result[Nn.LayerNorm]
  gamma = try weight(c, "#{prefix}.weight", 1, backend)
  beta = try weight(c, "#{prefix}.bias", 1, backend)
  if Tensor.shape(beta) != Tensor.shape(gamma)
    return error("#{prefix}: weight #{Tensor.shape(gamma)} and bias #{Tensor.shape(beta)} differ")
  end

  ok(Nn.LayerNorm(gamma: gamma, beta: beta, eps: eps))
end

def embedding(c: Hub.Checkpoint, name: String, backend: Tensor.Backend) -> Result[Nn.Embedding]
  ok(Nn.Embedding(table: try weight(c, name, 2, backend)))
end

def encoder_block(
  c: Hub.Checkpoint,
  names: BlockNames,
  heads: Int,
  activation: Nn.Activation,
  eps: Float,
  backend: Tensor.Backend,
) -> Result[Nn.EncoderBlock]
  ok(
    Nn.EncoderBlock(
      norm1: try layer_norm(c, names.norm1, eps, backend),
      attention: try attention(c, names.query, names.key, names.value, names.output, heads, backend),
      norm2: try layer_norm(c, names.norm2, eps, backend),
      mlp: Nn.Mlp(
        up: try dense(c, names.up, backend),
        down: try dense(c, names.down, backend),
        activation: activation,
      ),
    ),
  )
end

def decoder_block(
  c: Hub.Checkpoint,
  names: DecoderBlockNames,
  heads: Int,
  activation: Nn.Activation,
  eps: Float,
  backend: Tensor.Backend,
) -> Result[Nn.DecoderBlock]
  ok(
    Nn.DecoderBlock(
      norm1: try layer_norm(c, names.norm1, eps, backend),
      self_attention: try attention(c, names.query, names.key, names.value, names.output, heads, backend),
      norm2: try layer_norm(c, names.norm2, eps, backend),
      cross_attention: try attention(
        c,
        names.cross_query,
        names.cross_key,
        names.cross_value,
        names.cross_output,
        heads,
        backend,
      ),
      norm3: try layer_norm(c, names.norm3, eps, backend),
      mlp: Nn.Mlp(
        up: try dense(c, names.up, backend),
        down: try dense(c, names.down, backend),
        activation: activation,
      ),
    ),
  )
end

private

## Frees `t` now on the backends whose tensors hold native memory, so a load
## does not hold a weight twice.
def close(t: Tensor.Tensor) -> Void
  case t.handle
    Tensor.Handle.Native(h) -> Native.close(h)
    Tensor.Handle.Ggml(h) -> Native.close(h)
    _ -> nil
  end
end

## `{prefix}.bias` of `[rows]`, or `none` when the checkpoint has none.
def bias(c: Hub.Checkpoint, prefix: String, rows: Int, backend: Tensor.Backend) -> Result[Optional[Tensor.Tensor]]
  return ok(none) if not Hub.has?(c, "#{prefix}.bias")
  b = try weight(c, "#{prefix}.bias", 1, backend)
  return error("#{prefix}.bias: needs [#{rows}], got #{Tensor.shape(b)}") if Tensor.shape(b)[0] != rows
  ok(some(b))
end

## The four projections of one attention, by prefix.
def attention(
  c: Hub.Checkpoint,
  query: String,
  key: String,
  value: String,
  output: String,
  heads: Int,
  backend: Tensor.Backend,
) -> Result[Nn.Attention]
  ok(
    Nn.Attention(
      query: try dense(c, query, backend),
      key: try dense(c, key, backend),
      value: try dense(c, value, backend),
      output: try dense(c, output, backend),
      heads: heads,
    ),
  )
end
