## `nn/` layers from a checkpoint, by tensor name (`hub/DESIGN.md`).
##
## Names follow Hugging Face: a dense layer is `{prefix}.weight` stored
## `[out, in]` (transposed here to `nn/`'s `[in, out]`) and an optional
## `{prefix}.bias`; a layer norm is `{prefix}.weight` and `{prefix}.bias`.
## Float weights become `F32`, the compute dtype (`tensor/DESIGN.md` §6).

import Hub from "hub"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"

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
  b: Optional[Tensor.Tensor] = none
  if Hub.has?(c, "#{prefix}.bias")
    bias = try weight(c, "#{prefix}.bias", 1, backend)
    if Tensor.shape(bias)[0] != Tensor.shape(w)[0]
      return error("#{prefix}.bias: needs [#{Tensor.shape(w)[0]}], got #{Tensor.shape(bias)}")
    end

    b = some(bias)
  end

  ok(Nn.Dense(w: try Tensor.transpose(w), b: b))
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
  attention = Nn.Attention(
    query: try dense(c, names.query, backend),
    key: try dense(c, names.key, backend),
    value: try dense(c, names.value, backend),
    output: try dense(c, names.output, backend),
    heads: heads,
  )

  ok(
    Nn.EncoderBlock(
      norm1: try layer_norm(c, names.norm1, eps, backend),
      attention: attention,
      norm2: try layer_norm(c, names.norm2, eps, backend),
      mlp: Nn.Mlp(
        up: try dense(c, names.up, backend),
        down: try dense(c, names.down, backend),
        activation: activation,
      ),
    ),
  )
end
