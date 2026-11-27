## The weights of an encoder by name, for training it (`Params.split` of `train`): a map from the names of the
## checkpoint to the tensors of the layers, and the layers back from such a map. DistilBERT and BERT share it.

import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"

## An error unless `given` has the names of `own`, no more and no fewer, each with the shape `own` holds.
def check(own: Map[String, Tensor.Tensor], given: Map[String, Tensor.Tensor]) -> Result[Void]
  for name in given.keys.sort
    old = try own.get(name).to_result("the model has no weight #{name}")
    if Tensor.shape(given[name]) != Tensor.shape(old)
      return error("#{name}: the model holds #{Tensor.shape(old)}, got #{Tensor.shape(given[name])}")
    end
  end

  for name in own.keys.sort
    return error("the weights have no #{name}") if !given.has_key?(name)
  end

  ok()
end

def dense(out: Map[String, Tensor.Tensor], name: String, d: Nn.Dense) -> Map[String, Tensor.Tensor]
  named = out.put("#{name}.weight", d.w)
  case d.b
    some(b) -> named.put("#{name}.bias", b)
    none -> named
  end
end

def norm(out: Map[String, Tensor.Tensor], name: String, n: Nn.LayerNorm) -> Map[String, Tensor.Tensor]
  out.put("#{name}.weight", n.gamma).put("#{name}.bias", n.beta)
end

## The eight layers of the block `b` under `n`, added to `out`.
def block(out: Map[String, Tensor.Tensor], n: Load.BlockNames, b: Nn.EncoderBlock) -> Map[String, Tensor.Tensor]
  out = norm(out, n.norm1, b.norm1)
  out = dense(out, n.query, b.attention.query)
  out = dense(out, n.key, b.attention.key)
  out = dense(out, n.value, b.attention.value)
  out = dense(out, n.output, b.attention.output)
  out = norm(out, n.norm2, b.norm2)
  out = dense(out, n.up, b.mlp.up)
  dense(out, n.down, b.mlp.down)
end

## `d` and `n` with the tensors of `weights`, whose names `check` has checked.
def with_dense(weights: Map[String, Tensor.Tensor], name: String, d: Nn.Dense) -> Nn.Dense
  d.with(w: weights["#{name}.weight"], b: d.b.map { |_| weights["#{name}.bias"] })
end

def with_norm(weights: Map[String, Tensor.Tensor], name: String, n: Nn.LayerNorm) -> Nn.LayerNorm
  n.with(gamma: weights["#{name}.weight"], beta: weights["#{name}.bias"])
end

def with_block(weights: Map[String, Tensor.Tensor], n: Load.BlockNames, b: Nn.EncoderBlock) -> Nn.EncoderBlock
  a = b.attention
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
  )
end
