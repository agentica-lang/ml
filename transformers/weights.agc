## The tensors of a model checkpoint by name, for the models whose layers `Load` has no helper for: the names and
## shapes are checked first, then every tensor is loaded (and all closed again if one fails), and the layers are built
## from the map.

import Hub from "pkg/hub/hub"
import Lift from "lift"
import Load from "pkg/hub/load"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"

## A tensor of the checkpoint and the shape that `config.json` gives it.
struct Spec
  name: String
  shape: List[Int]
end

## An error unless the checkpoint has every tensor of `specs` with its shape.
def check(c: Hub.Checkpoint, specs: List[Spec]) -> Result[Void]
  for s in specs
    return error("#{s.name}: the checkpoint has no such tensor") if !Hub.has?(c, s.name)
    got = try Hub.shape(c, s.name)
    return error("#{s.name}: needs #{s.shape}, got #{got}") if got != s.shape
  end

  ok()
end

## Every tensor of `specs` from the checkpoint, by name. When one fails to load, the ones before it are closed.
def load(c: Hub.Checkpoint, specs: List[Spec], backend: Tensor.Backend) -> Result[Map[String, Tensor.Tensor]]
  out: Map[String, Tensor.Tensor] = {}
  for s in specs
    case Load.weight(c, s.name, s.shape.size, backend)
      ok(t) -> out = out.put(s.name, t)
      error(e) ->
        Lift.close(out.values)
        return error(e)
    end
  end

  ok(out)
end

## The tensor `name`, which `check` has checked and `load` has loaded.
def at(t: Map[String, Tensor.Tensor], name: String) -> Tensor.Tensor = t.get(name).unwrap!

## A dense layer as the checkpoint stores it (`[out, in]`), with a bias when the checkpoint has one.
def dense_of(t: Map[String, Tensor.Tensor], prefix: String) -> Nn.Dense
  Nn.Dense(w: at(t, "#{prefix}.weight"), b: t.get("#{prefix}.bias"), out_in: true)
end

def norm_of(t: Map[String, Tensor.Tensor], prefix: String, eps: Float) -> Nn.LayerNorm
  Nn.LayerNorm(gamma: at(t, "#{prefix}.weight"), beta: at(t, "#{prefix}.bias"), eps: eps)
end

## The specs of one encoder block named by `n`: two layer norms, the four projections of the attention (the biases of
## the query, key and value only with `qkv_bias`) and the two dense layers of the MLP, `width` wide and `inner` inside.
def block_specs(n: Load.BlockNames, width: Int, inner: Int, qkv_bias: Bool = true) -> List[Spec]
  specs: List[Spec] = []
  for norm in [n.norm1, n.norm2]
    specs = specs + [Spec(name: "#{norm}.weight", shape: [width]), Spec(name: "#{norm}.bias", shape: [width])]
  end

  for p in [n.query, n.key, n.value, n.output]
    specs = specs.push(Spec(name: "#{p}.weight", shape: [width, width]))
    if qkv_bias or p == n.output
      specs = specs.push(Spec(name: "#{p}.bias", shape: [width]))
    end
  end

  for p in [n.up, n.down]
    out = inner
    into = width
    if p == n.down
      out = width
      into = inner
    end

    specs = specs + [Spec(name: "#{p}.weight", shape: [out, into]), Spec(name: "#{p}.bias", shape: [out])]
  end

  specs
end

## The pre-norm encoder block named by `n`, from the tensors `t` that `block_specs` listed.
def block_of(
  t: Map[String, Tensor.Tensor],
  n: Load.BlockNames,
  heads: Int,
  activation: Nn.Activation,
  eps: Float,
) -> Nn.EncoderBlock
  Nn.EncoderBlock(
    norm1: norm_of(t, n.norm1, eps),
    attention: Nn.Attention(
      query: dense_of(t, n.query),
      key: dense_of(t, n.key),
      value: dense_of(t, n.value),
      output: dense_of(t, n.output),
      heads: heads,
    ),
    norm2: norm_of(t, n.norm2, eps),
    mlp: Nn.Mlp(up: dense_of(t, n.up), down: dense_of(t, n.down), activation: activation),
  )
end
