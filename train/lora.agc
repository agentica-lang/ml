## LoRA (`train/DESIGN.md`): a weight `W` of a linear layer, `[out, in]`, stays frozen and two small matrices are
## trained beside it, `A` of `[r, in]` and `B` of `[out, r]`. The layer then computes with `W + (alpha / r) · B A`.
## `applied` builds that sum, so the model's own forward runs on it: inside a loss it is the training forward, and
## outside it is the merge for inference. `B` starts at zero, so a new adapter changes nothing.

import Init from "pkg/nn/init"
import Tensor from "pkg/tensor/tensor"
import Trace from "pkg/tensor/trace"

## `targets` name the layers as PEFT's `target_modules` do (a list of names, not a regular expression): a layer is
## adapted when its name (its weight's name without `.weight`) is a target or ends with `.` and a target (`q_proj`
## takes every `...self_attn.q_proj`).
## `dropout` is what a PEFT adapter was trained with; an adapter with a dropout above 0 runs, and is not trained here.
struct Config
  targets: List[String]
  r: Int = 8
  alpha: Float = 16.0
  dropout: Float = 0.0
end

## `names` are the weights the adapter changes, sorted; `a[i]` and `b[i]` belong to `names[i]`.
struct Adapter
  config: Config
  names: List[String]
  a: List[Tensor.Tensor]
  b: List[Tensor.Tensor]
end

## A new adapter for the layers of `weights` that `config.targets` name: every `A` uniform in `±1 / sqrt(in)` from
## `seed` (PEFT's start) and every `B` zero. An error when a target names no layer, or a layer whose weight is not a
## matrix.
def init(weights: Map[String, Tensor.Tensor], config: Config, seed: Int, backend: Tensor.Backend) -> Result[Adapter]
  return error("a LoRA rank is at least 1, got #{config.r}") if config.r < 1
  return error("a LoRA alpha is above 0, got #{config.alpha}: the adapter would add nothing") if config.alpha <= 0.0
  names = try adapted(weights.keys, config.targets)
  a: List[Tensor.Tensor] = []
  b: List[Tensor.Tensor] = []
  for i in 0...names.size
    shape = Tensor.shape(weights[names[i]])
    return error("#{names[i]}: LoRA adapts a matrix, got #{shape}") if shape.size != 2
    try float32(names[i], weights[names[i]])
    a = a.push(try Init.uniform([config.r, shape[1]], 1.0 / Math.sqrt(shape[1].to_f), Init.derive(seed, i), backend))
    b = b.push(try Tensor.zeros([shape[0], config.r], Tensor.Dtype.F32, backend))
  end

  ok(Adapter(config: config, names: names, a: a, b: b))
end

## What `fit` trains: `A` then `B` of each adapted weight, in the order of `names`. An adapter with a dropout above 0
## is refused: a step here has no dropout (ml#209).
def params(adapter: Adapter) -> Result[List[Tensor.Tensor]]
  if adapter.config.dropout > 0.0
    return error(
      "the adapter has a dropout of #{adapter.config.dropout}, and LoRA trains without dropout here (ml#209): " +
      "set it to 0 to train",
    )
  end

  out: List[Tensor.Tensor] = []
  for i in 0...adapter.names.size
    out = out.push(adapter.a[i]).push(adapter.b[i])
  end

  ok(out)
end

## `adapter` with `params` as its tensors: what `params` gave, a loss got or `fit` answered.
def with_params(adapter: Adapter, params: List[Tensor.Tensor]) -> Result[Adapter]
  if params.size != 2 * adapter.names.size
    return error("the adapter has #{2 * adapter.names.size} tensors, got #{params.size}")
  end

  a: List[Tensor.Tensor] = []
  b: List[Tensor.Tensor] = []
  for i in 0...adapter.names.size
    a = a.push(params[2 * i])
    b = b.push(params[2 * i + 1])
  end

  ok(adapter.with(a: a, b: b))
end

## Every weight by name, the adapted ones as `W + (alpha / r) · B A`. Inside a loss, where the adapter is traced, the
## weights of the model are constants of the trace. A name `weights` lacks or a shape that does not fit is an error.
def applied(weights: Map[String, Tensor.Tensor], adapter: Adapter) -> Result[Map[String, Tensor.Tensor]]
  traced = adapter.a.any? { |t: Tensor.Tensor| t.backend == Tensor.Backend.Trace }
  out: Map[String, Tensor.Tensor] = {}
  for name in weights.keys
    w = weights[name]
    if traced
      w = Trace.constant(w)
    end

    out = out.put(name, w)
  end

  for i in 0...adapter.names.size
    name = adapter.names[i]
    w = try out.get(name).to_result("the model has no weight #{name}")
    try float32(name, w)
    a = adapter.a[i]
    b = adapter.b[i]
    scale = adapter.config.alpha / Tensor.shape(a)[0].to_f
    want = [Tensor.shape(b)[0], Tensor.shape(a)[1]]
    return error("#{name}: the model holds #{Tensor.shape(w)}, the adapter is for #{want}") if Tensor.shape(w) != want
    scaled = try Tensor.mul(
      b,
      try Tensor.full(Tensor.shape(b), Tensor.Values.Float(scale), Tensor.dtype(b), b.backend),
    )

    out = out.put(name, try Tensor.add(w, try Tensor.matmul(scaled, a)))
  end

  ok(out)
end

private

## The weights of `names` that `targets` take, sorted. Each target must take one.
def adapted(names: List[String], targets: List[String]) -> Result[List[String]]
  return error("a LoRA config names no target") if targets.empty?
  out: List[String] = []
  for target in targets
    taken = names.select { |name: String| takes?(target, name) }
    return error("the target #{target} names no layer of the model") if taken.empty?
    out = out + taken
  end

  for name in out
    if name.include?("embed")
      return error("#{name} is an embedding: PEFT adapts it with lora_embedding_A and B, which LoRA here does not")
    end
  end

  ok(out.uniq.sort)
end

## The adapter is `F32`: a weight of another dtype (`weights: F16` of a model's `open`) is refused, not cast.
def float32(name: String, w: Tensor.Tensor) -> Result[Void]
  if Tensor.dtype(w) != Tensor.Dtype.F32
    return error("#{name} is #{Tensor.dtype(w)}: LoRA adapts F32 weights, so open the model with weights F32")
  end

  ok()
end

def takes?(target: String, name: String) -> Bool = name == "#{target}.weight" or name.end_with?(".#{target}.weight")
