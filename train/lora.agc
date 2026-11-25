## LoRA (`train/DESIGN.md`): a weight `W` of a linear layer, `[out, in]`, stays frozen and two small matrices are
## trained beside it, `A` of `[r, in]` and `B` of `[out, r]`. The layer then computes with `W + (alpha / r) · B A`.
## `applied` builds that sum, so the model's own forward runs on it: inside a loss it is the training forward, and
## outside it is the merge for inference. `B` starts at zero, so a new adapter changes nothing. `save` and `open`
## keep an adapter in PEFT's files, so an adapter trained with PEFT runs here and one trained here loads in PEFT.

import Hub from "pkg/hub/hub"
import Init from "pkg/nn/init"
import SafetensorsWrite from "pkg/hub/safetensors_write"
import Shards from "pkg/hub/shard"
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

## Writes `adapter` into the folder `dir` (made when it is not there) as PEFT does: `adapter_config.json` and
## `adapter_model.safetensors`, where the weight `x.weight` has `base_model.model.x.lora_A.weight` and
## `...lora_B.weight`. Every adapted weight is written as a linear layer's: PEFT keeps the adapter of an `Embedding`
## under other names and another layout, which an adapter here does not know, so one that adapts an embedding table
## is not for PEFT. An `A` whose rank is not `config.r` is an error (PEFT's `rank_pattern` is not written).
def save(dir: String, adapter: Adapter) -> Result[Void]
  tensors: Map[String, Tensor.Tensor] = {}
  for i in 0...adapter.names.size
    name = adapter.names[i]
    return error("#{name}: an adapted weight is named x.weight") if not name.end_with?(".weight")
    rank = Tensor.shape(adapter.a[i])[0]
    if rank != adapter.config.r
      return error("#{name}: the adapter has rank #{rank} and its config #{adapter.config.r}")
    end

    layer = "#{PREFIX}#{name.slice(0, name.length - ".weight".length)}"
    tensors = tensors.put("#{layer}.lora_A.weight", adapter.a[i]).put("#{layer}.lora_B.weight", adapter.b[i])
  end

  file = Peft(
    r: adapter.config.r,
    lora_alpha: adapter.config.alpha,
    lora_dropout: adapter.config.dropout,
    target_modules: adapter.config.targets,
  )

  try Dir.mkdir(dir)
  try SafetensorsWrite.write("#{dir}/#{WEIGHTS}", tensors, {"format": "pt"})
  File.write_text("#{dir}/#{CONFIG}", JSON.encode(file))
end

## The adapter of the folder `dir`, written by PEFT or by `save`, on `backend` (as `F32`). Its `names` are the
## weights it adapts, from the names of its tensors. It reads a LoRA adapter of linear layers alone; an error names
## what it does not compute: another `peft_type`, DoRA, rsLoRA, `fan_in_fan_out`, a trained bias, `modules_to_save`,
## a rank or alpha pattern, `target_modules` as a regular expression, and the adapter of an `Embedding`.
def open(dir: String, backend: Tensor.Backend) -> Result[Adapter]
  file = try Shards.within("#{dir}/#{CONFIG}", config("#{dir}/#{CONFIG}"))
  c = try Hub.open("#{dir}/#{WEIGHTS}")
  names: List[String] = []
  for key in Hub.names(c)
    names = names.push(try Shards.within("#{dir}/#{WEIGHTS}", adapted_by(key)))
  end

  names = names.uniq.sort
  a: List[Tensor.Tensor] = []
  b: List[Tensor.Tensor] = []
  for name in names
    layer = "#{PREFIX}#{name.slice(0, name.length - ".weight".length)}"
    one = try Hub.tensor(c, "#{layer}.lora_A.weight", backend)
    two = try Hub.tensor(c, "#{layer}.lora_B.weight", backend)
    shapes = [Tensor.shape(one), Tensor.shape(two)]
    if shapes[0].size != 2 or shapes[1].size != 2 or shapes[0][0] != shapes[1][1] or shapes[0][0] != file.r
      return error("#{dir}/#{WEIGHTS}: #{layer} holds A #{shapes[0]} and B #{shapes[1]}, not a pair of rank #{file.r}")
    end

    a = a.push(one)
    b = b.push(two)
  end

  return error("#{dir}/#{WEIGHTS} holds no adapter") if names.empty?
  made = Config(targets: file.target_modules, r: file.r, alpha: file.lora_alpha, dropout: file.lora_dropout)
  ok(Adapter(config: made, names: names, a: a, b: b))
end

private

CONFIG = "adapter_config.json"
WEIGHTS = "adapter_model.safetensors"

## What PEFT puts before the name of a layer.
PREFIX = "base_model.model."

## `adapter_config.json`: the keys PEFT needs to load an adapter and the ones that change what it computes. PEFT
## fills the rest with its defaults, and `open` ignores the rest.
struct Peft
  r: Int
  lora_alpha: Float
  target_modules: List[String]
  peft_type: String = "LORA"
  lora_dropout: Float = 0.0
  bias: String = "none"
  fan_in_fan_out: Bool = false
  use_dora: Bool = false
  use_rslora: Bool = false
  lora_bias: Bool = false
  modules_to_save: Optional[List[String]] = none
  rank_pattern: Map[String, Int] = {}
  alpha_pattern: Map[String, Float] = {}
  inference_mode: Bool = true
  base_model_name_or_path: Optional[String] = none
  task_type: Optional[String] = none
end

## The config of the file `path`, or what in it LoRA here does not compute.
def config(path: String) -> Result[Peft]
  json = try JSON.parse(try File.read_text(path))
  kind = try JSON.string(try JSON.get(json, "peft_type"))
  return error("the adapter is #{kind}, not LORA") if kind != "LORA"
  if JSON.kind(try JSON.get(json, "target_modules")) == "string"
    return error("target_modules is a regular expression: a list of names is read")
  end

  file = try JSON.decode(json, Peft)
  return error("use_dora: a DoRA adapter is not read") if file.use_dora
  return error("use_rslora: the scale alpha / sqrt(r) is not read") if file.use_rslora
  return error("fan_in_fan_out: an adapter of transposed weights is not read") if file.fan_in_fan_out
  return error("bias is #{file.bias}: an adapter with trained biases is not read") if file.bias != "none"
  return error("lora_bias: an adapter with a bias on B is not read") if file.lora_bias
  return error("modules_to_save: whole trained layers are not read") if not file.modules_to_save.unwrap_or([]).empty?

  if not file.rank_pattern.empty? or not file.alpha_pattern.empty?
    return error("rank_pattern or alpha_pattern: a rank or alpha per layer is not read")
  end

  return error("a LoRA rank is at least 1, got #{file.r}") if file.r < 1
  ok(file)
end

## The weight that the tensor `key` of an adapter file adapts.
def adapted_by(key: String) -> Result[String]
  return error("#{key} is the adapter of an Embedding, which is not read") if key.include?("lora_embedding_")

  for ending in [".lora_A.weight", ".lora_B.weight"]
    if key.start_with?(PREFIX) and key.end_with?(ending)
      return ok("#{key.slice(PREFIX.length, key.length - PREFIX.length - ending.length)}.weight")
    end
  end

  error("#{key} is not the lora_A or lora_B of a layer")
end

## The weights of `names` that `targets` take, sorted. Each target must take one.
def adapted(names: List[String], targets: List[String]) -> Result[List[String]]
  return error("a LoRA config names no target") if targets.empty?
  out: List[String] = []
  for target in targets
    taken = names.select { |name: String| takes?(target, name) }
    return error("the target #{target} names no layer of the model") if taken.empty?
    out = out + taken
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
