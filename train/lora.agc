## LoRA (`train/DESIGN.md`): a weight `W` of a linear layer, `[out, in]`, stays frozen and two small matrices are
## trained beside it, `A` of `[r, in]` and `B` of `[out, r]`. The layer then computes with `W + (alpha / r) · B A`.
## `applied` builds that sum, so the model's own forward runs on it: inside a loss it is the training forward, and
## `merged` computes it once for inference. `B` starts at zero, so a new adapter changes nothing. `save` and `open`
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
## A target may name an embedding table, which is a matrix too: such an adapter trains and runs here, and `save`
## cannot write it for PEFT (see `save`). `dropout` is what a PEFT adapter was trained with; an adapter with a
## dropout above 0 runs, and is not trained here.
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
## `seed` (PEFT's start) and every `B` zero. A target may be an embedding table (see `Config`). An error when a target names no layer, or a layer whose weight is not a
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

## `weights` with the adapter merged in, for inference: the adapted weights are new tensors on their backend, each
## `W + (alpha / r) · B A`, and the others are the tensors of `weights`. The sums run as one traced graph, so this
## works on ggml too, which `applied` outside a loss does not. A model made from them runs at the cost of the model
## alone.
def merged(weights: Map[String, Tensor.Tensor], adapter: Adapter) -> Result[Map[String, Tensor.Tensor]]
  sums = try Trace.eval_many(
    adapter.a + adapter.b,
    ->(ps: List[Tensor.Tensor]) { sums_of(weights, adapter, ps) },
    false,
  )

  out = weights
  for i in 0...adapter.names.size
    out = out.put(adapter.names[i], sums[i])
  end

  ok(out)
end

## Writes `adapter` into the folder `dir` (made when it is not there) as PEFT does: `adapter_config.json` and
## `adapter_model.safetensors`, where the weight `x.weight` has `base_model.model.x.lora_A.weight` and
## `...lora_B.weight`. Each file gets its name when it is whole. Every adapted weight is written as a linear
## layer's. PEFT keeps the adapter of an `Embedding` under other names and in another layout, and an adapter here
## does not know which of its weights is one: `embeddings` are the names of the model's embedding tables (a model
## says its own), and an adapter of one of them is an error. Also an error: a tensor whose rank is not `config.r`
## (PEFT's `rank_pattern` is not written) and a weight that no target takes. The config does not say which model
## the adapter is for (PEFT's `base_model_name_or_path` is written as null).
def save(dir: String, adapter: Adapter, embeddings: List[String] = []) -> Result[Void]
  tensors: Map[String, Tensor.Tensor] = {}
  for i in 0...adapter.names.size
    name = adapter.names[i]
    taken = adapter.config.targets.any? { |target: String| takes?(target, name) }
    return error("#{name}: no target of the config takes it (an adapted weight is named x.weight)") if not taken

    if embeddings.include?(name)
      return error("#{name} is an embedding table: PEFT keeps its adapter in another layout, which is not written")
    end

    r = adapter.config.r
    if Tensor.shape(adapter.a[i])[0] != r or Tensor.shape(adapter.b[i])[1] != r
      shapes = "A #{Tensor.shape(adapter.a[i])} and B #{Tensor.shape(adapter.b[i])}"
      return error("#{name}: the adapter holds #{shapes}, and its config has rank #{r}")
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
  try File.write_text("#{dir}/#{CONFIG}.part", JSON.encode(file))
  Path.rename("#{dir}/#{CONFIG}.part", "#{dir}/#{CONFIG}")
end

## The adapter of the folder `dir`, written by PEFT or by `save`, on `backend` (as `F32`: a half-float adapter is
## widened). Its `names` are the weights it adapts, from the names of its tensors. It reads a LoRA adapter of linear
## layers alone, and an error names what is not computed here: another `peft_type`; DoRA, rsLoRA, aLoRA, QALoRA or
## BD-LoRA; `fan_in_fan_out`; a trained bias; `modules_to_save`, `target_parameters` or `trainable_token_indices`;
## `layer_replication`; a rank or alpha pattern; `target_modules` as a regular expression; and the adapter of an
## `Embedding`. Which model the adapter is for (`base_model_name_or_path`) is not kept.
def open(dir: String, backend: Tensor.Backend) -> Result[Adapter]
  file = try Shards.within("#{dir}/#{CONFIG}", config("#{dir}/#{CONFIG}"))
  c = try Hub.open("#{dir}/#{WEIGHTS}")
  names: List[String] = []
  for key in Hub.names(c)
    names = names.push(try Shards.within("#{dir}/#{WEIGHTS}", adapted_by(key)))
  end

  names = names.uniq.sort
  return error("#{dir}/#{WEIGHTS} holds no adapter") if names.empty?
  a: List[Tensor.Tensor] = []
  b: List[Tensor.Tensor] = []
  for name in names
    layer = "#{PREFIX}#{name.slice(0, name.length - ".weight".length)}"
    both = try Shards.within("#{dir}/#{WEIGHTS}", pair(c, layer, file.r, backend))
    a = a.push(both[0])
    b = b.push(both[1])
  end

  made = Config(targets: file.target_modules, r: file.r, alpha: file.lora_alpha, dropout: file.lora_dropout)
  ok(Adapter(config: made, names: names, a: a, b: b))
end

private

CONFIG = "adapter_config.json"
WEIGHTS = "adapter_model.safetensors"

## What PEFT puts before the name of a layer.
PREFIX = "base_model.model."

## The keys of PEFT's config that change what an adapter computes, each with what it is: `open` refuses a file
## where one is set (not null, false, `[]` or `{}`).
NOT_READ = [
  ["use_dora", "a DoRA adapter"],
  ["use_rslora", "the scale alpha / sqrt(r)"],
  ["use_qalora", "a QALoRA adapter"],
  ["use_bdlora", "a BD-LoRA adapter"],
  ["alora_invocation_tokens", "an aLoRA adapter"],
  ["fan_in_fan_out", "an adapter of transposed weights"],
  ["lora_bias", "an adapter with a bias on B"],
  ["modules_to_save", "whole trained layers"],
  ["target_parameters", "an adapter of parameters that are not layers"],
  ["trainable_token_indices", "trained token embeddings"],
  ["layer_replication", "a model with repeated layers"],
  ["rank_pattern", "a rank per layer"],
  ["alpha_pattern", "an alpha per layer"],
]

## `adapter_config.json` as `save` writes it: the keys PEFT needs to load an adapter and the ones that change what
## it computes. PEFT fills the rest with its defaults.
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

## What `open` takes from `adapter_config.json`.
struct Read
  r: Int
  lora_alpha: Float
  target_modules: List[String]
  lora_dropout: Float = 0.0
end

## The config of the file `path`, or what in it LoRA here does not compute.
def config(path: String) -> Result[Read]
  json = try JSON.parse(try File.read_text(path))
  for key in ["peft_type", "r", "lora_alpha", "target_modules"]
    return error("#{key} is missing or null") if not set?(json, key)
  end

  kind = try JSON.string(try JSON.get(json, "peft_type"))
  return error("the adapter is #{kind}, not LORA") if kind != "LORA"
  if JSON.kind(try JSON.get(json, "target_modules")) == "string"
    return error("target_modules is a regular expression: a list of names is read")
  end

  for entry in NOT_READ
    return error("#{entry[0]}: #{entry[1]} is not read") if set?(json, entry[0])
  end

  if set?(json, "bias") and JSON.encode(try JSON.get(json, "bias")) != "\"none\""
    return error("bias: an adapter with trained biases is not read")
  end

  file = try JSON.decode(json, Read)
  return error("a LoRA rank is at least 1, got #{file.r}") if file.r < 1
  return error("a LoRA alpha is above 0, got #{file.lora_alpha}") if file.lora_alpha <= 0.0
  ok(file)
end

## Whether the config `json` gives `key` a value that is not null, false or empty.
def set?(json: JSON.Value, key: String) -> Bool
  case JSON.get(json, key)
    ok(value) -> not ["null", "false", "[]", "{}"].include?(JSON.encode(value))
    error(_) -> false
  end
end

## `A` and `B` of `layer` in the adapter file `c`: `F32` matrices of rank `r`.
def pair(c: Hub.Checkpoint, layer: String, r: Int, backend: Tensor.Backend) -> Result[List[Tensor.Tensor]]
  a = try Hub.tensor(c, "#{layer}.lora_A.weight", backend)
  b = try Hub.tensor(c, "#{layer}.lora_B.weight", backend)
  for t in [a, b]
    if Tensor.dtype(t) != Tensor.Dtype.F32
      return error("#{layer} holds #{Tensor.dtype(t)} values, and an adapter holds floats")
    end
  end

  shapes = [Tensor.shape(a), Tensor.shape(b)]
  if shapes[0].size != 2 or shapes[1].size != 2 or shapes[0][0] != r or shapes[1][1] != r
    return error("#{layer} holds A #{shapes[0]} and B #{shapes[1]}, not a pair of rank #{r}")
  end

  ok([a, b])
end

## The weight that the tensor `key` of an adapter file adapts.
def adapted_by(key: String) -> Result[String]
  return error("#{key} is the adapter of an Embedding, which is not read") if key.include?("lora_embedding_")

  for ending in [".lora_A.weight", ".lora_B.weight"]
    if key.start_with?(PREFIX) and key.end_with?(ending) and key.length > PREFIX.length + ending.length
      layer = key.slice(PREFIX.length, key.length - PREFIX.length - ending.length)
      return error("#{key} names no layer") if layer.split(".").include?("")
      return ok("#{layer}.weight")
    end
  end

  error("#{key} is not the lora_A or lora_B of a layer")
end

## The adapted weights of `applied`, in the order of `names`, with `ps` (every `A`, then every `B`) as the adapter.
def sums_of(
  weights: Map[String, Tensor.Tensor],
  adapter: Adapter,
  ps: List[Tensor.Tensor],
) -> Result[List[Tensor.Tensor]]
  count = adapter.names.size
  with = try applied(weights, adapter.with(a: ps.take(count), b: ps.drop(count)))
  ok(adapter.names.map { |name: String| with[name] })
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
