## `Lora.open` and `Lora.save` against PEFT's own files: `fixtures/lora_peft/` is the adapter of `lora_golden.py`
## as `PeftModel.save_pretrained` wrote it (peft 0.21.2), with a dropout in its config. The adapter read from it gives
## PEFT's output, `save` writes the same tensors under the same names and the same config values, what `save` wrote
## reads back as it was, and what LoRA here does not compute is refused (`lora_files.py` makes the files PEFT would
## not write, with the standard library). The other way, PEFT loading what `save`
## wrote, is `lora_golden.py check` (manual: it needs torch and peft). Run from `train/`.

import Hub from "pkg/hub/hub"
import Lora from "../lora"
import Nn from "pkg/nn/nn"
import S from "support"
import Tensor from "pkg/tensor/tensor"
import Write from "pkg/hub/safetensors_write"

PEFT = "test/fixtures/lora_peft"
WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"

## The reference computes in doubles and the peer in `F32`.
ABS = 0.00001
REL = 0.0001

## The part of `fixtures/lora_golden.json` this file reads.
struct Golden
  shapes: Map[String, List[Int]]
  weights: Map[String, List[Float]]
  inputs: List[Float]
  output: List[Float]
end

## The keys of `adapter_config.json` that `save` writes and PEFT's file must agree on.
struct Written
  peft_type: String
  r: Int
  lora_alpha: Float
  lora_dropout: Float
  target_modules: List[String]
  bias: String
  fan_in_fan_out: Bool
  use_dora: Bool
  use_rslora: Bool
  lora_bias: Bool
  modules_to_save: Optional[List[String]]
  rank_pattern: Map[String, Int]
  alpha_pattern: Map[String, Float]
  inference_mode: Bool
  base_model_name_or_path: Optional[String]
  task_type: Optional[String]
end

private

def with_folder(body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-train-lora-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = body.call(dir)
  _ = Path.remove(dir, 5000)
  result
end

def bits(t: Tensor.Tensor) -> Result[List[String]] = ok((try S.floats_of(t)).map { |x: Float| "#{Float.to_bits(x)}" })

def written(dir: String) -> Result[Written]
  w = try JSON.decode(try File.read_text("#{dir}/#{CONFIG}"), Written)
  ok(w.with(target_modules: w.target_modules.sort))
end

## An adapter folder with `tensors` and `config` as its config.
def folder(dir: String, name: String, tensors: Map[String, Tensor.Tensor], config: String) -> Result[String]
  made = "#{dir}/#{name}"
  try Dir.mkdir(made)
  try Write.write("#{made}/#{WEIGHTS}", tensors)
  try File.write_text("#{made}/#{CONFIG}", config)
  ok(made)
end

def test_an_adapter_of_peft_gives_its_output() -> Result[Void]
  g = try JSON.decode(try File.read_text("test/fixtures/lora_golden.json"), Golden)
  weights: Map[String, Tensor.Tensor] = {}
  for name in g.weights.keys
    weights = weights.put(name, try S.tensor(g.weights[name], g.shapes[name], Tensor.Dtype.F32))
  end

  adapter = try Lora.open(PEFT, S.REF)
  try Test.eq(adapter.names, ["first.weight", "second.weight"])
  try Test.eq(adapter.config.targets.sort, ["first", "second"])
  try Test.eq(adapter.config.r, 2)
  try Test.eq(adapter.config.alpha, 4.0)
  try Test.assert((adapter.config.dropout - 0.1).abs < 0.000001, "the dropout PEFT trained with")
  try Test.assert(!Lora.params(adapter).ok?, "an adapter with a dropout is not trained")
  w = try Lora.applied(weights, adapter)
  first = Nn.Dense(w: w["first.weight"], b: some(w["first.bias"]), out_in: true)
  second = Nn.Dense(w: w["second.weight"], b: some(w["second.bias"]), out_in: true)
  x = try S.tensor(g.inputs, [6, 3], Tensor.Dtype.F32)
  out = try Nn.dense(second, try Tensor.relu(try Nn.dense(first, x)))
  S.expect_close(try S.floats_of(out), g.output, ABS, REL, "the output")
end

def test_save_writes_the_files_of_peft_and_reads_them_back() -> Result[Void]
  with_folder do |dir|
    adapter = try Lora.open(PEFT, S.REF)
    try Lora.save("#{dir}/saved", adapter)
    ours = try Hub.open("#{dir}/saved/#{WEIGHTS}")
    theirs = try Hub.open("#{PEFT}/#{WEIGHTS}")
    try Test.eq(Hub.names(ours), Hub.names(theirs))
    try Test.eq(Hub.metadata(ours)[WEIGHTS], Hub.metadata(theirs)[WEIGHTS])
    for name in Hub.names(theirs)
      try Test.eq(try Hub.shape(ours, name), try Hub.shape(theirs, name))
      try Test.eq(try Hub.dtype(ours, name), try Hub.dtype(theirs, name))
      try Test.eq(try bits(try Hub.tensor(ours, name, S.REF)), try bits(try Hub.tensor(theirs, name, S.REF)))
    end

    try Test.eq(try written("#{dir}/saved"), try written(PEFT))
    back = try Lora.open("#{dir}/saved", S.REF)
    try Test.eq(back.config, adapter.config.with(targets: back.config.targets))
    try Test.eq(back.config.targets.sort, adapter.config.targets.sort)
    try Test.eq(back.names, adapter.names)
    for i in 0...adapter.names.size
      try Test.eq(try bits(back.a[i]), try bits(adapter.a[i]))
      try Test.eq(try bits(back.b[i]), try bits(adapter.b[i]))
    end

    try Lora.save("#{dir}/again", back)
    try Test.eq(try File.sha256("#{dir}/again/#{WEIGHTS}"), try File.sha256("#{dir}/saved/#{WEIGHTS}"))
    Test.eq(try File.read_text("#{dir}/again/#{CONFIG}"), try File.read_text("#{dir}/saved/#{CONFIG}"))
  end
end

def test_a_config_that_changes_the_math_is_refused() -> Result[Void]
  with_folder do |dir|
    config = try File.read_text("#{PEFT}/#{CONFIG}")
    tensors = try Hub.tensors(try Hub.open("#{PEFT}/#{WEIGHTS}"), S.REF)
    try Test.assert(Lora.open(try folder(dir, "same", tensors, config), S.REF).ok?, "PEFT's own config")
    changes = [
      ["\"peft_type\": \"LORA\"", "\"peft_type\": \"IA3\"", "LORA"],
      ["\"use_dora\": false", "\"use_dora\": true", "use_dora"],
      ["\"use_rslora\": false", "\"use_rslora\": true", "use_rslora"],
      ["\"fan_in_fan_out\": false", "\"fan_in_fan_out\": true", "fan_in_fan_out"],
      ["\"bias\": \"none\"", "\"bias\": \"all\"", "bias"],
      ["\"lora_bias\": false", "\"lora_bias\": true", "lora_bias"],
      ["\"modules_to_save\": null", "\"modules_to_save\": [\"head\"]", "modules_to_save"],
      ["\"rank_pattern\": {}", "\"rank_pattern\": {\"first\": 4}", "rank_pattern"],
      ["\"alpha_pattern\": {}", "\"alpha_pattern\": {\"first\": 8}", "alpha_pattern"],
      ["\"r\": 2", "\"r\": 3", "rank 3"],
      ["\"r\": 2", "\"r\": 0", "at least 1"],
      ["\"lora_alpha\": 4", "\"lora_alpha\": 0", "above 0"],
      ["\"lora_alpha\": 4", "\"lora_alpha\": null", "lora_alpha is missing"],
      ["\"lora_alpha\": 4", "\"alpha\": 4", "lora_alpha is missing"],
      ["\"use_qalora\": false", "\"use_qalora\": true", "use_qalora"],
      ["\"use_bdlora\": null", "\"use_bdlora\": {\"target_modules_bd_a\": [\"first\"]}", "use_bdlora"],
      ["\"alora_invocation_tokens\": null", "\"alora_invocation_tokens\": [1, 2]", "alora_invocation_tokens"],
      ["\"layer_replication\": null", "\"layer_replication\": [[0, 1], [0, 1]]", "layer_replication"],
      ["\"target_parameters\": null", "\"target_parameters\": [\"first.weight\"]", "target_parameters"],
      ["\"trainable_token_indices\": null", "\"trainable_token_indices\": [0, 1]", "trainable_token_indices"],
    ]

    for i in 0...changes.size
      changed = config.replace(changes[i][0], changes[i][1])
      try Test.assert(changed != config, "PEFT's config has #{changes[i][0]}")
      try S.expect_error(Lora.open(try folder(dir, "config-#{i}", tensors, changed), S.REF), changes[i][2])
    end

    start = config.index_of("\"target_modules\": [").unwrap!
    stop = start + config.slice(start, config.length - start).index_of("]").unwrap! + 1
    rest = config.slice(stop, config.length - stop)
    pattern = "#{config.slice(0, start)}\"target_modules\": \"first|second\"#{rest}"
    try S.expect_error(Lora.open(try folder(dir, "pattern", tensors, pattern), S.REF), "regular expression")
    absent = "#{config.slice(0, start)}\"target_modules\": null#{rest}"
    try S.expect_error(Lora.open(try folder(dir, "absent", tensors, absent), S.REF), "target_modules is missing")
    empty = config.replace("\"rank_pattern\": {}", "\"rank_pattern\": null")
    try Test.assert(empty != config, "PEFT's config has a rank_pattern")
    opened = Lora.open(try folder(dir, "empty", tensors, empty), S.REF)
    Test.assert(opened.ok?, "a null rank_pattern is no pattern")
  end
end

def test_tensors_that_are_not_a_linear_adapter_are_refused() -> Result[Void]
  with_folder do |dir|
    config = try File.read_text("#{PEFT}/#{CONFIG}")
    a = try Tensor.zeros([2, 3], Tensor.Dtype.F32, S.REF)
    b = try Tensor.zeros([4, 2], Tensor.Dtype.F32, S.REF)
    p = "base_model.model."
    pair = {"#{p}first.lora_A.weight": a, "#{p}first.lora_B.weight": b}
    try Test.assert(Lora.open(try folder(dir, "pair", pair, config), S.REF).ok?, "one pair")
    embedding = pair.put("#{p}emb.lora_embedding_A", a).put("#{p}emb.lora_embedding_B", b)
    try S.expect_error(Lora.open(try folder(dir, "embedding", embedding, config), S.REF), "Embedding")
    try S.expect_error(
      Lora.open(try folder(dir, "other", pair.put("#{p}head.weight", a), config), S.REF),
      "head.weight",
    )

    try S.expect_error(
      Lora.open(try folder(dir, "half", pair.delete("#{p}first.lora_B.weight"), config), S.REF),
      "lora_B",
    )

    wide = pair.put("#{p}first.lora_B.weight", try Tensor.zeros([4, 3], Tensor.Dtype.F32, S.REF))
    try S.expect_error(Lora.open(try folder(dir, "wide", wide, config), S.REF), "rank")
    for key in ["#{p}lora_A.weight", "#{p}.lora_A.weight", "#{p}a..b.lora_A.weight", "first.lora_A.weight"]
      try S.expect_error(Lora.open(try folder(dir, "key-#{key.length}", pair.put(key, a), config), S.REF), key)
    end

    S.expect_error(Lora.open("#{dir}/missing", S.REF), "adapter_config.json")
  end
end

def test_save_refuses_what_peft_would_not_load() -> Result[Void]
  with_folder do |dir|
    made = try Lora.open(PEFT, S.REF)
    try S.expect_error(Lora.save("#{dir}/rank", made.with(config: made.config.with(r: 4))), "rank 4")
    narrow = made.with(b: made.b.set(0, try Tensor.zeros([4, 3], Tensor.Dtype.F32, S.REF)))
    try S.expect_error(Lora.save("#{dir}/narrow", narrow), "first.weight")
    try S.expect_error(Lora.save("#{dir}/table", made, ["second.weight"]), "embedding table")
    try Test.assert(Lora.save("#{dir}/other", made, ["tokens.weight"]).ok?, "an embedding it does not adapt")
    untaken = made.with(config: made.config.with(targets: ["first"]))
    try S.expect_error(Lora.save("#{dir}/untaken", untaken), "second.weight")
    for name in ["rank", "narrow", "table", "untaken"]
      try Test.assert(!File.stat("#{dir}/#{name}/#{CONFIG}").ok?, "#{name} leaves no config")
    end

    Test.assert(!File.stat("#{dir}/other/#{CONFIG}.part").ok?, "the config has its name")
  end
end

def test_a_half_float_adapter_is_widened_and_an_integer_one_refused() -> Result[Void]
  p = try Process.spawn_captured("python3", ["-I", "test/lora_files.py"])
  out = try Process.output(p, 30000)
  code = try Process.wait(p, 30000)
  return error("lora_files.py exited with #{code}: #{out.stderr}") if code != 0
  dir = out.stdout.strip.split("\n").last
  return error("lora_files.py printed no folder: #{dir}") if not dir.start_with?("/")
  result = odd_files(dir)
  _ = Path.remove(dir, 5000)
  result
end

def odd_files(dir: String) -> Result[Void]
  made = try Lora.open(PEFT, S.REF)
  half = try Lora.open("#{dir}/half", S.REF)
  try Test.eq(half.names, made.names)
  for i in 0...made.names.size
    try Test.eq(Tensor.dtype(half.a[i]), Tensor.Dtype.F32)
    try S.expect_close(try S.floats_of(half.a[i]), try S.floats_of(made.a[i]), 0.001, 0.001, "A #{i} from F16")
    try S.expect_close(try S.floats_of(half.b[i]), try S.floats_of(made.b[i]), 0.001, 0.001, "B #{i} from F16")
  end

  S.expect_error(Lora.open("#{dir}/integer", S.REF), "floats")
end
