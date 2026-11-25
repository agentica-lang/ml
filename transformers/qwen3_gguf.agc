## Qwen3 as a GGUF file, from a Hugging Face model folder (`transformers/DESIGN.md`): the config as the key-values of
## llama.cpp's converter (`qwen3.*`), the tokenizer of `tokenizer.json` as `tokenizer.ggml.*` (the tokens in the order
## of their ids, their types and the merges, as `Bpe.open_gguf` reads them), the chat template, and every weight in the
## format asked for. A matrix is written as `F32`, `F16` or `BF16`; a vector (a norm) is always `F32`, as llama.cpp
## does. A tied model writes no `output.weight`: the table is the head, which llama.cpp and `Qwen3.open` read as tied.
##
## The weights are not read into Agentica: `GgufWrite` writes them with a native range conversion (`Tensor.convert_file`).

import Gguf from "pkg/hub/gguf"
import GgufWrite from "pkg/hub/gguf_write"
import Hub from "pkg/hub/hub"
import Tensor from "pkg/tensor/tensor"
import Added from "tokenizer/added"
import Bpe from "tokenizer/bpe"
import Qwen3 from "qwen3"
import Source from "tokenizer/source"

## The type of a token in a GGUF: a normal token, a control one (special, or written `<|...|>`), a user defined one
## and a padding id that no token has.
NORMAL = 1
CONTROL = 3
USER_DEFINED = 4
UNUSED = 5

## The `general.file_type` of llama.cpp for a model of one format.
FILE_TYPES = {"F32": 0, "F16": 1, "BF16": 32}

## The fields of `tokenizer_config.json` that are written.
struct TokenizerConfig
  chat_template: Optional[String]
  pad_token: Optional[String]
end

## The fields of `config.json` that the config of the model does not have.
struct Ids
  bos_token_id: Optional[Int]
  eos_token_id: Optional[Int]
end

## Writes the model of `dir` (`config.json`, `model.safetensors` or its index, `tokenizer.json` and, if there is one,
## `tokenizer_config.json`) to the GGUF file `out`. `format` is that of the matrices.
def convert(dir: String, out: String, format: Tensor.FloatFormat) -> Result[Void]
  config = try Qwen3.with_head_dim(try Qwen3.read_config(dir))
  try Qwen3.check_config(config)
  return error("the model type is #{config.model_type}: only qwen3 is written") if config.model_type != "qwen3"
  c = try Qwen3.checkpoint(dir)
  tok = try Bpe.open("#{dir}/tokenizer.json")
  fields = config_fields(config, format)
  fields = fields.concat(try tokenizer_fields(dir, config, tok))
  GgufWrite.write(out, fields, try tensors(c, config, format))
end

private

def key(name: String, value: GgufWrite.Meta) -> GgufWrite.Field = GgufWrite.Field(key: name, value: value)

def config_fields(config: Qwen3.Config, format: Tensor.FloatFormat) -> List[GgufWrite.Field]
  kind = "F32"
  case format
    Tensor.FloatFormat.F16 -> kind = "F16"
    Tensor.FloatFormat.BF16 -> kind = "BF16"
    _ -> nil
  end

  [
    key("general.architecture", GgufWrite.Meta.Text("qwen3")),
    key("general.type", GgufWrite.Meta.Text("model")),
    key("general.file_type", GgufWrite.Meta.U32(FILE_TYPES[kind])),
    key("qwen3.block_count", GgufWrite.Meta.U32(config.num_hidden_layers)),
    key("qwen3.context_length", GgufWrite.Meta.U32(config.max_position_embeddings)),
    key("qwen3.embedding_length", GgufWrite.Meta.U32(config.hidden_size)),
    key("qwen3.feed_forward_length", GgufWrite.Meta.U32(config.intermediate_size)),
    key("qwen3.attention.head_count", GgufWrite.Meta.U32(config.num_attention_heads)),
    key("qwen3.attention.head_count_kv", GgufWrite.Meta.U32(config.num_key_value_heads)),
    key("qwen3.attention.key_length", GgufWrite.Meta.U32(config.head_dim)),
    key("qwen3.attention.value_length", GgufWrite.Meta.U32(config.head_dim)),
    key("qwen3.attention.layer_norm_rms_epsilon", GgufWrite.Meta.F32(config.rms_norm_eps)),
    key("qwen3.rope.freq_base", GgufWrite.Meta.F32(config.rope_theta)),
  ]
end

## Whether llama.cpp's converter calls an added token control: it is special, or written `<|...|>`.
def control?(a: Added.Added) -> Bool = a.special or (a.content.start_with?("<|") and a.content.end_with?("|>"))

## The tokens of ids 0 to `vocab_size`, their types, and the merges by rank.
def tokenizer_fields(dir: String, config: Qwen3.Config, tok: Bpe.ByteLevelBpe) -> Result[List[GgufWrite.Field]]
  if tok.pieces.size > config.vocab_size
    return error("the tokenizer has ids up to #{tok.pieces.size - 1}, but the vocabulary has #{config.vocab_size}")
  end

  added: Map[Int, Added.Added] = {}
  for a in tok.added
    return error("the added token #{a.content} has the id #{a.id}, beyond the vocabulary") if a.id >= config.vocab_size
    added = added.put(a.id, a)
  end

  tokens: List[String] = []
  types: List[Int] = []
  for id in 0...config.vocab_size
    if added.key?(id)
      tokens = tokens.push(added[id].content)
      if control?(added[id])
        types = types.push(CONTROL)
      else
        types = types.push(USER_DEFINED)
      end
    elsif id < tok.pieces.size and tok.pieces[id] != ""
      tokens = tokens.push(tok.pieces[id])
      types = types.push(NORMAL)
    else
      tokens = tokens.push("[PAD#{id}]")
      types = types.push(UNUSED)
    end
  end

  merges: List[String] = List.repeat("", tok.ranks.size)
  for pair in tok.ranks.keys
    rank = tok.ranks[pair]
    if rank < 0 or rank >= merges.size
      return error("the merge #{pair.inspect} has the rank #{rank}, but the ranks are 0 to #{merges.size - 1}")
    end

    return error("the merge #{pair.inspect} has the rank #{rank} twice") if merges[rank] != ""
    merges = merges.set(rank, pair)
  end

  fields = [
    key("tokenizer.ggml.model", GgufWrite.Meta.Text("gpt2")),
    key("tokenizer.ggml.pre", GgufWrite.Meta.Text("qwen2")),
    key("tokenizer.ggml.tokens", GgufWrite.Meta.Texts(tokens)),
    key("tokenizer.ggml.token_type", GgufWrite.Meta.Ints(types)),
    key("tokenizer.ggml.merges", GgufWrite.Meta.Texts(merges)),
  ]

  ok(fields.concat(try ids_fields(dir, tok)))
end

## The bos, eos and padding ids, and the chat template, of the files that name them.
def ids_fields(dir: String, tok: Bpe.ByteLevelBpe) -> Result[List[GgufWrite.Field]]
  fields: List[GgufWrite.Field] = []
  ids = try Source.within("#{dir}/config.json", JSON.decode(try Source.read_text("#{dir}/config.json"), Ids))
  if ids.eos_token_id.some?
    fields = fields.push(key("tokenizer.ggml.eos_token_id", GgufWrite.Meta.U32(ids.eos_token_id.unwrap!)))
  end

  if ids.bos_token_id.some?
    fields = fields.push(key("tokenizer.ggml.bos_token_id", GgufWrite.Meta.U32(ids.bos_token_id.unwrap!)))
  end

  template: Optional[String] = none
  settings = "#{dir}/tokenizer_config.json"
  if Path.real(settings).ok?
    text = try Source.within(settings, JSON.decode(try Source.read_text(settings), TokenizerConfig))
    template = text.chat_template
    if text.pad_token.some?
      found = tok.added.find { |a| a.content == text.pad_token.unwrap! }
      case found
        ok(a) -> fields = fields.push(key("tokenizer.ggml.padding_token_id", GgufWrite.Meta.U32(a.id)))
        error(_) -> return error("the pad token #{text.pad_token.unwrap!} is not an added token")
      end
    end
  end

  fields = fields.push(key("tokenizer.ggml.add_bos_token", GgufWrite.Meta.Bool(false)))
  fields = fields.push(key("tokenizer.ggml.add_eos_token", GgufWrite.Meta.Bool(false)))
  file = "#{dir}/chat_template.jinja"
  if !template.some? and Path.real(file).ok?
    template = some(try Source.read_text(file))
  end

  if template.some?
    fields = fields.push(key("tokenizer.chat_template", GgufWrite.Meta.Text(template.unwrap!)))
  end

  ok(fields)
end

## The weights of the model in the order of a file of llama.cpp, with their GGUF names.
def tensors(c: Hub.Checkpoint, config: Qwen3.Config, format: Tensor.FloatFormat) -> Result[List[GgufWrite.Out]]
  names = ["model.embed_tokens.weight"]
  for i in 0...config.num_hidden_layers
    for part in [
      "input_layernorm.weight",
      "self_attn.q_proj.weight",
      "self_attn.k_proj.weight",
      "self_attn.v_proj.weight",
      "self_attn.o_proj.weight",
      "self_attn.q_norm.weight",
      "self_attn.k_norm.weight",
      "post_attention_layernorm.weight",
      "mlp.gate_proj.weight",
      "mlp.up_proj.weight",
      "mlp.down_proj.weight",
    ]
      names = names.push("model.layers.#{i}.#{part}")
    end
  end

  names = names.push("model.norm.weight")
  if not config.tie_word_embeddings
    names = names.push("lm_head.weight")
  end

  out: List[GgufWrite.Out] = []
  for name in names
    source = try Hub.source(c, name)
    want = try shape_of(name, config)
    return error("#{name} has the shape #{source.shape}, but the config needs #{want}") if source.shape != want

    kind = format
    if source.shape.size == 1
      kind = Tensor.FloatFormat.F32
    end

    out = out.push(GgufWrite.Out(name: try Gguf.gguf_name(name), source: source, format: kind))
  end

  ok(out)
end

## The shape the config gives the weight `name` of a `qwen3` checkpoint.
def shape_of(name: String, config: Qwen3.Config) -> Result[List[Int]]
  hidden = config.hidden_size
  heads = config.num_attention_heads * config.head_dim
  kv_heads = config.num_key_value_heads * config.head_dim
  parts = name.split(".")
  return ok([config.vocab_size, hidden]) if name == "model.embed_tokens.weight" or name == "lm_head.weight"
  return ok([hidden]) if name == "model.norm.weight"
  return error("the weight #{name} has no shape here") if parts.size != 6 and parts.size != 5
  case parts.drop(3).join(".")
    "input_layernorm.weight" -> ok([hidden])
    "post_attention_layernorm.weight" -> ok([hidden])
    "self_attn.q_proj.weight" -> ok([heads, hidden])
    "self_attn.k_proj.weight" -> ok([kv_heads, hidden])
    "self_attn.v_proj.weight" -> ok([kv_heads, hidden])
    "self_attn.o_proj.weight" -> ok([hidden, heads])
    "self_attn.q_norm.weight" -> ok([config.head_dim])
    "self_attn.k_norm.weight" -> ok([config.head_dim])
    "mlp.gate_proj.weight" -> ok([config.intermediate_size, hidden])
    "mlp.up_proj.weight" -> ok([config.intermediate_size, hidden])
    "mlp.down_proj.weight" -> ok([hidden, config.intermediate_size])
    _ -> error("the weight #{name} has no shape here")
  end
end
