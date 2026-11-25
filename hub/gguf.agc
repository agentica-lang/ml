## One GGUF file (`hub/DESIGN.md`): `GGUF`, a version, the number of tensors and of key-values, the key-values, the
## tensor infos, padding to the alignment, then one byte buffer that the infos cut into tensors.
##
## `open` reads the header and nothing else, and checks it whole: every length against the file, every tensor inside
## the data and clear of the others. A float tensor (F32, F16, BF16) is read from its byte range by the backend, as
## for safetensors (`shard.agc`). Only the version 3 of the format and the `qwen3` architecture are read.

import Binary from "std/binary"
import Shards from "shard"

## The bytes read first. A header that is longer is read again from the start with twice as many, up to `MAX_HEADER`.
FIRST_READ = 8388608
MAX_HEADER = 268435456
## What `parse` says when the bytes it was given end inside the header, and the file goes on.
TRUNCATED = "the header continues past the bytes read"
## The key-values, the tensors and the items of an array read by `strings` or `ints` that a header may count.
MAX_COUNT = 1000000
## ggml's `GGML_MAX_DIMS`.
MAX_DIMS = 4

## The name of each type of ggml that this reader sizes, and its values and bytes per block.
KIND_NAMES = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
  12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 30: "BF16",
}
KIND_BLOCKS = {
  0: [1, 4],
  1: [1, 2],
  2: [32, 18],
  3: [32, 20],
  6: [32, 22],
  7: [32, 24],
  8: [32, 34],
  9: [32, 36],
  10: [256, 84],
  11: [256, 110],
  12: [256, 144],
  13: [256, 176],
  14: [256, 210],
  15: [256, 292],
  24: [1, 1],
  25: [1, 2],
  26: [1, 4],
  27: [1, 8],
  28: [1, 8],
  30: [1, 2],
}

## The bytes of one scalar of each value type that has a fixed width.
WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
## The item types that `ints` reads.
INT_KINDS = [0, 1, 2, 3, 4, 5, 10, 11]

## The GGUF names of one `qwen3` block and of the rest, to the names of Hugging Face's checkpoint.
BLOCK_NAMES = {
  "attn_norm.weight": "input_layernorm.weight",
  "attn_q.weight": "self_attn.q_proj.weight",
  "attn_k.weight": "self_attn.k_proj.weight",
  "attn_v.weight": "self_attn.v_proj.weight",
  "attn_output.weight": "self_attn.o_proj.weight",
  "attn_q_norm.weight": "self_attn.q_norm.weight",
  "attn_k_norm.weight": "self_attn.k_norm.weight",
  "ffn_norm.weight": "post_attention_layernorm.weight",
  "ffn_gate.weight": "mlp.gate_proj.weight",
  "ffn_up.weight": "mlp.up_proj.weight",
  "ffn_down.weight": "mlp.down_proj.weight",
}
TOP_NAMES = {
  "token_embd.weight": "model.embed_tokens.weight",
  "output_norm.weight": "model.norm.weight",
  "output.weight": "lm_head.weight",
}

## The value types of the format that this reader holds.
enum Value
  Int(Int)
  Float(Float)
  Bool(Bool)
  Text(String)
  Array(Items)
end

## An array of the header is not read when the file opens: its item type, how many items there are, and the bytes
## they take (from `at` to `stop`).
struct Items
  kind: Int
  count: Int
  at: Int
  stop: Int
end

## A parsed file: `base` is the byte where the data buffer starts, `tensors` hold the names of the file and the shapes
## the other way round (the first dimension of GGUF is the last of a row-major shape).
struct Header
  path: String
  base: Int
  values: Map[String, Value]
  tensors: List[Shards.Entry]
end

## The file at `path`: its header parsed and checked, no tensor read.
def open(path: String) -> Result[Header]
  size = (try File.stat(path)).size_bytes
  want = [size, FIRST_READ].min.unwrap!
  while true
    bytes = try Shards.within(path, Shards.read_range(path, 0, want))
    case parse(bytes, size)
      ok(h) -> return ok(h.with(path: path))
      error(e) ->
        if e.message != TRUNCATED or want >= size or want >= MAX_HEADER
          return error("#{path}: #{e.message}") if e.message != TRUNCATED
          return error("#{path}: the header is longer than #{want} bytes, past the limit or the end of the file")
        end
    end

    want = [size, want * 2].min.unwrap!
  end

  error("#{path}: unreachable")
end

## The file as a `Shard` that `Hub` reads tensors from, the names those of Hugging Face's checkpoint of the same
## architecture. An architecture other than `qwen3` is an error that names it.
def shard(h: Header) -> Result[Shards.Shard]
  arch = try string(h, "general.architecture")
  return error("#{h.path}: the architecture #{arch} has no tensor names here (only qwen3)") if arch != "qwen3"
  entries: Map[String, Shards.Entry] = {}
  origin: Map[String, String] = {}
  for e in h.tensors
    name = hf_name(e.name)
    return error("#{h.path}: the tensors #{origin[name]} and #{e.name} both become #{name}") if entries.key?(name)
    entries = entries.put(name, e.with(name: name))
    origin = origin.put(name, e.name)
  end

  ok(Shards.Shard(path: h.path, base: h.base, entries: entries, metadata: {}))
end

## The GGUF name of the Hugging Face tensor `name` of a `qwen3` model (`model.layers.0.self_attn.q_proj.weight` is
## `blk.0.attn_q.weight`): the other way of `shard`. A name that has no GGUF one is an error.
def gguf_name(name: String) -> Result[String]
  for top in TOP_NAMES.keys
    return ok(top) if TOP_NAMES[top] == name
  end

  parts = name.split(".")
  if parts.size > 3 and parts[0] == "model" and parts[1] == "layers"
    case parts[2].to_i
      ok(n) ->
        if n < 0 or "#{n}" != parts[2]
          return error("the layer #{parts[2]} of #{name} is not a number without a sign or leading zeros")
        end

      error(_) -> return error("the layer #{parts[2]} of #{name} is not a number")
    end

    rest = parts.drop(3).join(".")
    for block in BLOCK_NAMES.keys
      return ok("blk.#{parts[2]}.#{block}") if BLOCK_NAMES[block] == rest
    end
  end

  error("the tensor #{name} has no GGUF name here (only qwen3)")
end

## The string, boolean, integer or float under `key`. A key that is missing, or of another type, is an error naming it.
def string(h: Header, key: String) -> Result[String]
  case try value(h, key)
    Value.Text(s) -> ok(s)
    _ -> error("#{h.path}: #{key} is not a string")
  end
end

def bool(h: Header, key: String) -> Result[Bool]
  case try value(h, key)
    Value.Bool(b) -> ok(b)
    _ -> error("#{h.path}: #{key} is not a boolean")
  end
end

def int(h: Header, key: String) -> Result[Int]
  case try value(h, key)
    Value.Int(n) -> ok(n)
    _ -> error("#{h.path}: #{key} is not an integer")
  end
end

def float(h: Header, key: String) -> Result[Float]
  case try value(h, key)
    Value.Float(x) -> ok(x)
    _ -> error("#{h.path}: #{key} is not a float")
  end
end

## The strings of the array under `key`, read from the file now (a vocabulary is megabytes, so `open` skips it).
def strings(h: Header, key: String) -> Result[List[String]]
  items = try array_of(h, key)
  return error("#{h.path}: #{key} is not an array of strings") if items.kind != 8
  b = try items_bytes(h, items)
  where = "#{h.path}: #{key}"
  out: List[String] = []
  at = 0
  for _ in 0...items.count
    t = try Shards.within(where, text(b, at, b.size))
    out = out.push(t.text)
    at = t.at
  end

  ok(out)
end

## The integers of the array under `key`, read from the file now.
def ints(h: Header, key: String) -> Result[List[Int]]
  items = try array_of(h, key)
  return error("#{h.path}: #{key} is not an array of integers") if not INT_KINDS.include?(items.kind)
  b = try items_bytes(h, items)
  out: List[Int] = []
  at = 0
  for _ in 0...items.count
    r = try item(b, at, items.kind, b.size)
    case r.value
      Value.Int(n) -> out = out.push(n)
      _ -> return error("#{h.path}: #{key} has an item that is not an integer")
    end

    at = r.at
  end

  ok(out)
end

## Whether the file has a key-value `key`.
def has?(h: Header, key: String) -> Bool = h.values.key?(key)

private

struct Text
  text: String
  at: Int
end

struct Read
  value: Value
  at: Int
end

def array_of(h: Header, key: String) -> Result[Items]
  case try value(h, key)
    Value.Array(items) ->
      return error("#{h.path}: #{key} has #{items.count} items, more than #{MAX_COUNT}") if items.count > MAX_COUNT
      ok(items)
    _ -> error("#{h.path}: #{key} is not an array")
  end
end

def items_bytes(h: Header, items: Items) -> Result[Bytes]
  Shards.within(h.path, Shards.read_range(h.path, items.at, items.stop - items.at))
end

def value(h: Header, key: String) -> Result[Value] = h.values.get(key).to_result("#{h.path}: no key #{key}")

def hf_name(name: String) -> String
  return TOP_NAMES[name] if TOP_NAMES.key?(name)
  parts = name.split(".")
  if parts.size > 2 and parts[0] == "blk"
    rest = parts.drop(2).join(".")
    return "model.layers.#{parts[1]}.#{BLOCK_NAMES[rest]}" if BLOCK_NAMES.key?(rest)
  end

  name
end

## The header of the first bytes of a file of `size` bytes.
def parse(b: Bytes, size: Int) -> Result[Header]
  return error("the file ends inside the header") if size < 24
  try need(b, 0, 24, size)
  return error("not a GGUF file (it does not start with GGUF)") if b[0] != 71 or b[1] != 71 or b[2] != 85 or b[3] != 70

  version = try Binary.read_u32_le(b, 4)
  return error("GGUF version #{version} is not supported (only 3)") if version != 3
  tensor_count = try Binary.read_u64_le(b, 8)
  value_count = try Binary.read_u64_le(b, 16)
  if count_bad?(tensor_count, size) or count_bad?(value_count, size)
    return error("#{tensor_count} tensors and #{value_count} key-values do not fit in the header")
  end

  at = 24
  values: Map[String, Value] = {}
  for _ in 0...value_count
    key = try text(b, at, size)
    kind = try u32(b, key.at, size)
    read = try item(b, key.at + 4, kind, size)
    return error("the key #{key.text} appears twice") if values.key?(key.text)
    values = values.put(key.text, read.value)
    at = read.at
  end

  alignment = 32
  if values.key?("general.alignment")
    case values["general.alignment"]
      Value.Int(n) -> alignment = n
      _ -> return error("general.alignment is not an integer")
    end
  end

  if alignment < 1 or alignment.bit_and(alignment - 1) != 0
    return error("the alignment #{alignment} is not a positive power of two")
  end

  infos: List[Shards.Entry] = []
  types: List[Int] = []
  for _ in 0...tensor_count
    name = try text(b, at, size)
    dims = try u32(b, name.at, size)
    return error("tensor #{name.text} has #{dims} dimensions (1 to #{MAX_DIMS})") if dims < 1 or dims > MAX_DIMS
    shape: List[Int] = []
    pos = name.at + 4
    for _ in 0...dims
      d = try u64(b, pos, size)
      return error("tensor #{name.text} has a dimension of #{d}") if d < 1
      shape = [d].concat(shape)
      pos = pos + 8
    end

    ggml = try u32(b, pos, size)
    return error("tensor #{name.text} has the ggml type #{ggml}, which is not supported") if not KIND_BLOCKS.key?(ggml)

    offset = try u64(b, pos + 4, size)
    at = pos + 12
    infos = infos.push(
      Shards.Entry(name: name.text, dtype: KIND_NAMES[ggml], shape: shape, start: offset, stop: offset),
    )

    types = types.push(ggml)
  end

  base = ((at + alignment - 1) / alignment) * alignment
  return error("the data starts at #{base}, past the end of the file") if base > size
  checked = try check_tensors(infos, types, alignment, size - base)
  ok(Header(path: "", base: base, values: values, tensors: checked))
end

def count_bad?(n: Int, size: Int) -> Bool = n > MAX_COUNT or n > size / 12

## Each tensor's size from its shape and type, inside the `room` bytes of data, aligned, and clear of the others.
def check_tensors(infos: List[Shards.Entry], kinds: List[Int], alignment: Int, room: Int) -> Result[List[Shards.Entry]]
  out: List[Shards.Entry] = []
  names: Set[String] = Set.new([])
  for i in 0...infos.size
    e = infos[i]
    return error("tensor #{e.name} appears twice") if names.include?(e.name)
    names = names.add(e.name)
    block = KIND_BLOCKS[kinds[i]]
    last = e.shape[e.shape.size - 1]
    return error("tensor #{e.name}: #{last} values per row is not a multiple of #{block[0]}") if last % block[0] != 0
    # counted in blocks, so that a quantized tensor that fills the data is not refused
    blocks = last / block[0]
    fits = room / block[1]
    for d in e.shape.take(e.shape.size - 1)
      return error("tensor #{e.name}: #{e.shape} is past the #{room} bytes of data") if blocks > fits / d
      blocks = blocks * d
    end

    return error("tensor #{e.name}: #{e.shape} is past the #{room} bytes of data") if blocks > fits
    need = blocks * block[1]
    start = e.start
    return error("tensor #{e.name}: offset #{start} is not a multiple of #{alignment}") if start % alignment != 0
    if need > room - start
      return error("tensor #{e.name}: #{need} bytes at #{start} is past the #{room} bytes of data")
    end

    out = out.push(e.with(stop: start + need))
  end

  at = 0
  for e in out.sort_by(&order)
    return error("tensor #{e.name} overlaps the tensor before it") if e.start < at
    at = e.stop
  end

  ok(out)
end

def order(e: Shards.Entry) -> Int = e.start

## A read of `n` bytes at `at` is inside the bytes, which end before the file does only when `TRUNCATED`.
def need(b: Bytes, at: Int, n: Int, size: Int) -> Result[Void]
  return error("a length of #{n} is past the end of the file") if n < 0 or n > size
  return error("the file ends inside the header") if at + n > size
  return error(TRUNCATED) if at + n > b.size
  ok()
end

def u32(b: Bytes, at: Int, size: Int) -> Result[Int]
  try need(b, at, 4, size)
  Binary.read_u32_le(b, at)
end

def u64(b: Bytes, at: Int, size: Int) -> Result[Int]
  try need(b, at, 8, size)
  Binary.read_u64_le(b, at)
end

## The string (a length of 8 bytes, then that many bytes of UTF-8) at `at`.
def text(b: Bytes, at: Int, size: Int) -> Result[Text]
  n = try u64(b, at, size)
  try need(b, at + 8, n, size)
  s = try String.from_bytes(b.slice(at + 8, n))
  ok(Text(text: s, at: at + 8 + n))
end

## The value of `kind` at `at`, and where it ends.
def item(b: Bytes, at: Int, kind: Int, size: Int) -> Result[Read]
  case kind
    0 -> scalar(b, at, 1, size)
    1 -> signed(b, at, 1, size)
    2 -> scalar(b, at, 2, size)
    3 -> signed(b, at, 2, size)
    4 -> scalar(b, at, 4, size)
    5 -> signed(b, at, 4, size)
    6 -> real(b, at, 4, size)
    7 -> flag(b, at, size)
    8 -> string_item(b, at, size)
    9 -> array(b, at, size)
    10 -> scalar(b, at, 8, size)
    11 -> signed(b, at, 8, size)
    12 -> real(b, at, 8, size)
    _ -> error("the value type #{kind} is not known")
  end
end

def scalar(b: Bytes, at: Int, width: Int, size: Int) -> Result[Read]
  try need(b, at, width, size)
  n = 0
  case width
    1 -> n = b[at]
    2 -> n = try Binary.read_u16_le(b, at)
    4 -> n = try Binary.read_u32_le(b, at)
    _ -> n = try Binary.read_u64_le(b, at)
  end

  ok(Read(value: Value.Int(n), at: at + width))
end

def signed(b: Bytes, at: Int, width: Int, size: Int) -> Result[Read]
  try need(b, at, width, size)
  n = 0
  case width
    1 -> n = try Binary.read_i8(b, at)
    2 -> n = try Binary.read_i16_le(b, at)
    4 -> n = try Binary.read_i32_le(b, at)
    _ -> n = try Binary.read_i64_le(b, at)
  end

  ok(Read(value: Value.Int(n), at: at + width))
end

def real(b: Bytes, at: Int, width: Int, size: Int) -> Result[Read]
  try need(b, at, width, size)
  x = 0.0
  if width == 4
    x = try Binary.read_f32_le(b, at)
  else
    x = try Binary.read_f64_le(b, at)
  end

  ok(Read(value: Value.Float(x), at: at + width))
end

def flag(b: Bytes, at: Int, size: Int) -> Result[Read]
  try need(b, at, 1, size)
  return error("a boolean is #{b[at]}, not 0 or 1") if b[at] > 1
  ok(Read(value: Value.Bool(b[at] == 1), at: at + 1))
end

def string_item(b: Bytes, at: Int, size: Int) -> Result[Read]
  t = try text(b, at, size)
  ok(Read(value: Value.Text(t.text), at: t.at))
end

## An array: its item type, its count, then the items, which are skipped here (a vocabulary is read when it is
## asked for). A string item takes a length each, any other has a fixed width.
def array(b: Bytes, at: Int, size: Int) -> Result[Read]
  kind = try u32(b, at, size)
  count = try u64(b, at + 4, size)
  start = at + 12
  return error("an array of arrays is not supported") if kind == 9
  return error("an array of #{count} items does not fit in the file") if count > size
  if kind == 8
    pos = start
    for _ in 0...count
      n = try u64(b, pos, size)
      try need(b, pos + 8, n, size)
      pos = pos + 8 + n
    end

    return ok(Read(value: Value.Array(Items(kind: kind, count: count, at: start, stop: pos)), at: pos))
  end

  return error("the array item type #{kind} is not known") if not WIDTHS.key?(kind)
  width = WIDTHS[kind]
  try need(b, start, count * width, size)
  ok(
    Read(
      value: Value.Array(Items(kind: kind, count: count, at: start, stop: start + count * width)),
      at: start + count * width,
    ),
  )
end
