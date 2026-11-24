## Writes a GGUF file (`hub/DESIGN.md`, "GGUF"): the key-values and tensors the caller names, version 3, little-endian,
## the data aligned to 32 as llama.cpp does. The header is built here and written first; each tensor is then converted
## from its range of its source file, by the backend's native call (`Tensor.convert_file`), straight into its place in
## the output, so no value goes through an Agentica list and a model of several GB is never in memory. The file is
## written under `path + ".part"` and renamed when it is whole, so a failure leaves no half file.

import Binary from "std/binary"
import Shards from "shard"
import Tensor from "pkg/tensor/tensor"

## The alignment of the data buffer and of every tensor in it, the default of the format.
ALIGNMENT = 32
## The bytes written at a time (`File.write_bytes` takes 1 MiB at most).
PART = 1048576

## The value of a key.
enum Meta
  U32(Int)
  F32(Float)
  Bool(Bool)
  Text(String)
  ## An array of strings (a vocabulary, the merges).
  Texts(List[String])
  ## An array of 32-bit signed integers (the types of the tokens).
  Ints(List[Int])
end

struct Field
  key: String
  value: Meta
end

## A tensor to write: `name` in the new file, the place of its values, and the format they take there.
struct Out
  name: String
  source: Shards.Source
  format: Tensor.FloatFormat
end

## Writes `fields` and `tensors` to `path`. A tensor has 1 to 4 dimensions, none of them 0, and its name is not used
## twice; the values are converted from `source` to `format` (a value beyond 65504 is refused for `F16`).
def write(path: String, fields: List[Field], tensors: List[Out]) -> Result[Void]
  header = try Shards.within(path, header_of(fields, tensors))
  part = path + ".part"
  result = fill(part, header, tensors)
  case result
    ok(_) -> Path.replace(part, path)
    error(e) ->
      _ = Path.remove(part)
      error("#{path}: #{e.message}")
  end
end

private

## The header with its padding, the byte where the data buffer starts, where each tensor goes in it, and the size of
## the whole file: the data ends padded to the alignment after the last tensor too, as ggml and gguf-py write it and
## ggml reads it.
struct Layout
  header: Bytes
  base: Int
  offsets: List[Int]
  size: Int
end

def fill(part: String, header: Layout, tensors: List[Out]) -> Result[Void]
  try File.open(part, "write") do |f|
    at = 0
    while at < header.header.size
      n = [PART, header.header.size - at].min.unwrap!
      try File.write_bytes(f, header.header.slice(at, n))
      at = at + n
    end

    ok()
  end

  for i in 0...tensors.size
    t = tensors[i]
    src = t.source
    count = volume(src.shape)
    try Shards.within(
      "tensor #{t.name}",
      Tensor.convert_file(src.path, src.offset, src.format, count, part, header.base + header.offsets[i], t.format),
    )
  end

  info = try File.stat(part)
  if info.size_bytes < header.size
    try File.open(part, "append") do |f|
      try File.write_bytes(f, try Bytes.from_list(List.repeat(0, header.size - info.size_bytes)))
      ok()
    end
  end

  ok()
end

def volume(shape: List[Int]) -> Int
  n = 1
  for d in shape
    n = n * d
  end

  n
end

def header_of(fields: List[Field], tensors: List[Out]) -> Result[Layout]
  parts: List[Bytes] = ["GGUF".bytes, try Binary.write_u32_le(3), try Binary.write_u64_le(tensors.size)]
  parts = parts.push(try Binary.write_u64_le(fields.size))
  keys: Set[String] = Set.new([])
  for f in fields
    return error("the key #{f.key} appears twice") if keys.include?(f.key)
    keys = keys.add(f.key)
    parts = parts.push(try string(f.key))
    parts = parts.concat(try encoded(f.value))
  end

  names: Set[String] = Set.new([])
  offsets: List[Int] = []
  at = 0
  for t in tensors
    return error("the tensor #{t.name} appears twice") if names.include?(t.name)
    names = names.add(t.name)
    shape = t.source.shape
    if t.format == Tensor.FloatFormat.Q8_0
      return error("the tensor #{t.name} is written as F32, F16 or BF16, not Q8_0")
    end

    return error("the tensor #{t.name} has #{shape.size} dimensions (1 to 4)") if shape.size < 1 or shape.size > 4
    parts = parts.push(try string(t.name))
    parts = parts.push(try Binary.write_u32_le(shape.size))
    for d in shape.reverse
      return error("the tensor #{t.name} has a dimension of #{d}") if d < 1
      parts = parts.push(try Binary.write_u64_le(d))
    end

    at = align(at)
    offsets = offsets.push(at)
    parts = parts.push(try Binary.write_u32_le(ggml_type(t.format)))
    parts = parts.push(try Binary.write_u64_le(at))
    at = at + volume(shape) * width(t.format)
  end

  size = 0
  for p in parts
    size = size + p.size
  end

  base = align(size)
  parts = parts.push(Bytes.from_list(List.repeat(0, base - size)).unwrap!)
  ok(Layout(header: Bytes.concat(parts), base: base, offsets: offsets, size: base + align(at)))
end

## The type number of ggml of a float format.
def ggml_type(format: Tensor.FloatFormat) -> Int
  case format
    Tensor.FloatFormat.F32 -> 0
    Tensor.FloatFormat.F16 -> 1
    Tensor.FloatFormat.BF16 -> 30
    Tensor.FloatFormat.Q8_0 -> 8
  end
end

## The bytes of one value of a float format.
def width(format: Tensor.FloatFormat) -> Int
  case format
    Tensor.FloatFormat.F32 -> 4
    _ -> 2
  end
end

def flag(b: Bool) -> Int
  return 1 if b
  0
end

def align(n: Int) -> Int = ((n + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT

## A string: a length of 8 bytes, then the UTF-8.
def string(s: String) -> Result[Bytes] = ok(Bytes.concat([try Binary.write_u64_le(s.bytes.size), s.bytes]))

## The type of a value and the value, as the parts that follow its key.
def encoded(v: Meta) -> Result[List[Bytes]]
  case v
    Meta.U32(n) -> ok([try Binary.write_u32_le(4), try Binary.write_u32_le(n)])
    Meta.F32(x) -> ok([try Binary.write_u32_le(6), Binary.write_f32_le(x)])
    Meta.Bool(b) -> ok([try Binary.write_u32_le(7), try Binary.write_u8(flag(b))])
    Meta.Text(s) -> ok([try Binary.write_u32_le(8), try string(s)])
    Meta.Texts(list) ->
      parts: List[Bytes] = [try Binary.write_u32_le(9), try Binary.write_u32_le(8), try Binary.write_u64_le(list.size)]
      for s in list
        parts = parts.push(try string(s))
      end

      ok(parts)
    Meta.Ints(list) ->
      parts: List[Bytes] = [try Binary.write_u32_le(9), try Binary.write_u32_le(5), try Binary.write_u64_le(list.size)]
      for n in list
        parts = parts.push(try Binary.write_i32_le(n))
      end

      ok(parts)
  end
end
