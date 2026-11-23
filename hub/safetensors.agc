## One safetensors file (`hub/DESIGN.md`): an 8-byte little-endian header
## length, a JSON header, then one byte buffer that the header's
## `data_offsets` cut into little-endian, row-major tensors.
##
## `open` reads the whole file: the File API has no seek, so a ranged read per
## tensor would read the file again up to that tensor.

import Binary from "std/binary"
import Tensor from "pkg/tensor/tensor"

## The largest header the format allows, so a hostile length is refused
## before anything is read past it.
MAX_HEADER = 100000000
CHUNK = 1048576

## The bytes per element of every safetensors dtype, so a file with a dtype
## `tensor/` lacks still opens and only that tensor is refused.
WIDTHS = {
  "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4,
  "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}

## The `tensor/` dtype of each safetensors dtype that has one.
DTYPES = {
  "F32": Tensor.Dtype.F32, "F16": Tensor.Dtype.F16, "F64": Tensor.Dtype.F64, "I8": Tensor.Dtype.I8,
  "I16": Tensor.Dtype.I16, "I32": Tensor.Dtype.I32, "I64": Tensor.Dtype.I64, "U8": Tensor.Dtype.U8,
  "BOOL": Tensor.Dtype.Bool,
}

## A tensor of the header: `start` and `stop` index the data buffer.
struct Entry
  name: String
  dtype: String
  shape: List[Int]
  start: Int
  stop: Int
end

## A parsed file: the whole file's `bytes`, with the buffer at `base`.
struct Shard
  path: String
  entries: Map[String, Entry]
  metadata: Map[String, String]
  bytes: Bytes
  base: Int
end

## The JSON form of one header entry.
struct Header
  dtype: String
  shape: List[Int]
  data_offsets: List[Int]
end

def open(path: String) -> Result[Shard]
  bytes = try within(path, read_all(path))
  within(path, parse(bytes, path))
end

## `r`, with `prefix: ` before its error.
def within[T](prefix: String, r: Result[T]) -> Result[T]
  case r
    ok(v) -> ok(v)
    error(e) -> error("#{prefix}: #{e.message}")
  end
end

def read_all(path: String) -> Result[Bytes]
  File.open(path) do |f|
    parts: List[Bytes] = []
    chunk = try File.read_bytes(f, CHUNK)
    while not chunk.empty?
      parts = parts.push(chunk)
      chunk = try File.read_bytes(f, CHUNK)
    end

    ok(Bytes.concat(parts))
  end
end

## Parses and validates the header of a whole file's `bytes`.
def parse(bytes: Bytes, path: String) -> Result[Shard]
  return error("#{bytes.size} bytes is too short for the header length") if bytes.size < 8
  n = try Binary.read_u64_le(bytes, 0)
  return error("the header length #{n} is over the limit #{MAX_HEADER}") if n > MAX_HEADER
  return error("the header length #{n} is past the end of the file") if n > bytes.size - 8
  text = try String.from_bytes(bytes.slice(8, n))
  return error("the header does not start with {") if not text.start_with?("{")
  fields: Map[String, JSON.Value] = {}
  case JSON.node(try JSON.parse(text))
    JSON.Node.Object(m) -> fields = m
    _ -> return error("the header is not a JSON object")
  end

  size = bytes.size - 8 - n
  entries: Map[String, Entry] = {}
  metadata: Map[String, String] = {}
  for name in fields.keys
    if name == "__metadata__"
      metadata = try within("__metadata__", JSON.decode(fields[name], Map[String, String]))
    else
      entries = entries.put(name, try entry(name, fields[name], size))
    end
  end

  try check_layout(entries.values, size)
  ok(Shard(path: path, entries: entries, metadata: metadata, bytes: bytes, base: 8 + n))
end

def entry(name: String, json: JSON.Value, size: Int) -> Result[Entry]
  h = try within("tensor #{name}", JSON.decode(json, Header))

  width = try WIDTHS.get(h.dtype).to_result("tensor #{name}: unknown dtype #{h.dtype}")
  return error("tensor #{name}: data_offsets must be [begin, end]") if h.data_offsets.size != 2
  start = h.data_offsets[0]
  stop = h.data_offsets[1]
  if start < 0 or stop < start or stop > size
    return error("tensor #{name}: data_offsets [#{start}, #{stop}] are outside the #{size} data bytes")
  end

  span = stop - start
  need = width
  for d in h.shape
    return error("tensor #{name}: negative dimension in #{h.shape}") if d < 0
    return error("tensor #{name}: #{h.dtype} #{h.shape} does not fit in #{span} bytes") if d > 0 and need > span / d
    need = need * d
  end

  return error("tensor #{name}: #{h.dtype} #{h.shape} needs #{need} bytes, data_offsets hold #{span}") if need != span

  ok(Entry(name: name, dtype: h.dtype, shape: h.shape, start: start, stop: stop))
end

## The tensors tile the buffer: sorted by offset, each starts where the one
## before stops, and the last stops at its end (no gap, overlap or tail).
def check_layout(entries: List[Entry], size: Int) -> Result[Void]
  at = 0
  for e in entries.sort_by(&order)
    return error("tensor #{e.name} overlaps the tensor before it") if e.start < at
    return error("the data has a gap of #{e.start - at} bytes before tensor #{e.name}") if e.start > at
    at = e.stop
  end

  return error("the tensors cover #{at} of the #{size} data bytes") if at != size
  ok()
end

## Empty tensors sort before a non-empty one at the same offset.
def order(e: Entry) -> Int
  return e.start * 2 if e.stop == e.start
  e.start * 2 + 1
end

## The tensor `name` of `shard`, with the file's dtype.
def tensor(shard: Shard, name: String, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  e = try shard.entries.get(name).to_result("#{shard.path}: no tensor #{name}")
  values = try within("#{shard.path}: tensor #{name}", decode(shard.bytes, shard.base + e.start, e))

  Tensor.from_list(values, e.shape, DTYPES[e.dtype], backend)
end

def decode(b: Bytes, at: Int, e: Entry) -> Result[List[Tensor.Values]]
  count = (e.stop - e.start) / WIDTHS[e.dtype]
  case e.dtype
    "F32" -> ok((0...count).map { |i| Tensor.Values.Float(Float.from_bits32(u32(b, at + 4 * i))) })
    "F16" -> ok((0...count).map { |i| Tensor.Values.Float(half(u16(b, at + 2 * i))) })
    "F64" -> ok((0...count).map { |i| Tensor.Values.Float(Float.from_bits(u64(b, at + 8 * i))) })
    "I8" -> ok((0...count).map { |i| Tensor.Values.Int(signed(b[at + i], 8)) })
    "I16" -> ok((0...count).map { |i| Tensor.Values.Int(signed(u16(b, at + 2 * i), 16)) })
    "I32" -> ok((0...count).map { |i| Tensor.Values.Int(signed(u32(b, at + 4 * i), 32)) })
    "I64" -> ok((0...count).map { |i| Tensor.Values.Int(u64(b, at + 8 * i)) })
    "U8" -> ok((0...count).map { |i| Tensor.Values.Int(b[at + i]) })
    "BOOL" -> bools(b, at, count)
    _ -> error("dtype #{e.dtype} has no tensor/ dtype")
  end
end

def bools(b: Bytes, at: Int, count: Int) -> Result[List[Tensor.Values]]
  out: List[Tensor.Values] = []
  for i in 0...count
    return error("BOOL byte #{i} is #{b[at + i]}, not 0 or 1") if b[at + i] > 1
    out = out.push(Tensor.Values.Bool(b[at + i] == 1))
  end

  ok(out)
end

def u16(b: Bytes, at: Int) -> Int = b[at].bit_or(b[at + 1].shift_left(8))

def u32(b: Bytes, at: Int) -> Int = u16(b, at).bit_or(u16(b, at + 2).shift_left(16))

## All 64 bits: the top byte wraps into the sign, as `I64` wants.
def u64(b: Bytes, at: Int) -> Int = u32(b, at).bit_or(u32(b, at + 4).shift_left(32))

def signed(v: Int, bits: Int) -> Int
  return v - 1.shift_left(bits) if v >= 1.shift_left(bits - 1)
  v
end

## IEEE binary16 to a Float, rebuilt as binary64 bits (exact).
def half(bits: Int) -> Float
  sign = bits.shift_right(15).shift_left(63)
  exp = bits.shift_right(10).bit_and(0x1F)
  mant = bits.bit_and(0x3FF)
  if exp == 0
    # Subnormal: mant * 2^-24.
    m = mant.to_f * Float.from_bits((1023 - 24).shift_left(52))
    return -m if sign != 0
    return m
  end

  return Float.from_bits(sign.bit_or(0x7FF0000000000000).bit_or(mant.shift_left(42))) if exp == 31
  Float.from_bits(sign.bit_or((exp - 15 + 1023).shift_left(52)).bit_or(mant.shift_left(42)))
end
