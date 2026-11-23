## One safetensors file (`hub/DESIGN.md`): an 8-byte little-endian header
## length, a JSON header, then one byte buffer that the header's
## `data_offsets` cut into little-endian, row-major tensors.
##
## `open` reads the header and nothing else. A float tensor (F32, F16, BF16)
## is read by the backend from its byte range of the file
## (`Tensor.from_file`), so no value goes through an Agentica list; any other
## dtype is read here, from its offset (`File.seek`).

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

## The `tensor/` dtype of each safetensors dtype that is read here, as the file has it.
DTYPES = {
  "F64": Tensor.Dtype.F64, "I8": Tensor.Dtype.I8, "I16": Tensor.Dtype.I16, "I32": Tensor.Dtype.I32,
  "I64": Tensor.Dtype.I64, "U8": Tensor.Dtype.U8, "BOOL": Tensor.Dtype.Bool,
}

## The float formats a backend reads from the file itself.
FORMATS = {"F32": Tensor.FloatFormat.F32, "F16": Tensor.FloatFormat.F16, "BF16": Tensor.FloatFormat.BF16}

## A tensor of the header: `start` and `stop` index the data buffer.
struct Entry
  name: String
  dtype: String
  shape: List[Int]
  start: Int
  stop: Int
end

## A parsed file: its header, and `base`, the byte of the file where the data buffer starts.
struct Shard
  path: String
  base: Int
  entries: Map[String, Entry]
  metadata: Map[String, String]
end

## The JSON form of one header entry.
struct Header
  dtype: String
  shape: List[Int]
  data_offsets: List[Int]
end

def open(path: String) -> Result[Shard] = within(path, read(path))

## The tensor `name` of `shard`: F32, F16 and BF16 as `F32`, any other dtype as the file has it.
def tensor(shard: Shard, name: String, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  e = try shard.entries.get(name).to_result("#{shard.path}: no tensor #{name}")
  at = shard.base + e.start
  if FORMATS.key?(e.dtype)
    return within(
      "#{shard.path}: tensor #{name}",
      Tensor.from_file(shard.path, at, e.shape, FORMATS[e.dtype], backend),
    )
  end

  return error("#{shard.path}: tensor #{name}: dtype #{e.dtype} has no tensor/ dtype") if not DTYPES.key?(e.dtype)
  b = try within("#{shard.path}: tensor #{name}", read_range(shard.path, at, e.stop - e.start))
  values = try within("#{shard.path}: tensor #{name}", decode(b, e))
  Tensor.from_list(values, e.shape, DTYPES[e.dtype], backend)
end

private

## Parses and validates a header's JSON `text` over a buffer of `size` bytes.
def parse(text: String, size: Int) -> Result[Shard]
  return error("the header does not start with {") if not text.start_with?("{")
  fields: Map[String, JSON.Value] = {}
  case JSON.node(try JSON.parse(text))
    JSON.Node.Object(m) -> fields = m
    _ -> return error("the header is not a JSON object")
  end

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
  ok(Shard(path: "", base: 0, entries: entries, metadata: metadata))
end

## The header length is checked against the file's size before the header
## is read, and the header before the data offsets are.
def read(path: String) -> Result[Shard]
  size = (try File.stat(path)).size_bytes
  return error("#{size} bytes is too short for the header length") if size < 8
  File.open(path) do |f|
    n = try Binary.read_u64_le(try read_exact(f, 8), 0)
    return error("the header length #{n} is over the limit #{MAX_HEADER}") if n > MAX_HEADER
    return error("the header length #{n} is past the end of the file") if n > size - 8
    shard = try parse(try String.from_bytes(try read_exact(f, n)), size - 8 - n)
    ok(shard.with(path: path, base: 8 + n))
  end
end

## `length` bytes at `offset` of `path`.
def read_range(path: String, offset: Int, length: Int) -> Result[Bytes]
  File.open(path) do |f|
    try File.seek(f, offset)
    read_exact(f, length)
  end
end

## Exactly `count` bytes from `f`, in chunks.
def read_exact(f: File.Handle, count: Int) -> Result[Bytes]
  parts: List[Bytes] = []
  left = count
  while left > 0
    chunk = try File.read_bytes(f, [left, CHUNK].min.unwrap!)
    return error("the file ended #{left} bytes early") if chunk.empty?
    parts = parts.push(chunk)
    left = left - chunk.size
  end

  ok(Bytes.concat(parts))
end

## `r`, with `prefix: ` before its error.
def within[T](prefix: String, r: Result[T]) -> Result[T]
  case r
    ok(v) -> ok(v)
    error(e) -> error("#{prefix}: #{e.message}")
  end
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

  for d in h.shape
    return error("tensor #{name}: negative dimension in #{h.shape}") if d < 0
  end

  # A zero anywhere makes the tensor empty, so later dimensions cannot overflow.
  span = stop - start
  need = 0
  if not h.shape.include?(0)
    need = width
    for d in h.shape
      return error("tensor #{name}: #{h.dtype} #{h.shape} does not fit in #{span} bytes") if need > span / d
      need = need * d
    end
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

## The values of the bytes `b` of one tensor, for the dtypes the backends do not read from the file.
def decode(b: Bytes, e: Entry) -> Result[List[Tensor.Values]]
  at = 0
  count = (e.stop - e.start) / WIDTHS[e.dtype]
  case e.dtype
    "F64" -> ok((0...count).map { |i| Tensor.Values.Float(Float.from_bits(i64_at(b, at + 8 * i))) })
    "I8" -> ok((0...count).map { |i| Tensor.Values.Int(signed(b[at + i], 8)) })
    "I16" -> ok((0...count).map { |i| Tensor.Values.Int(signed(u16(b, at + 2 * i), 16)) })
    "I32" -> ok((0...count).map { |i| Tensor.Values.Int(signed(u32(b, at + 4 * i), 32)) })
    "I64" -> ok((0...count).map { |i| Tensor.Values.Int(i64_at(b, at + 8 * i)) })
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

# The readers below index without checks: `check_layout` already put every
# tensor inside the buffer, so `at` is in range.
def u16(b: Bytes, at: Int) -> Int = b[at].bit_or(b[at + 1].shift_left(8))

def u32(b: Bytes, at: Int) -> Int = u16(b, at).bit_or(u16(b, at + 2).shift_left(16))

## All 64 bits: the top byte wraps into the sign, as `I64` wants.
def i64_at(b: Bytes, at: Int) -> Int = u32(b, at).bit_or(u32(b, at + 4).shift_left(32))

def signed(v: Int, bits: Int) -> Int
  return v - 1.shift_left(bits) if v >= 1.shift_left(bits - 1)
  v
end
