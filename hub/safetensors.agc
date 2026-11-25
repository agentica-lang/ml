## One safetensors file (`hub/DESIGN.md`): an 8-byte little-endian header
## length, a JSON header, then one byte buffer that the header's
## `data_offsets` cut into little-endian, row-major tensors.
##
## `open` reads the header and nothing else. A float tensor (F32, F16, BF16)
## is read by the backend from its byte range of the file
## (`Tensor.from_file`), so no value goes through an Agentica list; any other
## dtype is read here, from its offset (`File.seek`).

import Binary from "std/binary"
import Shards from "shard"

## The largest header the format allows, so a hostile length is refused
## before anything is read past it.
MAX_HEADER = 100000000

## The JSON form of one header entry.
struct Header
  dtype: String
  shape: List[Int]
  data_offsets: List[Int]
end

## The file at `path`: its header parsed and checked, no tensor read.
def open(path: String) -> Result[Shards.Shard] = Shards.within(path, read(path))

private

## Parses and validates a header's JSON `text` over a buffer of `size` bytes.
def parse(text: String, size: Int) -> Result[Shards.Shard]
  return error("the header does not start with {") if not text.start_with?("{")
  fields: Map[String, JSON.Value] = {}
  case JSON.node(try JSON.parse(text))
    JSON.Node.Object(m) -> fields = m
    _ -> return error("the header is not a JSON object")
  end

  entries: Map[String, Shards.Entry] = {}
  metadata: Map[String, String] = {}
  for name in fields.keys
    if name == "__metadata__"
      metadata = try Shards.within("__metadata__", JSON.decode(fields[name], Map[String, String]))
    else
      entries = entries.put(name, try entry(name, fields[name], size))
    end
  end

  try check_layout(entries.values, size)
  ok(Shards.Shard(path: "", base: 0, entries: entries, metadata: metadata))
end

## The header length is checked against the file's size before the header
## is read, and the header before the data offsets are.
def read(path: String) -> Result[Shards.Shard]
  size = (try File.stat(path)).size_bytes
  return error("#{size} bytes is too short for the header length") if size < 8
  File.open(path) do |f|
    n = try (try Binary.read_u64_le(try Shards.read_exact(f, 8), 0)).to_i
    return error("the header length #{n} is over the limit #{MAX_HEADER}") if n > MAX_HEADER
    return error("the header length #{n} is past the end of the file") if n > size - 8
    shard = try parse(try String.from_bytes(try Shards.read_exact(f, n)), size - 8 - n)
    ok(shard.with(path: path, base: 8 + n))
  end
end

def entry(name: String, json: JSON.Value, size: Int) -> Result[Shards.Entry]
  h = try Shards.within("tensor #{name}", JSON.decode(json, Header))

  width = try Shards.WIDTHS.get(h.dtype).to_result("tensor #{name}: unknown dtype #{h.dtype}")
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

  ok(Shards.Entry(name: name, dtype: h.dtype, shape: h.shape, start: start, stop: stop))
end

## The tensors tile the buffer: sorted by offset, each starts where the one
## before stops, and the last stops at its end (no gap, overlap or tail).
def check_layout(entries: List[Shards.Entry], size: Int) -> Result[Void]
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
def order(e: Shards.Entry) -> Int
  return e.start * 2 if e.stop == e.start
  e.start * 2 + 1
end
