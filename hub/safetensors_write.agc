## Writes a safetensors file (`hub/DESIGN.md`): the `F32` tensors the caller names, sorted by name, each one
## written from its backend straight after the header (`Tensor.append_file`), so no value goes through an Agentica
## list. The header is padded with spaces to a multiple of 8 bytes, as the `safetensors` library writes it. The file is
## written under `path + ".part"` and renamed when it is whole, so a failure leaves no half file.

import Binary from "std/binary"
import Shards from "shard"
import Tensor from "pkg/tensor/tensor"

## The header bytes written at a time (`File.write_bytes` takes 1 MiB at most).
PART = 1048576

## Writes `tensors` and `metadata` (the `__metadata__` of the header) to `path`. The same tensors and metadata give
## the same bytes.
def write(path: String, tensors: Map[String, Tensor.Tensor], metadata: Map[String, String] = {}) -> Result[Void]
  names = tensors.keys.sort
  head = try Shards.within(path, header(names, tensors, metadata))
  part = path + ".part"
  result = fill(part, head, names.map { |name: String| tensors[name] })
  case result
    ok(_) -> Path.replace(part, path)
    error(e) ->
      _ = Path.remove(part)
      error("#{path}: #{e.message}")
  end
end

private

## The 8 bytes of the header's length and the header, padded.
def header(names: List[String], tensors: Map[String, Tensor.Tensor], metadata: Map[String, String]) -> Result[Bytes]
  fields: List[String] = []
  if not metadata.empty?
    pairs = metadata.keys.sort.map { |key: String| "#{JSON.encode(key)}:#{JSON.encode(metadata[key])}" }
    fields = fields.push("\"__metadata__\":{#{pairs.join(",")}}")
  end

  at = 0
  for name in names
    return error("a tensor cannot be named __metadata__") if name == "__metadata__"
    t = tensors[name]
    return error("tensor #{name}: only F32 tensors are written") if Tensor.dtype(t) != Tensor.Dtype.F32

    stop = at + 4 * volume(Tensor.shape(t))
    shape = JSON.encode(Tensor.shape(t))
    fields = fields.push(
      "#{JSON.encode(name)}:{\"dtype\":\"F32\",\"shape\":#{shape},\"data_offsets\":[#{at},#{stop}]}",
    )

    at = stop
  end

  text = "{" + fields.join(",") + "}"
  text = text + " " * ((8 - text.bytes.size % 8) % 8)
  ok(Bytes.concat([Binary.write_u64_le(text.bytes.size.to_uint64!), text.bytes]))
end

def fill(part: String, head: Bytes, tensors: List[Tensor.Tensor]) -> Result[Void]
  try File.open(part, "write") do |f|
    at = 0
    while at < head.size
      n = [PART, head.size - at].min.unwrap!
      try File.write_bytes(f, head.slice(at, n))
      at = at + n
    end

    ok()
  end

  size = head.size
  for t in tensors
    try Tensor.append_file(t, part)
    size = size + 4 * volume(Tensor.shape(t))
  end

  written = (try File.stat(part)).size_bytes
  return error("the file holds #{written} bytes, not the #{size} of its header and tensors") if written != size
  ok()
end

def volume(shape: List[Int]) -> Int
  n = 1
  for d in shape
    n = n * d
  end

  n
end
