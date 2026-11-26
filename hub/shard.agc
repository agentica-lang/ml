## What every checkpoint file gives `hub/`: a `Shard` (the byte ranges of its tensors) and the reads of them. The
## formats (`safetensors.agc`, `gguf.agc`) only parse their header into entries. A float tensor (F32, F16, BF16) is
## then read by the backend from its byte range of the file (`Tensor.from_file`), so no value goes through an Agentica
## list; any other dtype is read here, from its offset (`File.seek`).

import Binary from "std/binary"
import Tensor from "pkg/tensor/tensor"

CHUNK = 1048576

## The bytes per element of every dtype of the formats, so a file with a dtype `tensor/` lacks still opens and only
## that tensor is refused.
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

## The quantized types of a GGUF file that a backend reads from the file itself: the format of their blocks, and the
## dtype of a weight that keeps them.
BLOCKS = {"Q8_0": Tensor.FloatFormat.Q8_0, "Q4_K": Tensor.FloatFormat.Q4_K, "Q6_K": Tensor.FloatFormat.Q6_K}
KEPT = {"Q8_0": Tensor.Dtype.Q8_0, "Q4_K": Tensor.Dtype.Q4_K, "Q6_K": Tensor.Dtype.Q6_K}

## A tensor of a file: `start` and `stop` index the data buffer.
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

## Where the values of a float tensor are: its file, the byte they start at, and their format.
struct Source
  path: String
  offset: Int
  format: Tensor.FloatFormat
  shape: List[Int]
end

## The place of the float tensor `name` of `shard`, for a native call that reads or copies the range. A tensor of
## another dtype is an error.
def source(shard: Shard, name: String) -> Result[Source]
  e = try shard.entries.get(name).to_result("#{shard.path}: no tensor #{name}")
  return error("#{shard.path}: tensor #{name} is #{e.dtype}, not F32, F16 or BF16") if not FORMATS.key?(e.dtype)

  ok(Source(path: shard.path, offset: shard.base + e.start, format: FORMATS[e.dtype], shape: e.shape))
end

## The dtype of the tensor `name` of `shard` as the file names it (`F32`, `Q8_0`, `Q4_K`...), read from the header.
def dtype(shard: Shard, name: String) -> Result[String]
  e = try shard.entries.get(name).to_result("#{shard.path}: no tensor #{name}")
  ok(e.dtype)
end

## The shape of the tensor `name` of `shard`, read from the header.
def shape(shard: Shard, name: String) -> Result[List[Int]]
  e = try shard.entries.get(name).to_result("#{shard.path}: no tensor #{name}")
  ok(e.shape)
end

## The tensor `name` of `shard`: F32, F16 and BF16 as `F32` (as `F16` with `dtype: F16`, a half weight of the ggml
## backend, `Tensor.from_file`), a Q8_0, Q4_K or Q6_K one dequantized to `F32` (as the dtype of its own type, a quantized
## weight of the ggml backend that keeps the blocks; no other file dtype has that form), any other dtype as the file has it, whatever
## `dtype` says (only a float file has a half form), except an F64 tensor on ggml, which holds no F64: it comes as `F32`,
## silently (a value beyond F32 is an infinity).
def tensor(
  shard: Shard,
  name: String,
  backend: Tensor.Backend,
  dtype: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[Tensor.Tensor]
  e = try shard.entries.get(name).to_result("#{shard.path}: no tensor #{name}")
  at = shard.base + e.start
  case kept_type(dtype)
    some(type) ->
      if type != e.dtype
        return error("#{shard.path}: tensor #{name} is #{e.dtype}: only a #{type} tensor is kept as #{type}")
      end

    none -> nil
  end

  if BLOCKS.key?(e.dtype)
    kept = Tensor.Dtype.F32
    if dtype == KEPT[e.dtype]
      kept = dtype
    end

    return within(
      "#{shard.path}: tensor #{name}",
      Tensor.from_file(shard.path, at, e.shape, BLOCKS[e.dtype], backend, kept),
    )
  end

  if FORMATS.key?(e.dtype)
    return within(
      "#{shard.path}: tensor #{name}",
      Tensor.from_file(shard.path, at, e.shape, FORMATS[e.dtype], backend, dtype),
    )
  end

  return error("#{shard.path}: tensor #{name}: dtype #{e.dtype} has no tensor/ dtype") if not DTYPES.key?(e.dtype)
  b = try within("#{shard.path}: tensor #{name}", read_range(shard.path, at, e.stop - e.start))
  values = try within("#{shard.path}: tensor #{name}", decode(b, e))
  Tensor.from_list(values, e.shape, held(e.dtype, backend), backend)
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

private

## The file type (`Q8_0`, `Q4_K` or `Q6_K`) of the quantized weight `dtype`, `none` for another dtype.
def kept_type(dtype: Tensor.Dtype) -> Optional[String]
  for type in KEPT.keys
    return some(type) if KEPT[type] == dtype
  end

  none
end

## The dtype a tensor of the file dtype `name` has on `backend`: as the file has it, but `F32` for F64 on ggml.
def held(name: String, backend: Tensor.Backend) -> Tensor.Dtype
  case backend
    Tensor.Backend.Ggml(_) -> return Tensor.Dtype.F32 if name == "F64"
    _ -> nil
  end

  DTYPES[name]
end

## The values of the bytes `b` of one tensor, for the dtypes the backends do not read from the file. The format's check
## (`check_layout` of safetensors, `check_tensors` of GGUF) already put every tensor inside the buffer, so a read
## cannot fail.
def decode(b: Bytes, e: Entry) -> Result[List[Tensor.Values]]
  at = 0
  count = (e.stop - e.start) / WIDTHS[e.dtype]
  case e.dtype
    "F64" -> ok((0...count).map { |i| Tensor.Values.Float(Binary.read_f64_le(b, at + 8 * i).unwrap!) })
    "I8" -> ok((0...count).map { |i| Tensor.Values.Int(b[at + i].wrap_int8.to_i) })
    "I16" -> ok((0...count).map { |i| Tensor.Values.Int(Binary.read_i16_le(b, at + 2 * i).unwrap!.to_i) })
    "I32" -> ok((0...count).map { |i| Tensor.Values.Int(Binary.read_i32_le(b, at + 4 * i).unwrap!.to_i) })
    "I64" -> ok((0...count).map { |i| Tensor.Values.Int(Binary.read_i64_le(b, at + 8 * i).unwrap!) })
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
