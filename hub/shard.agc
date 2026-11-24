## What every checkpoint file gives `hub/`: a `Shard` (the byte ranges of its tensors) and the reads of them. The
## formats (`safetensors.agc`, `gguf.agc`) only parse their header into entries. A float tensor (F32, F16, BF16) is
## then read by the backend from its byte range of the file (`Tensor.from_file`), so no value goes through an Agentica
## list; any other dtype is read here, from its offset (`File.seek`).

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

## The shape of the tensor `name` of `shard`, read from the header.
def shape(shard: Shard, name: String) -> Result[List[Int]]
  e = try shard.entries.get(name).to_result("#{shard.path}: no tensor #{name}")
  ok(e.shape)
end

## The tensor `name` of `shard`: F32, F16 and BF16 as `F32` (as `F16` with `dtype: F16`, a half weight of the ggml
## backend, `Tensor.from_file`), any other dtype as the file has it, whatever `dtype` says (only a float file has a
## half form), except an F64 tensor on ggml, which holds no F64: it comes as `F32`, silently (a value beyond F32 is
## an infinity).
def tensor(
  shard: Shard,
  name: String,
  backend: Tensor.Backend,
  dtype: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[Tensor.Tensor]
  e = try shard.entries.get(name).to_result("#{shard.path}: no tensor #{name}")
  at = shard.base + e.start
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

## The dtype a tensor of the file dtype `name` has on `backend`: as the file has it, but `F32` for F64 on ggml.
def held(name: String, backend: Tensor.Backend) -> Tensor.Dtype
  case backend
    Tensor.Backend.Ggml(_) -> return Tensor.Dtype.F32 if name == "F64"
    _ -> nil
  end

  DTYPES[name]
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

# The readers below index without checks: the format's check (`check_layout` of
# safetensors, `check_tensors` of GGUF) already put every tensor inside the
# buffer, so `at` is in range.
def u16(b: Bytes, at: Int) -> Int = b[at].bit_or(b[at + 1].shift_left(8))

def u32(b: Bytes, at: Int) -> Int = u16(b, at).bit_or(u16(b, at + 2).shift_left(16))

## All 64 bits: the top byte wraps into the sign, as `I64` wants.
def i64_at(b: Bytes, at: Int) -> Int = u32(b, at).bit_or(u32(b, at + 4).shift_left(32))

def signed(v: Int, bits: Int) -> Int
  return v - 1.shift_left(bits) if v >= 1.shift_left(bits - 1)
  v
end
