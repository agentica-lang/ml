## `Tensor.from_file`: F32, F16 and BF16 ranges of one file read the same on
## every backend this machine has (the reference reads them in Agentica, libtorch
## and ggml in C), and a bad range is an `error` on each.
##
## The file is one buffer with bytes before the first range, so the offsets
## are not zero. Without a native module its backend is not in the list: the
## availability of each is asserted by its own test file.

import Conform from "torch_support"
import Tensor from "../tensor"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

## Three padding bytes, then 3 F32 (1.5, -2.25, 0.0) at 3, 5 F16 (1.0, -2.0,
## 0.5, the smallest subnormal, 65504.0) at 15, and 3 BF16 (1.0, -3.0, 0.5) at 25.
BYTES = [
  9, 9, 9, 0, 0, 192, 63, 0, 0, 16, 192, 0, 0, 0, 0, 0, 60, 0, 192, 0, 56, 1, 0, 255, 123, 128, 63, 64, 192, 0, 63,
]

private

def backends() -> List[Tensor.Backend]
  all = [Conform.REF, Conform.TORCH, CPU, VULKAN]
  all.select { |b: Tensor.Backend| Tensor.available(b) }
end

## Runs `body` with a new folder that holds `data.bin`, then removes the folder.
def with_file(body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-from-file-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  path = "#{dir}/data.bin"
  result = within(path, body)
  _ = Path.remove(dir, 5000)
  result
end

def within(path: String, body: Fn[String, Result[Void]]) -> Result[Void]
  try File.open(path, "write", 5000) do |f|
    try File.write_bytes(f, try Bytes.from_list(BYTES), 5000)
    ok()
  end

  body.call(path)
end

def test_every_format_reads_the_same_on_each_backend() -> Result[Void]
  with_file do |path|
    for b in backends()
      f32 = try Tensor.from_file(path, 3, [3, 1], Tensor.FloatFormat.F32, b)
      try Test.eq(Tensor.dtype(f32), Tensor.Dtype.F32)
      try Test.eq(Tensor.shape(f32), [3, 1])
      try Test.eq(try floats(f32), [1.5, -2.25, 0.0])
      f16 = try Tensor.from_file(path, 15, [5], Tensor.FloatFormat.F16, b)
      try Test.eq(Tensor.dtype(f16), Tensor.Dtype.F32)
      try Test.eq(try floats(f16), [1.0, -2.0, 0.5, 1.0 / 16777216.0, 65504.0])
      bf16 = try Tensor.from_file(path, 25, [1, 3], Tensor.FloatFormat.BF16, b)
      try Test.eq(try floats(bf16), [1.0, -3.0, 0.5])
      empty = try Tensor.from_file(path, 31, [0, 4], Tensor.FloatFormat.F32, b)
      try Test.eq(Tensor.shape(empty), [0, 4])
    end

    ok()
  end
end

def test_a_bad_range_is_an_error_on_each_backend() -> Result[Void]
  with_file do |path|
    for b in backends()
      try Test.error?(Tensor.from_file(path, 29, [3], Tensor.FloatFormat.BF16, b))
      try Test.error?(Tensor.from_file(path, 0, [8], Tensor.FloatFormat.F32, b))
      try Test.error?(Tensor.from_file(path, 32, [1], Tensor.FloatFormat.BF16, b))
      try Test.error?(Tensor.from_file(path, -1, [1], Tensor.FloatFormat.F32, b))
      try Test.error?(Tensor.from_file(path, 0, [-1], Tensor.FloatFormat.F32, b))
      try Test.error?(Tensor.from_file("#{path}.missing", 0, [1], Tensor.FloatFormat.F32, b))
      try Test.error?(Tensor.from_file(path, 0, [4611686018427387904, 4], Tensor.FloatFormat.F32, b))
      try Test.error?(Tensor.from_file(path, 9223372036854775807, [1], Tensor.FloatFormat.F32, b))
      try Test.error?(Tensor.from_file(path.split("/").take(3).join("/"), 0, [1], Tensor.FloatFormat.F32, b))
    end

    Test.error?(Tensor.from_file(path, 0, [1], Tensor.FloatFormat.F32, Tensor.Backend.Trace))
  end
end

## One F16 range of more than the 1 MiB chunk the native readers widen at a time, from an odd offset,
## made of finite halves of both signs.
def test_a_range_of_several_chunks_reads_the_same_on_each_backend() -> Result[Void]
  count = 524291
  bytes = [9].concat((0...(count * 2)).map { |i| chunk_byte(i) })
  dir = "/tmp/agentica-from-file-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  path = "#{dir}/chunks.bin"
  try File.open(path, "write", 5000) do |f|
    try File.write_bytes(f, try Bytes.from_list(bytes), 5000)
    ok()
  end

  want = try floats(try Tensor.from_file(path, 1, [count], Tensor.FloatFormat.F16, Conform.REF))
  for b in backends()
    got = try floats(try Tensor.from_file(path, 1, [count], Tensor.FloatFormat.F16, b))
    try Test.assert(got == want, "the range read differs on #{b}")
  end

  _ = Path.remove(dir, 5000)
  ok()
end

## The low byte of a half is arbitrary; the high one keeps the exponent below 30, so no half is infinite or NaN.
def chunk_byte(i: Int) -> Int
  return (i * 7) % 256 if i % 2 == 0
  high = (i / 2) % 124
  return high + 128 if (i / 2) % 3 == 0
  high
end

def floats(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end
