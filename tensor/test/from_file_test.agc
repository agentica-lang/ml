## `Tensor.from_file`: F32, F16 and BF16 ranges of one file read the same on
## every backend this machine has (the reference reads them in Agentica, libtorch,
## ggml and XLA in C), and a bad range is an `error` on each.
##
## The file is one buffer with bytes before the first range, so the offsets
## are not zero. Without a native module its backend is not in the list: the
## availability of each is asserted by its own test file.

import Conform from "torch_support"
import FileRange from "../src/file_range"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
XLA = Tensor.Backend.Xla(Tensor.Device.Cpu)

## Three padding bytes, then 3 F32 (1.5, -2.25, 0.0) at 3, 5 F16 (1.0, -2.0,
## 0.5, the smallest subnormal, 65504.0) at 15, and 3 BF16 (1.0, -3.0, 0.5) at 25.
BYTES = [
  9, 9, 9, 0, 0, 192, 63, 0, 0, 16, 192, 0, 0, 0, 0, 0, 60, 0, 192, 0, 56, 1, 0, 255, 123, 128, 63, 64, 192, 0, 63,
]

private

def backends() -> List[Tensor.Backend]
  all = [Conform.REF, Conform.TORCH, CPU, VULKAN, XLA]
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

## `Tensor.from_bytes` reads the bytes of a range as `from_file` does, on each backend, and a size that is not the
## shape's is an `error`.
def test_from_bytes_reads_what_from_file_reads() -> Result[Void]
  all = BYTES
  try with_file do |path|
    try Test.error?(Tensor.from_file(path, 3, [3], Tensor.FloatFormat.U8))
    try Test.error?(Tensor.convert_file(path, 0, Tensor.FloatFormat.U8, 3, path, 0, Tensor.FloatFormat.F32))
    ok()
  end

  for b in backends()
    f32 = try Tensor.from_bytes(try Bytes.from_list(all.drop(3).take(12)), [3, 1], Tensor.FloatFormat.F32, b)
    try Test.eq(Tensor.dtype(f32), Tensor.Dtype.F32)
    try Test.eq(try floats(f32), [1.5, -2.25, 0.0])
    f16 = try Tensor.from_bytes(try Bytes.from_list(all.drop(15).take(10)), [5], Tensor.FloatFormat.F16, b)
    try Test.eq(try floats(f16), [1.0, -2.0, 0.5, 1.0 / 16777216.0, 65504.0])
    empty = try Tensor.from_bytes(try Bytes.from_list([]), [0, 4], Tensor.FloatFormat.F32, b)
    try Test.eq(Tensor.shape(empty), [0, 4])
    u8 = try Tensor.from_bytes(try Bytes.from_list([0, 1, 127, 128, 255, 9]), [2, 3], Tensor.FloatFormat.U8, b)
    try Test.eq(Tensor.dtype(u8), Tensor.Dtype.F32)
    try Test.eq(try floats(u8), [0.0, 1.0, 127.0, 128.0, 255.0, 9.0])
    try Test.error?(Tensor.from_bytes(try Bytes.from_list([0, 1, 2]), [2, 2], Tensor.FloatFormat.U8, b))
    try Test.error?(Tensor.from_bytes(try Bytes.from_list(all.drop(3).take(11)), [3, 1], Tensor.FloatFormat.F32, b))
    try Test.error?(Tensor.from_bytes(try Bytes.from_list(all.drop(3).take(12)), [2, 1], Tensor.FloatFormat.F32, b))
  end

  Test.error?(Tensor.from_bytes(try Bytes.from_list([0, 0, 0, 0]), [1], Tensor.FloatFormat.F32, Tensor.Backend.Trace))
end

## One Q8_0 block (the scale 0.5 as a half, then the quants -16..15) is dequantized to F32 on each backend, by
## `from_bytes` and by `from_file`, and a length that is not whole blocks is an `error`.
def test_a_q8_0_block_is_dequantized_on_each_backend() -> Result[Void]
  quants = (0...32).map { |i: Int| (i + 240) % 256 }
  block = try Bytes.from_list([0, 56] + quants)
  want = (0...32).map { |i: Int| (i - 16).to_f * 0.5 }
  with_file do |path|
    try File.open("#{path}.q8", "write", 5000) do |f|
      try File.write_bytes(f, block, 5000)
      ok()
    end

    for b in backends()
      try Test.eq(try floats(try Tensor.from_bytes(block, [32], Tensor.FloatFormat.Q8_0, b)), want)
      try Test.eq(try floats(try Tensor.from_file("#{path}.q8", 0, [4, 8], Tensor.FloatFormat.Q8_0, b)), want)
      try Test.error?(Tensor.from_bytes(block, [31], Tensor.FloatFormat.Q8_0, b))
      try Test.error?(Tensor.from_file("#{path}.q8", 0, [64], Tensor.FloatFormat.Q8_0, b))
    end

    ok()
  end
end

## `Tensor.append_file` writes what `from_file` read: after the bytes the file had, the same bytes as the range, on
## each backend, and only an `F32` tensor into a file that exists.
def test_append_file_writes_back_the_bytes_of_a_range() -> Result[Void]
  with_file do |path|
    all = backends()
    for i in 0...all.size
      b = all[i]
      out = "#{path}.#{i}"
      try File.open(out, "write", 5000) do |f|
        try File.write_bytes(f, try Bytes.from_list([7, 7]), 5000)
        ok()
      end

      t = try Tensor.from_file(path, 3, [3, 1], Tensor.FloatFormat.F32, b)
      try Tensor.append_file(t, out)
      try Tensor.append_file(try Tensor.from_file(path, 31, [0, 4], Tensor.FloatFormat.F32, b), out)
      try Test.eq((try FileRange.read_range(out, 0, 14)).to_a, [7, 7] + BYTES.drop(3).take(12))
      try Test.eq((try File.stat(out)).size_bytes, 14)
      try Test.error?(Tensor.append_file(t, "#{path}.missing"))
      try Test.error?(Tensor.append_file(try Tensor.arange(0, 3, 1, Tensor.Dtype.I32, b), out))
    end

    Test.error?(Tensor.append_file(Trace.constant(try Tensor.zeros([1], Tensor.Dtype.F32, Conform.REF)), path))
  end
end

## `TENSOR_LOAD` is what turns the Agentica read on, so a misspelled name in CI
## fails here instead of testing the native path twice.
def test_the_agentica_read_follows_tensor_load() -> Result[Void]
  Test.eq(FileRange.agentica_read?(), System.env("TENSOR_LOAD") == ok("agentica"))
end

## A zero beside dimensions whose product overflows an Int is an empty tensor,
## not a crash.
def test_a_shape_that_overflows_an_int_is_no_crash() -> Result[Void]
  with_file do |path|
    for b in backends()
      _ = Tensor.from_file(path, 31, [4294967296, 4294967296, 0], Tensor.FloatFormat.F32, b)
    end

    ok()
  end
end

## With `TENSOR_REQUIRE_XLA` the XLA backend is among those the tests above run on, so they cannot drop it silently.
def test_xla_is_among_the_backends_when_it_is_required() -> Result[Void]
  return ok() if System.env("TENSOR_REQUIRE_XLA") != ok("1")
  Test.assert(backends().include?(XLA), "TENSOR_REQUIRE_XLA is set and the XLA backend is not in backends()")
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

## One F16 range of more than the 1 MiB chunk the native readers widen at a time, from an odd offset: the halves
## 1, 2, 4 and 8 repeat, so a chunk read twice or skipped changes the sum, which stays exact in F32. Only the
## native backends run it: the reference would spend the step budget on 524291 values.
def test_a_range_of_several_chunks_sums_the_same_on_each_native_backend() -> Result[Void]
  count = 524291
  quad = [0, 60, 0, 64, 0, 68, 0, 72]
  tail = [0, 60, 0, 64, 0, 68]
  group = Bytes.concat(List.repeat(try Bytes.from_list(quad), 16384))
  dir = "/tmp/agentica-from-file-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  path = "#{dir}/chunks.bin"
  try File.open(path, "write", 5000) do |f|
    try File.write_bytes(f, try Bytes.from_list([9]), 5000)
    for _ in 0...8
      try File.write_bytes(f, group, 5000)
    end

    try File.write_bytes(f, try Bytes.from_list(tail), 5000)
    ok()
  end

  for b in backends()
    if b != Conform.REF
      t = try Tensor.from_file(path, 1, [count], Tensor.FloatFormat.F16, b)
      total = try Trace.eval([t], ->(xs: List[Tensor.Tensor]) { Tensor.sum(xs[0]) })
      try Test.eq(try floats(total), [1966087.0])
      last = try Tensor.from_file(path, 1 + 2 * 524288, [3], Tensor.FloatFormat.F16, b)
      try Test.eq(try floats(last), [1.0, 2.0, 4.0])
    end
  end

  _ = Path.remove(dir, 5000)
  ok()
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
