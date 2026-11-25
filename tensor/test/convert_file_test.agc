## `Tensor.convert_file`: a range of floats of one file written as another format into another file, in C and in
## chunks on several threads. The expected bytes come from Python (`struct` "e" rounds to nearest even; bfloat16 is the
## top half of the binary32 after the round-to-nearest-even add), and a range of several chunks equals the small one.
##
## Without the ggml module the checks pass by asserting nothing, unless `TENSOR_REQUIRE_GGML` is set (CI sets it).

import Binary from "std/binary"
import Tensor from "../tensor"

F32 = Tensor.FloatFormat.F32
F16 = Tensor.FloatFormat.F16
BF16 = Tensor.FloatFormat.BF16

## Nine values, in F32 after three padding bytes: 1.5, -2.25, 0.0, 3.14159274, 1e-8, 65504.0, 0.1, -1e-5 and
## 1.00390625 (a tie for BF16, which rounds to the even 1.0).
SRC = [
  9, 9, 9, 0, 0, 192, 63, 0, 0, 16, 192, 0, 0, 0, 0, 219, 15, 73, 64, 119, 204, 43, 50, 0, 224, 127, 71, 205, 204, 204,
  61, 172, 197, 39, 183, 0, 128, 128, 63,
]
AS_F16 = [0, 62, 128, 192, 0, 0, 72, 66, 0, 0, 255, 123, 102, 46, 168, 128, 4, 60]
AS_BF16 = [192, 63, 16, 192, 0, 0, 73, 64, 44, 50, 128, 71, 205, 61, 40, 183, 128, 63]
## The F16 and the BF16 values above, as F32.
F16_AS_F32 = [
  0, 0, 192, 63, 0, 0, 16, 192, 0, 0, 0, 0, 0, 0, 73, 64, 0, 0, 0, 0, 0, 224, 127, 71, 0, 192, 204, 61, 0, 0, 40, 183,
  0, 128, 128, 63,
]
BF16_AS_F32 = [
  0, 0, 192, 63, 0, 0, 16, 192, 0, 0, 0, 0, 0, 0, 73, 64, 0, 0, 44, 50, 0, 0, 128, 71, 0, 0, 205, 61, 0, 0, 40, 183, 0,
  0, 128, 63,
]
F16_AS_BF16 = [192, 63, 16, 192, 0, 0, 73, 64, 0, 0, 128, 71, 205, 61, 40, 183, 128, 63]
## 256 values per block of the long range, 524288 values (`FF_VALUES`) per chunk of C.
BLOCK = 256

private

def devices() -> Result[Bool]
  return ok(true) if Tensor.available(Tensor.Backend.Ggml(Tensor.Device.Cpu))
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok(false) if !flag.ok?
  return ok(false) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end

## Runs `body` with the paths of `src.bin` (holding `data`) and `dst.bin` (five bytes of 7), then removes the folder.
def with_files(data: List[Int], body: Fn[String, String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-convert-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = prepare(dir, data, body)
  _ = Path.remove(dir, 5000)
  result
end

def prepare(dir: String, data: List[Int], body: Fn[String, String, Result[Void]]) -> Result[Void]
  try put("#{dir}/src.bin", try Bytes.from_list(data))
  try put("#{dir}/dst.bin", try Bytes.from_list([7, 7, 7, 7, 7]))
  body.call("#{dir}/src.bin", "#{dir}/dst.bin")
end

## A new file `path` holding `data` (up to 1 MiB).
def put(path: String, data: Bytes) -> Result[Void]
  File.open(path, "write", 5000) do |f|
    try File.write_bytes(f, data, 5000)
    ok()
  end
end

## The bytes of `path` from byte `from` on.
def read_from(path: String, from: Int) -> Result[Bytes]
  File.open(path) do |f|
    try File.seek(f, from)
    parts: List[Bytes] = []
    while true
      part = try File.read_bytes(f, 1048576)
      return ok(Bytes.concat(parts)) if part.length == 0
      parts = parts.push(part)
    end

    ok(Bytes.concat(parts))
  end
end

def tail(path: String, from: Int) -> Result[List[Int]] = ok((try read_from(path, from)).to_a)

## The `count` values at 5 of `dst`, as the bytes `to` gives them (earlier conversions leave longer bytes behind).
def convert(
  src: String,
  from: Int,
  src_format: Tensor.FloatFormat,
  count: Int,
  dst: String,
  to: Tensor.FloatFormat,
) -> Result[List[Int]]
  try Tensor.convert_file(src, from, src_format, count, dst, 5, to)
  width = 2
  if to == F32
    width = 4
  end

  ok((try tail(dst, 5)).take(count * width))
end

def test_every_pair_of_formats_gives_the_bytes_python_does() -> Result[Void]
  return ok() if !try devices()
  with_files(SRC) do |src, dst|
    try Test.eq(try convert(src, 3, F32, 9, dst, F32), SRC.drop(3))
    try Test.eq(try convert(src, 3, F32, 9, dst, F16), AS_F16)
    try Test.eq(try convert(src, 3, F32, 9, dst, BF16), AS_BF16)
    ok()
  end
end

def test_a_half_range_widens_or_converts() -> Result[Void]
  return ok() if !try devices()
  with_files(AS_F16) do |src, dst|
    try Test.eq(try convert(src, 0, F16, 9, dst, F32), F16_AS_F32)
    try Test.eq(try convert(src, 0, F16, 9, dst, BF16), F16_AS_BF16)
    try Test.eq(try convert(src, 0, F16, 9, dst, F16), AS_F16)
    ok()
  end
end

def test_a_bfloat_range_widens_or_copies() -> Result[Void]
  return ok() if !try devices()
  with_files(AS_BF16) do |src, dst|
    try Test.eq(try convert(src, 0, BF16, 9, dst, F32), BF16_AS_F32)
    try Test.eq(try convert(src, 0, BF16, 9, dst, BF16), AS_BF16)
    ok()
  end
end

## The range may start anywhere in the destination, which grows to hold it, and the bytes before it stay.
def test_the_destination_keeps_what_is_before_and_grows() -> Result[Void]
  return ok() if !try devices()
  with_files(SRC) do |src, dst|
    try Tensor.convert_file(src, 3, F32, 9, dst, 9, F16)
    try Test.eq(try tail(dst, 0), [7, 7, 7, 7, 7, 0, 0, 0, 0].concat(AS_F16))
    ok()
  end
end

def test_a_bad_range_or_value_is_an_error() -> Result[Void]
  return ok() if !try devices()
  try with_files(SRC) do |src, dst|
    try Test.error?(Tensor.convert_file(src, 3, F32, 10, dst, 5, F32))
    try Test.error?(Tensor.convert_file(src, -1, F32, 1, dst, 5, F32))
    try Test.error?(Tensor.convert_file(src, 3, F32, 1, dst, -1, F32))
    try Test.error?(Tensor.convert_file(src, 3, F32, -1, dst, 5, F32))
    try Test.error?(Tensor.convert_file("#{src}.missing", 0, F32, 1, dst, 5, F32))
    try Test.error?(Tensor.convert_file(src, 3, F32, 1, "#{dst}.missing", 5, F32))
    try Test.error?(Tensor.convert_file(src, 3, Tensor.FloatFormat.Q8_0, 32, dst, 5, F32))
    try Test.error?(Tensor.convert_file(src, 3, F32, 1, dst, 5, Tensor.FloatFormat.Q8_0))
    ok()
  end

  # 65519.0 is below the half of a step over 65504 that F16 rounds to, yet it is beyond what F16 holds
  with_files([0, 224, 127, 71, 0, 239, 127, 71]) do |src, dst|
    try Test.eq(try convert(src, 0, F32, 1, dst, F16), [255, 123])
    try Test.error?(Tensor.convert_file(src, 0, F32, 2, dst, 5, F16))
    Test.eq(try convert(src, 0, F32, 2, dst, BF16), [128, 71, 128, 71])
  end
end

## A NaN stays a NaN in F32 (the bytes are copied), F16 and BF16, whatever its sign.
def test_a_nan_stays_a_nan() -> Result[Void]
  return ok() if !try devices()
  with_files([0, 0, 192, 127, 0, 0, 192, 255]) do |src, dst|
    try Test.eq(try convert(src, 0, F32, 2, dst, F32), [0, 0, 192, 127, 0, 0, 192, 255])
    half = try convert(src, 0, F32, 2, dst, F16)
    try Test.assert(nan_half?(half[0], half[1]) and nan_half?(half[2], half[3]), "F16 gave #{half}")
    brain = try convert(src, 0, F32, 2, dst, BF16)
    Test.assert(nan_bfloat?(brain[0], brain[1]) and nan_bfloat?(brain[2], brain[3]), "BF16 gave #{brain}")
  end
end

## Whether the F16 of the bytes `low` and `high` is a NaN: all exponent bits and some mantissa bit.
def nan_half?(low: Int, high: Int) -> Bool = high % 128 >= 124 and (high % 4 != 0 or low != 0)

## The same for a BF16.
def nan_bfloat?(low: Int, high: Int) -> Bool = high % 128 == 127 and low >= 128 and low % 128 != 0

def f32_block() -> Result[Bytes]
  parts: List[Bytes] = []
  for i in 0...BLOCK
    parts = parts.push(Binary.write_f32_le(i.to_float32))
  end

  ok(Bytes.concat(parts))
end

## A range of three chunks of C, run on several threads, is the small range repeated: the integers below 256 are exact
## in F16 and in BF16, so the block converted alone is what every block of the long range must be.
def test_a_range_of_several_chunks_equals_the_small_one() -> Result[Void]
  return ok() if !try devices()
  block = try f32_block()
  blocks = 6144
  dir = "/tmp/agentica-convert-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = long_range(dir, block, blocks)
  _ = Path.remove(dir, 5000)
  result
end

def long_range(dir: String, block: Bytes, blocks: Int) -> Result[Void]
  src = "#{dir}/src.bin"
  try File.open(src, "write", 5000) do |f|
    for _ in 0...blocks
      try File.write_bytes(f, block, 5000)
    end

    ok()
  end

  small = "#{dir}/small.bin"
  try put(small, Bytes.concat([]))
  for format in [F16, BF16]
    dst = "#{dir}/#{format}.bin"
    try put(dst, Bytes.concat([]))
    try Tensor.convert_file(src, 0, F32, BLOCK, small, 0, format)
    want = try read_from(small, 0)
    try Test.eq(want.length, BLOCK * 2)
    try Tensor.convert_file(src, 0, F32, BLOCK * blocks, dst, 0, format)
    got = try read_from(dst, 0)
    try Test.eq(got.length, BLOCK * blocks * 2)
    for i in 0...blocks
      try Test.assert(got.slice(i * BLOCK * 2, BLOCK * 2) == want, "block #{i} of #{format} differs")
    end
  end

  ok()
end
