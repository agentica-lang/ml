## Half weights on ggml (`Tensor.from_file(..., dtype: F16)`): the load converts F32, F16 and BF16 ranges and refuses a
## value beyond 65504; a compiled graph reads a half weight as the weight of `linear` and the table of `embedding`
## and gives what the F32 weight gives; any other use of one is an error naming the op.
##
## The values are multiples of 1/8, so F16 holds them exactly and the two paths differ by the sum order only. Without the ggml
## module the checks pass by asserting nothing, unless `TENSOR_REQUIRE_GGML` is set (CI sets it).

import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Device.Cpu
VULKAN = Tensor.Device.Vulkan(0)
TOL = 0.00001

## Three padding bytes, then 3 F32 (1.5, -2.25, 0.0) at 3, 5 F16 (1.0, -2.0, 0.5, the smallest subnormal, 65504.0)
## at 15, and 3 BF16 (1.0, -3.0, 0.5) at 25.
BYTES = [
  9, 9, 9, 0, 0, 192, 63, 0, 0, 16, 192, 0, 0, 0, 0, 0, 60, 0, 192, 0, 56, 1, 0, 255, 123, 128, 63, 64, 192, 0, 63,
]

## F32 70000.0, then 1.0: the first is beyond F16.
OVERFLOW = [0, 184, 136, 71, 0, 0, 128, 63]

private

def devices() -> Result[List[Tensor.Device]]
  found = [CPU, VULKAN].select { |d: Tensor.Device| Tensor.available(Tensor.Backend.Ggml(d)) }
  return ok(found) if !found.empty?
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok([]) if !flag.ok?
  return ok([]) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end

def with_file(bytes: List[Int], body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-half-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  path = "#{dir}/data.bin"
  result = within(path, bytes, body)
  _ = Path.remove(dir, 5000)
  result
end

def within(path: String, bytes: List[Int], body: Fn[String, Result[Void]]) -> Result[Void]
  try File.open(path, "write", 5000) do |f|
    try File.write_bytes(f, try Bytes.from_list(bytes), 5000)
    ok()
  end

  body.call(path)
end

def floats(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(x) -> out = out.push(x)
      _ -> return error("a float tensor holds #{v}")
    end
  end

  ok(out)
end

## The little-endian bytes of the F32 `x`.
def f32_bytes(x: Float) -> List[Int]
  bits = Float.to_bits32(x)
  [bits % 256, (bits / 256) % 256, (bits / 65536) % 256, bits / 16777216]
end

## 12 weights, each a multiple of 1/8, as F32 bytes.
def weight_bytes() -> List[Int]
  out: List[Int] = []
  for i in 0...12
    out = out + f32_bytes(((i * 5) % 11 - 5).to_f / 8.0)
  end

  out
end

def run_model(path: String, device: Tensor.Device, dtype: Tensor.Dtype) -> Result[List[Tensor.Tensor]]
  backend = Tensor.Backend.Ggml(device)
  w = try Tensor.from_file(path, 0, [4, 3], Tensor.FloatFormat.F32, backend, dtype)
  x = try Tensor.from_list(
    [1.0, -0.5, 0.25, 2.0, 0.125, -1.0].map { |v: Float| Tensor.Values.Float(v) },
    [2, 3],
    Tensor.Dtype.F32,
    backend,
  )

  ids = try Tensor.from_list([3, 0, 2].map { |v: Int| Tensor.Values.Int(v) }, [3], Tensor.Dtype.I32, backend)
  Trace.eval_many([x, ids, w]) do |a|
    ok([try Tensor.linear(a[0], a[2], none, true), try Tensor.embedding(a[1], a[2])])
  end
end

def refused(r: Result[Tensor.Tensor], text: String) -> Result[Void]
  case r
    ok(_) -> error("ran where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

def test_a_half_load_converts_each_format_and_refuses_a_value_beyond_f16() -> Result[Void]
  for device in try devices()
    backend = Tensor.Backend.Ggml(device)
    try with_file(BYTES) do |path|
      f32 = try Tensor.from_file(path, 3, [3, 1], Tensor.FloatFormat.F32, backend, Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(f32), Tensor.Dtype.F16)
      try Test.eq(try floats(f32), [1.5, -2.25, 0.0])
      f16 = try Tensor.from_file(path, 15, [5], Tensor.FloatFormat.F16, backend, Tensor.Dtype.F16)
      try Test.eq(try floats(f16), [1.0, -2.0, 0.5, 1.0 / 16777216.0, 65504.0])
      bf16 = try Tensor.from_file(path, 25, [1, 3], Tensor.FloatFormat.BF16, backend, Tensor.Dtype.F16)
      try Test.eq(try floats(bf16), [1.0, -3.0, 0.5])
      ok()
    end

    try with_file(OVERFLOW) do |path|
      try refused(
        Tensor.from_file(path, 0, [2], Tensor.FloatFormat.F32, backend, Tensor.Dtype.F16),
        "outside what F16 holds",
      )

      ok()
    end
  end

  ok()
end

def test_only_ggml_holds_a_half_weight() -> Result[Void]
  try with_file(BYTES) do |path|
    try refused(
      Tensor.from_file(path, 3, [3], Tensor.FloatFormat.F32, Tensor.Backend.Reference, Tensor.Dtype.F16),
      "ggml",
    )

    try refused(
      Tensor.from_file(path, 3, [3], Tensor.FloatFormat.F32, Tensor.Backend.Reference, Tensor.Dtype.I32),
      "F32 or F16",
    )

    ok()
  end

  ok()
end

def test_a_compiled_graph_reads_a_half_weight_as_the_f32_weight() -> Result[Void]
  for device in try devices()
    try with_file(weight_bytes()) do |path|
      half = try run_model(path, device, Tensor.Dtype.F16)
      whole = try run_model(path, device, Tensor.Dtype.F32)
      for i in 0...2
        try Test.eq(Tensor.shape(half[i]), Tensor.shape(whole[i]))
        try Test.eq(Tensor.dtype(half[i]), Tensor.Dtype.F32)
        a = try floats(half[i])
        b = try floats(whole[i])
        for j in 0...a.size
          try Test.assert(
            (a[j] - b[j]).abs < TOL,
            "output #{i} value #{j} is #{a[j]} with a half weight, #{b[j]} with F32",
          )
        end
      end

      ok()
    end
  end

  ok()
end

def test_a_half_weight_anywhere_else_is_an_error_naming_the_op() -> Result[Void]
  for device in try devices()
    backend = Tensor.Backend.Ggml(device)
    try with_file(weight_bytes()) do |path|
      w = try Tensor.from_file(path, 0, [4, 3], Tensor.FloatFormat.F32, backend, Tensor.Dtype.F16)
      try refused(Trace.eval([w, w]) do |a| Tensor.add(a[0], a[1]) end, "add")
      try refused(Trace.eval([w]) do |a| Tensor.transpose(a[0]) end, "F16")
      x = try Tensor.from_list(
        [1.0, 2.0, 3.0, 4.0].map { |v: Float| Tensor.Values.Float(v) },
        [1, 4],
        Tensor.Dtype.F32,
        backend,
      )

      try refused(Trace.eval([x, w]) do |a| Tensor.matmul(a[1], a[0]) end, "matmul")
      ok()
    end
  end

  ok()
end
