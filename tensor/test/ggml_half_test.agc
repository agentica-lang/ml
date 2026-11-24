## Half weights on ggml (`Tensor.from_file(..., dtype: F16)`): the load converts F32, F16 and BF16 ranges and refuses a
## value beyond 65504, naming its element; a compiled graph reads a half weight as the weight of `linear` and the table of
## `embedding`; any other use of one, and a gradient for one, is an error. An `F16` table still gathers on the reference.
##
## A backend may round the activations of a half-weight product to F16 (ggml's Vulkan does with `vulkan_fp16` on; its CPU
## kernels did not here), so the activations here are checked at 2^-11 of the products, and the weights are multiples
## of 1/8, which F16 holds exactly. Without the ggml
## module the checks pass by asserting nothing, unless `TENSOR_REQUIRE_GGML` is set (CI sets it).

import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Device.Cpu
VULKAN = Tensor.Device.Vulkan(0)
## One F16 mantissa step: the relative error of rounding an activation to F16.
STEP = 0.00048828125
## The load reads this many values at a time (`FF_VALUES` of `native/float_file.h`).
CHUNK = 524288

## Three padding bytes, then 3 F32 (1.5, -2.25, 0.0) at 3, 5 F16 (1.0, -2.0, 0.5, the smallest subnormal, 65504.0)
## at 15, and 3 BF16 (1.0, -3.0, 0.5) at 25.
BYTES = [
  9, 9, 9, 0, 0, 192, 63, 0, 0, 16, 192, 0, 0, 0, 0, 0, 60, 0, 192, 0, 56, 1, 0, 255, 123, 128, 63, 64, 192, 0, 63,
]

## F32 65504.0 and -65504.0, the largest values F16 holds.
LIMITS = [0, 224, 127, 71, 0, 224, 127, 199]

private

def devices() -> Result[List[Tensor.Device]]
  found = [CPU, VULKAN].select { |d: Tensor.Device| Tensor.available(Tensor.Backend.Ggml(d)) }
  return ok(found) if !found.empty?
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok([]) if !flag.ok?
  return ok([]) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end

def with_file(data: List[Bytes], body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-half-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  path = "#{dir}/data.bin"
  result = within(path, data, body)
  _ = Path.remove(dir, 5000)
  result
end

## A write takes at most 1 MiB, so the file is written in parts.
def within(path: String, data: List[Bytes], body: Fn[String, Result[Void]]) -> Result[Void]
  try File.open(path, "write", 5000) do |f|
    for part in data
      try File.write_bytes(f, part, 5000)
    end

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

## 12 weights, each a multiple of 1/8: row `j` of the `[4, 3]` weight is `weights()[3 j...3 j + 3]`.
def weights() -> List[Float]
  out: List[Float] = []
  for i in 0...12
    out = out.push(((i * 5) % 11 - 5).to_f / 8.0)
  end

  out
end

def weight_bytes() -> Result[List[Bytes]] = ok([try Bytes.from_list(weights().flat_map { |x: Float| f32_bytes(x) })])

def bytes_of(values: List[Int]) -> Result[List[Bytes]] = ok([try Bytes.from_list(values)])

## `count` zero F32 values (a power of two, at least 262144), in parts of 1 MiB, then the F32 `tail`.
def zeros_then(count: Int, tail: List[Int]) -> Result[List[Bytes]]
  block = try Bytes.from_list([0, 0, 0, 0])
  for _ in 0...18
    block = Bytes.concat([block, block])
  end

  parts: List[Bytes] = []
  for _ in 0...(count / 262144)
    parts = parts.push(block)
  end

  ok(parts.push(try Bytes.from_list(tail)))
end

def run_model(path: String, device: Tensor.Device, dtype: Tensor.Dtype, xs: List[Float]) -> Result[List[Tensor.Tensor]]
  backend = Tensor.Backend.Ggml(device)
  w = try Tensor.from_file(path, 0, [4, 3], Tensor.FloatFormat.F32, backend, dtype)
  x = try Tensor.from_list(xs.map { |v: Float| Tensor.Values.Float(v) }, [2, 3], Tensor.Dtype.F32, backend)
  ids = try Tensor.from_list([3, 0, 2].map { |v: Int| Tensor.Values.Int(v) }, [3], Tensor.Dtype.I32, backend)
  Trace.eval_many([x, ids, w]) do |a|
    rows = try Tensor.embedding(a[1], a[2])
    ok([try Tensor.linear(a[0], a[2], none, true), try Tensor.to_dtype(rows, Tensor.Dtype.F32)])
  end
end

def refused(r: Result[Tensor.Tensor], text: String) -> Result[Void]
  case r
    ok(_) -> error("ran where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

## Loads the `count` values of `format` in `data` as a half tensor and expects an error that names `element`.
def overflows(
  data: List[Bytes],
  format: Tensor.FloatFormat,
  count: Int,
  element: Int,
  backend: Tensor.Backend,
) -> Result[Void]
  with_file(data) do |path|
    try refused(Tensor.from_file(path, 0, [count], format, backend, Tensor.Dtype.F16), "outside what F16 holds")
    try refused(Tensor.from_file(path, 0, [count], format, backend, Tensor.Dtype.F16), "element #{element} ")
    ok()
  end
end

def test_a_half_load_converts_each_format() -> Result[Void]
  for device in try devices()
    backend = Tensor.Backend.Ggml(device)
    try with_file(try bytes_of(BYTES)) do |path|
      f32 = try Tensor.from_file(path, 3, [3, 1], Tensor.FloatFormat.F32, backend, Tensor.Dtype.F16)
      try Test.eq(Tensor.dtype(f32), Tensor.Dtype.F16)
      try Test.eq(try floats(f32), [1.5, -2.25, 0.0])
      f16 = try Tensor.from_file(path, 15, [5], Tensor.FloatFormat.F16, backend, Tensor.Dtype.F16)
      try Test.eq(try floats(f16), [1.0, -2.0, 0.5, 1.0 / 16777216.0, 65504.0])
      bf16 = try Tensor.from_file(path, 25, [1, 3], Tensor.FloatFormat.BF16, backend, Tensor.Dtype.F16)
      try Test.eq(try floats(bf16), [1.0, -3.0, 0.5])
      ok()
    end

    try with_file(try bytes_of(LIMITS)) do |path|
      limits = try Tensor.from_file(path, 0, [2], Tensor.FloatFormat.F32, backend, Tensor.Dtype.F16)
      try Test.eq(try floats(limits), [65504.0, -65504.0])
      ok()
    end
  end

  ok()
end

def test_a_value_beyond_f16_is_an_error_naming_its_element() -> Result[Void]
  for device in try devices()
    backend = Tensor.Backend.Ggml(device)
    # 70000.0 at element 1, after 1.0
    try overflows(try bytes_of([0, 0, 128, 63, 0, 184, 136, 71]), Tensor.FloatFormat.F32, 2, 1, backend)
    # -70000.0 at element 1, after 0.5
    try overflows(try bytes_of([0, 0, 0, 63, 0, 184, 136, 199]), Tensor.FloatFormat.F32, 2, 1, backend)
    # 65505.0, one past the largest F16
    try overflows(try bytes_of([0, 0, 128, 63, 0, 225, 127, 71]), Tensor.FloatFormat.F32, 2, 1, backend)
    # BF16 1.0, then 99840.0
    try overflows(try bytes_of([128, 63, 195, 71]), Tensor.FloatFormat.BF16, 2, 1, backend)
    # the first value of the second chunk
    try overflows(try zeros_then(CHUNK, [0, 184, 136, 71]), Tensor.FloatFormat.F32, CHUNK + 1, CHUNK, backend)
  end

  ok()
end

def test_only_ggml_holds_a_half_weight() -> Result[Void]
  try with_file(try bytes_of(BYTES)) do |path|
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
  w = weights()
  exact = [1.0, -0.5, 0.25, 2.0, 0.125, -1.0]
  thirds = [1.0 / 3.0, -2.0 / 3.0, 0.1, 3.7, -0.9, 0.0123]
  for device in try devices()
    try with_file(try weight_bytes()) do |path|
      for xs in [exact, thirds]
        half = try run_model(path, device, Tensor.Dtype.F16, xs)
        whole = try run_model(path, device, Tensor.Dtype.F32, xs)
        for i in 0...2
          try Test.eq(Tensor.shape(half[i]), Tensor.shape(whole[i]))
          try Test.eq(Tensor.dtype(half[i]), Tensor.Dtype.F32)
        end

        # linear: the activations may round to F16, so each product is good to 2^-11; ggml's CPU backend keeps
        # them F32, so there only the order of the sums differs
        step = STEP
        if device == CPU
          step = 0.0
        end

        a = try floats(half[0])
        b = try floats(whole[0])
        for r in 0...2
          for j in 0...4
            bound = 0.000001
            for k in 0...3
              bound += step * (xs[r * 3 + k] * w[j * 3 + k]).abs
            end

            try Test.assert(
              (a[r * 4 + j] - b[r * 4 + j]).abs <= bound,
              "linear [#{r}, #{j}] is #{a[r * 4 + j]} with a half weight, #{b[r * 4 + j]} with F32",
            )
          end
        end

        # embedding: the rows are the weights themselves, exact
        try Test.eq(try floats(half[1]), try floats(whole[1]))
      end

      ok()
    end
  end

  ok()
end

def test_a_half_weight_anywhere_else_is_an_error_naming_the_op() -> Result[Void]
  for device in try devices()
    backend = Tensor.Backend.Ggml(device)
    try with_file(try weight_bytes()) do |path|
      w = try Tensor.from_file(path, 0, [4, 3], Tensor.FloatFormat.F32, backend, Tensor.Dtype.F16)
      try refused(Trace.eval([w, w]) do |a| Tensor.add(a[0], a[1]) end, "add")
      try refused(Trace.eval([w]) do |a| Tensor.transpose(a[0]) end, "output 0")
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

def test_no_gradient_is_taken_for_a_half_weight() -> Result[Void]
  for device in try devices()
    backend = Tensor.Backend.Ggml(device)
    try with_file(try weight_bytes()) do |path|
      w = try Tensor.from_file(path, 0, [4, 3], Tensor.FloatFormat.F32, backend, Tensor.Dtype.F16)
      x = try Tensor.from_list(
        [1.0, 2.0, 3.0].map { |v: Float| Tensor.Values.Float(v) },
        [1, 3],
        Tensor.Dtype.F32,
        backend,
      )

      asked = Trace.value_and_grad([x, w]) do |a| Tensor.sum(try Tensor.linear(a[0], a[1], none, true)) end
      case asked
        ok(_) -> return error("a gradient was taken for a half weight")
        error(e) -> try Test.assert(e.message.include?("half weight"), "the error said \"#{e.message}\"")
      end

      # a weight that the output does not reach, whose gradient is a fill of zeros
      unreached = Trace.eval_many([x, w]) do |a|
        out = try Tensor.sum(a[0])
        Trace.gradients(out, [a[0], a[1]])
      end

      case unreached
        ok(_) -> return error("a gradient was taken for a half weight")
        error(e) -> try Test.assert(e.message.include?("no gradient is taken"), "the error said \"#{e.message}\"")
      end

      ok()
    end
  end

  ok()
end

def test_an_f16_table_still_gathers_on_the_reference() -> Result[Void]
  table = try Tensor.from_list(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0].map { |v: Float| Tensor.Values.Float(v) },
    [3, 2],
    Tensor.Dtype.F16,
    Tensor.Backend.Reference,
  )

  ids = try Tensor.from_list(
    [2, 0].map { |v: Int| Tensor.Values.Int(v) },
    [2],
    Tensor.Dtype.I32,
    Tensor.Backend.Reference,
  )

  rows = try Trace.eval([ids, table]) do |a| Tensor.embedding(a[0], a[1]) end
  try Test.eq(Tensor.dtype(rows), Tensor.Dtype.F16)
  Test.eq(try floats(rows), [5.0, 6.0, 1.0, 2.0])
end
