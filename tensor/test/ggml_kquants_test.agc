## Q4_K and Q6_K weights (`Tensor.from_file(..., format: Q4_K | Q6_K)`): ggml's blocks of 256 values, read like Q8_0's
## (`ggml_q8_test.agc`). The blocks are random bytes and the values they dequantize to come from the official `gguf`
## package (`hub/test/kquants_golden.py`, `fixtures/kquants.json`): every backend that reads a file dequantizes them to
## `F32` within float32 rounding of the peer's values (ggml's own readback of a kept weight too), and on ggml a weight
## keeps its blocks as the weight of `linear` (with `out_in`) and the table of `embedding`. Any other use of one, and a
## gradient for one, is an error. ggml rounds the activations of a product to Q8_K blocks of 256 values (Vulkan to F16
## as well), so a product is checked at half a step of that rounding against the dequantized `F32` weight. Without the
## ggml module the ggml checks pass by asserting nothing, unless `TENSOR_REQUIRE_GGML` is set (CI sets it).

import Files from "ggml_file_support"
import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Device.Cpu
VULKAN = Tensor.Device.Vulkan(0)
ROWS = 2
COLS = 512
## The values of the blocks stay under 16 and float32 holds them to 2^-20 at most: the readback of the peer's float32 and
## ours agree to this many units of absolute error.
TOL = 0.000004
## One F16 mantissa step: the relative error of a Vulkan product that rounds to F16.
STEP = 0.00048828125

struct Block
  shape: List[Int]
  hex: String
  bits: List[Int]
end

struct Golden
  q4_k: Block
  q6_k: Block
end

## One type: its format and dtype, and the blocks and values of the peer.
struct Kind
  name: String
  format: Tensor.FloatFormat
  dtype: Tensor.Dtype
  block: Block
end

private

def golden() -> Result[Golden] = JSON.decode(try File.read_text("test/fixtures/kquants.json"), Golden)

def kinds() -> Result[List[Kind]]
  g = try golden()
  ok([
    Kind(name: "Q4_K", format: Tensor.FloatFormat.Q4_K, dtype: Tensor.Dtype.Q4_K, block: g.q4_k),
    Kind(name: "Q6_K", format: Tensor.FloatFormat.Q6_K, dtype: Tensor.Dtype.Q6_K, block: g.q6_k),
  ])
end

def devices() -> Result[List[Tensor.Device]] = Files.devices([CPU, VULKAN])

## Three padding bytes, then the blocks of the peer.
def file_of(k: Kind) -> Result[Bytes]
  ok(Bytes.concat([Bytes.from_list([9, 9, 9]).unwrap!, try OpenSSL.Digest.hexdecode(k.block.hex)]))
end

def with_kind(k: Kind, body: Fn[String, Result[Void]]) -> Result[Void] = Files.with_file("kq", try file_of(k), body)

## The values the peer dequantized to.
def expected(k: Kind) -> List[Float] = k.block.bits.map { |b: Int| Float32.from_bits(b.to_uint32!).to_f }

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

def close_to(got: List[Float], want: List[Float], what: String) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    try Test.assert((got[i] - want[i]).abs <= TOL, "#{what} value #{i} is #{got[i]}, the peer's is #{want[i]}")
  end

  ok()
end

def refused(r: Result[Tensor.Tensor], text: String) -> Result[Void]
  case r
    ok(_) -> error("ran where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

def backends() -> Result[List[Tensor.Backend]]
  out = [Tensor.Backend.Reference]
  if Tensor.available(Tensor.Backend.Torch(CPU))
    out = out.push(Tensor.Backend.Torch(CPU))
  end

  if Tensor.available(Tensor.Backend.Xla(CPU))
    out = out.push(Tensor.Backend.Xla(CPU))
  end

  for device in try devices()
    out = out.push(Tensor.Backend.Ggml(device))
  end

  ok(out)
end

## Two rows of 512 activations: multiples of 1/8 and 1/4, and a large one in each row.
def activations() -> List[Float]
  first = (0...COLS).map { |i: Int| ((i * 5) % 11 - 5).to_f / 8.0 }
  second = (0...COLS).map { |i: Int| ((i * 3) % 7 - 3).to_f / 4.0 }
  first.take(100) + [6.0] + first.drop(101) + second.take(300) + [-5.0] + second.drop(301)
end

## Row `r` of a `[ROWS, COLS]` weight.
def row(values: List[Float], r: Int) -> List[Float] = (0...COLS).map { |i: Int| values[r * COLS + i] }

def test_a_k_quant_file_loads_as_f32_on_every_backend() -> Result[Void]
  for k in try kinds()
    try with_kind(k) do |path|
      for backend in try backends()
        w = try Tensor.from_file(path, 3, [ROWS, COLS], k.format, backend)
        try Test.eq(Tensor.dtype(w), Tensor.Dtype.F32)
        try close_to(try floats(w), expected(k), "#{k.name} on backend #{backend}")
      end

      ok()
    end
  end

  ok()
end

def test_a_k_quant_weight_keeps_its_blocks_and_serves_linear_and_embedding() -> Result[Void]
  xs = activations()
  for k in try kinds()
    for device in try devices()
      backend = Tensor.Backend.Ggml(device)
      try with_kind(k) do |path|
        kept = try Tensor.from_file(path, 3, [ROWS, COLS], k.format, backend, k.dtype)
        whole = try Tensor.from_file(path, 3, [ROWS, COLS], k.format, backend)
        try Test.eq(Tensor.dtype(kept), k.dtype)
        try Test.eq(Tensor.shape(kept), [ROWS, COLS])
        try close_to(try floats(kept), expected(k), "kept #{k.name}")

        x = try Tensor.from_list(xs.map { |v: Float| Tensor.Values.Float(v) }, [2, COLS], Tensor.Dtype.F32, backend)
        ids = try Tensor.from_list([1, 0, 1].map { |v: Int| Tensor.Values.Int(v) }, [3], Tensor.Dtype.I32, backend)
        run = try Trace.eval_many([x, ids, kept, whole]) do |a|
          rows = try Tensor.embedding(a[1], a[2])
          ok([
            try Tensor.linear(a[0], a[2], none, true),
            try Tensor.linear(a[0], a[3], none, true),
            try Tensor.to_dtype(rows, Tensor.Dtype.F32),
            try Tensor.embedding(a[1], a[3]),
          ])
        end

        got = try floats(run[0])
        want = try floats(run[1])
        w = try floats(whole)
        step = 0.0
        if device != CPU
          step = STEP
        end

        for r in 0...2
          for j in 0...ROWS
            bound = 0.000001
            for b in 0...(COLS / 256)
              amax = 0.0
              for i in 0...256
                amax = [amax, xs[r * COLS + b * 256 + i].abs].max.unwrap!
              end

              for i in 0...256
                bound += w[j * COLS + b * 256 + i].abs * amax * (1.0 / 254.0)
              end
            end

            for i in 0...COLS
              bound += (step + 0.000001) * (xs[r * COLS + i] * w[j * COLS + i]).abs
            end

            try Test.assert(
              (got[r * ROWS + j] - want[r * ROWS + j]).abs <= bound,
              "linear [#{r}, #{j}] is #{got[r * ROWS + j]} with a #{k.name} weight, #{want[r * ROWS + j]} with F32 (bound #{bound})",
            )
          end
        end

        # embedding: ggml widens the rows of the quantized table
        try Test.eq(try floats(run[2]), try floats(run[3]))
        try close_to(try floats(run[3]), row(w, 1) + row(w, 0) + row(w, 1), "embedding")
        ok()
      end
    end
  end

  ok()
end

def test_a_k_quant_weight_anywhere_else_is_an_error() -> Result[Void]
  for k in try kinds()
    try with_kind(k) do |path|
      name = k.name
      try refused(Tensor.from_file(path, 3, [ROWS, COLS], k.format, Tensor.Backend.Reference, k.dtype), "ggml")
      try refused(Tensor.from_file(path, 3, [ROWS, COLS], Tensor.FloatFormat.Q8_0, Tensor.Backend.Reference), "past")
      for device in try devices()
        backend = Tensor.Backend.Ggml(device)
        try refused(Tensor.from_file(path, 3, [ROWS, COLS], Tensor.FloatFormat.F32, backend, k.dtype), "#{name} file")

        try refused(Tensor.from_file(path, 3, [ROWS, COLS], k.format, backend, Tensor.Dtype.F16), "not as F16")
        try refused(Tensor.from_file(path, 3, [ROWS * COLS], k.format, backend, k.dtype), "rank-2")
        try refused(Tensor.from_file(path, 3, [ROWS * 2, COLS], k.format, backend, k.dtype), "past")
        try refused(Tensor.from_file(path, 3, [ROWS, 33], k.format, backend), "blocks of 256")
        try refused(Tensor.from_file(path, 3, [ROWS, 255], k.format, backend, k.dtype), "blocks of 256")
        w = try Tensor.from_file(path, 3, [ROWS, COLS], k.format, backend, k.dtype)
        x = try Tensor.from_list(
          (0...COLS).map { |i: Int| Tensor.Values.Float(i.to_f) },
          [1, COLS],
          Tensor.Dtype.F32,
          backend,
        )

        try refused(Trace.eval([w, w]) do |a| Tensor.add(a[0], a[1]) end, "add")
        try refused(Trace.eval([w]) do |a| Tensor.transpose(a[0]) end, "output 0")
        narrow = try Tensor.from_list(
          (0...ROWS).map { |i: Int| Tensor.Values.Float(i.to_f) },
          [1, ROWS],
          Tensor.Dtype.F32,
          backend,
        )

        try refused(Trace.eval([narrow, w]) do |a| Tensor.linear(a[0], a[1], none, false) end, "out_in")
        asked = Trace.value_and_grad([x, w]) do |a| Tensor.sum(try Tensor.linear(a[0], a[1], none, true)) end
        case asked
          ok(_) -> return error("a gradient was taken for a #{name} weight")
          error(e) -> try Test.assert(e.message.include?("quantized weight"), "the error said \"#{e.message}\"")
        end

        frozen = Trace.value_and_grad([x]) do |a|
          Tensor.sum(try Tensor.linear(a[0], Trace.constant(w), none, true))
        end

        case frozen
          ok(_) -> return error("a gradient was taken through a #{name} weight")
          error(e) ->
            try Test.assert(e.message.include?("#{name} weight is not supported"), "the error said \"#{e.message}\"")
        end
      end

      ok()
    end
  end

  ok()
end

## A quantized tensor comes only from `from_file`: no creation or `to_dtype` makes one, on any backend.
def test_only_from_file_makes_a_k_quant_tensor() -> Result[Void]
  for dtype in [Tensor.Dtype.Q4_K, Tensor.Dtype.Q6_K]
    for backend in try backends()
      try refused(Tensor.zeros([256], dtype, backend), "from_file")
      ones = try Tensor.floats([1.0], [1], backend)
      try refused(Tensor.to_dtype(ones, dtype), "from_file")
    end
  end

  ok()
end
