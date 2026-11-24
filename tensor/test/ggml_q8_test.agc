## Q8_0 weights (`Tensor.from_file(..., format: Q8_0)`): a Q8_0 range, ggml's blocks of 32 values (a binary16 scale and 32
## signed bytes), loads as `F32` on every backend (dequantized as ggml does: the scale times the byte), or with
## `dtype: Q8_0` on ggml it keeps its blocks, as the weight of `linear` (with `out_in`) and the table of `embedding`. Any
## other use of one, and a gradient for one, is an error.
##
## ggml's `mul_mat` rounds the activations to Q8_0 blocks (and Vulkan to F16 as well), so the product of a quantized weight
## is checked at the half step of that rounding plus the rounding of its scale to F16, `amax * (1 / 254 + 2^-11)` per
## activation of a block, against the dequantized `F32`
## weight, whose rows are exact. Without the ggml module the ggml checks pass by asserting nothing, unless
## `TENSOR_REQUIRE_GGML` is set (CI sets it).

import Tensor from "../tensor"
import Trace from "../trace"

CPU = Tensor.Device.Cpu
VULKAN = Tensor.Device.Vulkan(0)
Q8_0 = Tensor.FloatFormat.Q8_0
ROWS = 4
COLS = 64
## One F16 mantissa step: the relative error of a Vulkan product that rounds to F16.
STEP = 0.00048828125
## The block scale `d = amax / 127` of an activation is stored as F16: a value of up to 127 steps is off by 127 * d * 2^-11
## = amax * 2^-11, on top of the half step.
SCALE_STEP = 0.00048828125

## The binary16 bits of each block's scale, all exact: 0.5, 0.25, 1.0, -0.75, 0.0625, 0.125, 2^-8, 0.03125.
SCALES = [0x3800, 0x3400, 0x3C00, 0xBA00, 0x2C00, 0x3000, 0x1C00, 0x2800]
VALUES = [0.5, 0.25, 1.0, -0.75, 0.0625, 0.125, 0.00390625, 0.03125]

private

def devices() -> Result[List[Tensor.Device]]
  found = [CPU, VULKAN].select { |d: Tensor.Device| Tensor.available(Tensor.Backend.Ggml(d)) }
  return ok(found) if !found.empty?
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok([]) if !flag.ok?
  return ok([]) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end

def with_file(data: Bytes, body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-q8-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  path = "#{dir}/data.bin"
  try File.open(path, "write", 5000) do |f|
    try File.write_bytes(f, data, 5000)
    ok()
  end

  result = body.call(path)
  _ = Path.remove(dir, 5000)
  result
end

## The byte of value `j` of block `b`: every signed byte from -127 to 127, and -128 once.
def byte_of(b: Int, j: Int) -> Int
  v = (j * 7 + b * 13) % 255 - 127
  return -128 if b == 3 and j == 5
  v
end

## Three padding bytes, then the 8 blocks of the `[4, 64]` weight: each a scale and 32 bytes.
def file_bytes() -> Result[Bytes]
  out = [9, 9, 9]
  for b in 0...8
    out = out.push(SCALES[b] % 256)
    out = out.push(SCALES[b] / 256)
    for j in 0...32
      out = out.push((byte_of(b, j) + 256) % 256)
    end
  end

  Bytes.from_list(out)
end

## What the file holds, row-major: the scale times the byte.
def expected() -> List[Float]
  out: List[Float] = []
  for b in 0...8
    for j in 0...32
      out = out.push(VALUES[b] * byte_of(b, j).to_f)
    end
  end

  out
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

def refused(r: Result[Tensor.Tensor], text: String) -> Result[Void]
  case r
    ok(_) -> error("ran where an error naming \"#{text}\" was due")
    error(e) -> Test.assert(e.message.include?(text), "the error said \"#{e.message}\"")
  end
end

## Two rows of 64 activations: a few large ones, then multiples of 1/8 and 1/4.
## Row `r` of a `[4, 64]` weight.
def row(values: List[Float], r: Int) -> List[Float] = (0...COLS).map { |i: Int| values[r * COLS + i] }

def activations() -> List[Float]
  first = (0...56).map { |i: Int| ((i * 5) % 11 - 5).to_f / 8.0 }
  second = (0...64).map { |i: Int| ((i * 3) % 7 - 3).to_f / 4.0 }
  [1.0, -0.5, 0.25, 3.0, 0.125, -2.0, 0.75, 1.5] + first + second
end

def test_a_q8_0_file_loads_as_f32_on_every_backend() -> Result[Void]
  try with_file(try file_bytes()) do |path|
    backends = [Tensor.Backend.Reference]
    if Tensor.available(Tensor.Backend.Torch(CPU))
      backends = backends.push(Tensor.Backend.Torch(CPU))
    end

    for device in try devices()
      backends = backends.push(Tensor.Backend.Ggml(device))
    end

    for backend in backends
      w = try Tensor.from_file(path, 3, [ROWS, COLS], Q8_0, backend)
      try Test.eq(Tensor.dtype(w), Tensor.Dtype.F32)
      try Test.eq(try floats(w), expected())
    end

    ok()
  end

  ok()
end

def test_a_q8_0_weight_keeps_its_blocks_and_serves_linear_and_embedding() -> Result[Void]
  xs = activations()
  for device in try devices()
    backend = Tensor.Backend.Ggml(device)
    try with_file(try file_bytes()) do |path|
      kept = try Tensor.from_file(path, 3, [ROWS, COLS], Q8_0, backend, Tensor.Dtype.Q8_0)
      whole = try Tensor.from_file(path, 3, [ROWS, COLS], Q8_0, backend)
      try Test.eq(Tensor.dtype(kept), Tensor.Dtype.Q8_0)
      try Test.eq(Tensor.shape(kept), [ROWS, COLS])
      try Test.eq(try floats(kept), expected())

      x = try Tensor.from_list(xs.map { |v: Float| Tensor.Values.Float(v) }, [2, COLS], Tensor.Dtype.F32, backend)
      ids = try Tensor.from_list([3, 0, 2].map { |v: Int| Tensor.Values.Int(v) }, [3], Tensor.Dtype.I32, backend)
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
      w = expected()
      step = 0.0
      if device != CPU
        step = STEP
      end

      for r in 0...2
        for j in 0...ROWS
          bound = 0.000001
          for k in 0...COLS
            amax = 0.0
            for i in 0...32
              amax = [amax, xs[r * COLS + (k / 32) * 32 + i].abs].max.unwrap!
            end

            bound += w[j * COLS + k].abs * amax * (1.0 / 254.0 + SCALE_STEP)
            bound += step * (xs[r * COLS + k] * w[j * COLS + k]).abs
          end

          try Test.assert(
            (got[r * ROWS + j] - want[r * ROWS + j]).abs <= bound,
            "linear [#{r}, #{j}] is #{got[r * ROWS + j]} with a Q8_0 weight, #{want[r * ROWS + j]} with F32 (bound #{bound})",
          )
        end
      end

      # embedding: ggml widens the rows of the quantized table exactly
      try Test.eq(try floats(run[2]), try floats(run[3]))
      try Test.eq(try floats(run[3]), row(w, 3) + row(w, 0) + row(w, 2))
      ok()
    end
  end

  ok()
end

def test_a_q8_0_weight_anywhere_else_is_an_error() -> Result[Void]
  try with_file(try file_bytes()) do |path|
    try refused(Tensor.from_file(path, 3, [ROWS, COLS], Q8_0, Tensor.Backend.Reference, Tensor.Dtype.Q8_0), "ggml")
    try refused(Tensor.from_file(path, 3, [ROWS, COLS], Tensor.FloatFormat.F32, Tensor.Backend.Reference), "past")
    for device in try devices()
      backend = Tensor.Backend.Ggml(device)
      try refused(
        Tensor.from_file(path, 3, [ROWS, COLS], Tensor.FloatFormat.F32, backend, Tensor.Dtype.Q8_0),
        "Q8_0 file",
      )

      try refused(Tensor.from_file(path, 3, [ROWS, COLS], Q8_0, backend, Tensor.Dtype.F16), "not as F16")
      try refused(Tensor.from_file(path, 3, [ROWS * COLS], Q8_0, backend, Tensor.Dtype.Q8_0), "rank-2")
      try refused(Tensor.from_file(path, 3, [ROWS * 2, COLS], Q8_0, backend, Tensor.Dtype.Q8_0), "past")
      try refused(Tensor.from_file(path, 3, [ROWS, 33], Q8_0, backend), "blocks of 32")
      try refused(Tensor.from_file(path, 3, [ROWS, 31], Q8_0, backend, Tensor.Dtype.Q8_0), "blocks of 32")
      w = try Tensor.from_file(path, 3, [ROWS, COLS], Q8_0, backend, Tensor.Dtype.Q8_0)
      x = try Tensor.from_list(
        (0...COLS).map { |i: Int| Tensor.Values.Float(i.to_f) },
        [1, COLS],
        Tensor.Dtype.F32,
        backend,
      )

      try refused(Trace.eval([w, w]) do |a| Tensor.add(a[0], a[1]) end, "add")
      try refused(Trace.eval([w]) do |a| Tensor.transpose(a[0]) end, "output 0")
      ids = try Tensor.from_list([1, 3].map { |v: Int| Tensor.Values.Int(v) }, [2], Tensor.Dtype.I32, backend)
      try refused(Trace.eval([ids, w]) do |a| Tensor.embedding(a[0], try Tensor.transpose(a[1])) end, "Q8_0 operand")
      try refused(Trace.eval([w]) do |a| Tensor.transpose(try Tensor.transpose(a[0])) end, "Q8_0 operand")
      narrow = try Tensor.from_list(
        (0...ROWS).map { |i: Int| Tensor.Values.Float(i.to_f) },
        [1, ROWS],
        Tensor.Dtype.F32,
        backend,
      )

      try refused(Trace.eval([narrow, w]) do |a| Tensor.linear(a[0], a[1], none, false) end, "out_in")
      asked = Trace.value_and_grad([x, w]) do |a| Tensor.sum(try Tensor.linear(a[0], a[1], none, true)) end
      case asked
        ok(_) -> return error("a gradient was taken for a Q8_0 weight")
        error(e) -> try Test.assert(e.message.include?("quantized weight"), "the error said \"#{e.message}\"")
      end
    end

    ok()
  end

  ok()
end

## A `Q8_0` tensor comes only from `from_file`: no creation or `to_dtype` makes one, on any backend.
def test_only_from_file_makes_a_q8_0_tensor() -> Result[Void]
  q8 = Tensor.Dtype.Q8_0
  backends = [Tensor.Backend.Reference]
  if Tensor.available(Tensor.Backend.Torch(CPU))
    backends = backends.push(Tensor.Backend.Torch(CPU))
  end

  for device in try devices()
    backends = backends.push(Tensor.Backend.Ggml(device))
  end

  for backend in backends
    try refused(Tensor.zeros([32], q8, backend), "from_file")
    try refused(Tensor.ones([32], q8, backend), "from_file")
    try refused(Tensor.empty([32], q8, backend), "from_file")
    try refused(Tensor.full([32], Tensor.Values.Float(1.0), q8, backend), "from_file")
    try refused(Tensor.from_list([Tensor.Values.Float(1.0)], [1], q8, backend), "from_file")
    ones = try Tensor.floats([1.0], [1], backend)
    try refused(Tensor.to_dtype(ones, q8), "from_file")
  end

  try with_file(try file_bytes()) do |path|
    for device in try devices()
      kept = try Tensor.from_file(path, 3, [ROWS, COLS], Q8_0, Tensor.Backend.Ggml(device), q8)
      try refused(Tensor.like(kept), "from_file")
    end

    ok()
  end

  ok()
end
