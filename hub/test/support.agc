## Shared helpers of the `hub/` tests (not a test file itself): the Python
## peer `peer.py` writes one set of fixtures and the values it expects into a
## new temporary folder, which each test removes when it ends. Run the tests
## from `hub/`.

import Hub from "../hub"
import Shards from "../shard"
import Tensor from "pkg/tensor/tensor"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)
GGML = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

## The backends of this machine that read a float range from the file in C.
def native() -> Result[List[Tensor.Backend]]
  try Tensor.vulkan_fp16(false)
  ok([TORCH, GGML, VULKAN].select { |b: Tensor.Backend| Tensor.available(b) })
end

## The ggml backends of this machine: none when ggml is not built, an error when it is required
## (`TENSOR_REQUIRE_GGML`) and not built.
def ggml_backends() -> Result[List[Tensor.Backend]]
  try Tensor.vulkan_fp16(false)
  if not Tensor.available(GGML)
    flag = System.env("TENSOR_REQUIRE_GGML")
    return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)") if flag.ok? and try flag != ""
    return ok([])
  end

  return ok([GGML, VULKAN]) if Tensor.available(VULKAN)
  ok([GGML])
end

## Runs `body` on the folder the peer wrote `set` into, then removes it.
def with_fixtures(set: String, body: Fn[String, Result[Void]]) -> Result[Void]
  p = try Process.spawn_captured("python3", ["-I", "test/peer.py", set])
  out = try Process.output(p, 30000)
  code = try Process.wait(p, 30000)
  return error("peer.py exited with #{code}: #{out.stderr}") if code != 0
  dir = out.stdout.strip.split("\n").last
  if not dir.start_with?("/") or not dir.split("/").last.start_with?("agentica-hub-")
    return error("peer.py printed no fixture folder: #{dir}")
  end

  result = body.call(dir)
  _ = Path.remove(dir)
  result
end

## The JSON document `file` of the folder.
def json(dir: String, file: String) -> Result[JSON.Value] = JSON.parse(try File.read_text("#{dir}/#{file}"))

## A list of JSON numbers as Floats.
def floats(json: JSON.Value) -> Result[List[Float]]
  out: List[Float] = []
  for v in try JSON.list(json)
    out = out.push(try JSON.float(v))
  end

  ok(out)
end

## A list of JSON rows as Floats, flattened row after row.
def rows(json: JSON.Value) -> Result[List[Float]]
  out: List[Float] = []
  for row in try JSON.list(json)
    out = out + try floats(row)
  end

  ok(out)
end

def floats_of(t: Tensor.Tensor) -> Result[List[Float]]
  out: List[Float] = []
  for v in try Tensor.to_list(t)
    case v
      Tensor.Values.Float(f) -> out = out.push(f)
      _ -> return error("a float tensor holds Float values")
    end
  end

  ok(out)
end

def f32(data: List[Float], shape: List[Int], backend: Tensor.Backend) -> Result[Tensor.Tensor]
  Tensor.floats(data, shape, backend)
end

## `got` matches `want` within `abs + rel * |want|`, element by element.
def expect_close(got: List[Float], want: List[Float], abs: Float, rel: Float, what: String) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    gap = (got[i] - want[i]).abs
    return error("#{what}: value #{i} is #{got[i]}, expected #{want[i]}") if gap > abs + rel * want[i].abs
  end

  ok()
end

def same_as_peer(c: Hub.Checkpoint, expected: JSON.Value) -> Result[Void]
  fields: Map[String, JSON.Value] = {}
  case JSON.node(expected)
    JSON.Node.Object(m) -> fields = m
    _ -> return error("the expected values are not a JSON object")
  end

  try Test.eq(Hub.names(c), fields.keys.sort)
  for name in fields.keys
    want = fields[name]
    t = try Hub.tensor(c, name, REF)
    try Test.eq(Tensor.dtype(t), loaded_dtype(try JSON.string(try JSON.get(want, "dtype"))))
    try Test.eq(Tensor.shape(t), try JSON.decode(try JSON.get(want, "shape"), List[Int]))
    try Test.eq(bits(try Tensor.to_list(t)), bits(try values(try JSON.get(want, "values"), Tensor.dtype(t))))
  end

  ok()
end

## F32, F16 and BF16 load as `F32`; the other dtypes as the file has them.
def loaded_dtype(file_dtype: String) -> Tensor.Dtype
  return Tensor.Dtype.F32 if ["F32", "F16", "BF16"].include?(file_dtype)
  Shards.DTYPES[file_dtype]
end

## The peer's values: floats come as text, so `inf` and `nan` fit in JSON.
def values(json: JSON.Value, dtype: Tensor.Dtype) -> Result[List[Tensor.Values]]
  out: List[Tensor.Values] = []
  for v in try JSON.list(json)
    case dtype
      Tensor.Dtype.F32 -> out = out.push(Tensor.Values.Float(try float(try JSON.string(v))))
      Tensor.Dtype.F64 -> out = out.push(Tensor.Values.Float(try float(try JSON.string(v))))
      Tensor.Dtype.Bool -> out = out.push(Tensor.Values.Bool(try JSON.bool(v)))
      _ -> out = out.push(Tensor.Values.Int(try JSON.int(v)))
    end
  end

  ok(out)
end

def float(text: String) -> Result[Float]
  inf = Float.from_bits(0x7FF0000000000000)
  case text
    "inf" -> ok(inf)
    "-inf" -> ok(-inf)
    "nan" -> ok(inf - inf)
    _ -> text.to_f
  end
end

## Floats as their bits, so `-0.0` differs from `0.0` and NaN equals NaN
## (`to_bits` gives every NaN one pattern: payloads are not compared).
def bits(vs: List[Tensor.Values]) -> List[String]
  vs.map do |v|
    case v
      Tensor.Values.Float(f) -> "f#{Float.to_bits(f)}"
      Tensor.Values.Int(i) -> "i#{i}"
      Tensor.Values.Bool(b) -> "b#{b}"
    end
  end
end

def expect_error[T](r: Result[T], phrase: String) -> Result[Void]
  case r
    ok(_) -> error("expected an error with \"#{phrase}\", got a value")
    error(e) -> Test.assert(e.message.include?(phrase), "\"#{e.message}\" does not hold \"#{phrase}\"")
  end
end
