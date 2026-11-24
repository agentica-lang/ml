## Shared helpers of the `hub/` tests (not a test file itself): the Python
## peer `peer.py` writes one set of fixtures and the values it expects into a
## new temporary folder, which each test removes when it ends. Run the tests
## from `hub/`.

import Tensor from "pkg/tensor/tensor"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)
GGML = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

## The backends of this machine that read a float range from the file in C.
def native() -> List[Tensor.Backend]
  _ = Tensor.vulkan_fp16(false)
  [TORCH, GGML, VULKAN].select { |b: Tensor.Backend| Tensor.available(b) }
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
