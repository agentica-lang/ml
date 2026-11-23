## Shared helpers of the `hub/` tests (not a test file itself): the Python
## peer `peer.py` writes the fixtures and the values it expects into a fresh
## folder, which each test removes when it ends.

import Tensor from "pkg/tensor/tensor"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)

## Runs `body` on a folder of fresh fixtures, then removes the folder.
def with_fixtures(tag: String, body: Fn[String, Result[Void]]) -> Result[Void]
  tmp = System.env("TMPDIR").unwrap_or("")
  dir = "#{if_empty(tmp, "/tmp")}/agentica-hub-#{System.pid}-#{tag}"
  try Path.remove(dir)
  p = try Process.spawn_captured("python3", ["-I", "test/peer.py", dir])
  out = try Process.output(p, 30000)
  code = try Process.wait(p, 30000)
  return error("peer.py exited with #{code}: #{out.stderr}") if code != 0
  result = body.call(dir)
  try Path.remove(dir)
  result
end

def if_empty(s: String, default: String) -> String
  return default if s.empty?
  s
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
  Tensor.from_list(data.map { |x: Float| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, backend)
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
