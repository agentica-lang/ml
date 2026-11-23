## Shared helpers of the `hub/` tests (not a test file itself): the Python
## peer `peer.py` writes one set of fixtures and the values it expects into a
## new temporary folder, which each test removes when it ends. Run the tests
## from `hub/`.

import Tensor from "pkg/tensor/tensor"

REF = Tensor.Backend.Reference
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)

## Runs `body` on the folder the peer wrote `set` into, then removes it.
def with_fixtures(set: String, body: Fn[String, Result[Void]]) -> Result[Void]
  p = try Process.spawn_captured("python3", ["-I", "test/peer.py", set])
  out = try Process.output(p, 30000)
  code = try Process.wait(p, 30000)
  return error("peer.py exited with #{code}: #{out.stderr}") if code != 0
  dir = out.stdout.strip
  return error("peer.py printed no fixture folder: #{dir}") if not dir.include?("agentica-hub-")
  result = body.call(dir)
  try Path.remove(dir)
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
