## Helpers of the tests that write a file for ggml to read (not a test file itself).

import Tensor from "../tensor"

## The ggml devices of `wanted` that are there. With none, the ggml checks pass by asserting nothing, unless
## `TENSOR_REQUIRE_GGML` is set (CI sets it): then a missing module is an error.
def devices(wanted: List[Tensor.Device]) -> Result[List[Tensor.Device]]
  found = wanted.select { |d: Tensor.Device| Tensor.available(Tensor.Backend.Ggml(d)) }
  return ok(found) if !found.empty?
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok([]) if !flag.ok?
  return ok([]) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end

## Runs `body` with the path of a file that holds `data`, in a folder of `prefix` that is removed afterwards, whether
## the file could be written or not.
def with_file(prefix: String, data: Bytes, body: Fn[String, Result[Void]]) -> Result[Void]
  dir = "/tmp/agentica-#{prefix}-#{Random.int(0, 999999999)}"
  try Dir.mkdir_private(dir, 5000)
  result = written(dir, data, body)
  _ = Path.remove(dir, 5000)
  result
end

## The little-endian bytes of the F32 `x`.
def f32_bytes(x: Float) -> List[Int]
  bits = Float32.to_bits(x.to_float32).to_i
  [bits % 256, (bits / 256) % 256, (bits / 65536) % 256, bits / 16777216]
end

private

def written(dir: String, data: Bytes, body: Fn[String, Result[Void]]) -> Result[Void]
  path = "#{dir}/data.bin"
  try File.open(path, "write", 5000) do |f|
    try File.write_bytes(f, data, 5000)
    ok()
  end

  body.call(path)
end
