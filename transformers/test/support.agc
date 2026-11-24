## Helpers of the tests (not a test file itself): editing a fixture file, a
## folder of fixtures from a Python peer, tensors from lists, and comparing floats.

import Tensor from "pkg/tensor/tensor"

REF = Tensor.Backend.Reference
GGML = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))

## `text` with the `n`th (from 0) occurrence of `needle` replaced by `with`;
## an error when there is no such occurrence, so a test never "refuses" a file
## it did not change.
def edited(text: String, needle: String, with: String, n: Int) -> Result[String]
  parts = text.split(needle)
  return error("no occurrence #{n} of #{needle}") if parts.size <= n + 1
  out = parts[0]
  for i in 1...parts.size
    if i - 1 == n
      out = out + with + parts[i]
    else
      out = out + needle + parts[i]
    end
  end

  ok(out)
end

## Every occurrence of `needle` replaced, an error when there is none.
def replaced(text: String, needle: String, with: String) -> Result[String]
  return error("no #{needle} in the file") if not text.include?(needle)
  ok(text.replace(needle, with))
end

## The ggml backends of this machine: none when ggml is not built, an error
## when it is required (`TENSOR_REQUIRE_GGML`) and not built.
def ggml_backends() -> Result[List[Tensor.Backend]]
  if not Tensor.available(GGML)
    flag = System.env("TENSOR_REQUIRE_GGML")
    return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)") if flag.ok? and try flag != ""
    return ok([])
  end

  return ok([GGML, VULKAN]) if Tensor.available(VULKAN)
  ok([GGML])
end

## Runs `body` with the folder that `script` (run from `transformers/`)
## writes, named `prefix...`, and removes the folder after.
def with_fixtures(script: String, prefix: String, body: Fn[String, Result[Void]]) -> Result[Void]
  p = try Process.spawn_captured("python3", ["-I", script])
  out = try Process.output(p, 30000)
  code = try Process.wait(p, 30000)
  return error("#{script} exited with #{code}: #{out.stderr}") if code != 0
  dir = out.stdout.strip.split("\n").last
  if not dir.start_with?("/") or not dir.split("/").last.start_with?(prefix)
    return error("#{script} printed no fixture folder: #{dir}")
  end

  result = body.call(dir)
  _ = Path.remove(dir)
  result
end

def f32(data: List[Float], shape: List[Int], backend: Tensor.Backend = REF) -> Result[Tensor.Tensor]
  Tensor.from_list(data.map { |x: Float| Tensor.Values.Float(x) }, shape, Tensor.Dtype.F32, backend)
end

## JSON numbers nested to any depth, flattened in order.
def flat(json: JSON.Value) -> Result[List[Float]]
  case JSON.list(json)
    ok(items) ->
      out: List[Float] = []
      for item in items
        out = out + try flat(item)
      end

      ok(out)
    error(_) -> ok([try JSON.float(json)])
  end
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

## `got` matches `want` within `abs + rel * |want|`, element by element.
def expect_close(got: List[Float], want: List[Float], abs: Float, rel: Float, what: String) -> Result[Void]
  try Test.eq(got.size, want.size)
  for i in 0...want.size
    gap = (got[i] - want[i]).abs
    return error("#{what}: value #{i} is #{got[i]}, expected #{want[i]}") if gap > abs + rel * want[i].abs
  end

  ok()
end
