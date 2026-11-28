## Helpers of the tests (not a test file itself): editing a fixture file, a
## folder of fixtures from a Python peer, tensors from lists, and comparing floats.

import Generate from "../generate"
import Tensor from "pkg/tensor/tensor"

REF = Tensor.Backend.Reference
GGML = Tensor.Backend.Ggml(Tensor.Device.Cpu)
VULKAN = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
TORCH = Tensor.Backend.Torch(Tensor.Device.Cpu)
XLA = Tensor.Backend.Xla(Tensor.Device.Cpu)

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
  try Tensor.vulkan_fp16(false)
  if not Tensor.available(GGML)
    flag = System.env("TENSOR_REQUIRE_GGML")
    return error("ggml required but not built (TENSOR_REQUIRE_GGML is set)") if flag.ok? and try flag != ""
    return ok([])
  end

  return ok([GGML, VULKAN]) if Tensor.available(VULKAN)
  ok([GGML])
end

## `[XLA]` when the plugin is there (`tensor/tools/install_xla.agc`), none when it is not, an error when it is required
## (`TENSOR_REQUIRE_XLA`) and missing.
def xla_backends() -> Result[List[Tensor.Backend]]
  return ok([XLA]) if Tensor.available(XLA)
  flag = System.env("TENSOR_REQUIRE_XLA")
  return error("XLA required but not installed (TENSOR_REQUIRE_XLA is set)") if flag.ok? and try flag != ""
  ok([])
end

## The libtorch backend when it is built, none when it is not, and an error when it is required
## (`TENSOR_REQUIRE_TORCH`, set by the torch job of CI) and not built.
def torch_backends() -> Result[List[Tensor.Backend]]
  return ok([TORCH]) if Tensor.available(TORCH)
  flag = System.env("TENSOR_REQUIRE_TORCH")
  return error("libtorch required but not built (TENSOR_REQUIRE_TORCH is set)") if flag.ok? and try flag != ""
  ok([])
end

## The ggml devices the pairing cases of `generate_test.agc` and `serving_test.agc` run on: the CPU engine, here and in
## the two CPU jobs of CI. A Vulkan device pairs too (`Generate.pairable`), but its software form is an interpreter of
## shaders, roughly twenty times slower on these fixtures, so `vulkan_pair_test.agc` is the case that runs it, and these
## files keep their step budget.
def paired_backends() -> Result[List[Tensor.Backend]]
  out: List[Tensor.Backend] = [REF]
  for backend in try ggml_backends()
    if Generate.pairable(backend) and not vulkan?(backend)
      out = out.push(backend)
    end
  end

  ok(out)
end

## Whether `backend` is a Vulkan device of ggml, the software one included.
def vulkan?(backend: Tensor.Backend) -> Bool
  case backend
    Tensor.Backend.Ggml(Tensor.Device.Vulkan(_)) -> true
    _ -> false
  end
end

## The Vulkan devices of this machine, for `vulkan_pair_test.agc`: the engine must have the backend and the loader a
## device. An error when `TENSOR_REQUIRE_VULKAN` is set and there is none, so the software-Vulkan job fails instead of
## skipping the case; without the flag, an empty list and a note.
def vulkan_backends() -> Result[List[Tensor.Backend]]
  out: List[Tensor.Backend] = []
  for backend in try ggml_backends()
    if vulkan?(backend)
      out = out.push(backend)
    end
  end

  return ok(out) if out.size > 0
  flag = System.env("TENSOR_REQUIRE_VULKAN")
  if flag.ok? and try flag != ""
    return error("Vulkan required but no Vulkan device is visible (TENSOR_REQUIRE_VULKAN is set)")
  end

  IO.warn("SKIP no Vulkan device: the pairing case of `vulkan_pair_test.agc` ran nowhere")
  ok(out)
end

## The backends of this machine that do not pair, where a scheduler must step its rows alone: XLA, when it is installed.
## libtorch takes the same singleton path but its job runs no serving test, so it is not here (`serving_test.agc` checks
## the rule for both).
def unpaired_backends() -> Result[List[Tensor.Backend]]
  out: List[Tensor.Backend] = []
  for backend in try xla_backends()
    if not Generate.pairable(backend)
      out = out.push(backend)
    end
  end

  ok(out)
end

## `ggml_backends` for a model with a gated delta rule of head width `width` (Qwen3.5): ggml-vulkan runs that op for
## widths 16, 32, 64 and 128 only (`tensor/DESIGN.md`), so for another width Vulkan is left out, with a note.
def ggml_backends_for_delta(width: Int) -> Result[List[Tensor.Backend]]
  all = try ggml_backends()
  return ok(all) if [16, 32, 64, 128].include?(width) or all.size < 2
  IO.warn("SKIP Vulkan: its gated delta rule needs a head width of 16, 32, 64 or 128, the model has #{width}")
  ok(all.take(1))
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

## `r` is an error whose message holds `phrase`.
def expect_error[T](r: Result[T], phrase: String) -> Result[Void]
  case r
    ok(_) -> error("expected an error with \"#{phrase}\", got a value")
    error(e) -> Test.assert(e.message.include?(phrase), "\"#{e.message}\" does not hold \"#{phrase}\"")
  end
end
