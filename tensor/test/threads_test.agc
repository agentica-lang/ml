## The threads of the ggml CPU backend: `TENSOR_THREADS` sets them, and anything that is not a positive integer, or is
## above ggml's 512 threads, leaves the default (the physical cores), and asking does not fix Vulkan's precision. The choice is made once per process, so each case is a process of its own. Without
## the ggml module the checks pass by asserting nothing, unless `TENSOR_REQUIRE_GGML` is set (CI sets it).

import Tensor from "../tensor"

def test_tensor_threads_sets_the_threads_and_a_bad_value_leaves_the_default() -> Result[Void]
  return ok() if !try present()
  default = try threads_with("")
  try Test.assert(default >= 1, "the default is #{default}")
  try Test.eq(try threads_with("#{default + 1}"), default + 1)
  for bad in ["0", "many", "3x", "100000"]
    try Test.eq(try threads_with(bad), default)
  end

  ok()
end

private

def present() -> Result[Bool]
  return ok(true) if Tensor.available(Tensor.Backend.Ggml(Tensor.Device.Cpu))
  flag = System.env("TENSOR_REQUIRE_GGML")
  return ok(false) if !flag.ok?
  return ok(false) if try flag == ""
  error("ggml required but not built (TENSOR_REQUIRE_GGML is set)")
end

## The threads a fresh process gets with `TENSOR_THREADS` set to `value`.
def threads_with(value: String) -> Result[Int]
  p = try Process.spawn_captured(
    "agentica",
    ["test/threads_probe.agc"],
    Process.Options(env: {"TENSOR_THREADS": value}),
  )

  out = try Process.output(p, 60000)
  code = try Process.wait(p, 60000)
  return error("the probe exited with #{code}: #{out.stdout} #{out.stderr}") if code != 0
  text = out.stdout.strip
  return error("the probe failed: #{text}") if text.start_with?("error")
  text.to_i
end
