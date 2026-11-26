## Opt-in loader regression for a CUDA PJRT plugin. It has its own file (and process) so the CPU plugin is not
## already loaded. Set TENSOR_TEST_XLA_CUDA_PLUGIN to the path of xla_cuda_plugin.so to run it.

import Xla from "../src/xla"

private

def test_the_cuda_plugin_loads_without_cpu_client_options() -> Result[Void]
  case System.env("TENSOR_TEST_XLA_CUDA_PLUGIN")
    ok(path) ->
      return ok() if path.empty?
      _ = try Xla.open_at(path, "CUDA loader test")
      ok()
    error(_) -> ok()
  end
end
