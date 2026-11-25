## Prints the threads of the ggml CPU backend of this process, for `threads_test.agc`. Asking does not fix the
## precision of Vulkan: `vulkan_fp16(true)` after it is still allowed.

import Lower from "../src/lower"
import Tensor from "../tensor"

def main() -> Void
  case Lower.cpu_threads()
    ok(n) -> puts(report(n))
    error(message) -> puts("error: #{message}")
  end
end

def report(n: Int) -> String
  case Tensor.vulkan_fp16(true)
    ok(_) -> "#{n}"
    error(message) -> "error: #{message}"
  end
end
