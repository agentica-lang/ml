## Prints the threads of the ggml CPU backend of this process, for `threads_test.agc`.

import Lower from "../src/lower"

def main() -> Void
  case Lower.cpu_threads()
    ok(n) -> puts(n)
    error(message) -> puts("error: #{message}")
  end
end
