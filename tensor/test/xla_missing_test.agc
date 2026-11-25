## A missing XLA plugin is an error that names the file and the install step. A file of its own, so no plugin is
## loaded in its process before it runs.

import Xla from "../src/xla"

private

def test_a_missing_plugin_names_the_file_and_the_install_step() -> Result[Void]
  # Without the native module there is no loader to ask.
  return ok() if !Native.load("xla").ok?
  r = Xla.open_at("/nonexistent/xla_cpu_pjrt.so", "zeros")
  try Test.assert(!r.ok?, "a missing plugin loaded")
  try Test.assert(r.message.start_with?("zeros: backend not available"), r.message)
  try Test.assert(r.message.include?("/nonexistent/xla_cpu_pjrt.so"), r.message)
  Test.assert(r.message.include?("agentica tools/install_xla.agc"), r.message)
end
