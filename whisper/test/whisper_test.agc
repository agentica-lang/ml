## The native layer answers error until whisper.cpp is linked (Fase W, Via A):
## this pins the scaffold contract and proves the FFI wiring (the `extern`
## module block, the generated header, the C file) builds and loads.

import Whisper from "../whisper"

private

def test_load_model_errors_until_the_engine_is_linked() -> Result[Void]
  case Whisper.load_model("model.gguf")
    ok(_) -> Test.assert(false, "load_model worked with no engine linked")
    error(e) -> Test.eq(e.message, "whisper is not linked yet: no model can load")
  end
end
