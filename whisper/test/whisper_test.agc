## The error paths of the native layer, with no model and no network: loading
## names nothing the engine can read, so every call answers `error`. These run
## in plain `agentica test` (they still link the engine built by
## `tools/build-engine.sh`, which CI builds first). Real transcription lives
## in `live/whisper_live_test.agc`.

import Whisper from "../whisper"

private

def test_load_model_errors_on_a_missing_file() -> Result[Void]
  case Whisper.load_model("no/such/model.bin")
    ok(_) -> Test.assert(false, "load_model read a file that is not there")
    error(e) ->
      Test.eq(e.message, "cannot load the model at no/such/model.bin: whisper_init_from_file_with_params gave NULL")
  end
end

def test_load_model_errors_on_a_path_past_4096_bytes() -> Result[Void]
  path = ""
  for _i in 0...500
    path = path + "0123456789"
  end

  case Whisper.load_model(path)
    ok(_) -> Test.assert(false, "load_model took a path past 4096 bytes")
    error(e) -> Test.eq(e.message, "load_model takes a path of 4096 bytes at most")
  end
end
