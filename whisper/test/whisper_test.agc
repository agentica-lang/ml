## The error paths of the native layer, with no model and no network: loading
## names nothing the engine can read, so every call answers `error`. These run
## in plain `agentica test` (they still link the engine built by
## `tools/build-engine.sh`, which CI builds first). Real transcription lives
## in `live/whisper_live_test.agc`.

import Whisper from "../whisper"

private

def test_widen_keeps_each_sample_exactly() -> Result[Void]
  packed: List[Float32] = [0.0, 0.1, -1.0, 32_767.0 / 32_768.0]
  wide = Whisper.widen(packed)
  try Test.eq(wide.size, 4)
  try Test.eq(wide[1], 0.1.to_float32.to_f)
  try Test.eq(wide[3], 32_767.0 / 32_768.0)
  try Test.eq(wide.map { |x| x.to_float32 }, packed)
  Test.eq(Whisper.widen([]), [])
end

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

def test_language_name_spells_out_a_known_code() -> Result[Void]
  try Test.eq(Whisper.language_name("es"), some("spanish"))
  Test.eq(Whisper.language_name("en"), some("english"))
end

def test_language_name_is_none_for_an_unknown_code() -> Result[Void]
  try Test.eq(Whisper.language_name("xx"), none)
  try Test.eq(Whisper.language_name(""), none)
  Test.eq(Whisper.language_name("auto"), none)
end
