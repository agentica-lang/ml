## The model-bound checks with the pinned tiny model: real weights load
## through the FFI, real inference runs on them, and the empty-audio error
## answers against a loaded model. Needs the engine
## (`tools/build-engine.sh`, which CI builds first) and the downloaded
## weights (`tools/fetch-model.sh`, at `models/ggml-tiny.bin`, never in git).
## From `whisper/`:
##
## ```sh
## agentica test live/
## ```
##
## Decoding the 11 s sample takes ~14 M Agentica steps, past what one test
## file may spend, so the transcription itself runs as
## `live/transcribe_jfk.agc` (`agentica live/transcribe_jfk.agc`), where runs
## are not step-capped.

import Whisper from "../whisper"

private

## The model `tools/fetch-model.sh` downloads, relative to `whisper/`.
MODEL_PATH = "models/ggml-tiny.bin"

def test_the_tiny_model_transcribes_silence_to_empty() -> Result[Void]
  zeros: List[Float] = []
  for _i in 0...1600
    zeros = zeros.push(0.0)
  end

  model = try Whisper.load_model(MODEL_PATH)
  text = try Whisper.transcribe(model, zeros)
  Test.eq(text, "")
end

def test_transcribe_errors_without_samples() -> Result[Void]
  model = try Whisper.load_model(MODEL_PATH)
  case Whisper.transcribe(model, [])
    ok(_) -> Test.assert(false, "transcribe took no samples")
    error(e) -> Test.eq(e.message, "transcribe needs at least one sample at 16000 Hz")
  end
end
