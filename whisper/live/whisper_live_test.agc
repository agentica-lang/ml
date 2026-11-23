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

def test_silence_has_no_text_to_join() -> Result[Void]
  zeros: List[Float] = []
  for _i in 0...1600
    zeros = zeros.push(0.0)
  end

  model = try Whisper.load_model(MODEL_PATH)
  segments = try Whisper.transcribe_with(model, zeros, Whisper.Options())
  Test.eq(Whisper.join_text(segments), "")
end

def test_transcribe_errors_without_samples() -> Result[Void]
  model = try Whisper.load_model(MODEL_PATH)
  case Whisper.transcribe(model, [])
    ok(_) -> Test.assert(false, "transcribe took no samples")
    error(e) -> Test.eq(e.message, "transcribe_with needs at least one sample at 16000 Hz")
  end
end

def test_detailed_silence_detects_a_language_and_has_no_text() -> Result[Void]
  zeros: List[Float] = []
  for _i in 0...1600
    zeros = zeros.push(0.0)
  end

  model = try Whisper.load_model(MODEL_PATH)
  options = Whisper.Options(language: "auto", language_probabilities: true)
  transcript = try Whisper.transcribe_detailed(model, zeros, options)
  case transcript.detection
    some(detection) ->
      try Test.eq(detection.probabilities.get(detection.language), some(detection.probability))
      Test.assert(transcript.segments.all? do |segment| segment.text.strip.empty? end, "silence gave text")

    none -> Test.assert(false, "language_probabilities gave no detection")
  end
end

def test_detailed_has_no_detection_unless_asked() -> Result[Void]
  model = try Whisper.load_model(MODEL_PATH)
  transcript = try Whisper.transcribe_detailed(model, [0.0], Whisper.Options())
  Test.eq(transcript.detection, none)
end

def test_transcribe_errors_on_a_beam_size_past_8() -> Result[Void]
  model = try Whisper.load_model(MODEL_PATH)
  case Whisper.transcribe_detailed(model, [0.0], Whisper.Options(beam_size: 9))
    ok(_) -> Test.assert(false, "transcribe_detailed took 9 beams")
    error(e) -> Test.eq(e.message, "transcribe_with takes a beam_size from 0 to 8")
  end
end

def test_transcribe_errors_on_a_negative_max_len() -> Result[Void]
  model = try Whisper.load_model(MODEL_PATH)
  case Whisper.transcribe_with(model, [0.0], Whisper.Options(max_len: -1))
    ok(_) -> Test.assert(false, "transcribe_with took a negative max_len")
    error(e) -> Test.eq(e.message, "transcribe_with takes a max_len from 0 to 100000")
  end
end

def test_transcribe_errors_on_an_unknown_language() -> Result[Void]
  model = try Whisper.load_model(MODEL_PATH)
  case Whisper.transcribe_detailed(model, [0.0], Whisper.Options(language: "xx"))
    ok(_) -> Test.assert(false, "transcribe_detailed took the language xx")
    error(e) -> Test.eq(e.message, "transcribe_with does not know that language code")
  end
end
