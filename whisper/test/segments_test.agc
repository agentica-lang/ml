## The model-free half of the binding's surface: `Options` and `Segment` are
## plain Agentica structs, and `join_text` is pure, so the path from the
## segments `transcribe_with` returns to the text `transcribe` gives is checked
## with no model, no engine call and no network. The native error paths are in
## `whisper_test.agc`; real inference is in `live/`.

import Whisper from "../whisper"

private

def test_options_default_to_the_engine_language_without_translation() -> Result[Void]
  options = Whisper.Options()
  try Test.eq(options.language, "")
  Test.eq(options.translate, false)
end

def test_options_take_a_language_and_translation() -> Result[Void]
  options = Whisper.Options(language: "es", translate: true)
  try Test.eq(options.language, "es")
  Test.eq(options.translate, true)
end

def test_join_text_trims_each_segment_and_leaves_one_space_between() -> Result[Void]
  segments = [
    Whisper.Segment(start_ms: 0, end_ms: 1_000, text: " And so my "),
    Whisper.Segment(start_ms: 1_000, end_ms: 2_000, text: "\tfellow "),
    Whisper.Segment(start_ms: 2_000, end_ms: 3_000, text: " Americans.\n"),
  ]

  Test.eq(Whisper.join_text(segments), "And so my fellow Americans.")
end

def test_join_text_drops_whitespace_only_segments() -> Result[Void]
  segments = [
    Whisper.Segment(start_ms: 0, end_ms: 500, text: "   "),
    Whisper.Segment(start_ms: 500, end_ms: 1_000, text: "hola"),
    Whisper.Segment(start_ms: 1_000, end_ms: 1_500, text: "\n"),
  ]

  Test.eq(Whisper.join_text(segments), "hola")
end

def test_join_text_of_no_segments_is_empty() -> Result[Void]
  empty: List[Whisper.Segment] = []
  Test.eq(Whisper.join_text(empty), "")
end
