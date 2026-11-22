## Validates the options-driven path end to end: 2.6 s of Spanish through the
## pinned small multilingual model, first with the language pinned to `"es"`
## and then with `translate: true`. Unlike `live/transcribe_jfk.agc`, it pins
## the shape of the timed segments and a couple of stable words, not an exact
## transcript: the fixture is short and synthesised, so which words survive is
## the one thing not worth betting on. Decoding is past the steps one test
## file may spend (1 M), so this runs as a program, where runs are not
## step-capped. From `whisper/`, after `tools/build-engine.sh` and
## `tools/fetch-model.sh`:
##
## ```sh
## agentica live/transcribe_spanish.agc
## ```
##
## Prints the segments of both passes and exits 0 when every check holds, or
## the first failure and exits 1. CI runs exactly this after building the
## engine and downloading the models.

import Whisper from "../whisper"

## The multilingual model `tools/fetch-model.sh` downloads for this case.
MODEL_PATH = "models/ggml-small.bin"
## 2.6 s of Spanish synthesised for this test: 42 353 samples at 16 kHz, mono
## 16-bit PCM, 84 KB (its provenance is in `README.md`).
FIXTURE_PATH = "live/fixtures/es-hola.wav"
## How far the segments may fall short of an edge of the audio: the engine
## times the speech it heard, not the silence around it (this fixture's one
## segment runs from 0 ms to 2560 ms of its 2647 ms).
EDGE_MS = 500

def main() -> Void
  case Whisper.load(FIXTURE_PATH)
    ok(pcm) -> run(pcm)
    error(e) ->
      puts("transcribe_spanish: cannot read #{FIXTURE_PATH}: #{e.message}")
      System.exit(1)
  end
end

def run(pcm: List[Float]) -> Void
  case Whisper.load_model(MODEL_PATH)
    ok(model) -> spanish(model, pcm)
    error(e) ->
      puts("transcribe_spanish: cannot load #{MODEL_PATH}: #{e.message}")
      System.exit(1)
  end
end

# The Spanish pass, then the same audio translated.
def spanish(model: Whisper.Model, pcm: List[Float]) -> Void
  case Whisper.transcribe_with(model, pcm, Whisper.Options(language: "es"))
    ok(segments) -> check_spanish(segments, pcm.size / 16, model, pcm)
    error(e) -> fail("cannot transcribe #{FIXTURE_PATH} in Spanish: #{e.message}")
  end
end

def check_spanish(segments: List[Whisper.Segment], duration_ms: Int, model: Whisper.Model, pcm: List[Float]) -> Void
  show(segments)
  check(!segments.empty?, "the Spanish decode has no segments")
  first = segments[0]
  last = segments[segments.size - 1]
  check(
    first.start_ms <= EDGE_MS,
    "the first segment starts at #{first.start_ms} ms, past the #{EDGE_MS} ms the audio may lead in by",
  )

  check(
    last.end_ms >= duration_ms - EDGE_MS,
    "the last segment ends at #{last.end_ms} ms, short of the #{duration_ms} ms of audio",
  )

  # Ordered, and every segment a range: none starts before the one it follows,
  # and none starts at or after its own end.
  at = 0
  for segment in segments
    check(segment.start_ms < segment.end_ms, "a segment is no range: #{segment.start_ms} ms to #{segment.end_ms} ms")
    check(
      segment.start_ms >= at,
      "a segment starts at #{segment.start_ms} ms, before the #{at} ms of the one before it",
    )

    at = segment.start_ms
  end

  text = Whisper.join_text(segments)
  check(!text.empty?, "the Spanish decode is empty")
  # The two words this fixture opens and closes on, case aside: not a
  # transcript pin, but enough to tell Spanish from a silent or garbled run.
  check(text.downcase.include?("hola"), "the Spanish decode lost \"hola\": [#{text}]")
  check(text.downcase.include?("español"), "the Spanish decode lost \"español\": [#{text}]")
  puts(text)
  english(model, pcm)
end

# The same audio with `translate: true`, the other half of `Options`: the
# engine answers in English instead of the spoken language.
def english(model: Whisper.Model, pcm: List[Float]) -> Void
  case Whisper.transcribe_with(model, pcm, Whisper.Options(language: "es", translate: true))
    ok(segments) ->
      show(segments)
      text = Whisper.join_text(segments)
      check(!text.empty?, "the translated decode is empty")
      check(text.downcase.include?("hello"), "the translated decode is not English: [#{text}]")
      check(text.downcase.include?("spanish"), "the translated decode lost \"Spanish\": [#{text}]")
      puts(text)

    error(e) -> fail("cannot translate #{FIXTURE_PATH}: #{e.message}")
  end
end

def show(segments: List[Whisper.Segment]) -> Void
  for segment in segments
    puts("#{segment.start_ms} ms to #{segment.end_ms} ms: [#{segment.text}]")
  end
end

def check(condition: Bool, message: String) -> Void
  return if condition
  fail(message)
end

def fail(message: String) -> Void
  puts("transcribe_spanish: #{message}")
  System.exit(1)
end
