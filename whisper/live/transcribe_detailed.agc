## Validates the detailed decode end to end on the 11 s JFK sample with the
## pinned tiny model: language detection, timed tokens, segments cut at
## `max_len` characters and beam search. Like `transcribe_jfk.agc`, it runs as
## a program because decoding the sample is past what one test file may spend.
## From `whisper/`, after `tools/build-engine.sh` and `tools/fetch-model.sh`:
##
## ```sh
## agentica live/transcribe_detailed.agc
## ```
##
## Prints the segments and exits 0 when every check holds, or the first
## failure and exits 1.

import Whisper from "../whisper"

SAMPLE_PATH = "../third_party/whisper.cpp/samples/jfk.wav"
MODEL_PATH = "models/ggml-tiny.bin"

## The longest segment text `max_len: 60` allows: 60 characters plus the
## token that crossed the limit, which the engine keeps whole.
MAX_SEGMENT_CHARS = 80

def main() -> Void
  case run()
    ok(_) -> nil
    error(e) ->
      puts("transcribe_detailed: #{e.message}")
      System.exit(1)
  end
end

def run() -> Result[Void]
  pcm = try Whisper.load(SAMPLE_PATH)
  model = try Whisper.load_model(MODEL_PATH)
  options = Whisper.Options(language: "auto", token_timestamps: true, max_len: 60, detect_language: true)
  transcript = try Whisper.transcribe_detailed(model, pcm, options)
  try expect(transcript.language == "en", "decoded language #{transcript.language}, want en")
  detection = try transcript.detection.to_result("detect_language gave no detection")
  try expect(detection.language == "en", "detected #{detection.language}, want en")
  try expect(detection.probability > 0.5, "en has probability #{detection.probability}")
  try expect(transcript.segments.size > 1, "max_len 60 left #{transcript.segments.size} segment")
  for segment in transcript.segments
    try check_segment(segment)
  end

  text = transcript.segments.map do |segment| segment.text.strip end
  try expect(text.join(" ").downcase.include?("ask not what your country"), "text #{text.join(" ")}")
  beam = try Whisper.transcribe_with(model, pcm, Whisper.Options(beam_size: 5))
  try expect(Whisper.join_text(beam).downcase.include?("your country"), "beam text #{Whisper.join_text(beam)}")
  ok()
end

## A segment's tokens spell its text, are timed inside it and in order, and
## its probabilities are in range.
def check_segment(segment: Whisper.SegmentDetail) -> Result[Void]
  puts("#{segment.start_ms} ms to #{segment.end_ms} ms: #{segment.text}")
  spelled = segment.tokens.map do |token| token.text end
  try expect(spelled.join("").strip == segment.text.strip, "tokens spell [#{spelled.join("")}]")
  try expect(segment.text.length <= MAX_SEGMENT_CHARS, "segment of #{segment.text.length} characters")
  try expect(segment.avg_logprob <= 0.0, "avg_logprob #{segment.avg_logprob}")
  try expect(segment.no_speech_probability >= 0.0 and segment.no_speech_probability <= 1.0, "no_speech")
  for token in segment.tokens
    try expect(!token.ids.empty?, "a token with no ids")
    try expect(token.probability >= 0.0 and token.probability <= 1.0, "probability #{token.probability}")
    try expect(token.start_ms <= token.end_ms, "token #{token.text} ends before it starts")
    try expect(token.start_ms >= segment.start_ms - 10, "token #{token.text} starts before its segment")
  end

  ok()
end

def expect(condition: Bool, message: String) -> Result[Void]
  return error(message) if !condition

  ok()
end
