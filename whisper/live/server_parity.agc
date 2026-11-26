## Checks that `transcribe_detailed` agrees with whisper.cpp's own server: the
## `verbose_json` answers in `live/fixtures/server-*-medium.json` were
## recorded from a `whisper-server` (whisper.cpp 1.9.x, Vulkan, `-l auto`)
## running the medium multilingual model, for `jfk.wav` and
## `live/fixtures/es-hola.wav` (non-ASCII text: `¡`, `ñ`). This program decodes
## the same audio with the options that server uses for `verbose_json` and
## compares, within a float tolerance:
##
## - the decoded and detected language, the detected probability and the
##   three most probable languages;
## - per segment: the text, the times, the token ids, `avg_logprob` and
##   `no_speech_prob`;
## - per word (a `Token`): the text and the probability.
##
## Token times are not compared: the engine derives them from the timestamp
## tokens' probabilities, and they move by up to 0.7 s between the Vulkan
## server and a CPU run (segment times and token ids do not).
##
## It needs the medium model, which `tools/fetch-model.sh` does not fetch
## (1.5 GB; whisper.cpp's `models/download-ggml-model.sh medium`), so it runs
## by hand, not in CI. From `whisper/`, after `tools/build-engine.sh` (CPU) or
## `tools/build-engine.sh vulkan`:
##
## ```sh
## agentica live/server_parity.agc path/to/ggml-medium.bin
## ```
##
## Prints one line per sample and exits 0 when both agree, or the first
## difference and exits 1.

import Whisper from "../whisper"

SAMPLES = [
  ["../third_party/whisper.cpp/samples/jfk.wav", "live/fixtures/server-jfk-medium.json"],
  ["live/fixtures/es-hola.wav", "live/fixtures/server-es-hola-medium.json"],
]

## How far a time (ms) and a probability may drift: the server ran on a GPU,
## this run may not, and floating point order differs between backends.
TIME_TOLERANCE_MS = 50
PROBABILITY_TOLERANCE = 0.05

def main() -> Void
  args = System.args()
  if args.size != 1
    puts("usage: agentica live/server_parity.agc MODEL")
    System.exit(2)
  end

  case run(args[0])
    ok(_) -> nil
    error(e) ->
      puts("server_parity: #{e.message}")
      System.exit(1)
  end
end

def run(model_path: String) -> Result[Void]
  model = try Whisper.load_model(model_path)
  for sample in SAMPLES
    try compare_sample(model, sample[0], sample[1])
    puts("ok #{sample[0]}")
  end

  ok()
end

def compare_sample(model: Whisper.Model, audio: String, golden_path: String) -> Result[Void]
  golden = try JSON.parse(try File.read_text(golden_path))
  pcm = try Whisper.load(audio)
  options = Whisper.Options(language: "auto", token_timestamps: true, max_len: 60, language_probabilities: true)
  transcript = try Whisper.transcribe_detailed(model, Whisper.widen(pcm), options)
  try same_text("language", name(transcript.language), try string(golden, "language"))
  detection = try transcript.detection.to_result("no detection")
  try same_text("detected_language", name(detection.language), try string(golden, "detected_language"))
  try near("detected_language_probability", detection.probability, try number(golden, "detected_language_probability"))
  try compare_top_languages(detection, try JSON.get(golden, "language_probabilities"))
  segments = try JSON.list(try JSON.get(golden, "segments"))
  if segments.size != transcript.segments.size
    return error("#{segments.size} segments in the golden, #{transcript.segments.size} here")
  end

  for i in 0...segments.size
    try compare_segment(i, transcript.segments[i], segments[i])
  end

  ok()
end

## The three most probable languages are the same, in order, and each is near
## its golden probability.
def compare_top_languages(detection: Whisper.Detection, golden: JSON.Value) -> Result[Void]
  want = try top_three(golden)
  ours = detection.probabilities.keys.sort_by do |code| -detection.probabilities[code] end
  for i in 0...3
    code = want[i]
    try same_text("top language #{i}", ours[i], code)
    try near("probability of #{code}", detection.probabilities[code], try number(golden, code))
  end

  ok()
end

def top_three(golden: JSON.Value) -> Result[List[String]]
  codes = case JSON.node(golden)
    JSON.Node.Object(fields) -> fields.keys
    _ -> []
  end

  probabilities: Map[String, Float] = {}
  for code in codes
    probabilities = probabilities.put(code, try number(golden, code))
  end

  sorted = codes.sort_by do |code| -probabilities[code] end
  return error("fewer than three languages in the golden") if sorted.size < 3

  ok(sorted.take(3))
end

def compare_segment(i: Int, segment: Whisper.SegmentDetail, golden: JSON.Value) -> Result[Void]
  at = "segment #{i}"
  try same_text("#{at} text", segment.text, try string(golden, "text"))
  try near_ms("#{at} start", segment.start_ms, try number(golden, "start"), TIME_TOLERANCE_MS)
  try near_ms("#{at} end", segment.end_ms, try number(golden, "end"), TIME_TOLERANCE_MS)
  ids = segment.tokens.flat_map do |token| token.ids end
  want_ids = try JSON.list(try JSON.get(golden, "tokens"))
  want = want_ids.map do |id| JSON.int(id).unwrap_or(-1) end
  return error("#{at} tokens #{ids}, golden #{want}") if ids != want

  try near("#{at} avg_logprob", segment.avg_logprob, try number(golden, "avg_logprob"))
  try near("#{at} no_speech_prob", segment.no_speech_probability, try number(golden, "no_speech_prob"))
  words = try JSON.list(try JSON.get(golden, "words"))
  if words.size != segment.tokens.size
    return error("#{at}: #{words.size} words in the golden, #{segment.tokens.size} here")
  end

  for j in 0...words.size
    token = segment.tokens[j]
    word = words[j]
    try same_text("#{at} word #{j}", token.text, try string(word, "word"))
    try near("#{at} word #{j} probability", token.probability, try number(word, "probability"))
  end

  ok()
end

def name(code: String) -> String = Whisper.language_name(code).unwrap_or(code)

def string(json: JSON.Value, key: String) -> Result[String] = JSON.string(try JSON.get(json, key))

def number(json: JSON.Value, key: String) -> Result[Float] = JSON.float(try JSON.get(json, key))

def same_text(what: String, ours: String, golden: String) -> Result[Void]
  return error("#{what}: [#{ours}], golden [#{golden}]") if ours != golden

  ok()
end

def near(what: String, ours: Float, golden: Float) -> Result[Void]
  return error("#{what}: #{ours}, golden #{golden}") if (ours - golden).abs > PROBABILITY_TOLERANCE

  ok()
end

## `golden` is in seconds, as the server writes it.
def near_ms(what: String, ours_ms: Int, golden_s: Float, tolerance_ms: Int) -> Result[Void]
  golden_ms = (golden_s * 1000.0).round
  return error("#{what}: #{ours_ms} ms, golden #{golden_ms} ms") if (ours_ms - golden_ms).abs > tolerance_ms

  ok()
end
