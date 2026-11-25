## Transcribes the clips of `test/fixtures/transcribe_golden.json` with a real
## whisper-tiny and compares them with what Hugging Face answered: the token
## ids exactly, the text and the segments (times exactly; within 1e-9 in the
## long cases), the language and its probability within 1e-5. The two long
## cases (over 30 s, windows that seek) are built from the two clips as `test/transcribe_golden.py` builds
## them. It prints the real-time factor of each case and, on ggml, the
## graphs it compiled. Manual, not run by the tests (the checkpoint is not in
## git). From `asr/`:
##
## ```sh
## agentica tools/transcribe_check.agc MODEL_DIR ES_HOLA_WAV JFK_WAV [BACKEND] [CASE]
## ```
##
## MODEL_DIR holds `model.safetensors`, `config.json` and `tokenizer.json`.
## `WEIGHTS=f16` loads the big matrices as half weights (ggml only) and `VULKAN_FP16=1` lets Vulkan multiply in
## fp16 (`Tensor.vulkan_fp16`); a check against whisper.cpp leaves both off, which is exact `float32`.
## BACKEND is `torch` (libtorch, the default), `reference` (the pure one,
## slow), `ggml` (ggml on the CPU) or `vulkan` (ggml on Vulkan device 0); CASE
## runs only that case's index.

import Asr from "../asr"
import Binary from "std/binary"
import Load from "pkg/hub/load"
import Lower from "pkg/tensor/src/lower"
import Tensor from "pkg/tensor/tensor"
import Tokenizer from "../tokenizer"

GOLDEN = "test/fixtures/transcribe_golden.json"
## The gap allowed on the language probability.
CLOSE = 0.00001
## The gap allowed on a segment's time in the long cases, where it is a sum of an offset and a step.
TIME = 0.000000001

struct Golden
  cases: List[Case]
end

struct Case
  clip: String
  language: Optional[String]
  task: String
  timestamps: Bool
  tokens: List[Int]
  text: String
  detected: String
  probability: Float
  segments: List[Span]
end

struct Span
  start: Float
  stop: Float
  text: String
end

def main() -> Void
  case run(System.args())
    ok(same) ->
      if not same
        puts("transcribe_check: a case does not match")
        System.exit(1)
      end

      puts("transcribe_check: all cases match")
    error(e) ->
      puts("transcribe_check: #{e.message}")
      System.exit(2)
  end
end

def run(args: List[String]) -> Result[Bool]
  return error("usage: MODEL_DIR ES_HOLA_WAV JFK_WAV [BACKEND] [CASE]") if args.size < 3
  name = "torch"
  if args.size > 3
    name = args[3]
  end

  try Load.vulkan_fp16_from_env()
  backend = try backend_of(name)
  return error("#{backend} is not available") if not Tensor.available(backend)
  started = Time.monotonic_ms()
  m = try Asr.open("#{args[0]}/model.safetensors", backend, try Load.weights_from_env())
  tok = try Tokenizer.open("#{args[0]}/tokenizer.json")
  puts("loaded in #{Time.monotonic_ms() - started} ms")
  es = try pcm16(try read_bytes(args[1]))
  jfk = try pcm16(try read_bytes(args[2]))
  clips = {"es-hola": es, "jfk": jfk, "jfk-long": jfk_long(jfk), "es-long": es_long(es)}
  golden = try JSON.decode(try File.read_text(GOLDEN), Golden)
  all = true
  for i in 0...golden.cases.size
    next if args.size > 4 and args[4] != i.to_s
    c = golden.cases[i]
    pcm = clips.get(c.clip).unwrap_or([])
    options = Asr.Options(language: c.language, translate: c.task == "translate", timestamps: c.timestamps)
    compiled = try compiles(backend)
    started = Time.monotonic_ms()
    got = try Asr.transcribe(m, tok, pcm, options)
    ms = Time.monotonic_ms() - started
    compiled = try compiles(backend) - compiled
    rtf = ms.to_f / 1000.0 / (pcm.size.to_f / 16000.0)
    gap = (got.language_probability - c.probability).abs
    if c.language != none
      gap = 0.0
    end

    spans = got.segments.map { |s| Span(start: s.start, stop: s.stop, text: s.text) }
    same = got.tokens == c.tokens and got.text == c.text and got.language == c.detected and gap <= CLOSE
    tolerance = 0.0
    if c.clip.end_with?("-long")
      tolerance = TIME
    end

    same = same and spans_match(spans, c.segments, tolerance)
    puts("case #{i} #{c.clip} #{c.language.unwrap_or("auto")} #{c.task} timestamps=#{c.timestamps}: #{verdict(same)}")
    puts("  #{ms} ms, real-time factor #{rtf}, #{compiled} graphs compiled")
    puts("  language #{got.language} p=#{got.language_probability} (Hugging Face #{c.detected} p=#{c.probability})")
    puts("  text #{got.text}")
    puts("  avg_logprob #{got.avg_logprob}, no_speech_prob #{got.no_speech_prob}")
    if not same
      puts("  tokens #{got.tokens}")
      puts("  wanted #{c.tokens}")
      puts("  segments #{spans}")
      puts("  wanted #{c.segments}")
      all = false
    end
  end

  ok(all)
end

## `jfk` six times, with silence between, and 37 samples of silence more: 73.5 s, not a multiple of 160 samples.
def jfk_long(jfk: List[Float]) -> List[Float]
  out = jfk
  for gap in [1.0, 3.0, 0.5, 2.0, 1.0]
    out = out + silence(gap) + jfk
  end

  out + List.repeat(0.0, 37)
end

## `es` 24 times, with half a second of silence after each: 75.5 s.
def es_long(es: List[Float]) -> List[Float]
  out: List[Float] = []
  for _ in 0...24
    out = out + es + silence(0.5)
  end

  out
end

def silence(seconds: Float) -> List[Float] = List.repeat(0.0, (seconds * 16000.0).to_i!)

def spans_match(got: List[Span], want: List[Span], tolerance: Float) -> Bool
  return false if got.size != want.size
  for i in 0...got.size
    a = got[i]
    b = want[i]
    return false if a.text != b.text or (a.start - b.start).abs > tolerance or (a.stop - b.stop).abs > tolerance
  end

  true
end

def backend_of(name: String) -> Result[Tensor.Backend]
  case name
    "torch" -> ok(Tensor.Backend.Torch(Tensor.Device.Cpu))
    "reference" -> ok(Tensor.Backend.Reference)
    "ggml" -> ok(Tensor.Backend.Ggml(Tensor.Device.Cpu))
    "vulkan" -> ok(Tensor.Backend.Ggml(Tensor.Device.Vulkan(0)))
    _ -> error("BACKEND is torch, reference, ggml or vulkan, got #{name}")
  end
end

## The graphs ggml compiled so far in this process; 0 off ggml.
def compiles(backend: Tensor.Backend) -> Result[Int]
  case backend
    Tensor.Backend.Ggml(_) -> Lower.compiles()
    _ -> ok(0)
  end
end

def verdict(same: Bool) -> String
  return "match" if same
  "MISMATCH"
end

## The whole file at `path`, read in chunks.
def read_bytes(path: String) -> Result[Bytes]
  File.open(path) do |f|
    parts: List[Bytes] = []
    while true
      chunk = try File.read_bytes(f, 1048576)
      break if chunk.empty?
      parts = parts.push(chunk)
    end

    ok(Bytes.concat(parts))
  end
end

## The samples of a 16-bit PCM WAV file, scaled to -1..1.
def pcm16(wav: Bytes) -> Result[List[Float]]
  at = 12
  while at + 8 <= wav.size
    size = (try Binary.read_u32_le(wav, at + 4)).to_i
    if wav.slice(at, 4) == "data".bytes
      out: List[Float] = []
      for i in 0...(size / 2)
        out = out.push((try Binary.read_i16_le(wav, at + 8 + 2 * i)).to_f / 32768.0)
      end

      return ok(out)
    end

    at += 8 + size
  end

  error("no data chunk")
end
