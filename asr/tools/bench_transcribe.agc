## Times `Asr.transcribe` the way benchmarks/whisper_cpp_bench.cpp times whisper.cpp: the model is loaded once, one
## warm-up transcription, then RUNS timed ones (the whole pipeline: log-mel, encoder, greedy decoding, no timestamps).
## RUNS -1 stops after the load. It prints tab-separated lines: `load`, `warmup`, `run` (one per run, in ms), `text`,
## `tokens`, and `phase NAME MS STEPS` for one more pass split into its parts. Manual, like transcribe_check.agc;
## benchmarks/bench_whisper.sh runs it. The stage pass calls the stage functions (`Asr.encode`, `Asr.listen`,
## `Asr.step`) and repeats `transcribe`'s own order, so it must follow them when they change.
## From `asr/`:
##
## ```sh
## agentica tools/bench_transcribe.agc MODEL_DIR WAV BACKEND LANGUAGE [RUNS [stamps]]
## ```
##
## `stamps` decodes with timestamps (the default of `Asr.Options`); without it, as here, there are none.
## MODEL_DIR holds `model.safetensors`, `config.json` and `tokenizer.json`; BACKEND is `torch`, `ggml` or `vulkan`.

import Asr from "../asr"
import Load from "pkg/hub/load"
import Tensor from "pkg/tensor/tensor"
import Tokenizer from "../tokenizer"
import TranscribeCheck from "transcribe_check"

def main() -> Void
  case run(System.args())
    ok(_) -> nil
    error(e) ->
      puts("bench_transcribe: #{e.message}")
      System.exit(2)
  end
end

def run(args: List[String]) -> Result[Void]
  return error("usage: MODEL_DIR WAV BACKEND LANGUAGE [RUNS [stamps]]") if args.size < 4
  runs = 5
  if args.size > 4
    runs = try args[4].to_i
  end

  try Load.vulkan_fp16_from_env(true)
  backend = try TranscribeCheck.backend_of(args[2])
  return error("#{args[2]} is not available") if not Tensor.available(backend)
  started = Time.monotonic_ms()
  m = try Asr.open("#{args[0]}/model.safetensors", backend, try Load.weights_from_env())
  tok = try Tokenizer.open("#{args[0]}/tokenizer.json")
  puts("load\t#{Time.monotonic_ms() - started}")
  return ok() if runs < 0
  pcm = try TranscribeCheck.pcm16(try TranscribeCheck.read_bytes(args[1]))
  options = Asr.Options(language: some(args[3]), timestamps: args.size > 5 and args[5] == "stamps")
  started = Time.monotonic_ms()
  got = try Asr.transcribe(m, tok, pcm, options)
  puts("warmup\t#{Time.monotonic_ms() - started}")
  for _ in 0...runs
    started = Time.monotonic_ms()
    try Asr.transcribe(m, tok, pcm, options)
    puts("run\t#{Time.monotonic_ms() - started}")
  end

  puts("text\t#{got.text}")
  puts("tokens\t#{got.tokens.map { |t| t.to_s }.join(" ")}")
  # The stage pass is of one window.
  return ok() if pcm.size >= 2 * m.config.max_source_positions * 160 + 160
  phases(m, tok, pcm, got.tokens, args[3])
end

## One pass over the stages of `transcribe`, forcing the tokens the run produced, so each stage has its own time.
def phases(
  m: Asr.Model,
  tok: Tokenizer.Tokenizer,
  pcm: List[Float],
  tokens: List[Int],
  language: String,
) -> Result[Void]
  frames = 2 * m.config.max_source_positions
  started = mark()
  mel = try Asr.features(m, pcm, frames)
  lap("log_mel", started)
  started = mark()
  audio = try Asr.encode(m, mel)
  lap("encoder", started)
  started = mark()
  cache = try Asr.listen(m, audio)
  lap("cross_cache", started)
  prompt = try Tokenizer.prompt(tok, some(language), Tokenizer.Job.Transcribe, false)
  started = mark()
  stepped = try Asr.step(m, cache, prompt)
  lap("prompt_step", started)
  started = mark()
  for id in tokens
    stepped = try Asr.step(m, stepped.cache, [id])
  end

  lap("decode_steps(#{tokens.size})", started)
  ok()
end

## The clock and the VM step count now, for `lap`.
def mark() -> List[Int] = [Time.monotonic_ms(), System.Profile.steps()]

## Prints the stage's time in ms and its VM steps (the interpreter's own work: tracing, lowering, the greedy rules).
def lap(name: String, started: List[Int]) -> Void
  puts("phase\t#{name}\t#{Time.monotonic_ms() - started[0]}\t#{System.Profile.steps() - started[1]}")
end
