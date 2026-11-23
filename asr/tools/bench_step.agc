## Times one decoder step of `Asr.step` on its own, and splits it with the profile. The model is loaded once, the audio
## goes through the log-mel, the encoder and the cross-attention cache once, and then PASSES passes run the prompt and
## forced tokens through `Asr.step` (the cache restarts every pass). It prints tab-separated lines: `pass MS STEPS`
## (the wall time of the pass in ms and its steps) and `steps N MS`, the mean per step. With `AGENTICA_VM_TRACE=profile`
## the `ml.*` sections of the report hold the split of a step (`ml.trace`, `ml.plan`, `ml.inputs`, `ml.acquire`,
## `ml.bind`, `ml.run`, `ml.readback`); PASSES 0 runs everything but the steps, and `benchmarks/step_profile.py`
## subtracts the two reports. Manual, like bench_transcribe.agc. From `asr/`:
##
## ```sh
## agentica tools/bench_step.agc MODEL_DIR WAV BACKEND LANGUAGE PASSES [TOKENS]
## ```
##
## TOKENS is a comma-separated list of ids forced after the prompt (the default is jfk.wav's transcript).

import Asr from "../asr"
import Mel from "../mel"
import Tensor from "pkg/tensor/tensor"
import Tokenizer from "../tokenizer"
import TranscribeCheck from "transcribe_check"

## The ids of whisper-tiny's transcript of jfk.wav (asr/test/fixtures/transcribe_golden.json).
JFK = "400,370,452,7177,6280,1029,406,437,428,1941,393,360,337,291,1029,437,291,393,360,337,428,1941,13"

def main() -> Void
  case run(System.args())
    ok(_) -> nil
    error(e) ->
      puts("bench_step: #{e.message}")
      System.exit(2)
  end
end

def run(args: List[String]) -> Result[Void]
  return error("usage: MODEL_DIR WAV BACKEND LANGUAGE PASSES [TOKENS]") if args.size < 5
  passes = try args[4].to_i
  ids: List[Int] = []
  spelled = JFK
  if args.size > 5
    spelled = args[5]
  end

  for part in spelled.split(",")
    ids = ids.push(try part.to_i)
  end

  backend = try TranscribeCheck.backend_of(args[2])
  return error("#{args[2]} is not available") if not Tensor.available(backend)
  m = try Asr.open("#{args[0]}/model.safetensors", backend)
  tok = try Tokenizer.open("#{args[0]}/tokenizer.json")
  pcm = try TranscribeCheck.pcm16(try TranscribeCheck.read_bytes(args[1]))
  frames = 2 * m.config.max_source_positions
  features = try Mel.log_mel(pcm, m.config.num_mel_bins, frames)
  mel = try Tensor.from_list(
    features.map { |x| Tensor.Values.Float(x) },
    [1, m.config.num_mel_bins, frames],
    Tensor.Dtype.F32,
    Tensor.backend(m.tokens.table),
  )

  cache = try Asr.listen(m, try Asr.encode(m, mel))
  prompt = try Tokenizer.prompt(tok, some(args[3]), Tokenizer.Job.Transcribe, false)
  total = 0
  count = 0
  # The first pass compiles the graphs, so it is a warm-up and is not in the mean.
  for pass in 0...passes
    started = Time.monotonic_ms()
    stepped = try Asr.step(m, cache, prompt)
    for id in ids
      stepped = try Asr.step(m, stepped.cache, [id])
    end

    elapsed = Time.monotonic_ms() - started
    puts("pass\t#{elapsed}\t#{ids.size + 1}")
    if pass > 0
      total += elapsed
      count += ids.size + 1
    end
  end

  if count > 0
    puts("steps\t#{count}\t#{total.to_f / count.to_f}")
  end

  ok()
end
