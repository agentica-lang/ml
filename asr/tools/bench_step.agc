## Times one decoder step of `Asr.step` on its own, and splits it with the profile. The model is loaded once, the audio
## goes through the log-mel, the encoder and the cross-attention cache once, and then PASSES passes run the prompt and
## forced tokens through `Asr.step` (the cache restarts every pass). It prints tab-separated lines: `pass MS STEPS`
## (the wall time of the pass in ms and its steps) and `steps N MS`, the mean per step. With `AGENTICA_VM_TRACE=profile`
## the `ml.*` sections of the report hold the split of a step (`ml.trace`, `ml.plan`, `ml.inputs`, `ml.acquire`,
## `ml.bind`, `ml.run`, `ml.readback`); PASSES 0 runs everything but the steps, and `benchmarks/step_profile.py`
## subtracts the two reports. Manual, like bench_transcribe.agc. From `asr/`:
##
## ```sh
## agentica tools/bench_step.agc MODEL_DIR WAV BACKEND LANGUAGE PASSES [TOKENS [pick]]
## ```
##
## TOKENS is a comma-separated list of ids forced after the prompt (the default is jfk.wav's transcript; `-` is the
## default). With `pick` each step is `Asr.step_pick` with an all-zero mask: the choice runs on the device and only an
## id and a log-probability come back, where the default `Asr.step` reads the logits.

import Asr from "../asr"
import Load from "pkg/hub/load"
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
  return error("usage: MODEL_DIR WAV BACKEND LANGUAGE PASSES [TOKENS [pick]]") if args.size < 5
  passes = try args[4].to_i
  ids: List[Int] = []
  spelled = JFK
  if args.size > 5 and args[5] != "-"
    spelled = args[5]
  end

  pick = args.size > 6 and args[6] == "pick"

  for part in spelled.split(",")
    ids = ids.push(try part.to_i)
  end

  try Load.vulkan_fp16_from_env(true)
  backend = try TranscribeCheck.backend_of(args[2])
  return error("#{args[2]} is not available") if not Tensor.available(backend)
  m = try Asr.open("#{args[0]}/model.safetensors", backend, try Load.weights_from_env())
  tok = try Tokenizer.open("#{args[0]}/tokenizer.json")
  pcm = try TranscribeCheck.pcm16(try TranscribeCheck.read_bytes(args[1]))
  frames = 2 * m.config.max_source_positions
  mel = try Asr.features(m, pcm, frames)
  cache = try Asr.listen(m, try Asr.encode(m, mel))
  prompt = try Tokenizer.prompt(tok, some(args[3]), Tokenizer.Job.Transcribe, false)
  mask = try Tensor.from_list(
    List.repeat(Tensor.Values.Float(0.0), m.config.vocab_size),
    [m.config.vocab_size],
    Tensor.Dtype.F32,
    Tensor.backend(m.tokens.table),
  )

  total = 0
  count = 0
  # The first pass compiles the graphs, so it is a warm-up and is not in the mean.
  for pass in 0...passes
    started = Time.monotonic_ms()
    last = cache
    if pick
      stepped = try Asr.step_pick(m, cache, prompt, mask)
      for id in ids
        next = try Asr.step_pick(m, stepped.cache, [id], mask)
        release(stepped.cache)
        stepped = next
      end

      last = stepped.cache
    else
      stepped = try Asr.step(m, cache, prompt)
      for id in ids
        next = try Asr.step(m, stepped.cache, [id])
        release(stepped.cache)
        stepped = next
      end

      last = stepped.cache
    end

    release(last)
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

## Closes the self-attention keys and values of `cache`, as `Asr.transcribe` does once a step replaced them.
def release(cache: Asr.Cache) -> Void
  for kv in cache.own
    for t in [kv.key, kv.value]
      case t.handle
        Tensor.Handle.Native(h) -> Native.close(h)
        Tensor.Handle.Ggml(h) -> Native.close(h)
        _ -> nil
      end
    end
  end
end
