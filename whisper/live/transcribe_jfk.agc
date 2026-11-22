## Validates real transcription end to end: the 11 s JFK sample from the
## whisper.cpp submodule through the pinned tiny model, asserting the exact
## greedy transcript. Decoding 176000 samples takes ~14 M Agentica steps,
## past what one test file may spend, so this runs as a program (runs are not
## step-capped), not as a test. From `whisper/`, after `tools/build-engine.sh`
## and `tools/fetch-model.sh`:
##
## ```sh
## agentica live/transcribe_jfk.agc
## ```
##
## Prints the transcript and exits 0 when it matches, or the mismatch and
## exits 1. CI runs exactly this after building the engine and downloading
## the model.

import Whisper from "../whisper"

## The sample the whisper.cpp submodule carries and the model
## `tools/fetch-model.sh` downloads, relative to `whisper/`.
SAMPLE_PATH = "../third_party/whisper.cpp/samples/jfk.wav"
MODEL_PATH = "models/ggml-tiny.bin"

## The greedy transcript of the sample with the pinned tiny model, byte for
## byte. The leading space is genuine whisper output (its tokenizer starts
## the segment with a space), not a typo. The exact match holds because
## decoding is fully deterministic here: WHISPER_SAMPLING_GREEDY takes the
## argmax at every step (no temperature sampling, no beam search) over fixed
## weights and fixed 16 kHz PCM, so the same engine pin always yields these
## bytes; a whisper.cpp upgrade that retokens or rewords needs a new pin.
EXPECTED = (
  " And so my fellow Americans ask not what your country can do for you, ask what you can do for your country."
)

def main() -> Void
  case Whisper.load(SAMPLE_PATH)
    ok(pcm) -> run(pcm)
    error(e) ->
      puts("transcribe_jfk: cannot read #{SAMPLE_PATH}: #{e.message}")
      System.exit(1)
  end
end

def run(pcm: List[Float]) -> Void
  case Whisper.load_model(MODEL_PATH)
    ok(model) -> check(model, pcm)
    error(e) ->
      puts("transcribe_jfk: cannot load #{MODEL_PATH}: #{e.message}")
      System.exit(1)
  end
end

def check(model: Whisper.Model, pcm: List[Float]) -> Void
  case Whisper.transcribe(model, pcm)
    ok(text) ->
      if text == EXPECTED
        puts(text)
      else
        puts("transcribe_jfk: mismatch, want [#{EXPECTED}], got [#{text}]")
        System.exit(1)
      end

    error(e) ->
      puts("transcribe_jfk: cannot transcribe #{SAMPLE_PATH}: #{e.message}")
      System.exit(1)
  end
end
