## Whisper speech-to-text (Via A): a thin binding over whisper.cpp.
##
## The `.agc` file is the one source of the native signatures: after each
## change to the `extern module` block, make the header again and check that
## it is current. native-gen comes from `agentica install` (see
## `agentica.lock`); from this folder:
##
## ```sh
## agentica install
## agentica deps/native-gen/cli.agc whisper.agc --output native/whisper_native.h
## agentica deps/native-gen/cli.agc whisper.agc --output native/whisper_native.h --check
## ```
##
## Audio enters as 16 kHz mono PCM floats in `Wav` (a pure Agentica WAV
## reader); a loaded model is a `Model` resource. Both native calls run in
## the compute pool: loading reads and parses the whole weights file, and
## transcribing runs the full encoder/decoder. The engine is the pinned
## submodule at `third_party/whisper.cpp`, built CPU-only by
## `tools/build-engine.sh` (its libraries are linked, its sources are never
## copied); weights are downloaded by `tools/fetch-model.sh` and never enter
## git. What each test layer covers is in `README.md`. No microphone here:
## file input only (Fase W, Via A of `roadmap.md`).

import Wav from "wav"

export Wav.decode
export Wav.load

extern module "whisper"
  ## A loaded model: a resource whose state the native side owns (the
  ## whisper context plus the mutex that serializes inference on it).
  resource Model
  ## Loads the model at `path` (a ggml/gguf file); error when it cannot be read.
  compute def load_model(path: String) -> Result[Model]
  ## Transcribes 16 kHz mono PCM audio (at most five minutes: longer input is
  ## an error, so one call is bounded); error when the model cannot run it.
  compute def transcribe(model: Model, pcm: List[Float]) -> Result[String]
end
