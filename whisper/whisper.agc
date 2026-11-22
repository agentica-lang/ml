## Whisper speech-to-text (Via A): a thin binding over whisper.cpp.
##
## The `.agc` file is the one source of the native signatures: after each
## change to the `extern module` block, make the header again and check that
## it is current. With the native-gen dependency added (`agentica add
## native-gen https://github.com/agentica-lang/packages --tag native-gen-0.4.0
## --subdir native-gen`), from this folder:
##
## ```sh
## agentica deps/native-gen/cli.agc whisper.agc --output native/whisper_native.h
## agentica deps/native-gen/cli.agc whisper.agc --output native/whisper_native.h --check
## ```
##
## Audio enters as 16 kHz mono PCM floats in `Wav` (a pure Agentica WAV
## reader); a loaded model is a `Model` resource, and transcription runs in
## the compute pool. The native stub answers `error` until whisper.cpp is
## linked (Fase W, Via A of `roadmap.md`).

import Wav from "wav"

export Wav.decode
export Wav.load

extern module "whisper"
  ## A loaded model: a resource whose state the native side owns.
  resource Model
  ## Loads the model at `path` (a ggml/gguf file); error when it cannot be read.
  def load_model(path: String) -> Result[Model]
  ## Transcribes 16 kHz mono PCM audio; error when the model cannot run it.
  compute def transcribe(model: Model, pcm: List[Float]) -> Result[String]
end
