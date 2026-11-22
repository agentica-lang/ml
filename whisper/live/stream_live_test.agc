## The streaming layer against the pinned tiny model: a fake `Source` of
## silence transcribes to no segments. Every window is VAD-skipped, so no
## `transcribe_with` decode runs here: this proves the skip path and the
## empty merge against a real model load, while the decode-and-merge path is
## proven offline in `test/stream_test.agc` and by the live programs. The
## windows stay small on purpose: one test file shares 1 M steps per test,
## and a full five minutes of PCM would spend that on list building alone
## (which is also why the five-minute bound itself stays untested, as
## `README.md` says). The windowing of long audio is proven by the offline
## boundary tests in `test/stream_test.agc`; here the engine sees several
## windows, including an overlap and a short tail. Needs the engine (`tools/build-engine.sh`, which
## CI builds first) and the downloaded weights (`tools/fetch-model.sh`, at
## `models/ggml-tiny.bin`, never in git). From `whisper/`:
##
## ```sh
## agentica test live/
## ```

import Whisper from "../whisper"
import Mic from "../mic"
import Stream from "../stream"

private

## The model `tools/fetch-model.sh` downloads, relative to `whisper/`.
MODEL_PATH = "models/ggml-tiny.bin"

def test_stream_transcribes_silence_to_no_segments() -> Result[Void]
  silence = List.repeat(0.0, 48_000)
  model = try Whisper.load_model(MODEL_PATH)
  options = Stream.default_options().with(window_s: 1, overlap_s: 0)
  segments = try Stream.transcribe_stream(model, Mic.Source.Samples(silence), options)
  try Test.eq(segments, [])
  Test.eq(Whisper.join_text(segments), "")
end

def test_stream_covers_overlapping_windows_and_a_short_tail() -> Result[Void]
  silence = List.repeat(0.0, 80_000)
  model = try Whisper.load_model(MODEL_PATH)
  options = Stream.default_options().with(window_s: 2, overlap_s: 1)
  segments = try Stream.transcribe_stream(model, Mic.Source.Samples(silence), options)
  Test.eq(segments, [])
end
