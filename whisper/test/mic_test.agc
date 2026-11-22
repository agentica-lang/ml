## The microphone layer with no device: the pinned `arecord` invocation, the
## raw-byte conversion, the `Source` abstraction and every error that needs no
## hardware. A live device is only in `live/mic_live_test.agc`.

import Mic from "../mic"

private

def test_default_capture_is_arecord_at_16khz_mono() -> Result[Void]
  capture = Mic.default_capture()
  try Test.eq(capture.command, "arecord")
  try Test.eq(capture.args, ["-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"])
  try Test.eq(capture.timeout_ms, 30_000)
  Test.eq(capture.max_samples, 160_000)
end

def test_decode_s16le_converts_samples() -> Result[Void]
  data = Bytes.from_list([0, 0, 255, 127, 0, 128, 0, 4]).unwrap!
  samples = try Mic.decode_s16le(data)
  Test.eq(samples, [0.0, 32_767.0 / 32_768.0, -1.0, 1_024.0 / 32_768.0])
end

def test_decode_s16le_rejects_an_odd_trailing_byte() -> Result[Void]
  data = Bytes.from_list([0, 0, 1]).unwrap!
  case Mic.decode_s16le(data)
    ok(_) -> Test.assert(false, "decode_s16le took an odd byte")
    error(e) -> Test.eq(e.message, "mic: raw audio holds whole 16-bit samples")
  end
end

def test_decode_s16le_of_no_bytes_is_empty() -> Result[Void] = Test.eq(try Mic.decode_s16le("".bytes), [])

def test_read_returns_fake_samples_untouched() -> Result[Void]
  Test.eq(Mic.read(Mic.Source.Samples([0.5, -0.5])), ok([0.5, -0.5]))
end

def test_read_of_a_missing_command_is_an_error() -> Result[Void]
  source = Mic.Source.Command(Mic.default_capture().with(command: "no-such-mic-command-xyz"))
  case Mic.read(source)
    ok(_) -> Test.assert(false, "read spawned a command that is not there")
    error(e) -> Test.eq(e.message, "process: executable is unavailable")
  end
end

def test_capture_rejects_a_sample_bound_outside_five_minutes() -> Result[Void]
  case Mic.capture(Mic.default_capture().with(max_samples: 0))
    ok(_) -> Test.assert(false, "capture took zero samples")
    error(e) -> Test.eq(e.message, "mic: max_samples holds 1..4800000 samples (five minutes at 16000 Hz)")
  end
end

def test_capture_rejects_a_nonpositive_timeout() -> Result[Void]
  case Mic.capture(Mic.default_capture().with(timeout_ms: 0))
    ok(_) -> Test.assert(false, "capture took a zero timeout")
    error(e) -> Test.eq(e.message, "mic: timeout_ms is at least 1 ms")
  end
end

def test_capture_names_a_missing_command() -> Result[Void]
  spec = Mic.default_capture().with(command: "no-such-mic-command-xyz", max_samples: 8)
  case Mic.capture(spec)
    ok(_) -> Test.assert(false, "capture spawned a command that is not there")
    error(e) -> Test.eq(e.message, "process: executable is unavailable")
  end
end
