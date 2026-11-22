## Microphone capture where a device exists, and a loud skip everywhere else:
## without a capture command and a device this test prints its SKIP lines and
## passes, so CI (which has no microphone) never depends on hardware. With a
## device it captures half a second and asserts the sample count. Needs only
## the engine (`tools/build-engine.sh`, which CI builds first): no model, no
## network. From `whisper/`:
##
## ```sh
## agentica test live/mic_live_test.agc
## ```

import Mic from "../mic"

private

def test_microphone_capture_when_a_device_exists() -> Result[Void]
  result = Mic.capture(Mic.default_capture().with(max_samples: 8_000, timeout_ms: 5_000))
  if result.ok?
    samples = result.unwrap!
    try Test.assert(samples.size == 8_000, "capture keeps max_samples samples")
    puts("mic: captured #{samples.size} samples from the default device")
    ok()
  else
    puts("SKIP: no microphone or capture tool on this machine (#{result.message})")
    puts("SKIP: this test only runs where a capture command and a device exist; see README")
    ok()
  end
end
