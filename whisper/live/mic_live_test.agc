## Microphone capture where a device exists, and a loud skip everywhere else:
## without a capture command and a device this test prints its SKIP lines and
## passes, so CI (which has no microphone) never depends on hardware. With a
## device it captures half a second and asserts the sample count. The skip is
## structural, not just printed: only the device-less errors below skip (a
## missing command, no audio, a command that died mid-capture, a hung read),
## and anything else — notably a decode error from a broken capture — fails
## the test. The runner shows passing tests as PASS only, so the SKIP text is
## for a human running this file, not for CI logs. Needs only the engine
## (`tools/build-engine.sh`, which CI builds first): no model, no network.
## From `whisper/`:
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
    try Test.assert(device_less(result.message), "capture failed with a device present: #{result.message}")
    puts("SKIP: no microphone or capture tool on this machine (#{result.message})")
    puts("SKIP: this test only runs where a capture command and a device exist; see README")
    ok()
  end
end

# Whether a capture error means "no device here" rather than a broken
# capture: a missing command, no audio at all, a command that died mid-capture
# (ALSA xrun, unplug, exit 1), or a hung read. Anything else fails the test.
def device_less(message: String) -> Bool
  (
    message.include?("executable is unavailable") or
    message.include?("gave no samples") or
    message.include?("capture command failed") or
    message.include?("timed out")
  )
end
