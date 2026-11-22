## Microphone capture for whisper: 16 kHz mono s16le PCM from a command run
## through `Process`, plus the `Source` abstraction that lets tests feed
## synthetic PCM instead of a device.
##
## `Capture` is how one capture runs: the command and its arguments (by
## default `arecord -q -f S16_LE -r 16000 -c 1 -t raw`, which writes raw 16 kHz
## mono signed 16-bit samples to standard output), a per-read timeout and the
## maximum number of samples to keep. `capture` spawns the command, reads up
## to `max_samples` samples of its standard output, closes the process on
## every path and decodes the bytes with `decode_s16le`. The command's
## standard error passes through to ours, so a missing device explains itself
## there; keep the invocation quiet (`-q`, `-hide_banner -nostats -loglevel
## error`) or route its stderr away (`sh -c '... 2>/dev/null'`), since a
## chatty child fills that buffer and stalls the read until the timeout. No
## native audio module is involved: `Process` is the transport, and
## it keeps this layer testable with a fake source.
##
## Device selection stays in `args`: `arecord -D plughw:1,0 ...` picks a card,
## and any command that writes the same raw format works, for example
## `sox -d -t raw -r 16000 -c 1 -b 16 -e signed-integer -` or
## `ffmpeg -f alsa -i default -ar 16000 -ac 1 -f s16le -` (both untested here:
## this package only pins `arecord`). Nothing here resamples: the command
## must output 16 kHz mono s16le, which `-r 16000 -c 1` (or the equivalent
## flags) ask the tool for.
##
## `Source` is what `Stream.transcribe_stream` reads: `Samples` holds
## synthetic PCM (tests feed this, so no device is needed), `Command` runs a
## live capture once read.

import Binary from "std/binary"

# The audio this package moves: 16 kHz mono, two bytes per sample.
RATE_HZ = 16_000
BYTES_PER_SAMPLE = 2
# The most samples one `capture` keeps: five minutes at 16 kHz, the same bound
# one `transcribe_with` call decodes, so a capture always fits the engine.
MAX_SAMPLES = 4_800_000
# The most bytes read from the device process at once (16 384 samples, about
# one second of audio). Raw reads may split a sample, so `read_raw` gathers
# bytes and `decode_s16le` converts once the capture stops.
CHUNK_BYTES = 32_768
# How long `capture` waits for a command that ended on its own to report its
# exit status. The command already closed its output by then, so this returns
# at once; it only bounds a command that lingers after its output ended.
EXIT_WAIT_MS = 2_000

## How one capture runs: the command that writes raw 16 kHz mono s16le PCM to
## its standard output, the arguments it takes, the timeout of each read and
## the maximum number of samples to keep before the process is closed.
## `default_capture` builds the pinned `arecord` invocation.
struct Capture
  command: String = "arecord"
  args: List[String]
  timeout_ms: Int = 30_000
  max_samples: Int = 160_000
end

## Where the audio of one capture (or one `transcribe_stream` call) comes
## from: `Samples` holds synthetic PCM, `Command` runs a live capture once
## read.
enum Source
  Samples(List[Float])
  Command(Capture)
end

## The default capture: `arecord -q -f S16_LE -r 16000 -c 1 -t raw` (ten seconds
## at most), writing raw 16 kHz mono s16le PCM to standard output.
def default_capture() -> Capture
  Capture(
    command: "arecord",
    args: ["-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
    timeout_ms: 30_000,
    max_samples: 160_000,
  )
end

## Captures up to `spec.max_samples` samples (ten seconds by default) of 16
## kHz mono PCM from `spec.command`: error when the bound is not
## 1..4800000 samples, when the timeout is not at least 1 ms, when the
## command cannot start, when a read times out, when the command wrote no
## audio, or when the command died mid-capture (a nonzero exit on the natural
## end of its output discards the partial audio: it is never returned, so a
## truncated utterance cannot transcribe as if complete; only a command that
## exited 0 may leave partial audio, and the cap path kills deliberately, so
## its status is meaningless there). The process is closed on every path, so
## no call leaves one running.
def capture(spec: Capture) -> Result[List[Float]]
  if spec.max_samples < 1 or spec.max_samples > MAX_SAMPLES
    return error("mic: max_samples holds 1..4800000 samples (five minutes at 16000 Hz)")
  end

  return error("mic: timeout_ms is at least 1 ms") if spec.timeout_ms < 1
  process = try Process.spawn(spec.command, spec.args, Process.Options(timeout_ms: spec.timeout_ms))
  outcome = read_raw(process, spec.max_samples, spec.timeout_ms)
  failure = check_exit(process, outcome)
  Process.close(process)
  try failure
  raw = try outcome
  if raw.data.empty?
    return error(
      "mic: the capture gave no samples (check the device and the command; its own error is on standard error)",
    )
  end

  decode_s16le(raw.data)
end

## Reads `source` to its samples: `Samples` as they are, `Command` through
## `capture`.
def read(source: Source) -> Result[List[Float]]
  case source
    Source.Samples(samples) -> ok(samples)
    Source.Command(spec) -> capture(spec)
  end
end

## Decodes raw s16le PCM `data` to floats from -1.0 to just below 1.0, as
## `Wav` decodes its `data` chunk; error on an odd byte, which is a truncated
## stream.
def decode_s16le(data: Bytes) -> Result[List[Float]]
  return error("mic: raw audio holds whole 16-bit samples") if data.size % 2 == 1
  samples: List[Float] = []
  offset = 0
  while offset < data.size
    value = try Binary.read_i16_le(data, offset)
    samples = samples.push(value.to_f / 32768.0)
    offset += 2
  end

  ok(samples)
end

private

# How one read of the device process ended: the sample cap (the later kill
# is deliberate, so the exit status is meaningless) or the end of the output
# (the command exited on its own, so its status decides). `read_raw` reports
# which case it hit, so `capture` only checks the status where it means
# something.
struct RawRead
  data: Bytes
  hit_cap: Bool
end

# On the end-of-output path the command exited on its own: a nonzero status
# means the audio is truncated, so the partial audio is discarded with an
# error, never returned. On the cap path (or when the read itself failed)
# there is nothing to check.
def check_exit(process: Process, outcome: Result[RawRead]) -> Result[Void]
  if outcome.ok?
    raw = outcome.unwrap!
    if !raw.hit_cap
      status = try Process.wait(process, EXIT_WAIT_MS)
      if status != 0
        return error("mic: the capture command failed (exit #{status}; partial audio is discarded, never returned)")
      end
    end
  end

  ok()
end

# Reads up to `max_samples` samples of the standard output of `process`: up
# to `CHUNK_BYTES` bytes per read, until the cap or the end of the output (an
# empty read). A read timeout kills and closes the process, so the caller
# still closes it and answers the timeout error.
def read_raw(process: Process, max_samples: Int, timeout_ms: Int) -> Result[RawRead]
  parts: List[Bytes] = []
  total = 0
  want = max_samples * BYTES_PER_SAMPLE
  if want > CHUNK_BYTES
    want = CHUNK_BYTES
  end

  chunk = try Process.read_bytes(process, want, timeout_ms)
  while !chunk.empty? and total < max_samples
    parts = parts.push(chunk)
    total += chunk.size / BYTES_PER_SAMPLE
    if total < max_samples
      rest = (max_samples - total) * BYTES_PER_SAMPLE
      if rest > CHUNK_BYTES
        rest = CHUNK_BYTES
      end

      chunk = try Process.read_bytes(process, rest, timeout_ms)
    end
  end

  ok(RawRead(data: Bytes.concat(parts), hit_cap: total >= max_samples))
end
