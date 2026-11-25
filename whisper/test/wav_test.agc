## The WAV reader on synthetic bytes: every fixture is built in-test, so no
## binary file lives in git.

import Binary from "std/binary"
import Wav from "../wav"

private

def test_16khz_mono_pcm_decodes_to_floats() -> Result[Void]
  samples = try Wav.decode(try wav_bytes([0, 32_767, -32_768, 4_096]))
  Test.eq(samples, [0.0, 32_767.0 / 32_768.0, -1.0, 0.125])
end

def test_an_unknown_chunk_is_skipped() -> Result[Void]
  junk = Bytes.concat(["JUNK".bytes, Binary.write_u32_le(3), Bytes.from_list([1, 2, 3]).unwrap!])
  wav = try wav_bytes([1_000])
  # An odd chunk takes a pad byte; the data still decodes after it.
  with_junk = Bytes.concat([wav.slice(0, 12), junk, Bytes.from_list([0]).unwrap!, wav.slice(12, wav.size - 12)])
  samples = try Wav.decode(with_junk)
  Test.eq(samples, [1_000.0 / 32_768.0])
end

def test_a_wav_without_data_is_an_error() -> Result[Void]
  wav = try wav_bytes([1])
  head = wav.slice(0, wav.size - 10)
  Test.assert(not Wav.decode(head).ok?, "a header and fmt with no data chunk")
end

def test_what_is_not_a_wav_is_an_error() -> Result[Void]
  good = try wav_bytes([1])
  not_riff = Bytes.concat(["RIFX".bytes, good.slice(4, good.size - 4)])
  try Test.assert(not Wav.decode(not_riff).ok?, "another magic")
  try Test.assert(not Wav.decode(good.slice(0, 11)).ok?, "a truncated header")
  try Test.assert(not Wav.decode(try wav_bytes([1], 8_000)).ok?, "another rate")
  try Test.assert(not Wav.decode(try wav_bytes([1, 2], 16_000, 2)).ok?, "stereo")
  Test.assert(not Wav.decode("not a wav".bytes).ok?, "plain text")
end

def test_data_before_fmt_is_an_error() -> Result[Void]
  wav = try wav_bytes([1])
  fmt = wav.slice(12, 24)
  data = wav.slice(36, wav.size - 36)
  swapped = Bytes.concat([wav.slice(0, 12), data, fmt])
  Test.assert(not Wav.decode(swapped).ok?, "a data chunk with no fmt before it")
end

def test_load_reads_a_file() -> Result[Void]
  bytes = try wav_bytes([0, -4_096])
  path = "wav_test_#{OpenSSL.Random.hex(4)}.wav"
  handle = try File.open(path, "write")
  wrote = File.write_bytes(handle, bytes)
  _ = File.close(handle)
  try wrote
  samples = Wav.load(path)
  _ = Path.remove(path)
  Test.eq(try samples, [0.0, -0.125])
end

# A minimal 16-bit WAV of `samples` at `rate` Hz on `channels` channels.
def wav_bytes(samples: List[Int], rate: Int = 16_000, channels: Int = 1) -> Result[Bytes]
  pcm: List[Bytes] = []
  for sample in samples
    pcm = pcm.push(Binary.write_i16_le(try sample.to_int16))
  end

  body = Bytes.concat(pcm)
  fmt = Bytes.concat([
    Binary.write_u16_le(1),
    Binary.write_u16_le(try channels.to_uint16),
    Binary.write_u32_le(try rate.to_uint32),
    Binary.write_u32_le(try (rate * channels * 2).to_uint32),
    Binary.write_u16_le(try (channels * 2).to_uint16),
    Binary.write_u16_le(16),
  ])

  ok(
    Bytes.concat([
      "RIFF".bytes,
      Binary.write_u32_le(try (36 + body.size).to_uint32),
      "WAVE".bytes,
      "fmt ".bytes,
      Binary.write_u32_le(16),
      fmt,
      "data".bytes,
      Binary.write_u32_le(try body.size.to_uint32),
      body,
    ]),
  )
end
