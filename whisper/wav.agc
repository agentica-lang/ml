## A 16 kHz mono WAV reader in pure Agentica: the RIFF header and its chunks are
## parsed with `std/binary`, and the 16-bit PCM samples come back as floats
## from -1.0 to just below 1.0. Anything else (another format, channels,
## rate or bit depth, a truncated file) is an `error`, never a halt.

import Binary from "std/binary"

# The most bytes read from the file at once.
CHUNK_BYTES = 1_048_576
# What the `fmt ` chunk must hold: PCM, one channel, 16 kHz, 16 bits.
FORMAT_PCM = 1
CHANNELS_MONO = 1
RATE_HZ = 16_000
BITS_PER_SAMPLE = 16

## The format the `fmt ` chunk announced.
struct Format
  channels: Int
  rate: Int
  bits: Int
end

## Reads the WAV file at `path` and decodes its samples.
def load(path: String) -> Result[List[Float]]
  handle = try File.open(path)
  bytes = read_all(handle)
  _ = File.close(handle)
  decode(try bytes)
end

## Decodes the WAV in `data`: its RIFF header, then its chunks up to `data`.
def decode(data: Bytes) -> Result[List[Float]]
  return error("a WAV needs its 12 bytes of RIFF header") if data.size < 12
  return error("not a RIFF file") if data.slice(0, 4) != "RIFF".bytes
  return error("not a WAVE file") if data.slice(8, 4) != "WAVE".bytes
  scan(data, 12, none)
end

private

def read_all(handle: File.Handle) -> Result[Bytes]
  parts: List[Bytes] = []
  chunk = try File.read_bytes(handle, CHUNK_BYTES)
  while !chunk.empty?
    parts = parts.push(chunk)
    chunk = try File.read_bytes(handle, CHUNK_BYTES)
  end

  ok(Bytes.concat(parts))
end

# The chunks from `offset` on, with the format seen so far: `fmt ` sets it,
# `data` decodes against it, any other chunk is skipped (plus its pad byte).
def scan(data: Bytes, offset: Int, format: Optional[Format]) -> Result[List[Float]]
  return error("a WAV has no data chunk") if offset + 8 > data.size
  id = data.slice(offset, 4)
  size = (try Binary.read_u32_le(data, offset + 4)).to_i
  start = offset + 8
  return error("a WAV chunk runs past its bytes") if start + size > data.size
  if id == "fmt ".bytes
    format = some(try read_format(data, start, size))
  elsif id == "data".bytes
    _ = try have_format(format)
    return pcm16(data, start, size)
  end

  rest = start + size
  if size % 2 == 1
    rest += 1
  end

  scan(data, rest, format)
end

# The format of a `fmt ` chunk of `size` bytes at `start`: only PCM mono
# 16 kHz 16-bit decodes.
def read_format(data: Bytes, start: Int, size: Int) -> Result[Format]
  return error("a fmt chunk holds 16 bytes at least") if size < 16
  format = (try Binary.read_u16_le(data, start)).to_i
  return error("only PCM WAV decodes") if format != FORMAT_PCM
  channels = (try Binary.read_u16_le(data, start + 2)).to_i
  return error("only mono WAV decodes") if channels != CHANNELS_MONO
  rate = (try Binary.read_u32_le(data, start + 4)).to_i
  return error("only 16 kHz WAV decodes") if rate != RATE_HZ
  bits = (try Binary.read_u16_le(data, start + 14)).to_i
  return error("only 16-bit WAV decodes") if bits != BITS_PER_SAMPLE
  ok(Format(channels: channels, rate: rate, bits: bits))
end

def have_format(format: Optional[Format]) -> Result[Format]
  case format
    some(found) -> ok(found)
    none -> error("a WAV needs its fmt chunk before its data")
  end
end

# The `size` bytes at `start` as 16-bit samples over 32768.
def pcm16(data: Bytes, start: Int, size: Int) -> Result[List[Float]]
  return error("a data chunk holds whole 16-bit samples") if size % 2 == 1
  samples: List[Float] = []
  offset = start
  while offset < start + size
    value = try Binary.read_i16_le(data, offset)
    samples = samples.push(value.to_f / 32768.0)
    offset += 2
  end

  ok(samples)
end
