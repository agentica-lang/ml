## TensorBoard event files: the TFRecord framing and the protobuf of an `Event` with a scalar `Summary`, written by hand
## (`train/DESIGN.md`, "Decisions", "Metrics live in `train/`"). A file is the records of `file_version` and then of `scalar`, each framed by `record`.
##
## ```text
## Event   { double wall_time = 1; int64 step = 2; string file_version = 3; Summary summary = 5 }
## Summary { repeated Value value = 1 }
## Value   { string tag = 1; float simple_value = 2 }
## ```

import Binary from "std/binary"
import Crc32c from "crc32c"

VARINT = 0
FIXED_64 = 1
LENGTH_DELIMITED = 2
FIXED_32 = 5

## The data of the first record of a file, which names the format.
def file_version(wall_time: Float) -> Bytes = Bytes.concat([double(1, wall_time), text(3, "brain.Event:2")])

## The data of the record of a scalar `value` (a float32 on the wire) of the series `tag`, at `step`.
def scalar(wall_time: Float, step: Int, tag: String, value: Float) -> Bytes
  entry = Bytes.concat([text(1, tag), key(2, FIXED_32), Binary.write_f32_le(value.to_float32)])
  summary = nested(1, entry)
  Bytes.concat([double(1, wall_time), key(2, VARINT), varint(step), nested(5, summary)])
end

## `data` as a TFRecord: its length as a u64, the masked CRC-32C of that, the data, and its masked CRC-32C. All are
## little-endian.
def record(data: Bytes) -> Result[Bytes]
  length = Binary.write_i64_le(data.size)
  head = Binary.write_u32_le(try Crc32c.masked(Crc32c.checksum(length)).to_uint32)
  tail = Binary.write_u32_le(try Crc32c.masked(Crc32c.checksum(data)).to_uint32)
  ok(Bytes.concat([length, head, data, tail]))
end

private

def key(field: Int, wire: Int) -> Bytes = varint(field.shift_left(3).bit_or(wire))

# Seven bits at a time, the lowest first, a set high bit on every byte but the last. The value is 0 or more (callers
# check), so every item is a byte and `unwrap!` cannot halt.
def varint(n: Int) -> Bytes
  out: List[Int] = []
  rest = n
  while rest >= 128
    out = out.push(rest.bit_and(127).bit_or(128))
    rest = rest.shift_right(7)
  end

  Bytes.from_list(out.push(rest)).unwrap!
end

def double(field: Int, x: Float) -> Bytes = key(field, FIXED_64) + Binary.write_f64_le(x)

def text(field: Int, s: String) -> Bytes = nested(field, s.bytes)

def nested(field: Int, payload: Bytes) -> Bytes
  Bytes.concat([key(field, LENGTH_DELIMITED), varint(payload.size), payload])
end
