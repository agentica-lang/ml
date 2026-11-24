## CRC-32C (Castagnoli), the checksum of TFRecord framing (`metrics.agc`, `events.agc`).

POLY = 0x82F63B78
LOW_32 = 0xFFFFFFFF
MASK_DELTA = 0xA282EAD8

# The reflected polynomial, one byte at a time.
TABLE = (0...256).to_a.map do |n|
  (0...8).reduce(n) do |c, _| c.shift_right(1).bit_xor(POLY.bit_and(0 - c.bit_and(1))) end
end

## The CRC-32C of `data`: `"123456789"` gives `0xE3069283`.
def checksum(data: Bytes) -> Int
  crc = LOW_32
  for b in data
    crc = TABLE[crc.bit_xor(b).bit_and(255)].bit_xor(crc.shift_right(8))
  end

  crc.bit_xor(LOW_32)
end

## The checksum as TFRecord stores it: rotated right by 15 bits, plus a constant, in 32 bits.
def masked(crc: Int) -> Int
  rotated = crc.shift_right(15).bit_or(crc.shift_left(17)).bit_and(LOW_32)
  (rotated + MASK_DELTA).bit_and(LOW_32)
end
