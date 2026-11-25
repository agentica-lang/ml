## CRC-32C (Castagnoli), the checksum of TFRecord framing (`metrics.agc`, `events.agc`).

POLY: UInt32 = 0x82F63B78
MASK_DELTA: UInt32 = 0xA282EAD8

# The reflected polynomial, one byte at a time.
TABLE: List[UInt32] = (0...256).to_a.map do |n|
  (0...8).reduce(n.to_uint32!) do |c, _| c.shift_right(1).bit_xor(POLY.bit_and(c.bit_and(1) * 0xFFFFFFFF)) end
end

## The CRC-32C of `data`: `"123456789"` gives `0xE3069283`.
def checksum(data: Bytes) -> UInt32
  crc: UInt32 = 0xFFFFFFFF
  for b in data
    crc = TABLE[crc.bit_xor(b.to_uint32!).bit_and(255).to_i].bit_xor(crc.shift_right(8))
  end

  crc.bit_not
end

## The checksum as TFRecord stores it: rotated right by 15 bits, plus a constant, in 32 bits.
def masked(crc: UInt32) -> UInt32 = crc.shift_right(15).bit_or(crc.shift_left(17)).wrapping_add(MASK_DELTA)
