"""Reads a TensorBoard event file with the standard library only and prints its records as JSON.

    python3 -I tfevents_peer.py FILE

Each record's length and data are checked against their masked CRC-32C, computed here bit by bit, and the protobuf
`Event` is decoded by hand. The output is a list of objects: `wall_time`, `step`, and either `file_version` or
`tag` and `value` (a float32 read as a double).
"""

import json
import struct
import sys

POLY = 0x82F63B78


def crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (POLY if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def masked(crc):
    return ((((crc >> 15) | (crc << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF


def varint(buf, i):
    n, shift = 0, 0
    while True:
        byte = buf[i]
        i += 1
        n |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return n, i
        shift += 7


def fields(buf):
    i = 0
    while i < len(buf):
        key, i = varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = varint(buf, i)
        elif wire == 1:
            value, i = buf[i : i + 8], i + 8
        elif wire == 2:
            size, i = varint(buf, i)
            value, i = buf[i : i + size], i + size
        elif wire == 5:
            value, i = buf[i : i + 4], i + 4
        else:
            sys.exit("unknown wire type %d" % wire)
        yield number, wire, value


def event(data):
    out = {}
    for number, _, value in fields(data):
        if number == 1:
            out["wall_time"] = struct.unpack("<d", value)[0]
        elif number == 2:
            out["step"] = value
        elif number == 3:
            out["file_version"] = value.decode()
        elif number == 5:
            for _, _, entry in (f for f in fields(value) if f[0] == 1):
                for n, _, v in fields(entry):
                    if n == 1:
                        out["tag"] = v.decode()
                    elif n == 2:
                        out["value"] = struct.unpack("<f", v)[0]
    return out


def records(raw):
    i = 0
    while i < len(raw):
        head = raw[i : i + 8]
        (length_crc,) = struct.unpack("<I", raw[i + 8 : i + 12])
        if masked(crc32c(head)) != length_crc:
            sys.exit("bad CRC of the length at byte %d" % i)
        (length,) = struct.unpack("<Q", head)
        data = raw[i + 12 : i + 12 + length]
        (data_crc,) = struct.unpack("<I", raw[i + 12 + length : i + 16 + length])
        if len(data) != length or masked(crc32c(data)) != data_crc:
            sys.exit("bad record at byte %d" % i)
        yield data
        i += 16 + length


if crc32c(b"123456789") != 0xE3069283:
    sys.exit("the peer's CRC-32C is wrong")

with open(sys.argv[1], "rb") as f:
    print(json.dumps([event(data) for data in records(f.read())]))
