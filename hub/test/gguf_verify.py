"""The independent check of `gguf_write_test.agc`: reads the GGUF that the writer made, with the standard library
alone, and compares it with `spec.json` and the source safetensors file of the `gguf_write` set of `peer.py`.

Usage: python3 -I test/gguf_verify.py DIR  (DIR holds `out.gguf`, `src.safetensors` and `spec.json`). Prints `ok` and the
number of tensors, or exits 1 with the first difference.

It checks the header (magic, version 3, counts), every key-value, the names, the dimensions (the reverse of the
shape), the types, that every tensor starts where the one before it ends rounded up to 32 (no gap, no overlap), that
the data buffer starts at a multiple of 32 and the file ends at the end of the last tensor rounded up to 32 as well
(ggml reads the padded size), and that the bytes of every tensor are the source values rounded to the target format
(to nearest even), computed here with `struct` ("e" is binary16).
"""

import json
import struct
import sys
from pathlib import Path

SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
TYPES = {0: "F32", 1: "F16", 30: "BF16"}
WIDTH = {"F32": 4, "F16": 2, "BF16": 2}


class Reader:
    def __init__(self, data):
        self.data, self.at = data, 0

    def take(self, fmt):
        value = struct.unpack_from("<" + fmt, self.data, self.at)[0]
        self.at += struct.calcsize("<" + fmt)
        return value

    def string(self):
        n = self.take("Q")
        raw = self.data[self.at:self.at + n]
        self.at += n
        return raw.decode()

    def value(self, kind):
        if kind == 8:
            return self.string()
        if kind == 9:
            item, count = self.take("I"), self.take("Q")
            return [self.value(item) for _ in range(count)]
        return self.take(SCALARS[kind])


def align(n):
    return -(-n // 32) * 32


def fail(message):
    print(message)
    sys.exit(1)


def source_values(raw, dtype):
    if dtype == "F32":
        return list(struct.unpack("<%df" % (len(raw) // 4), raw))
    if dtype == "F16":
        return list(struct.unpack("<%de" % (len(raw) // 2), raw))
    bits = struct.unpack("<%dH" % (len(raw) // 2), raw)
    return [struct.unpack("<f", struct.pack("<I", b << 16))[0] for b in bits]


def bf16_bits(v):
    b = struct.unpack("<I", struct.pack("<f", v))[0]
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF


def encode(values, target):
    if target == "F32":
        return struct.pack("<%df" % len(values), *values)
    if target == "F16":
        return struct.pack("<%de" % len(values), *values)
    return struct.pack("<%dH" % len(values), *[bf16_bits(v) for v in values])


def safetensors(path):
    data = path.read_bytes()
    n = struct.unpack_from("<Q", data, 0)[0]
    header = json.loads(data[8:8 + n])
    out = {}
    for name, t in header.items():
        if name != "__metadata__":
            a, b = t["data_offsets"]
            out[name] = (t["dtype"], t["shape"], data[8 + n + a:8 + n + b])
    return out


def main():
    folder = Path(sys.argv[1])
    spec = json.loads((folder / "spec.json").read_text())
    sources = safetensors(folder / "src.safetensors")
    data = (folder / "out.gguf").read_bytes()
    r = Reader(data)
    if data[:4] != b"GGUF" or struct.unpack_from("<I", data, 4)[0] != 3:
        fail("not a GGUF of version 3")
    r.at = 8
    tensor_count, kv_count = r.take("Q"), r.take("Q")
    if tensor_count != len(spec["tensors"]) or kv_count != len(spec["kvs"]):
        fail("counts %d, %d" % (tensor_count, kv_count))
    for kv in spec["kvs"]:
        key, kind = r.string(), r.take("I")
        value = r.value(kind)
        if key != kv["key"]:
            fail("key %s, expected %s" % (key, kv["key"]))
        want = [v for k, v in kv.items() if k != "key"][0]
        if value != want:
            fail("%s is %r, expected %r" % (key, value, want))
    infos = []
    for t in spec["tensors"]:
        name, dims = r.string(), r.take("I")
        shape = list(reversed([r.take("Q") for _ in range(dims)]))
        ggml, offset = r.take("I"), r.take("Q")
        source = sources[t["src"]]
        if name != t["out"] or shape != source[1] or TYPES.get(ggml) != t["format"]:
            fail("tensor %s %s type %s, expected %s %s %s" % (name, shape, ggml, t["out"], source[1], t["format"]))
        infos.append((name, offset, t))
    base = -(-r.at // 32) * 32
    if any(data[r.at:base]):
        fail("the padding of the header is not zero")
    end = 0
    for name, offset, t in infos:
        dtype, shape, raw = sources[t["src"]]
        want = encode(source_values(raw, dtype), t["format"])
        if dtype == t["format"] and want != raw:
            fail("%s: a copy of the same format changed bytes" % name)
        if offset != align(end):
            fail("%s at %d, expected %d (aligned, no gap, no overlap)" % (name, offset, align(end)))
        got = data[base + offset:base + offset + len(want)]
        if got != want:
            fail("%s: the bytes differ from the source rounded to %s" % (name, t["format"]))
        if any(data[base + end:base + offset]):
            fail("the padding before %s is not zero" % name)
        end = offset + len(want)
    if len(data) != base + align(end):
        fail("the file has %d bytes, expected %d (the last tensor is padded too)" % (len(data), base + align(end)))
    if any(data[base + end:]):
        fail("the padding after the last tensor is not zero")
    print("ok", len(infos))


main()
