"""The independent peer of the hub/ tests: writes safetensors files with the
standard library alone (struct + json) and the values a reader must find.

Usage: python3 peer.py OUT_DIR
"""

import json
import math
import struct
import sys
from pathlib import Path

FORMATS = {"F32": "f", "F16": "e", "F64": "d", "I64": "q", "I32": "i",
           "I16": "h", "I8": "b", "U8": "B", "BOOL": "?"}


def tensor(dtype, shape, values):
    return {"dtype": dtype, "shape": shape,
            "raw": struct.pack("<%d%s" % (len(values), FORMATS[dtype]), *values)}


def write(path, tensors, metadata=None, order=None):
    """Lays the buffers out in `order` (default: reversed names) so the
    header's key order and the buffer order differ, and pads the header
    with spaces to 8 bytes, as the reference writer does."""
    header = {"__metadata__": metadata} if metadata else {}
    data = b""
    for name in order or sorted(tensors, reverse=True):
        t = tensors[name]
        header[name] = {"dtype": t["dtype"], "shape": t["shape"],
                        "data_offsets": [len(data), len(data) + len(t["raw"])]}
        data += t["raw"]
    header = dict(sorted(header.items()))
    raw_header(path, json.dumps(header).encode(), data)


def raw_header(path, text, data=b""):
    text += b" " * (-len(text) % 8)
    path.write_bytes(struct.pack("<Q", len(text)) + text + data)


def decoded(t):
    n = len(t["raw"]) // struct.calcsize(FORMATS[t["dtype"]])
    values = struct.unpack("<%d%s" % (n, FORMATS[t["dtype"]]), t["raw"])
    return {"dtype": t["dtype"], "shape": t["shape"], "values": list(values)}


def values_file(out):
    half_bits = [0x3C00, 0xC000, 0x0001, 0x83FF, 0x7BFF, 0x8000, 0x3555]
    halves = [struct.unpack("<e", struct.pack("<H", b))[0] for b in half_bits]
    tensors = {
        "f32": tensor("F32", [2, 3], [1.5, -2.25, 0.1, 3.4e38, -0.0, 1e-40]),
        "f16": tensor("F16", [7], halves),
        "f64": tensor("F64", [2], [0.1, -1e300]),
        "i64": tensor("I64", [3], [-2**63, 2**53, -5]),
        "i32": tensor("I32", [2, 2], [-2**31, 2**31 - 1, 0, -7]),
        "i16": tensor("I16", [2], [-32768, 32767]),
        "i8": tensor("I8", [3], [-128, 127, -1]),
        "u8": tensor("U8", [2], [0, 255]),
        "bool": tensor("BOOL", [3], [True, False, True]),
        "scalar": tensor("F32", [], [2.5]),
        "empty": tensor("F32", [0, 3], []),
    }
    write(out / "values.safetensors", tensors, {"format": "pt"})
    expected = {name: decoded(t) for name, t in tensors.items()}
    (out / "values.json").write_text(json.dumps(expected))
    write(out / "bf16.safetensors",
          {"w": {"dtype": "BF16", "shape": [2], "raw": b"\x80\x3f\x00\x40"},
           "ok": tensor("F32", [1], [1.0])})
    raw_header(out / "bad_bool.safetensors",
               b'{"b":{"dtype":"BOOL","shape":[2],"data_offsets":[0,2]}}', b"\x01\x02")


def entry(dtype, shape, start, stop):
    return {"dtype": dtype, "shape": shape, "data_offsets": [start, stop]}


def malformed(out):
    """Each refused file and a phrase its error must hold."""
    cases = {
        "short": (None, "too short", b"\x01\x00"),
        "past_end": (None, "past the end", struct.pack("<Q", 1000) + b"{}"),
        "huge": (None, "over the limit", struct.pack("<Q", 2**40) + b"{}"),
        "not_utf8": (b'{"\xff":1}', "UTF-8", b""),
        "not_object": (b"[]", "does not start with {", b""),
        "bad_json": (b'{"a": ', "invalid JSON", b""),
        "unknown_dtype": ({"a": entry("X9", [1], 0, 1)}, "unknown dtype X9", b"\x00"),
        "missing_shape": ({"a": {"dtype": "F32", "data_offsets": [0, 4]}}, "tensor a", b"\0" * 4),
        "negative_dim": ({"a": entry("F32", [-1], 0, 4)}, "negative dimension", b"\0" * 4),
        "reversed": ({"a": entry("F32", [0], 4, 0)}, "outside", b"\0" * 4),
        "out_of_range": ({"a": entry("F32", [2], 0, 8)}, "outside", b"\0" * 4),
        "size_mismatch": ({"a": entry("F32", [3], 0, 8)}, "does not fit", b"\0" * 8),
        "gap": ({"a": entry("F32", [1], 0, 4), "b": entry("F32", [1], 8, 12)}, "gap", b"\0" * 12),
        "overlap": ({"a": entry("F32", [2], 0, 8), "b": entry("F32", [2], 4, 12)}, "overlaps", b"\0" * 12),
        "tail": ({"a": entry("F32", [1], 0, 4)}, "cover 4 of the 8", b"\0" * 8),
        "bad_metadata": ({"__metadata__": {"k": 1}}, "__metadata__", b""),
        "short_tensor": ({"a": entry("F32", [1], 0, 8)}, "needs 4 bytes", b"\0" * 8),
        "huge_shape": ({"a": entry("F32", [2**40, 2**40], 0, 4)}, "does not fit", b"\0" * 4),
    }
    phrases = {}
    for name, (header, phrase, data) in cases.items():
        path = out / ("bad_%s.safetensors" % name)
        if header is None:
            path.write_bytes(data)
        else:
            text = header if isinstance(header, bytes) else json.dumps(header).encode()
            raw_header(path, text, data)
        phrases[path.name] = phrase
    (out / "malformed.json").write_text(json.dumps(phrases))


def sharded(out):
    one = {"enc.a": tensor("F32", [2], [1.0, -1.0]), "enc.b": tensor("I64", [1], [42])}
    two = {"enc.c": tensor("F16", [3], [0.5, 2.0, -4.0])}
    write(out / "model-00001-of-00002.safetensors", one)
    write(out / "model-00002-of-00002.safetensors", two)
    weight_map = {n: "model-00001-of-00002.safetensors" for n in one}
    weight_map.update({n: "model-00002-of-00002.safetensors" for n in two})
    index = {"metadata": {"total_size": 20}, "weight_map": weight_map}
    (out / "model.safetensors.index.json").write_text(json.dumps(index))
    expected = {n: decoded(t) for n, t in {**one, **two}.items()}
    (out / "sharded.json").write_text(json.dumps(expected))
    bad = {
        "missing": {"enc.z": "model-00001-of-00002.safetensors"},
        "escape": {"enc.a": "../model-00001-of-00002.safetensors"},
        "absent": {"enc.a": "model-00003-of-00002.safetensors"},
    }
    for name, wm in bad.items():
        (out / ("%s.index.json" % name)).write_text(json.dumps({"weight_map": wm}))


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def layer(out):
    """dense [out, in] then layer_norm, with the forward pass in doubles."""
    w = [[0.5, -1.0, 0.25, 2.0], [1.5, 0.0, -0.75, 0.125], [-0.5, 1.25, 1.0, -2.0]]
    b = [0.1, -0.2, 0.3]
    gamma = [1.0, 0.5, 2.0]
    beta = [0.0, 0.25, -1.0]
    x = [[1.0, 2.0, -1.0, 0.5], [-0.5, 0.25, 3.0, -2.0]]
    eps = 1e-5
    write(out / "layer.safetensors", {
        "enc.fc.weight": tensor("F32", [3, 4], [v for row in w for v in row]),
        "enc.fc.bias": tensor("F32", [3], b),
        "enc.ln.weight": tensor("F16", [3], gamma),
        "enc.ln.bias": tensor("F64", [3], beta),
        "enc.ids": tensor("I64", [3], [1, 2, 3]),
    })
    b = [f32(v) for v in b]
    y = []
    for row in x:
        h = [sum(row[i] * w[o][i] for i in range(4)) + b[o] for o in range(3)]
        mu = sum(h) / 3
        var = sum((v - mu) ** 2 for v in h) / 3
        y.append([(v - mu) / math.sqrt(var + eps) * g + be
                  for v, g, be in zip(h, gamma, beta)])
    (out / "layer.json").write_text(json.dumps({"x": x, "y": y, "eps": eps}))


def block(out):
    """One Whisper-style encoder block (width 4, hidden 8): `k_proj` has no
    bias, as in Whisper."""
    def dense(name, rows, cols, bias=True):
        n = rows * cols
        ts = {name + ".weight": tensor("F32", [rows, cols], [((i * 7) % 11 - 5) / 10 for i in range(n)])}
        if bias:
            ts[name + ".bias"] = tensor("F32", [rows], [i / 10 for i in range(rows)])
        return ts

    def norm(name):
        return {name + ".weight": tensor("F32", [4], [1.0] * 4),
                name + ".bias": tensor("F32", [4], [0.0] * 4)}

    p = "encoder.layers.0."
    tensors = {**norm(p + "self_attn_layer_norm"), **dense(p + "self_attn.q_proj", 4, 4),
               **dense(p + "self_attn.k_proj", 4, 4, bias=False), **dense(p + "self_attn.v_proj", 4, 4),
               **dense(p + "self_attn.out_proj", 4, 4), **norm(p + "final_layer_norm"),
               **dense(p + "fc1", 8, 4), **dense(p + "fc2", 4, 8)}
    write(out / "block.safetensors", tensors)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    values_file(out)
    malformed(out)
    sharded(out)
    layer(out)
    block(out)


if __name__ == "__main__":
    main()
