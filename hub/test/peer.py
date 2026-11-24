"""The independent peer of the hub/ tests: writes safetensors files with the
standard library alone (struct + json) and the values a reader must find.

Usage: python3 peer.py SET, with SET one of values, malformed, sharded, layer,
block or decoder. Writes that set into a new temporary folder and prints its path.
"""

import json
import math
import shutil
import struct
import sys
import tempfile
from pathlib import Path

# BF16 values are the 16-bit patterns, widened to a float32 by the reader.
FORMATS = {"F32": "f", "F16": "e", "BF16": "H", "F64": "d", "I64": "q", "I32": "i",
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
    """Floats as text ("inf", "-inf", "nan" or repr), so JSON can hold them."""
    n = len(t["raw"]) // struct.calcsize(FORMATS[t["dtype"]])
    values = list(struct.unpack("<%d%s" % (n, FORMATS[t["dtype"]]), t["raw"]))
    if t["dtype"] == "BF16":
        values = [struct.unpack("<f", struct.pack("<I", b << 16))[0] for b in values]
    if t["dtype"] in ("F32", "F16", "BF16", "F64"):
        values = [repr(v) for v in values]
    return {"dtype": t["dtype"], "shape": t["shape"], "values": values}


def values_file(out):
    half_bits = [0x3C00, 0xC000, 0x0001, 0x83FF, 0x7BFF, 0x8000, 0x3555, 0x7C00, 0xFC00, 0x7E00]
    halves = [struct.unpack("<e", struct.pack("<H", b))[0] for b in half_bits]
    tensors = {
        "f32": tensor("F32", [2, 3], [1.5, -2.25, 0.1, 3.4e38, -0.0, 1e-40]),
        "f16": tensor("F16", [10], halves),
        "bf16": tensor("BF16", [2, 4], [0x3F80, 0xC040, 0x0001, 0x7F80, 0xFF80, 0x7FC0, 0x8000, 0x7F7F]),
        "f64": tensor("F64", [2], [0.1, -1e300]),
        "i64": tensor("I64", [3], [-2**63, 2**53, -5]),
        "i32": tensor("I32", [2, 2], [-2**31, 2**31 - 1, 0, -7]),
        "i16": tensor("I16", [2], [-32768, 32767]),
        "i8": tensor("I8", [3], [-128, 127, -1]),
        "u8": tensor("U8", [2], [0, 255]),
        "bool": tensor("BOOL", [3], [True, False, True]),
        "scalar": tensor("F32", [], [2.5]),
        "empty": tensor("F32", [0, 3], []),
        "empty_last": tensor("F32", [3, 0], []),
    }
    write(out / "values.safetensors", tensors, {"format": "pt"})
    expected = {name: decoded(t) for name, t in tensors.items()}
    (out / "values.json").write_text(json.dumps(expected))
    write(out / "u16.safetensors",
          {"w": {"dtype": "U16", "shape": [2], "raw": b"\x01\x00\x02\x00"},
           "ok": tensor("F32", [1], [1.0])})
    raw_header(out / "bad_bool.safetensors",
               b'{"b":{"dtype":"BOOL","shape":[2],"data_offsets":[0,2]}}', b"\x01\x02")
    # A shape whose first dimensions overflow an Int, made empty by its last one.
    raw_header(out / "hostile.safetensors",
               json.dumps({"h": entry("F32", [2**62, 4, 0], 0, 0)}).encode())
    # Header and buffer both past the reader's 1 MiB chunk.
    big = 2 * 1048576 + 5
    header = {"big": entry("U8", [big], 0, big), "small": entry("I64", [2], big, big + 16)}
    raw_header(out / "chunks.safetensors", json.dumps(header).encode() + b" " * 1048576,
               bytes(i % 256 for i in range(big)) + struct.pack("<2q", 7, -9))


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
        "negative_start": ({"a": entry("F32", [1], -4, 0)}, "outside", b"\0" * 4),
        "three_offsets": ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4, 8]}}, "[begin, end]", b"\0" * 4),
        "zero_header": (None, "does not start with {", struct.pack("<Q", 0)),
        "leading_space": (b' {"a": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}', "does not start with {", b"\0" * 4),
        "metadata_list": ({"__metadata__": ["k"]}, "__metadata__", b""),
        "out_of_range": ({"a": entry("F32", [2], 0, 8)}, "outside", b"\0" * 4),
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
        "missing": ("enc.z", "model-00001-of-00002.safetensors", "not in its shard"),
        "escape": ("enc.a", "../model-00001-of-00002.safetensors", "not a file name"),
        "absolute": ("enc.a", str(out / "model-00001-of-00002.safetensors"), "not a file name"),
        "backslash": ("enc.a", "a\\b.safetensors", "not a file name"),
        "dotdot": ("enc.a", "..", "not a file name"),
        "absent": ("enc.a", "model-00003-of-00002.safetensors", "model-00003-of-00002"),
    }
    phrases = {}
    for name, (tensor_name, file, phrase) in bad.items():
        (out / ("%s.index.json" % name)).write_text(json.dumps({"weight_map": {tensor_name: file}}))
        phrases["%s.index.json" % name] = phrase
    (out / "bad_index.json").write_text(json.dumps(phrases))


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
        # The older spelling of a layer norm (BERT's checkpoints).
        "enc.old.gamma": tensor("F32", [3], [2.0, 1.0, 0.5]),
        "enc.old.beta": tensor("F32", [3], [1.0, 0.0, -1.0]),
        # A file with both spellings: the weight and bias win.
        "enc.both.weight": tensor("F32", [3], [1.0, 1.0, 1.0]),
        "enc.both.bias": tensor("F32", [3], [0.0, 0.0, 0.0]),
        "enc.both.gamma": tensor("F32", [3], [5.0, 5.0, 5.0]),
        "enc.both.beta": tensor("F32", [3], [6.0, 6.0, 6.0]),
        "enc.half.gamma": tensor("F32", [3], [1.0, 1.0, 1.0]),
        "enc.half.beta": tensor("F32", [2], [1.0, 1.0]),
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


class Weights:
    """Dense layers stored [out, in] and layer norms with deterministic
    values, and what the loader must find: each dense weight transposed to
    [in, out]. Every value is a multiple of 1/8, exact in F32."""

    def __init__(self):
        self.tensors, self.dense, self.norm = {}, {}, {}

    def add_dense(self, name, rows, cols, k, bias=True):
        w = [((i * 7 + k * 13) % 23 - 11) / 8 for i in range(rows * cols)]
        self.tensors[name + ".weight"] = tensor("F32", [rows, cols], w)
        b = [(i - k) / 4 for i in range(rows)] if bias else None
        if bias:
            self.tensors[name + ".bias"] = tensor("F32", [rows], b)
        self.dense[name] = {"w": [w[o * cols + i] for i in range(cols) for o in range(rows)], "b": b}

    def add_norm(self, name, k):
        gamma, beta = [1 + (k + i) / 4 for i in range(4)], [(k - i) / 8 for i in range(4)]
        self.tensors[name + ".weight"] = tensor("F32", [4], gamma)
        self.tensors[name + ".bias"] = tensor("F64", [4], beta)
        self.norm[name] = {"gamma": gamma, "beta": beta}

    def add_attention(self, prefix, k):
        """Whisper's q/k/v/out projections; `k_proj` has no bias."""
        for j, part in enumerate(["q_proj", "k_proj", "v_proj", "out_proj"]):
            self.add_dense(prefix + part, 4, 4, k + j, bias=part != "k_proj")

    def expected(self):
        return {"dense": self.dense, "norm": self.norm}

    # The forward pass, in doubles, over the [in, out] weights.

    def linear(self, name, x):
        d = self.dense[name]
        cols = len(d["w"]) // len(x[0])
        b = d["b"] or [0.0] * cols
        return [[sum(row[i] * d["w"][i * cols + o] for i in range(len(row))) + b[o]
                 for o in range(cols)] for row in x]

    def layer_norm(self, name, x, eps):
        n = self.norm[name]
        out = []
        for row in x:
            mu = sum(row) / len(row)
            sd = math.sqrt(sum((v - mu) ** 2 for v in row) / len(row) + eps)
            out.append([(v - mu) / sd * g + b for v, g, b in zip(row, n["gamma"], n["beta"])])
        return out

    def attention(self, prefix, xq, xkv, heads, causal):
        """A blocked key (after the query, when causal) gets no weight."""
        q = self.linear(prefix + "q_proj", xq)
        k = self.linear(prefix + "k_proj", xkv)
        v = self.linear(prefix + "v_proj", xkv)
        dh = len(q[0]) // heads
        ctx = [[] for _ in xq]
        for h in range(heads):
            cols = range(h * dh, (h + 1) * dh)
            for i in range(len(xq)):
                seen = range(i + 1) if causal else range(len(xkv))
                scores = [sum(q[i][c] * k[j][c] for c in cols) / math.sqrt(dh) for j in seen]
                top = max(scores)
                e = [math.exp(s - top) for s in scores]
                ctx[i] += [sum(e[n] * v[j][c] for n, j in enumerate(seen)) / sum(e) for c in cols]
        return self.linear(prefix + "out_proj", ctx)


def plus(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def block(out):
    """One Whisper-style encoder block (width 4, hidden 8), every layer with
    its own weights."""
    w = Weights()
    p = "encoder.layers.0."
    w.add_norm(p + "self_attn_layer_norm", 0)
    w.add_attention(p + "self_attn.", 1)
    w.add_norm(p + "final_layer_norm", 5)
    w.add_dense(p + "fc1", 8, 4, 6)
    w.add_dense(p + "fc2", 4, 8, 7)
    embed = [(i % 9 - 4) / 2 for i in range(24)]
    w.tensors["encoder.embed_positions.weight"] = tensor("F16", [6, 4], embed)
    write(out / "block.safetensors", w.tensors)
    (out / "block.json").write_text(json.dumps({**w.expected(), "embed": embed}))


def decoder(out):
    """One Whisper-style decoder block (width 4, 2 heads, hidden 8, relu)
    and its forward pass: 3 decoder positions over 5 encoder positions."""
    w = Weights()
    p = "decoder.layers.0."
    w.add_norm(p + "self_attn_layer_norm", 0)
    w.add_attention(p + "self_attn.", 1)
    w.add_norm(p + "encoder_attn_layer_norm", 2)
    w.add_attention(p + "encoder_attn.", 5)
    w.add_norm(p + "final_layer_norm", 3)
    w.add_dense(p + "fc1", 8, 4, 9)
    w.add_dense(p + "fc2", 4, 8, 10)
    write(out / "decoder.safetensors", w.tensors)
    eps = 1e-5
    x = [[((i * 4 + c) * 5 % 13 - 6) / 8 for c in range(4)] for i in range(3)]
    memory = [[((i * 4 + c) * 3 % 11 - 5) / 8 for c in range(4)] for i in range(5)]
    n = w.layer_norm(p + "self_attn_layer_norm", x, eps)
    h = plus(x, w.attention(p + "self_attn.", n, n, 2, True))
    n = w.layer_norm(p + "encoder_attn_layer_norm", h, eps)
    h = plus(h, w.attention(p + "encoder_attn.", n, memory, 2, False))
    n = w.layer_norm(p + "final_layer_norm", h, eps)
    up = [[max(v, 0.0) for v in row] for row in w.linear(p + "fc1", n)]
    y = plus(h, w.linear(p + "fc2", up))
    expected = {**w.expected(), "x": x, "memory": memory, "y": y, "eps": eps}
    (out / "decoder.json").write_text(json.dumps(expected))


SETS = {"values": values_file, "malformed": malformed, "sharded": sharded,
        "layer": layer, "block": block, "decoder": decoder}


def main():
    out = Path(tempfile.mkdtemp(prefix="agentica-hub-"))
    try:
        SETS[sys.argv[1]](out)
    except BaseException:
        shutil.rmtree(out)
        raise
    print(out)


if __name__ == "__main__":
    main()
