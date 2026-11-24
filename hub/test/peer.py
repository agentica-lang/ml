"""The independent peer of the hub/ tests: writes safetensors files with the
standard library alone (struct + json) and the values a reader must find.

Usage: python3 peer.py SET, with SET one of values, malformed, sharded, layer,
block, decoder, half, gguf or gguf_bad. The gguf sets are GGUF version 3 files,
written by hand from the format's description. Writes that set into a new temporary folder and prints its path.
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


def half(out):
    """Matrices over and under the 100000 elements that `weights: F16` keeps in 16 bits: one value pattern, in F32,
    F16 and BF16 (every value is a multiple of 1/16, which all three hold exactly), one with a value beyond F16,
    an F64 one that stays F32, and the two sides of the threshold (100000 and 99999 elements)."""
    def pattern(n):
        return [((i * 7) % 17 - 8) / 16.0 for i in range(n)]

    def bf16(values):
        return [struct.unpack("<I", struct.pack("<f", v))[0] >> 16 for v in values]

    big = pattern(320 * 320)
    huge = list(big)
    huge[5] = 70000.0
    write(out / "half.safetensors", {
        "big.weight": tensor("F32", [320, 320], big),
        "big.bias": tensor("F32", [320], [0.0] * 320),
        "f16.weight": tensor("F16", [320, 320], big),
        "bf16.weight": tensor("BF16", [320, 320], bf16(big)),
        "huge.weight": tensor("F32", [320, 320], huge),
        "f64.weight": tensor("F64", [320, 320], big),
        "tok.weight": tensor("F32", [400, 300], pattern(400 * 300)),
        "edge.weight": tensor("F32", [100, 1000], pattern(100 * 1000)),
        "under.weight": tensor("F32", [3, 33333], pattern(3 * 33333)),
        "small.weight": tensor("F32", [4, 3], pattern(12)),
        "norm.weight": tensor("F32", [320], [1.0] * 320),
    })
    # `linear(ones, big, out_in)`: the sum of each row of `big`, exact in F32
    (out / "half.json").write_text(json.dumps({"sums": [sum(big[j * 320:(j + 1) * 320]) for j in range(320)]}))


# GGUF: the value types are numbered as in the format, the tensor types as in ggml.
GGUF_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
GGML_TYPES = {"F32": 0, "F16": 1, "Q8_0": 8, "Q4_K": 12, "I32": 26, "F64": 28, "BF16": 30}


def g_string(s):
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def g_value(kind, v):
    if kind == 8:
        return g_string(v)
    if kind == 9:
        item, items = v
        return struct.pack("<IQ", item, len(items)) + b"".join(g_value(item, x) for x in items)
    return struct.pack("<" + GGUF_SCALARS[kind], v)


def gguf(path, kvs, tensors, alignment=32, version=3, offsets=None):
    """kvs: [(key, kind, value)]; tensors: [(name, dims as GGUF has them, ggml type, raw bytes)]; `offsets` puts each
    tensor's info at that offset instead of where its bytes are."""
    head = b"GGUF" + struct.pack("<IQQ", version, len(tensors), len(kvs))
    for key, kind, v in kvs:
        head += g_string(key) + struct.pack("<I", kind) + g_value(kind, v)
    data, infos = b"", b""
    for i, (name, dims, ggml, raw) in enumerate(tensors):
        data += b"\0" * (-len(data) % alignment)
        infos += g_string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", x) for x in dims)
        infos += struct.pack("<IQ", ggml, len(data) if offsets is None else offsets[i])
        data += raw
    head += infos
    head += b"\0" * (-len(head) % alignment)
    path.write_bytes(head + data)


def g_tensor(dtype, shape, values):
    """A tensor as GGUF has it: the dims are the shape reversed."""
    t = tensor(dtype, shape, values)
    return t["raw"], list(reversed(shape))


def gguf_file(out):
    """Every value type, an array of each kind, and tensors of F32, F16, BF16, F64, I32 and one Q8_0 block row."""
    kvs = [
        ("general.architecture", 8, "qwen3"),
        ("general.alignment", 4, 32),
        ("t.u8", 0, 200), ("t.i8", 1, -5), ("t.u16", 2, 65000), ("t.i16", 3, -30000), ("t.u32", 4, 4000000000),
        ("t.i32", 5, -2000000000), ("t.f32", 6, 0.5), ("t.bool", 7, True), ("t.u64", 10, 2**40),
        ("t.i64", 11, -2**40), ("t.f64", 12, 0.1), ("t.str", 8, "h\u00e9llo"),
        ("t.arr_u32", 9, (4, [1, 2, 3])), ("t.arr_str", 9, (8, ["a", "bc", ""])), ("t.arr_empty", 9, (4, [])),
        ("qwen3.rope.freq_base", 6, 1000000.0),
    ]
    f32 = [1.5, -2.25, 0.0, 3.0, 0.5, -0.125]
    f16 = [1.0, -2.0, 0.5, 65504.0]
    bf = [0x3F80, 0xC040, 0x0001, 0x7F7F]
    blocks = struct.pack("<e", 0.5) + bytes(range(32))
    entries = {
        "token_embd.weight": ("F32", [3, 2], f32),
        "output_norm.weight": ("F32", [2], [1.0, 2.0]),
        "output.weight": ("F16", [2, 2], f16[:4]),
        "blk.0.attn_q.weight": ("BF16", [2, 2], bf),
        "blk.0.ffn_down.weight": ("F64", [2], [0.1, -1e300]),
        "blk.1.attn_norm.weight": ("I32", [3], [-7, 0, 7]),
        "rope_freqs.weight": ("F32", [1], [9.0]),
    }
    tensors = []
    expected = {}
    names = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
             "output.weight": "lm_head.weight", "blk.0.attn_q.weight": "model.layers.0.self_attn.q_proj.weight",
             "blk.0.ffn_down.weight": "model.layers.0.mlp.down_proj.weight",
             "blk.1.attn_norm.weight": "model.layers.1.input_layernorm.weight", "rope_freqs.weight": "rope_freqs.weight"}
    for name, (dtype, shape, values) in entries.items():
        raw, dims = g_tensor(dtype, shape, values)
        tensors.append((name, dims, GGML_TYPES[dtype], raw))
        expected[names[name]] = decoded(tensor(dtype, shape, values))
    gguf(out / "model.gguf", kvs, tensors)
    # a Q8_0 row of one block (32 values, 34 bytes: a half scale, 32 bytes) has no dtype of tensor/: refused alone
    raw, dims = g_tensor("F32", [2], [1.0, 2.0])
    gguf(out / "q8.gguf", [kvs[0]], [("token_embd.weight", dims, 0, raw),
                                     ("blk.0.ffn_up.weight", [32, 1], GGML_TYPES["Q8_0"], blocks)])
    # one Q4_K block (256 values, 144 bytes) that is all of the data: the file opens, and reading the tensor is refused
    gguf(out / "q4k.gguf", [kvs[0]], [("blk.0.ffn_up.weight", [256, 1], GGML_TYPES["Q4_K"], bytes(144))])
    # big enough for `weights: F16`: a table, a BF16 matrix and an F32 one
    def pattern(n):
        return [((i * 7) % 17 - 8) / 16.0 for i in range(n)]
    bf = lambda vs: [struct.unpack("<I", struct.pack("<f", v))[0] >> 16 for v in vs]
    big = [("token_embd.weight", "F16", [400, 300], pattern(400 * 300)),
           ("output.weight", "BF16", [320, 320], bf(pattern(320 * 320))),
           ("blk.0.attn_q.weight", "F32", [320, 320], pattern(320 * 320))]
    gguf(out / "half.gguf", [kvs[0]],
         [(n, list(reversed(shape)), GGML_TYPES[dt], tensor(dt, shape, vs)["raw"]) for n, dt, shape, vs in big])
    (out / "model.json").write_text(json.dumps({k: v for k, v in expected.items()}))


def gguf_big_header(out):
    """A header over the first read (8 MiB): a u8 array of 9 MiB, then the tensors."""
    kvs = [("general.architecture", 8, "qwen3"), ("big", 9, (0, [7] * (9 * 1024 * 1024)))]
    raw, dims = g_tensor("F32", [2, 2], [1.0, 2.0, 3.0, 4.0])
    gguf(out / "big.gguf", kvs, [("token_embd.weight", dims, 0, raw)])
    (out / "big.json").write_text(json.dumps(
        {"model.embed_tokens.weight": {"dtype": "F32", "shape": [2, 2], "values": ["1.0", "2.0", "3.0", "4.0"]}}))


def gguf_bad(out):
    """One malformed file per reason: the reader must refuse each with a message that holds the word in `cases`."""
    arch = ("general.architecture", 8, "qwen3")
    good = g_tensor("F32", [2, 2], [1.0, 2.0, 3.0, 4.0])
    row = lambda name, dims=None, ggml=0, raw=None: (name, dims or good[1], ggml, good[0] if raw is None else raw)
    write = lambda name, kvs, tensors, **kw: gguf(out / (name + ".gguf"), kvs, tensors, **kw)
    cases = {}

    def case(name, word, kvs, tensors, **kw):
        write(name, kvs, tensors, **kw)
        cases[name + ".gguf"] = word

    case("version2", "version 2", [arch], [row("token_embd.weight")], version=2)
    case("alignment3", "power of two", [arch, ("general.alignment", 4, 3)], [row("token_embd.weight")])
    case("two_dims_zero", "dimension of 0", [arch], [row("token_embd.weight", dims=[0, 2])])
    case("five_dims", "5 dimensions", [arch], [row("token_embd.weight", dims=[1, 1, 1, 1, 2])])
    case("unknown_type", "ggml type 19", [arch], [row("token_embd.weight", ggml=19)])
    case("short_data", "past the", [arch], [row("token_embd.weight", dims=[2, 3])])
    case("same_name", "twice", [arch], [row("token_embd.weight"), row("token_embd.weight")])
    case("same_key", "twice", [arch, arch], [row("token_embd.weight")])
    case("nested_array", "array of arrays", [arch, ("a", 9, (9, []))], [row("token_embd.weight")])
    case("other_arch", "llama", [("general.architecture", 8, "llama")], [row("token_embd.weight")])
    case("alignment0", "power of two", [arch, ("general.alignment", 4, 0)], [row("token_embd.weight")])
    case("too_many_values", "too many values", [arch], [row("token_embd.weight", dims=[2**62, 4])])
    case("overlap", "overlaps", [arch], [row("token_embd.weight"), row("output.weight")], offsets=[0, 0])
    case("unaligned", "multiple of 32", [arch], [row("token_embd.weight")], offsets=[8])
    case("block_row", "multiple of 32", [arch], [row("token_embd.weight", dims=[31, 1], ggml=8, raw=b"\0" * 34)])
    # raw edits of a good file
    base = out / "ok.gguf"
    gguf(base, [arch], [row("token_embd.weight")])
    data = base.read_bytes()
    (out / "magic.gguf").write_bytes(b"GGUX" + data[4:])
    flag = g_string("b") + struct.pack("<I", 7)
    gguf(out / "bad_bool.gguf", [arch, ("b", 7, True)], [row("token_embd.weight")])
    raw = (out / "bad_bool.gguf").read_bytes()
    (out / "bad_bool.gguf").write_bytes(raw.replace(flag + b"\x01", flag + b"\x02"))
    cases["bad_bool.gguf"] = "boolean"
    cases["magic.gguf"] = "not a GGUF"
    (out / "cut.gguf").write_bytes(data[:30])
    cases["cut.gguf"] = "ends inside the header"
    (out / "empty.gguf").write_bytes(b"")
    cases["empty.gguf"] = "ends inside the header"
    huge = bytearray(data)
    huge[8:16] = struct.pack("<Q", 2**40)
    (out / "huge_count.gguf").write_bytes(bytes(huge))
    cases["huge_count.gguf"] = "do not fit"
    huge = bytearray(data)
    huge[24:32] = struct.pack("<Q", 2**62)
    (out / "huge_string.gguf").write_bytes(bytes(huge))
    cases["huge_string.gguf"] = "past the end"
    # a count of items past the end, a value type that does not exist, a string that is not UTF-8, a data start past
    # the end of the file
    gguf(out / "array_count.gguf", [arch, ("a", 9, (4, [1, 2, 3]))], [row("token_embd.weight")])
    raw = (out / "array_count.gguf").read_bytes()
    (out / "array_count.gguf").write_bytes(raw.replace(struct.pack("<IQ", 4, 3), struct.pack("<IQ", 4, 2**40)))
    cases["array_count.gguf"] = "does not fit in the file"
    gguf(out / "kv_type.gguf", [arch, ("u", 7, True)], [row("token_embd.weight")])
    raw = (out / "kv_type.gguf").read_bytes()
    (out / "kv_type.gguf").write_bytes(raw.replace(g_string("u") + struct.pack("<I", 7), g_string("u") + struct.pack("<I", 13)))
    cases["kv_type.gguf"] = "value type 13"
    gguf(out / "utf8.gguf", [arch, ("s", 8, "abcd")], [row("token_embd.weight")])
    raw = (out / "utf8.gguf").read_bytes()
    (out / "utf8.gguf").write_bytes(raw.replace(b"abcd", b"\xff\xfe\xfd\xfc"))
    cases["utf8.gguf"] = "UTF-8"
    gguf(out / "data_start.gguf", [arch], [], alignment=4096)
    raw = (out / "data_start.gguf").read_bytes()
    (out / "data_start.gguf").write_bytes(raw[:-100])
    cases["data_start.gguf"] = "past the end of the file"
    (out / "cases.json").write_text(json.dumps(cases))


SETS = {"values": values_file, "malformed": malformed, "sharded": sharded,
        "layer": layer, "block": block, "decoder": decoder, "half": half,
        "gguf": lambda out: (gguf_file(out), gguf_big_header(out)), "gguf_bad": gguf_bad}


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
