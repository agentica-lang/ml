"""The independent peer of `qwen3_test.agc`: a tiny random Qwen3 written as a
safetensors file with Hugging Face's names plus its config.json, and its
forward pass in doubles. Standard library only.

Usage: python3 qwen_peer.py. Writes into a new temporary folder, prints its
path, and leaves:

- `tiny/`: `bfloat16` weights (rounded as the converter rounds), tied
  embeddings, with an `lm_head.weight` of other values that the model must
  ignore (`Qwen3-0.6B`'s file has one too);
- `untied/`: `float32` weights and its own `lm_head.weight`;
- `missing/`: `tiny` without one weight;
- `tiny.gguf`, `tied.gguf` (no `output.weight`), `untied.gguf`, `half.gguf` (and `llama.gguf`, `scaled.gguf`, which the loader must refuse): the same models as GGUF version 3 with llama.cpp's names, `output.weight`
  written for the tied one too (a copy of the table, as its converter does), and `half.gguf` in `float16`;
- `half/`: `float32`, tied, one layer of width 256, so that the table and the
  MLP matrices reach the half-weight threshold (no expected logits: the test
  compares it with itself loaded as `F32`);
- `expected.json`: for the batch `IDS`, the logits of every position of every
  row, of `tiny` and of `untied`, and of `tiny` on the longer row `LONG`.

The forward is Hugging Face's `Qwen3ForCausalLM`: RMS norms, queries and keys
normalized per head before the rotation (`rotate_half`, `rope_theta`),
grouped-query attention under a causal mask, SwiGLU.
"""

import json
import math
import random
import shutil
import struct
import tempfile
from pathlib import Path

CONFIG = {
    "model_type": "qwen3", "hidden_act": "silu", "hidden_size": 8, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4,
    "intermediate_size": 12, "vocab_size": 13, "max_position_embeddings": 32,
    "rms_norm_eps": 1e-6, "rope_theta": 1000000, "tie_word_embeddings": True,
    "attention_bias": False, "use_sliding_window": False, "rope_scaling": None,
    "sliding_window": None,
}
IDS = [[1, 5, 2, 7, 3, 12], [4, 4, 9, 0, 6, 11]]
# A row that outgrows a cache of 8 and of 16 rows.
LONG = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4]
MISSING = "model.layers.1.mlp.down_proj.weight"


def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    return struct.unpack("<f", struct.pack("<I", (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000))[0]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class Model:
    """Weights by HF name (flat, row-major) already rounded to `dtype`."""

    def __init__(self, seed, dtype, head):
        self.head = head
        self.rng = random.Random(seed)
        self.round = bf16 if dtype == "BF16" else f32
        self.dtype = dtype
        self.shapes, self.values = {}, {}

    def add(self, name, shape, center=0.0, spread=0.5):
        self.shapes[name] = shape
        self.values[name] = [self.round(center + self.rng.uniform(-spread, spread)) for _ in range(math.prod(shape))]

    def save(self, path, skip=()):
        header, data = {}, b""
        for name in sorted(self.values):
            if name == skip:
                continue
            if self.dtype == "BF16":
                raw = b"".join(struct.pack("<I", struct.unpack("<I", struct.pack("<f", v))[0])[2:] for v in self.values[name])
            else:
                raw = struct.pack("<%df" % len(self.values[name]), *self.values[name])
            header[name] = {"dtype": self.dtype, "shape": self.shapes[name],
                            "data_offsets": [len(data), len(data) + len(raw)]}
            data += raw
        text = json.dumps(header).encode()
        text += b" " * (-len(text) % 8)
        path.write_bytes(struct.pack("<Q", len(text)) + text + data)

    # The forward pass, in doubles. A matrix is a list of rows.

    def rows(self, name):
        cols = self.shapes[name][-1]
        v = self.values[name]
        return [v[i:i + cols] for i in range(0, len(v), cols)]

    def linear(self, name, x):
        return [[sum(a * b for a, b in zip(r, w)) for w in self.rows(name)] for r in x]

    def rms(self, name, x):
        g = self.values[name]
        return [[v / math.sqrt(sum(t * t for t in r) / len(r) + CONFIG["rms_norm_eps"]) * gi for v, gi in zip(r, g)]
                for r in x]

    def heads(self, x, count):
        hd = CONFIG["head_dim"]
        return [[r[h * hd:(h + 1) * hd] for h in range(count)] for r in x]

    def rope(self, vec, pos):
        hd = CONFIG["head_dim"]
        half = hd // 2
        out = list(vec)
        for i in range(half):
            angle = pos * CONFIG["rope_theta"] ** (-2.0 * i / hd)
            c, s = math.cos(angle), math.sin(angle)
            out[i] = vec[i] * c - vec[i + half] * s
            out[i + half] = vec[i + half] * c + vec[i] * s
        return out

    def attention(self, p, x):
        n_heads, n_kv = CONFIG["num_attention_heads"], CONFIG["num_key_value_heads"]
        hd = CONFIG["head_dim"]
        q = self.heads(self.linear(p + "q_proj.weight", x), n_heads)
        k = self.heads(self.linear(p + "k_proj.weight", x), n_kv)
        v = self.heads(self.linear(p + "v_proj.weight", x), n_kv)
        for t in range(len(x)):
            q[t] = [self.rope(self.rms(p + "q_norm.weight", [h])[0], t) for h in q[t]]
            k[t] = [self.rope(self.rms(p + "k_norm.weight", [h])[0], t) for h in k[t]]
        ctx = []
        for t in range(len(x)):
            row = []
            for h in range(n_heads):
                kv = h // (n_heads // n_kv)
                s = [sum(a * b for a, b in zip(q[t][h], k[j][kv])) / math.sqrt(hd) for j in range(t + 1)]
                e = [math.exp(z - max(s)) for z in s]
                row += [sum(e[j] * v[j][kv][c] for j in range(t + 1)) / sum(e) for c in range(hd)]
            ctx.append(row)
        return self.linear(p + "o_proj.weight", ctx)

    def mlp(self, p, x):
        gate, up = self.linear(p + "gate_proj.weight", x), self.linear(p + "up_proj.weight", x)
        return self.linear(p + "down_proj.weight", [[g / (1 + math.exp(-g)) * u for g, u in zip(rg, ru)]
                                                     for rg, ru in zip(gate, up)])

    def logits(self, ids):
        table = self.rows("model.embed_tokens.weight")
        h = [table[t] for t in ids]
        for i in range(CONFIG["num_hidden_layers"]):
            p = "model.layers.%d." % i
            h = plus(h, self.attention(p + "self_attn.", self.rms(p + "input_layernorm.weight", h)))
            h = plus(h, self.mlp(p + "mlp.", self.rms(p + "post_attention_layernorm.weight", h)))
        return self.linear(self.head, self.rms("model.norm.weight", h))


def plus(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def qwen(dtype, untied):
    c = CONFIG
    m = Model(2026 if untied else 2025, dtype, "lm_head.weight" if untied else "model.embed_tokens.weight")
    d, hd = c["hidden_size"], c["head_dim"]
    m.add("model.embed_tokens.weight", [c["vocab_size"], d])
    for i in range(c["num_hidden_layers"]):
        p = "model.layers.%d." % i
        m.add(p + "input_layernorm.weight", [d], 1.0, 0.25)
        m.add(p + "self_attn.q_proj.weight", [c["num_attention_heads"] * hd, d])
        m.add(p + "self_attn.k_proj.weight", [c["num_key_value_heads"] * hd, d])
        m.add(p + "self_attn.v_proj.weight", [c["num_key_value_heads"] * hd, d])
        m.add(p + "self_attn.o_proj.weight", [d, c["num_attention_heads"] * hd])
        m.add(p + "self_attn.q_norm.weight", [hd], 1.0, 0.25)
        m.add(p + "self_attn.k_norm.weight", [hd], 1.0, 0.25)
        m.add(p + "post_attention_layernorm.weight", [d], 1.0, 0.25)
        m.add(p + "mlp.gate_proj.weight", [c["intermediate_size"], d])
        m.add(p + "mlp.up_proj.weight", [c["intermediate_size"], d])
        m.add(p + "mlp.down_proj.weight", [d, c["intermediate_size"]])
    m.add("model.norm.weight", [d], 1.0, 0.25)
    # The tied file also has an `lm_head.weight`, a different one: the model ignores it.
    m.add("lm_head.weight", [c["vocab_size"], d])
    return m


HALF_CONFIG = dict(CONFIG, hidden_size=256, num_hidden_layers=1, head_dim=64, intermediate_size=512,
                   vocab_size=400)


def half():
    c = HALF_CONFIG
    m = Model(2027, "F32", "model.embed_tokens.weight")
    d, hd = c["hidden_size"], c["head_dim"]
    m.add("model.embed_tokens.weight", [c["vocab_size"], d])
    p = "model.layers.0."
    m.add(p + "input_layernorm.weight", [d], 1.0, 0.25)
    m.add(p + "self_attn.q_proj.weight", [c["num_attention_heads"] * hd, d])
    m.add(p + "self_attn.k_proj.weight", [c["num_key_value_heads"] * hd, d])
    m.add(p + "self_attn.v_proj.weight", [c["num_key_value_heads"] * hd, d])
    m.add(p + "self_attn.o_proj.weight", [d, c["num_attention_heads"] * hd])
    m.add(p + "self_attn.q_norm.weight", [hd], 1.0, 0.25)
    m.add(p + "self_attn.k_norm.weight", [hd], 1.0, 0.25)
    m.add(p + "post_attention_layernorm.weight", [d], 1.0, 0.25)
    m.add(p + "mlp.gate_proj.weight", [c["intermediate_size"], d])
    m.add(p + "mlp.up_proj.weight", [c["intermediate_size"], d])
    m.add(p + "mlp.down_proj.weight", [d, c["intermediate_size"]])
    m.add("model.norm.weight", [d], 1.0, 0.25)
    return m


GGUF_TOP = {"model.embed_tokens.weight": "token_embd.weight", "model.norm.weight": "output_norm.weight",
            "lm_head.weight": "output.weight"}
GGUF_BLOCK = {"input_layernorm.weight": "attn_norm.weight", "self_attn.q_proj.weight": "attn_q.weight",
              "self_attn.k_proj.weight": "attn_k.weight", "self_attn.v_proj.weight": "attn_v.weight",
              "self_attn.o_proj.weight": "attn_output.weight", "self_attn.q_norm.weight": "attn_q_norm.weight",
              "self_attn.k_norm.weight": "attn_k_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
              "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight",
              "mlp.down_proj.weight": "ffn_down.weight"}


def gguf_name(name):
    if name in GGUF_TOP:
        return GGUF_TOP[name]
    parts = name.split(".")  # model.layers.N.<rest>
    return "blk.%s.%s" % (parts[2], GGUF_BLOCK[".".join(parts[3:])])


def g_string(s):
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path, model, config, dtype, head, arch="qwen3", extra=()):
    """GGUF version 3, written from the format's description. `head`: "copy" writes `output.weight` as a copy of the
    table (llama.cpp does for a tied model), "none" writes no head (the table is it), "own" the model's `lm_head`.
    The tensor values are `dtype` ("F32" or "F16")."""
    c = config
    kvs = [("general.architecture", 8, arch), ("qwen3.block_count", 4, c["num_hidden_layers"]),
           ("qwen3.context_length", 4, c["max_position_embeddings"]), ("qwen3.embedding_length", 4, c["hidden_size"]),
           ("qwen3.feed_forward_length", 4, c["intermediate_size"]),
           ("qwen3.attention.head_count", 4, c["num_attention_heads"]),
           ("qwen3.attention.head_count_kv", 4, c["num_key_value_heads"]),
           ("qwen3.attention.key_length", 4, c["head_dim"]), ("qwen3.attention.value_length", 4, c["head_dim"]),
           ("qwen3.attention.layer_norm_rms_epsilon", 6, c["rms_norm_eps"]),
           ("qwen3.rope.freq_base", 6, float(c["rope_theta"]))] + list(extra)
    buf = b"GGUF"
    tensors = []
    for name in sorted(model.values):
        if name == "lm_head.weight" and head != "own":
            continue
        tensors.append((gguf_name(name), model.shapes[name], model.values[name]))
    if head == "copy":
        tensors.append(("output.weight", model.shapes["model.embed_tokens.weight"],
                        model.values["model.embed_tokens.weight"]))
    buf += struct.pack("<IQQ", 3, len(tensors), len(kvs))
    for key, kind, v in kvs:
        body = g_string(v) if kind == 8 else struct.pack("<I" if kind == 4 else "<f", v)
        buf += g_string(key) + struct.pack("<I", kind) + body
    data, infos = b"", b""
    code, fmt = (0, "f") if dtype == "F32" else (1, "e")
    for name, shape, values in tensors:
        # norms stay F32 in llama.cpp's files, even in an F16 one
        norm = len(shape) == 1
        kind, f = (0, "f") if norm else (code, fmt)
        raw = struct.pack("<%d%s" % (len(values), f), *values)
        data += b"\0" * (-len(data) % 32)
        infos += g_string(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in reversed(shape))
        infos += struct.pack("<IQ", kind, len(data))
        data += raw
    buf += infos
    buf += b"\0" * (-len(buf) % 32)
    path.write_bytes(buf + data)


def write(out):
    tied, untied = qwen("BF16", False), qwen("F32", True)
    expected = {"ids": IDS, "tiny": [tied.logits(r) for r in IDS], "untied": [untied.logits(r) for r in IDS],
                "long": LONG, "tiny_long": tied.logits(LONG)}
    for folder, model, config, skip in [("tiny", tied, CONFIG, ()), ("untied", untied, dict(CONFIG, tie_word_embeddings=False), ()),
                                        ("missing", tied, CONFIG, MISSING)]:
        (out / folder).mkdir()
        model.save(out / folder / "model.safetensors", skip)
        (out / folder / "config.json").write_text(json.dumps(config))
    (out / "half").mkdir()
    half().save(out / "half" / "model.safetensors")
    (out / "half" / "config.json").write_text(json.dumps(HALF_CONFIG))
    write_gguf(out / "tiny.gguf", tied, CONFIG, "F32", "copy")
    write_gguf(out / "tied.gguf", tied, CONFIG, "F32", "none")
    write_gguf(out / "untied.gguf", untied, dict(CONFIG, tie_word_embeddings=False), "F32", "own")
    write_gguf(out / "half.gguf", half(), HALF_CONFIG, "F16", "copy")
    write_gguf(out / "llama.gguf", tied, CONFIG, "F32", "copy", arch="llama")
    write_gguf(out / "scaled.gguf", tied, CONFIG, "F32", "copy", extra=[("qwen3.rope.scaling.type", 8, "yarn")])
    (out / "expected.json").write_text(json.dumps(expected))


def main():
    out = Path(tempfile.mkdtemp(prefix="agentica-qwen-"))
    try:
        write(out)
    except BaseException:
        shutil.rmtree(out)
        raise
    print(out)


if __name__ == "__main__":
    main()
