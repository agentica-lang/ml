"""The independent peer of the asr/ tests: a tiny random Whisper written as a
safetensors file with Hugging Face's names plus its config.json, and its
forward pass in doubles. Standard library only.

Usage: python3 peer.py. Writes `tiny/` (the model) and `missing/` (the same
model without one tensor) into a new temporary folder and prints its path.
`expected.json` holds two mels of 6 frames with their token rows, and a mel
of 4 frames for the encoder alone.
The forward runs twice: with the tanh GELU the Agentica model uses, and with
the exact erf GELU Whisper uses, so the tests can measure the drift.
"""

import json
import math
import random
import shutil
import struct
import tempfile
from pathlib import Path

CONFIG = {
    "model_type": "whisper", "activation_function": "gelu", "d_model": 4,
    "encoder_layers": 1, "decoder_layers": 1, "encoder_attention_heads": 2,
    "decoder_attention_heads": 2, "encoder_ffn_dim": 8, "decoder_ffn_dim": 8,
    "num_mel_bins": 3, "vocab_size": 7, "max_source_positions": 3,
    "max_target_positions": 8,
}
FRAMES = 6
TOKENS = [1, 5, 2, 6]
SECOND_TOKENS = [3, 0, 4, 4]
EPS = 1e-5
MISSING = "model.decoder.layers.0.fc2.weight"


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class Model:
    """Weights by HF name (flat, row-major F32 values) and the forward."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.shapes, self.values = {}, {}

    def add(self, name, shape, center=0.0, spread=0.5):
        n = math.prod(shape)
        self.shapes[name] = shape
        self.values[name] = [f32(center + self.rng.uniform(-spread, spread)) for _ in range(n)]

    def dense(self, prefix, rows, cols, bias=True):
        self.add(prefix + ".weight", [rows, cols])
        if bias:
            self.add(prefix + ".bias", [rows])

    def norm(self, prefix, width):
        self.add(prefix + ".weight", [width], 1.0, 0.25)
        self.add(prefix + ".bias", [width], 0.0, 0.25)

    def attention(self, prefix, d):
        for part in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            self.dense(prefix + part, d, d, bias=part != "k_proj")

    def save(self, path, skip=()):
        header, data = {}, b""
        for name in sorted(self.values):
            if name in skip:
                continue
            raw = struct.pack("<%df" % len(self.values[name]), *self.values[name])
            header[name] = {"dtype": "F32", "shape": self.shapes[name],
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

    def linear(self, prefix, x):
        w = self.rows(prefix + ".weight")
        b = self.values.get(prefix + ".bias", [0.0] * len(w))
        return [[sum(r[i] * wo[i] for i in range(len(r))) + bo for wo, bo in zip(w, b)] for r in x]

    def layer_norm(self, prefix, x):
        g, b = self.values[prefix + ".weight"], self.values[prefix + ".bias"]
        out = []
        for r in x:
            mu = sum(r) / len(r)
            sd = math.sqrt(sum((v - mu) ** 2 for v in r) / len(r) + EPS)
            out.append([(v - mu) / sd * gi + bi for v, gi, bi in zip(r, g, b)])
        return out

    def conv1d(self, prefix, x, stride):
        """x is [channels][length]; kernel 3, one zero of padding a side."""
        w, b = self.values[prefix + ".weight"], self.values[prefix + ".bias"]
        out_c, in_c, k = self.shapes[prefix + ".weight"]
        padded = [[0.0] + row + [0.0] for row in x]
        length = (len(x[0]) + 2 - k) // stride + 1
        return [[b[o] + sum(w[(o * in_c + c) * k + j] * padded[c][t * stride + j]
                            for c in range(in_c) for j in range(k))
                 for t in range(length)] for o in range(out_c)]

    def attend(self, prefix, xq, xkv, heads, causal):
        q = self.linear(prefix + "q_proj", xq)
        k = self.linear(prefix + "k_proj", xkv)
        v = self.linear(prefix + "v_proj", xkv)
        dh = len(q[0]) // heads
        ctx = [[] for _ in xq]
        for h in range(heads):
            cols = range(h * dh, (h + 1) * dh)
            for i in range(len(xq)):
                seen = range(i + 1) if causal else range(len(xkv))
                s = [sum(q[i][c] * k[j][c] for c in cols) / math.sqrt(dh) for j in seen]
                e = [math.exp(x - max(s)) for x in s]
                ctx[i] += [sum(e[n] * v[j][c] for n, j in enumerate(seen)) / sum(e) for c in cols]
        return self.linear(prefix + "out_proj", ctx)

    def mlp(self, prefix, x, gelu):
        return self.linear(prefix + "fc2", [[gelu(v) for v in r] for r in self.linear(prefix + "fc1", x)])

    def encode(self, mel, gelu):
        h = [[gelu(v) for v in r] for r in self.conv1d("model.encoder.conv1", mel, 1)]
        h = [[gelu(v) for v in r] for r in self.conv1d("model.encoder.conv2", h, 2)]
        pos = self.rows("model.encoder.embed_positions.weight")
        h = plus([list(col) for col in zip(*h)], pos)
        heads = CONFIG["encoder_attention_heads"]
        for i in range(CONFIG["encoder_layers"]):
            p = "model.encoder.layers.%d." % i
            n = self.layer_norm(p + "self_attn_layer_norm", h)
            h = plus(h, self.attend(p + "self_attn.", n, n, heads, False))
            h = plus(h, self.mlp(p, self.layer_norm(p + "final_layer_norm", h), gelu))
        return self.layer_norm("model.encoder.layer_norm", h)

    def decode(self, tokens, audio, gelu):
        table = self.rows("model.decoder.embed_tokens.weight")
        h = plus([table[t] for t in tokens], self.rows("model.decoder.embed_positions.weight"))
        heads = CONFIG["decoder_attention_heads"]
        for i in range(CONFIG["decoder_layers"]):
            p = "model.decoder.layers.%d." % i
            n = self.layer_norm(p + "self_attn_layer_norm", h)
            h = plus(h, self.attend(p + "self_attn.", n, n, heads, True))
            n = self.layer_norm(p + "encoder_attn_layer_norm", h)
            h = plus(h, self.attend(p + "encoder_attn.", n, audio, heads, False))
            h = plus(h, self.mlp(p, self.layer_norm(p + "final_layer_norm", h), gelu))
        h = self.layer_norm("model.decoder.layer_norm", h)
        return [[sum(a * b for a, b in zip(r, row)) for row in table] for r in h]


def plus(a, b):
    """Row-wise a + b; b may have more rows (a positional table)."""
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def gelu_erf(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


def gelu_tanh(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def whisper():
    c = CONFIG
    d, m = c["d_model"], Model(1923)
    m.add("model.encoder.conv1.weight", [d, c["num_mel_bins"], 3])
    m.add("model.encoder.conv1.bias", [d])
    m.add("model.encoder.conv2.weight", [d, d, 3])
    m.add("model.encoder.conv2.bias", [d])
    m.add("model.encoder.embed_positions.weight", [c["max_source_positions"], d])
    for i in range(c["encoder_layers"]):
        p = "model.encoder.layers.%d." % i
        m.norm(p + "self_attn_layer_norm", d)
        m.attention(p + "self_attn.", d)
        m.norm(p + "final_layer_norm", d)
        m.dense(p + "fc1", c["encoder_ffn_dim"], d)
        m.dense(p + "fc2", d, c["encoder_ffn_dim"])
    m.norm("model.encoder.layer_norm", d)
    m.add("model.decoder.embed_tokens.weight", [c["vocab_size"], d])
    m.add("model.decoder.embed_positions.weight", [c["max_target_positions"], d])
    for i in range(c["decoder_layers"]):
        p = "model.decoder.layers.%d." % i
        m.norm(p + "self_attn_layer_norm", d)
        m.attention(p + "self_attn.", d)
        m.norm(p + "encoder_attn_layer_norm", d)
        m.attention(p + "encoder_attn.", d)
        m.norm(p + "final_layer_norm", d)
        m.dense(p + "fc1", c["decoder_ffn_dim"], d)
        m.dense(p + "fc2", d, c["decoder_ffn_dim"])
    m.norm("model.decoder.layer_norm", d)
    return m


def write(out):
    m = whisper()
    bins = CONFIG["num_mel_bins"]

    def mel(frames):
        return [[f32(m.rng.uniform(-1.5, 1.5)) for _ in range(frames)] for _ in range(bins)]

    first, second, short = mel(FRAMES), mel(FRAMES), mel(4)
    expected = {"mel": first, "tokens": TOKENS, "missing": MISSING,
                "second_mel": second, "second_tokens": SECOND_TOKENS, "short_mel": short}
    for name, gelu in [("tanh", gelu_tanh), ("erf", gelu_erf)]:
        audio = m.encode(first, gelu)
        expected[name] = {"encoder": audio, "logits": m.decode(TOKENS, audio, gelu)}
    audio = m.encode(second, gelu_tanh)
    expected["second"] = {"encoder": audio, "logits": m.decode(SECOND_TOKENS, audio, gelu_tanh)}
    expected["short"] = {"encoder": m.encode(short, gelu_tanh)}
    for folder, skip in [("tiny", ()), ("missing", (MISSING,))]:
        (out / folder).mkdir()
        m.save(out / folder / "model.safetensors", skip)
        (out / folder / "config.json").write_text(json.dumps(CONFIG))
    (out / "expected.json").write_text(json.dumps(expected))


def main():
    out = Path(tempfile.mkdtemp(prefix="agentica-asr-"))
    try:
        write(out)
    except BaseException:
        shutil.rmtree(out)
        raise
    print(out)


if __name__ == "__main__":
    main()
