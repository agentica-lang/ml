"""The independent peer of `distilbert_test.agc`: a tiny random DistilBERT written as a safetensors file with Hugging
Face's names plus its config.json, and its forward pass in doubles. Standard library only.

Usage: python3 distilbert_peer.py. Writes into a new temporary folder, prints its path, and leaves:

- `tiny/`: the model as `DistilBertModel.save_pretrained` writes it (no prefix);
- `prefixed/`: the same weights under `distilbert.`, with the head of a `DistilBertFor...` model that the loader
  must ignore;
- `missing/`: `tiny` without one weight;
- `expected.json`: for the batch of `IDS`, the hidden state, the mean-pooled embedding and that embedding over its
  length, with the exact erf GELU and the layer-norm eps of 1e-12 that Hugging Face's DistilBERT uses; and for the
  scored pairs of `PAIRS`, the cosine loss (`loss`) and its slope for one value of a few weights (`slopes`), each a
  central difference of the loss in doubles: an oracle of the gradient that shares no code with the one it checks.
"""

import json
import math
import random
import shutil
import struct
import tempfile
from pathlib import Path

CONFIG = {
    "model_type": "distilbert", "activation": "gelu", "dim": 8, "n_layers": 2, "n_heads": 2, "hidden_dim": 16,
    "vocab_size": 11, "max_position_embeddings": 8, "pad_token_id": 0, "sinusoidal_pos_embds": False,
}
# Two sentences, the second shorter (padded with 0).
IDS = [[1, 5, 2, 7, 3], [1, 4, 6, 3]]
EPS = 1e-12
# Two pairs of sentences with a score each (`distilbert_weights_test.agc` has the same, padded with 0), the eps of
# `Nn.cosine_similarity`, the step of the central difference and the values it is taken at: a name and an index.
PAIRS = [([1, 5, 2, 7, 3], [1, 6, 3]), ([1, 4, 6, 3], [1, 2, 2, 5, 3])]
SCORES = [0.8, 0.1]
COSINE_EPS = 1e-16
STEP = 1e-5
PROBES = [
    ("embeddings.word_embeddings.weight", 42), ("embeddings.position_embeddings.weight", 3),
    ("embeddings.LayerNorm.weight", 1), ("embeddings.LayerNorm.bias", 6),
    ("transformer.layer.0.attention.q_lin.weight", 5), ("transformer.layer.0.attention.k_lin.weight", 9),
    ("transformer.layer.0.attention.v_lin.bias", 3), ("transformer.layer.0.attention.out_lin.weight", 11),
    ("transformer.layer.0.sa_layer_norm.bias", 2), ("transformer.layer.0.ffn.lin1.weight", 20),
    ("transformer.layer.0.ffn.lin1.bias", 2), ("transformer.layer.1.attention.q_lin.bias", 4),
    ("transformer.layer.1.ffn.lin2.weight", 7), ("transformer.layer.1.output_layer_norm.weight", 4),
]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class Model:
    """Weights by name (flat, row-major F32 values)."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.shapes, self.values = {}, {}

    def add(self, name, shape, center=0.0, spread=0.5):
        self.shapes[name] = shape
        self.values[name] = [f32(center + self.rng.uniform(-spread, spread)) for _ in range(math.prod(shape))]

    def dense(self, prefix, rows, cols):
        self.add(prefix + ".weight", [rows, cols])
        self.add(prefix + ".bias", [rows])

    def norm(self, prefix, width):
        self.add(prefix + ".weight", [width], 1.0, 0.25)
        self.add(prefix + ".bias", [width], 0.0, 0.25)

    def save(self, path, rename=lambda n: n, skip=()):
        header, data = {}, b""
        for name in sorted(self.values, key=rename):
            if name.startswith(skip):
                continue
            raw = struct.pack("<%df" % len(self.values[name]), *self.values[name])
            header[rename(name)] = {"dtype": "F32", "shape": self.shapes[name],
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
        w, b = self.rows(prefix + ".weight"), self.values[prefix + ".bias"]
        return [[sum(r[i] * wo[i] for i in range(len(r))) + bo for wo, bo in zip(w, b)] for r in x]

    def layer_norm(self, prefix, x):
        g, b = self.values[prefix + ".weight"], self.values[prefix + ".bias"]
        out = []
        for r in x:
            mu = sum(r) / len(r)
            sd = math.sqrt(sum((v - mu) ** 2 for v in r) / len(r) + EPS)
            out.append([(v - mu) / sd * gi + bi for v, gi, bi in zip(r, g, b)])
        return out

    def attend(self, prefix, x, keep):
        q = self.linear(prefix + "q_lin", x)
        k = self.linear(prefix + "k_lin", x)
        v = self.linear(prefix + "v_lin", x)
        heads = CONFIG["n_heads"]
        dh = len(q[0]) // heads
        seen = [j for j in range(len(x)) if keep[j]]
        ctx = [[] for _ in x]
        for h in range(heads):
            cols = range(h * dh, (h + 1) * dh)
            for i in range(len(x)):
                s = [sum(q[i][c] * k[j][c] for c in cols) / math.sqrt(dh) for j in seen]
                e = [math.exp(t - max(s)) for t in s]
                ctx[i] += [sum(e[n] * v[j][c] for n, j in enumerate(seen)) / sum(e) for c in cols]
        return self.linear(prefix + "out_lin", ctx)

    def forward(self, ids, keep):
        words, pos = self.rows("embeddings.word_embeddings.weight"), self.rows("embeddings.position_embeddings.weight")
        h = [[a + b for a, b in zip(words[t], pos[i])] for i, t in enumerate(ids)]
        h = self.layer_norm("embeddings.LayerNorm", h)
        for n in range(CONFIG["n_layers"]):
            p = "transformer.layer.%d." % n
            h = self.layer_norm(p + "sa_layer_norm", plus(h, self.attend(p + "attention.", h, keep)))
            up = [[gelu_erf(v) for v in r] for r in self.linear(p + "ffn.lin1", h)]
            h = self.layer_norm(p + "output_layer_norm", plus(h, self.linear(p + "ffn.lin2", up)))
        return h


def pooled(m, ids, seq):
    """The mean of the token vectors of `ids`, run padded to `seq` as a batch does."""
    n = len(ids)
    h = m.forward(ids + [CONFIG["pad_token_id"]] * (seq - n), [True] * n + [False] * (seq - n))
    return [sum(r[c] for r in h[:n]) / n for c in range(CONFIG["dim"])]


def cosine_loss(m):
    """The mean of `(cos(a, b) - score)^2` over `PAIRS`, as `Sentence.cosine_similarity_loss`."""
    seq = max(len(s) for pair in PAIRS for s in pair)
    total = 0.0
    for (left, right), score in zip(PAIRS, SCORES):
        a, b = pooled(m, left, seq), pooled(m, right, seq)
        dot, aa, bb = sum(x * y for x, y in zip(a, b)), sum(x * x for x in a), sum(y * y for y in b)
        total += (dot / math.sqrt(aa * bb + COSINE_EPS) - score) ** 2
    return total / len(PAIRS)


def slope(m, name, at):
    """`(f(x + h) - f(x - h)) / 2h` of the loss for the value `at` of the weight `name`."""
    old = m.values[name][at]
    m.values[name][at] = old + STEP
    up = cosine_loss(m)
    m.values[name][at] = old - STEP
    down = cosine_loss(m)
    m.values[name][at] = old
    return (up - down) / (2 * STEP)


def plus(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def gelu_erf(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


def distilbert():
    c, m = CONFIG, Model(1820)
    d = c["dim"]
    m.add("embeddings.word_embeddings.weight", [c["vocab_size"], d])
    m.add("embeddings.position_embeddings.weight", [c["max_position_embeddings"], d])
    m.norm("embeddings.LayerNorm", d)
    for i in range(c["n_layers"]):
        p = "transformer.layer.%d." % i
        for part in ["q_lin", "k_lin", "v_lin", "out_lin"]:
            m.dense(p + "attention." + part, d, d)
        m.norm(p + "sa_layer_norm", d)
        m.dense(p + "ffn.lin1", c["hidden_dim"], d)
        m.dense(p + "ffn.lin2", d, c["hidden_dim"])
        m.norm(p + "output_layer_norm", d)
    return m


def prefixed_name(name):
    """The names of a `DistilBertFor...` checkpoint: the head keeps its own."""
    if name.startswith(("pre_classifier.", "classifier.")):
        return name
    return "distilbert." + name


def write(out):
    m = distilbert()
    seq = max(len(x) for x in IDS)
    hidden, mean, unit = [], [], []
    for ids in IDS:
        n = len(ids)
        pad = seq - n
        h = m.forward(ids + [CONFIG["pad_token_id"]] * pad, [True] * n + [False] * pad)
        hidden.append(h)
        row = [sum(r[c] for r in h[:n]) / n for c in range(CONFIG["dim"])]
        length = math.sqrt(sum(v * v for v in row))
        mean.append(row)
        unit.append([v / length for v in row])
    head = distilbert()
    head.dense("pre_classifier", CONFIG["dim"], CONFIG["dim"])
    head.dense("classifier", 2, CONFIG["dim"])
    for folder, model, rename, skip in [("tiny", m, lambda n: n, ()), ("prefixed", head, prefixed_name, ()),
                                        ("missing", m, lambda n: n, ("transformer.layer.1.ffn.lin2.weight",))]:
        (out / folder).mkdir()
        model.save(out / folder / "model.safetensors", rename, skip)
        (out / folder / "config.json").write_text(json.dumps(CONFIG))
    slopes = [dict(name=name, at=at, slope=slope(m, name, at)) for name, at in PROBES]
    (out / "expected.json").write_text(json.dumps(dict(
        ids=IDS, hidden=hidden, mean=mean, unit=unit, loss=cosine_loss(m), slopes=slopes)))


def main():
    out = Path(tempfile.mkdtemp(prefix="agentica-distilbert-"))
    try:
        write(out)
    except BaseException:
        shutil.rmtree(out)
        raise
    print(out)


if __name__ == "__main__":
    main()
