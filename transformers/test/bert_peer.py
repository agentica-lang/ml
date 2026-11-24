"""The independent peer of `bert_test.agc`: a tiny random BERT written as a
safetensors file with Hugging Face's names plus its config.json, and its
forward pass in doubles. Standard library only.

Usage: python3 bert_peer.py. Writes into a new temporary folder, prints its
path, and leaves:

- `tiny/`: the model as `BertModel.save_pretrained` writes it (no prefix,
  `weight` and `bias` norms);
- `legacy/`: the same weights as `bert-base-uncased` stores them (`bert.`
  prefix, `gamma` and `beta` norms, and the `cls.*` head the model ignores);
- `nopooler/`: `tiny` without the pooler; `missing/`: `tiny` without one weight;
- `expected.json`: for the batch of `BATCH`, the hidden state, the pooled
  output and the mean-pooled embedding, with the exact erf GELU BERT uses.
"""

import json
import math
import random
import shutil
import struct
import tempfile
from pathlib import Path

CONFIG = {
    "model_type": "bert", "hidden_act": "gelu", "hidden_size": 8, "num_hidden_layers": 2,
    "num_attention_heads": 2, "intermediate_size": 16, "vocab_size": 11,
    "max_position_embeddings": 8, "type_vocab_size": 2, "layer_norm_eps": 1e-12,
    "pad_token_id": 0, "position_embedding_type": "absolute",
}
# Two sentences, the second shorter (padded with 0), with token types.
BATCH = {"ids": [[1, 5, 2, 7, 3], [1, 4, 6, 3]], "type_ids": [[0, 0, 0, 1, 1], [0, 0, 0, 0]]}
EPS = CONFIG["layer_norm_eps"]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class Model:
    """Weights by name (flat, row-major F32 values), under HF's modern names."""

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
        q = self.linear(prefix + "self.query", x)
        k = self.linear(prefix + "self.key", x)
        v = self.linear(prefix + "self.value", x)
        heads = CONFIG["num_attention_heads"]
        dh = len(q[0]) // heads
        seen = [j for j in range(len(x)) if keep[j]]
        ctx = [[] for _ in x]
        for h in range(heads):
            cols = range(h * dh, (h + 1) * dh)
            for i in range(len(x)):
                s = [sum(q[i][c] * k[j][c] for c in cols) / math.sqrt(dh) for j in seen]
                e = [math.exp(t - max(s)) for t in s]
                ctx[i] += [sum(e[n] * v[j][c] for n, j in enumerate(seen)) / sum(e) for c in cols]
        return self.linear(prefix + "output.dense", ctx)

    def forward(self, ids, types, keep):
        words, pos = self.rows("embeddings.word_embeddings.weight"), self.rows("embeddings.position_embeddings.weight")
        kinds = self.rows("embeddings.token_type_embeddings.weight")
        h = [[a + b + c for a, b, c in zip(words[t], pos[i], kinds[y])] for i, (t, y) in enumerate(zip(ids, types))]
        h = self.layer_norm("embeddings.LayerNorm", h)
        for n in range(CONFIG["num_hidden_layers"]):
            p = "encoder.layer.%d." % n
            h = self.layer_norm(p + "attention.output.LayerNorm", plus(h, self.attend(p + "attention.", h, keep)))
            up = [[gelu_erf(v) for v in r] for r in self.linear(p + "intermediate.dense", h)]
            h = self.layer_norm(p + "output.LayerNorm", plus(h, self.linear(p + "output.dense", up)))
        return h

    def pooled(self, h):
        return [math.tanh(v) for v in self.linear("pooler.dense", [h[0]])[0]]


def plus(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def gelu_erf(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


def bert():
    c, m = CONFIG, Model(1810)
    d = c["hidden_size"]
    m.add("embeddings.word_embeddings.weight", [c["vocab_size"], d])
    m.add("embeddings.position_embeddings.weight", [c["max_position_embeddings"], d])
    m.add("embeddings.token_type_embeddings.weight", [c["type_vocab_size"], d])
    m.norm("embeddings.LayerNorm", d)
    for i in range(c["num_hidden_layers"]):
        p = "encoder.layer.%d." % i
        for part in ["query", "key", "value"]:
            m.dense(p + "attention.self." + part, d, d)
        m.dense(p + "attention.output.dense", d, d)
        m.norm(p + "attention.output.LayerNorm", d)
        m.dense(p + "intermediate.dense", c["intermediate_size"], d)
        m.dense(p + "output.dense", d, c["intermediate_size"])
        m.norm(p + "output.LayerNorm", d)
    m.dense("pooler.dense", d, d)
    return m


def legacy_name(name):
    """The names of `bert-base-uncased`: a `bert.` prefix, gamma and beta."""
    if name.startswith("cls."):
        return name
    return "bert." + name.replace("LayerNorm.weight", "LayerNorm.gamma").replace("LayerNorm.bias", "LayerNorm.beta")


def write(out):
    m = bert()
    seq = max(len(x) for x in BATCH["ids"])
    hidden, pooled, mean = [], [], []
    for ids, types in zip(BATCH["ids"], BATCH["type_ids"]):
        n = len(ids)
        pad = seq - n
        h = m.forward(ids + [CONFIG["pad_token_id"]] * pad, types + [0] * pad, [True] * n + [False] * pad)
        hidden.append(h)
        pooled.append(m.pooled(h))
        mean.append([sum(r[c] for r in h[:n]) / n for c in range(CONFIG["hidden_size"])])
    # The legacy file also holds the head of a BertFor... model, which the
    # loader must not need.
    head = bert()
    head.add("cls.predictions.bias", [CONFIG["vocab_size"]])
    head.add("cls.seq_relationship.weight", [2, CONFIG["hidden_size"]])
    for folder, model, rename, skip in [("tiny", m, lambda n: n, ()), ("legacy", head, legacy_name, ()),
                                        ("nopooler", m, lambda n: n, ("pooler",)),
                                        ("missing", m, lambda n: n, ("encoder.layer.1.output.dense.weight",))]:
        (out / folder).mkdir()
        model.save(out / folder / "model.safetensors", rename, skip)
        (out / folder / "config.json").write_text(json.dumps(CONFIG))
    (out / "expected.json").write_text(json.dumps(
        dict(BATCH, hidden=hidden, pooled=pooled, mean=mean)))


def main():
    out = Path(tempfile.mkdtemp(prefix="agentica-bert-"))
    try:
        write(out)
    except BaseException:
        shutil.rmtree(out)
        raise
    print(out)


if __name__ == "__main__":
    main()
