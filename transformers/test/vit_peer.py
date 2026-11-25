"""The independent peer of `vit_test.agc`: tiny random ViTs written as safetensors files with Hugging Face's names
(`ViTForImageClassification.save_pretrained`) plus their config.json, and their forward pass in doubles. Standard
library only.

Usage: python3 vit_peer.py. Writes into a new temporary folder, prints its path, and leaves one folder for each model
(`biased` and `unbiased`, a config with `qkv_bias` false and no biases on the query, key and value projections),
`missing` (`biased` without a weight) and `expected.json`: for each model, the batch of `images` (`[2, 3, 8, 8]`) and
its logits `[2, 5]`.
"""

import json
import math
import random
import shutil
import struct
import tempfile
from pathlib import Path

EPS = 1e-6
CLASSES = 5
SIZE, PATCH, CHANNELS = 8, 4, 3
WIDTH, HEADS, LAYERS, INNER = 8, 2, 2, 16


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def gelu(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def matvec(w, b, x):
    return [sum(w[o][i] * x[i] for i in range(len(x))) + (b[o] if b else 0.0) for o in range(len(w))]


def layer_norm(x, gamma, beta):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    return [(v - mean) / math.sqrt(var + EPS) * g + b for v, g, b in zip(x, gamma, beta)]


class Model:
    def __init__(self, seed, qkv_bias):
        self.rng = random.Random(seed)
        self.config = dict(model_type="vit", hidden_size=WIDTH, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                           intermediate_size=INNER, hidden_act="gelu", layer_norm_eps=EPS, image_size=SIZE,
                           patch_size=PATCH, num_channels=CHANNELS, qkv_bias=qkv_bias,
                           id2label={str(i): "class %d" % i for i in range(CLASSES)})
        self.shapes, self.values = {}, {}
        self.build()

    def add(self, name, shape, center=0.0, spread=0.5):
        self.shapes[name] = shape
        self.values[name] = [f32(center + self.rng.uniform(-spread, spread)) for _ in range(math.prod(shape))]

    def dense(self, prefix, out, inp, bias=True):
        self.add(prefix + ".weight", [out, inp], 0.0, 0.5)
        if bias:
            self.add(prefix + ".bias", [out], 0.0, 0.3)

    def norm(self, prefix):
        self.add(prefix + ".weight", [WIDTH], 1.0, 0.3)
        self.add(prefix + ".bias", [WIDTH], 0.0, 0.3)

    def build(self):
        grid = SIZE // PATCH
        e = "vit.embeddings."
        self.add(e + "patch_embeddings.projection.weight", [WIDTH, CHANNELS, PATCH, PATCH], 0.0, 0.4)
        self.add(e + "patch_embeddings.projection.bias", [WIDTH], 0.0, 0.3)
        self.add(e + "cls_token", [1, 1, WIDTH], 0.0, 0.8)
        self.add(e + "position_embeddings", [1, grid * grid + 1, WIDTH], 0.0, 0.8)
        for i in range(LAYERS):
            p = "vit.encoder.layer.%d." % i
            self.norm(p + "layernorm_before")
            for name in ("query", "key", "value"):
                self.dense(p + "attention.attention." + name, WIDTH, WIDTH, self.config["qkv_bias"])
            self.dense(p + "attention.output.dense", WIDTH, WIDTH)
            self.norm(p + "layernorm_after")
            self.dense(p + "intermediate.dense", INNER, WIDTH)
            self.dense(p + "output.dense", WIDTH, INNER)
        self.norm("vit.layernorm")
        self.dense("classifier", CLASSES, WIDTH)

    def save(self, path, skip=None):
        header, data = {}, b""
        for name in sorted(self.values):
            if name == skip:
                continue
            raw = struct.pack("<%df" % len(self.values[name]), *self.values[name])
            header[name] = {"dtype": "F32", "shape": self.shapes[name], "data_offsets": [len(data), len(data) + len(raw)]}
            data += raw
        text = json.dumps(header).encode()
        text += b" " * (-len(text) % 8)
        path.write_bytes(struct.pack("<Q", len(text)) + text + data)

    # The forward pass, in doubles. An image is [channel][row][column].

    def matrix(self, name):
        w, shape = self.values[name + ".weight"], self.shapes[name + ".weight"]
        return [w[o * shape[1]:(o + 1) * shape[1]] for o in range(shape[0])]

    def bias(self, name):
        return self.values.get(name + ".bias")

    def apply(self, name, x):
        return matvec(self.matrix(name), self.bias(name), x)

    def norm_of(self, name, x):
        return layer_norm(x, self.values[name + ".weight"], self.values[name + ".bias"])

    def attention(self, p, xs):
        q = [self.apply(p + "attention.attention.query", x) for x in xs]
        k = [self.apply(p + "attention.attention.key", x) for x in xs]
        v = [self.apply(p + "attention.attention.value", x) for x in xs]
        d = WIDTH // HEADS
        out = [[] for _ in xs]
        for h in range(HEADS):
            lo, hi = h * d, (h + 1) * d
            for i in range(len(xs)):
                scores = [sum(q[i][c] * k[j][c] for c in range(lo, hi)) / math.sqrt(d) for j in range(len(xs))]
                top = max(scores)
                weights = [math.exp(s - top) for s in scores]
                total = sum(weights)
                out[i] += [sum(weights[j] / total * v[j][c] for j in range(len(xs))) for c in range(lo, hi)]
        return [self.apply(p + "attention.output.dense", o) for o in out]

    def logits(self, image):
        e = "vit.embeddings."
        w = self.values[e + "patch_embeddings.projection.weight"]
        b = self.values[e + "patch_embeddings.projection.bias"]
        grid = SIZE // PATCH
        tokens = [self.values[e + "cls_token"][:]]
        for gy in range(grid):
            for gx in range(grid):
                tokens.append([b[o] + sum(w[((o * CHANNELS + c) * PATCH + ky) * PATCH + kx]
                                          * image[c][gy * PATCH + ky][gx * PATCH + kx]
                                          for c in range(CHANNELS) for ky in range(PATCH) for kx in range(PATCH))
                               for o in range(WIDTH)])
        pos = self.values[e + "position_embeddings"]
        xs = [[t + pos[i * WIDTH + c] for c, t in enumerate(tok)] for i, tok in enumerate(tokens)]
        for i in range(LAYERS):
            p = "vit.encoder.layer.%d." % i
            attended = self.attention(p, [self.norm_of(p + "layernorm_before", x) for x in xs])
            xs = [[a + b for a, b in zip(x, y)] for x, y in zip(xs, attended)]
            for n, x in enumerate(xs):
                hidden = [gelu(v) for v in self.apply(p + "intermediate.dense", self.norm_of(p + "layernorm_after", x))]
                xs[n] = [a + b for a, b in zip(x, self.apply(p + "output.dense", hidden))]
        return self.apply("classifier", self.norm_of("vit.layernorm", xs[0]))


def write(out):
    rng = random.Random(83)
    images = [[[[f32(rng.uniform(-1.5, 1.5)) for _ in range(SIZE)] for _ in range(SIZE)] for _ in range(CHANNELS)]
              for _ in range(2)]
    expected = {"images": images}
    for seed, (name, qkv_bias) in enumerate([("biased", True), ("unbiased", False)]):
        m = Model(seed + 1, qkv_bias)
        (out / name).mkdir()
        m.save(out / name / "model.safetensors")
        (out / name / "config.json").write_text(json.dumps(m.config))
        expected[name] = [m.logits(image) for image in images]
        if name == "biased":
            (out / "missing").mkdir()
            m.save(out / "missing" / "model.safetensors", skip="vit.encoder.layer.1.attention.attention.key.weight")
            (out / "missing" / "config.json").write_text(json.dumps(m.config))
    (out / "expected.json").write_text(json.dumps(expected))


def main():
    out = Path(tempfile.mkdtemp(prefix="agentica-vit-"))
    try:
        write(out)
    except BaseException:
        shutil.rmtree(out)
        raise
    print(out)


if __name__ == "__main__":
    main()
