"""The independent peer of `clip_test.agc`: a tiny random CLIP written as a safetensors file with Hugging Face's names
(`CLIPModel.save_pretrained`) plus its config.json, and its forward pass in doubles. Standard library only.

Usage: python3 clip_peer.py. Writes into a new temporary folder, prints its path, and leaves the folder `clip`,
`missing` (`clip` without a weight) and `expected.json`: the batch of `images` (`[2, 3, 8, 8]`), the `rows` of text ids
(four rows of 3 to 5 ids, each with the end token, the last one twice) and, for them, `image_embeds` `[2, 6]`, `text_embeds`
`[4, 6]`, `logits` `[2, 4]` (scaled cosines), `probabilities` `[2, 4]` and the `scale` (`exp(logit_scale)`).
"""

import json
import math
import random
import shutil
import struct
import tempfile
from pathlib import Path

EPS = 1e-5
SIZE, PATCH, CHANNELS = 8, 4, 3
PROJECTION = 6
VOCAB, MAX_POSITIONS = 20, 6
VISION = dict(hidden_size=8, intermediate_size=12, num_hidden_layers=2, num_attention_heads=2)
TEXT = dict(hidden_size=6, intermediate_size=10, num_hidden_layers=2, num_attention_heads=3)
# The last row has the end token (19, the largest id) twice: the vector read is the one at the first.
ROWS = [[18, 3, 5, 19], [18, 7, 19], [18, 4, 9, 11, 19], [18, 3, 19, 5, 19]]
LOGIT_SCALE = 2.3


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def gelu(x):
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def layer_norm(x, gamma, beta):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    return [(v - mean) / math.sqrt(var + EPS) * g + b for v, g, b in zip(x, gamma, beta)]


def softmax(xs):
    top = max(xs)
    e = [math.exp(x - top) for x in xs]
    return [v / sum(e) for v in e]


class Model:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.config = dict(model_type="clip", projection_dim=PROJECTION,
                           text_config=dict(TEXT, hidden_act="gelu", layer_norm_eps=EPS,
                                            max_position_embeddings=MAX_POSITIONS, vocab_size=VOCAB),
                           vision_config=dict(VISION, hidden_act="gelu", layer_norm_eps=EPS, image_size=SIZE,
                                              patch_size=PATCH, num_channels=CHANNELS))
        self.shapes, self.values = {}, {}
        self.build()

    def add(self, name, shape, center=0.0, spread=0.5):
        self.shapes[name] = shape
        self.values[name] = [f32(center + self.rng.uniform(-spread, spread)) for _ in range(math.prod(shape))]

    def dense(self, prefix, out, inp, bias=True):
        self.add(prefix + ".weight", [out, inp], 0.0, 0.5)
        if bias:
            self.add(prefix + ".bias", [out], 0.0, 0.3)

    def norm(self, prefix, width):
        self.add(prefix + ".weight", [width], 1.0, 0.3)
        self.add(prefix + ".bias", [width], 0.0, 0.3)

    def layers(self, prefix, tower):
        width, inner = tower["hidden_size"], tower["intermediate_size"]
        for i in range(tower["num_hidden_layers"]):
            p = "%sencoder.layers.%d." % (prefix, i)
            self.norm(p + "layer_norm1", width)
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                self.dense(p + "self_attn." + name, width, width)
            self.norm(p + "layer_norm2", width)
            self.dense(p + "mlp.fc1", inner, width)
            self.dense(p + "mlp.fc2", width, inner)

    def build(self):
        w = VISION["hidden_size"]
        grid = SIZE // PATCH
        v = "vision_model."
        self.add(v + "embeddings.class_embedding", [w], 0.0, 0.8)
        self.add(v + "embeddings.patch_embedding.weight", [w, CHANNELS, PATCH, PATCH], 0.0, 0.4)
        self.add(v + "embeddings.position_embedding.weight", [grid * grid + 1, w], 0.0, 0.8)
        self.norm(v + "pre_layrnorm", w)
        self.layers(v, VISION)
        self.norm(v + "post_layernorm", w)
        self.dense("visual_projection", PROJECTION, w, bias=False)
        t = "text_model."
        tw = TEXT["hidden_size"]
        self.add(t + "embeddings.token_embedding.weight", [VOCAB, tw], 0.0, 0.8)
        self.add(t + "embeddings.position_embedding.weight", [MAX_POSITIONS, tw], 0.0, 0.8)
        self.layers(t, TEXT)
        self.norm(t + "final_layer_norm", tw)
        self.dense("text_projection", PROJECTION, tw, bias=False)
        self.shapes["logit_scale"] = []
        self.values["logit_scale"] = [f32(LOGIT_SCALE)]

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

    # The forward pass, in doubles.

    def apply(self, name, x):
        w, shape = self.values[name + ".weight"], self.shapes[name + ".weight"]
        b = self.values.get(name + ".bias")
        return [sum(w[o * shape[1] + i] * x[i] for i in range(shape[1])) + (b[o] if b else 0.0) for o in range(shape[0])]

    def norm_of(self, name, x):
        return layer_norm(x, self.values[name + ".weight"], self.values[name + ".bias"])

    def attention(self, p, xs, heads, causal):
        q = [self.apply(p + "self_attn.q_proj", x) for x in xs]
        k = [self.apply(p + "self_attn.k_proj", x) for x in xs]
        v = [self.apply(p + "self_attn.v_proj", x) for x in xs]
        d = len(xs[0]) // heads
        out = [[] for _ in xs]
        for h in range(heads):
            lo, hi = h * d, (h + 1) * d
            for i in range(len(xs)):
                span = range(i + 1) if causal else range(len(xs))
                w = softmax([sum(q[i][c] * k[j][c] for c in range(lo, hi)) / math.sqrt(d) for j in span])
                out[i] += [sum(w[n] * v[j][c] for n, j in enumerate(span)) for c in range(lo, hi)]
        return [self.apply(p + "self_attn.out_proj", o) for o in out]

    def encoder(self, prefix, tower, xs, causal):
        for i in range(tower["num_hidden_layers"]):
            p = "%sencoder.layers.%d." % (prefix, i)
            attended = self.attention(p, [self.norm_of(p + "layer_norm1", x) for x in xs],
                                      tower["num_attention_heads"], causal)
            xs = [[a + b for a, b in zip(x, y)] for x, y in zip(xs, attended)]
            for n, x in enumerate(xs):
                hidden = [gelu(v) for v in self.apply(p + "mlp.fc1", self.norm_of(p + "layer_norm2", x))]
                xs[n] = [a + b for a, b in zip(x, self.apply(p + "mlp.fc2", hidden))]
        return xs

    def image_embed(self, image):
        e = "vision_model.embeddings."
        w = self.values[e + "patch_embedding.weight"]
        width = VISION["hidden_size"]
        grid = SIZE // PATCH
        tokens = [self.values[e + "class_embedding"][:]]
        for gy in range(grid):
            for gx in range(grid):
                tokens.append([sum(w[((o * CHANNELS + c) * PATCH + ky) * PATCH + kx] * image[c][gy * PATCH + ky][gx * PATCH + kx]
                                   for c in range(CHANNELS) for ky in range(PATCH) for kx in range(PATCH))
                               for o in range(width)])
        pos = self.values[e + "position_embedding.weight"]
        xs = [[t + pos[i * width + c] for c, t in enumerate(tok)] for i, tok in enumerate(tokens)]
        xs = [self.norm_of("vision_model.pre_layrnorm", x) for x in xs]
        xs = self.encoder("vision_model.", VISION, xs, False)
        return self.apply("visual_projection", self.norm_of("vision_model.post_layernorm", xs[0]))

    def text_embed(self, ids):
        e = "text_model.embeddings."
        width = TEXT["hidden_size"]
        table, pos = self.values[e + "token_embedding.weight"], self.values[e + "position_embedding.weight"]
        xs = [[table[t * width + c] + pos[i * width + c] for c in range(width)] for i, t in enumerate(ids)]
        xs = self.encoder("text_model.", TEXT, xs, True)
        at = ids.index(max(ids))
        return self.apply("text_projection", self.norm_of("text_model.final_layer_norm", xs[at]))


def unit(x):
    n = math.sqrt(sum(v * v for v in x))
    return [v / n for v in x]


def write(out):
    rng = random.Random(83)
    images = [[[[f32(rng.uniform(-1.5, 1.5)) for _ in range(SIZE)] for _ in range(SIZE)] for _ in range(CHANNELS)]
              for _ in range(2)]
    m = Model(1)
    (out / "clip").mkdir()
    m.save(out / "clip" / "model.safetensors")
    (out / "clip" / "config.json").write_text(json.dumps(m.config))
    (out / "missing").mkdir()
    m.save(out / "missing" / "model.safetensors", skip="text_model.encoder.layers.1.self_attn.k_proj.weight")
    (out / "missing" / "config.json").write_text(json.dumps(m.config))
    image_embeds = [m.image_embed(i) for i in images]
    text_embeds = [m.text_embed(r) for r in ROWS]
    scale = math.exp(f32(LOGIT_SCALE))
    logits = [[scale * sum(a * b for a, b in zip(unit(i), unit(t))) for t in text_embeds] for i in image_embeds]
    (out / "expected.json").write_text(json.dumps(dict(
        images=images, rows=ROWS, image_embeds=image_embeds, text_embeds=text_embeds, logits=logits,
        probabilities=[softmax(r) for r in logits], scale=scale)))


def main():
    out = Path(tempfile.mkdtemp(prefix="agentica-clip-"))
    try:
        write(out)
    except BaseException:
        shutil.rmtree(out)
        raise
    print(out)


if __name__ == "__main__":
    main()
