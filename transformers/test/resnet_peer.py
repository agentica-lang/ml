"""The independent peer of `resnet_test.agc`: tiny random ResNets written as safetensors files with Hugging Face's names
(`ResNetForImageClassification.save_pretrained`) plus their config.json, and their forward pass in doubles with the
batch norms as they are in the checkpoint (not folded). Standard library only.

Usage: python3 resnet_peer.py. Writes into a new temporary folder, prints its path, and leaves one folder for each model
(`bottleneck`, `basic`, `strided`: a bottleneck with `downsample_in_first_stage` and `downsample_in_bottleneck`),
`missing` (`basic` without a weight) and `expected.json`: for each model, the batch of `images` (`[2, 3, 21, 19]`) and
its logits `[2, 5]`. The batch norms have their own statistics, so a fold that dropped the mean, the variance or a shift
would show.
"""

import json
import math
import random
import shutil
import struct
import tempfile
from pathlib import Path

EPS = 1e-5
CLASSES = 5
HEIGHT, WIDTH = 21, 19

MODELS = {
    "bottleneck": dict(embedding_size=4, hidden_sizes=[8, 12], depths=[1, 2], layer_type="bottleneck",
                       downsample_in_first_stage=False, downsample_in_bottleneck=False),
    "basic": dict(embedding_size=4, hidden_sizes=[4, 8, 8], depths=[1, 1, 2], layer_type="basic",
                  downsample_in_first_stage=False, downsample_in_bottleneck=False),
    "strided": dict(embedding_size=4, hidden_sizes=[8, 16], depths=[2, 1], layer_type="bottleneck",
                    downsample_in_first_stage=True, downsample_in_bottleneck=True),
}


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class Model:
    def __init__(self, seed, config):
        self.rng = random.Random(seed)
        self.config = dict(config, model_type="resnet", num_channels=3, hidden_act="relu",
                           id2label={str(i): "class %d" % i for i in range(CLASSES)})
        self.shapes, self.values = {}, {}
        self.build()

    def add(self, name, shape, center=0.0, spread=0.5):
        self.shapes[name] = shape
        self.values[name] = [f32(center + self.rng.uniform(-spread, spread)) for _ in range(math.prod(shape))]

    def conv(self, prefix, out, inp, kernel):
        self.add(prefix + ".convolution.weight", [out, inp, kernel, kernel], 0.0, 0.4)
        n = prefix + ".normalization"
        self.add(n + ".weight", [out], 1.0, 0.4)
        self.add(n + ".bias", [out], 0.0, 0.4)
        self.add(n + ".running_mean", [out], 0.0, 0.5)
        self.add(n + ".running_var", [out], 1.0, 0.6)

    def build(self):
        c = self.config
        self.conv("resnet.embedder.embedder", c["embedding_size"], 3, 7)
        width = c["embedding_size"]
        self.plan = []
        for s, (out, depth) in enumerate(zip(c["hidden_sizes"], c["depths"])):
            for l in range(depth):
                stride = 2 if l == 0 and (s > 0 or c["downsample_in_first_stage"]) else 1
                p = "resnet.encoder.stages.%d.layers.%d" % (s, l)
                self.plan.append((p, width, out, stride))
                if c["layer_type"] == "bottleneck":
                    mid = out // 4
                    self.conv(p + ".layer.0", mid, width, 1)
                    self.conv(p + ".layer.1", mid, mid, 3)
                    self.conv(p + ".layer.2", out, mid, 1)
                else:
                    self.conv(p + ".layer.0", out, width, 3)
                    self.conv(p + ".layer.1", out, out, 3)
                if width != out or stride != 1:
                    self.conv(p + ".shortcut", out, width, 1)
                width = out
        self.add("classifier.1.weight", [CLASSES, width])
        self.add("classifier.1.bias", [CLASSES])

    def save(self, path, skip=()):
        header, data = {}, b""
        for name in sorted(self.values):
            if name == skip:
                continue
            raw = struct.pack("<%df" % len(self.values[name]), *self.values[name])
            header[name] = {"dtype": "F32", "shape": self.shapes[name], "data_offsets": [len(data), len(data) + len(raw)]}
            data += raw
            if name.endswith("running_var"):
                counter = name[:-len("running_var")] + "num_batches_tracked"
                header[counter] = {"dtype": "I64", "shape": [], "data_offsets": [len(data), len(data) + 8]}
                data += struct.pack("<q", 7)
        text = json.dumps(header).encode()
        text += b" " * (-len(text) % 8)
        path.write_bytes(struct.pack("<Q", len(text)) + text + data)

    # The forward pass, in doubles. An image is [channel][row][column].

    def conv_bn(self, prefix, x, stride, kernel, relu):
        w, shape = self.values[prefix + ".convolution.weight"], self.shapes[prefix + ".convolution.weight"]
        out, inp = shape[0], shape[1]
        pad = kernel // 2
        h, wd = len(x[0]), len(x[0][0])
        oh, ow = (h + 2 * pad - kernel) // stride + 1, (wd + 2 * pad - kernel) // stride + 1
        n = prefix + ".normalization"
        gamma, beta = self.values[n + ".weight"], self.values[n + ".bias"]
        mean, var = self.values[n + ".running_mean"], self.values[n + ".running_var"]
        y = []
        for o in range(out):
            plane = []
            for r in range(oh):
                row = []
                for q in range(ow):
                    acc = 0.0
                    for i in range(inp):
                        for ky in range(kernel):
                            for kx in range(kernel):
                                yy, xx = r * stride + ky - pad, q * stride + kx - pad
                                if 0 <= yy < h and 0 <= xx < wd:
                                    acc += x[i][yy][xx] * w[((o * inp + i) * kernel + ky) * kernel + kx]
                    v = (acc - mean[o]) / math.sqrt(var[o] + EPS) * gamma[o] + beta[o]
                    row.append(max(v, 0.0) if relu else v)
                plane.append(row)
            y.append(plane)
        return y

    def max_pool(self, x):
        h, w = len(x[0]), len(x[0][0])
        oh, ow = (h + 2 - 3) // 2 + 1, (w + 2 - 3) // 2 + 1
        return [[[max(p[yy][xx] for yy in range(r * 2 - 1, r * 2 + 2) for xx in range(q * 2 - 1, q * 2 + 2)
                      if 0 <= yy < h and 0 <= xx < w) for q in range(ow)] for r in range(oh)] for p in x]

    def layer(self, p, x, stride):
        c = self.config
        if c["layer_type"] == "bottleneck":
            first, second = (stride, 1) if c["downsample_in_bottleneck"] else (1, stride)
            h = self.conv_bn(p + ".layer.0", x, first, 1, True)
            h = self.conv_bn(p + ".layer.1", h, second, 3, True)
            h = self.conv_bn(p + ".layer.2", h, 1, 1, False)
        else:
            h = self.conv_bn(p + ".layer.0", x, stride, 3, True)
            h = self.conv_bn(p + ".layer.1", h, 1, 3, False)
        skip = self.conv_bn(p + ".shortcut", x, stride, 1, False) if p + ".shortcut.convolution.weight" in self.values else x
        return [[[max(a + b, 0.0) for a, b in zip(ra, rb)] for ra, rb in zip(pa, pb)] for pa, pb in zip(h, skip)]

    def logits(self, image):
        x = self.max_pool(self.conv_bn("resnet.embedder.embedder", image, 2, 7, True))
        for p, _, _, stride in self.plan:
            x = self.layer(p, x, stride)
        pooled = [sum(sum(r) for r in plane) / (len(plane) * len(plane[0])) for plane in x]
        w, b = self.values["classifier.1.weight"], self.values["classifier.1.bias"]
        return [sum(w[k * len(pooled) + i] * pooled[i] for i in range(len(pooled))) + b[k] for k in range(CLASSES)]


def write(out):
    rng = random.Random(83)
    images = [[[[f32(rng.uniform(-1.5, 1.5)) for _ in range(WIDTH)] for _ in range(HEIGHT)] for _ in range(3)]
              for _ in range(2)]
    expected = {"images": images}
    for seed, (name, config) in enumerate(MODELS.items()):
        m = Model(seed + 1, config)
        (out / name).mkdir()
        m.save(out / name / "model.safetensors")
        (out / name / "config.json").write_text(json.dumps(m.config))
        expected[name] = [m.logits(image) for image in images]
        if name == "basic":
            (out / "missing").mkdir()
            m.save(out / "missing" / "model.safetensors", skip="resnet.encoder.stages.1.layers.0.shortcut.convolution.weight")
            (out / "missing" / "config.json").write_text(json.dumps(m.config))
    (out / "expected.json").write_text(json.dumps(expected))


def main():
    out = Path(tempfile.mkdtemp(prefix="agentica-resnet-"))
    try:
        write(out)
    except BaseException:
        shutil.rmtree(out)
        raise
    print(out)


if __name__ == "__main__":
    main()
