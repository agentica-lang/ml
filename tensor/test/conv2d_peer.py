"""The independent peer of `conv2d_test.agc`: `conv2d`, `max_pool2d` and `avg_pool2d` written as plain loops in doubles
(no shared code with `tensor/`), with the gradient of `sum(out * ct)` for the input and the weight. Standard library
only.

Usage: python3 conv2d_peer.py. Writes `cases.json` into a new temporary folder and prints the folder. Every input is
rounded to F32, so the F32 backends read the same numbers. The max pool inputs repeat their values (zeros, as after a
relu, and equal maxima): a gradient goes to the first position of a window that holds the maximum, in row-major order,
as libtorch does.
"""

import json
import random
import struct
import tempfile
from pathlib import Path


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


class Grid:
    """A row-major [N, C, H, W] array of doubles."""

    def __init__(self, shape, values):
        self.shape, self.values = shape, values

    def at(self, n, c, y, x):
        return ((n * self.shape[1] + c) * self.shape[2] + y) * self.shape[3] + x


def random_values(rng, count, ties=False):
    if not ties:
        return [f32(rng.uniform(-1.0, 1.0)) for _ in range(count)]
    pool = [0.0, 0.0, 0.0, 0.5, 0.5, -0.25, f32(rng.uniform(-1.0, 1.0)), f32(rng.uniform(0.0, 1.0))]
    return [rng.choice(pool) for _ in range(count)]


def windows(h, w, kh, kw, stride, padding):
    """Per output position, the (input y, input x, kernel y, kernel x) of the taps inside the input."""
    out_h, out_w = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    for oy in range(out_h):
        for ox in range(out_w):
            taps = []
            for ky in range(kh):
                for kx in range(kw):
                    y, x = oy * stride + ky - padding, ox * stride + kx - padding
                    if 0 <= y < h and 0 <= x < w:
                        taps.append((y, x, ky, kx))
            yield oy, ox, taps
    return


def conv_case(name, rng, n, c, h, w, o, kh, kw, stride, padding, groups):
    cg, og = c // groups, o // groups
    x = Grid([n, c, h, w], random_values(rng, n * c * h * w))
    wt = Grid([o, cg, kh, kw], random_values(rng, o * cg * kh * kw))
    out_h, out_w = (h + 2 * padding - kh) // stride + 1, (w + 2 * padding - kw) // stride + 1
    ct = Grid([n, o, out_h, out_w], random_values(rng, n * o * out_h * out_w))
    out = [0.0] * len(ct.values)
    dx, dw = [0.0] * len(x.values), [0.0] * len(wt.values)
    for b in range(n):
        for co in range(o):
            first = (co // og) * cg
            for oy, ox, taps in windows(h, w, kh, kw, stride, padding):
                at = ct.at(b, co, oy, ox)
                for ci in range(cg):
                    for y, xx, ky, kx in taps:
                        i, j = x.at(b, first + ci, y, xx), wt.at(co, ci, ky, kx)
                        out[at] += x.values[i] * wt.values[j]
                        dx[i] += ct.values[at] * wt.values[j]
                        dw[j] += ct.values[at] * x.values[i]
    return {"name": name, "kind": "conv2d", "stride": stride, "padding": padding, "groups": groups, "x": x.shape,
            "xv": x.values, "w": wt.shape, "wv": wt.values, "ct": ct.values, "out_shape": ct.shape, "out": out, "dx": dx, "dw": dw}


def pool_case(name, rng, maximum, n, c, h, w, kernel, stride, padding):
    x = Grid([n, c, h, w], random_values(rng, n * c * h * w, ties=maximum))
    out_h, out_w = (h + 2 * padding - kernel) // stride + 1, (w + 2 * padding - kernel) // stride + 1
    ct = Grid([n, c, out_h, out_w], random_values(rng, n * c * out_h * out_w))
    out = [0.0] * len(ct.values)
    dx = [0.0] * len(x.values)
    for b in range(n):
        for ch in range(c):
            for oy, ox, taps in windows(h, w, kernel, kernel, stride, padding):
                at = ct.at(b, ch, oy, ox)
                if maximum:
                    best = None
                    for y, xx, _, _ in taps:
                        i = x.at(b, ch, y, xx)
                        if best is None or x.values[i] > x.values[best]:
                            best = i
                    out[at] = x.values[best]
                    dx[best] += ct.values[at]
                else:
                    for y, xx, _, _ in taps:
                        i = x.at(b, ch, y, xx)
                        out[at] += x.values[i] / (kernel * kernel)
                        dx[i] += ct.values[at] / (kernel * kernel)
    return {"name": name, "kind": "max_pool2d" if maximum else "avg_pool2d", "kernel": kernel, "stride": stride,
            "padding": padding, "x": x.shape, "xv": x.values, "ct": ct.values, "out_shape": ct.shape, "out": out, "dx": dx}


def main():
    rng = random.Random(83)
    cases = [
        conv_case("3x3 stride 1 padding 1", rng, 2, 3, 5, 6, 4, 3, 3, 1, 1, 1),
        conv_case("3x3 stride 2 padding 1, an odd remainder", rng, 1, 2, 7, 6, 3, 3, 3, 2, 1, 1),
        conv_case("2x3 kernel, stride 2, two groups", rng, 2, 4, 6, 7, 4, 2, 3, 2, 1, 2),
        conv_case("1x1 stride 2", rng, 2, 3, 5, 5, 4, 1, 1, 2, 0, 1),
        conv_case("7x7 stride 2 padding 3", rng, 1, 2, 9, 9, 2, 7, 7, 2, 3, 1),
        pool_case("max 3x3 stride 2 padding 1", rng, True, 2, 2, 7, 7, 3, 2, 1),
        pool_case("max 2x2 stride 2", rng, True, 1, 2, 6, 6, 2, 2, 0),
        pool_case("avg 3x3 stride 2 padding 1", rng, False, 2, 2, 7, 6, 3, 2, 1),
        pool_case("avg 2x2 stride 1", rng, False, 1, 2, 4, 5, 2, 1, 0),
    ]
    folder = Path(tempfile.mkdtemp(prefix="agentica-conv2d-"))
    (folder / "cases.json").write_text(json.dumps({"cases": cases}))
    print(folder)


if __name__ == "__main__":
    main()
