"""Writes the images of `tools/resnet_check.agc` into the folder given as the argument: four images, bigger than the 256
pixels of the shorter side that the model's image processor resizes to, so they go through `Resize(256)` and
`CenterCrop(224)` as photos do. They are drawn, not photographed (smooth colors and a few discs with soft edges), so
they are small and have no license; what a model says of them is arbitrary, but the same for every implementation.
Soft edges because libvips and Pillow resize with different kernels and differ most at sharp ones. Standard library
only: `python3 -I test/resnet_images.py test/fixtures/resnet`.
"""

import math
import random
import struct
import sys
import zlib
from pathlib import Path

EDGE = 8.0
# Width, height and seed. The seeds are the ones where both models rank the best 5 classes with gaps wider than what
# the resize of libvips and Pillow can move.
SIZES = {"wide-1": (480, 360, 8), "wide-2": (500, 333, 17), "tall-1": (300, 400, 2), "tall-2": (300, 400, 22)}


def draw(seed, width, height):
    rng = random.Random(seed)
    base = [rng.uniform(0, 255) for _ in range(3)]
    freq = [rng.uniform(1.0, 4.0) for _ in range(3)]
    shapes = [(rng.uniform(0, width), rng.uniform(0, height), rng.uniform(15, 60), [rng.randrange(256) for _ in range(3)])
              for _ in range(5)]
    rows = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            px = [int(base[c] * 0.5 + 100 + 80 * math.sin(freq[c] * (x / width + y / height) * math.pi)) for c in range(3)]
            for cx, cy, r, color in shapes:
                cover = max(0.0, min(1.0, (r - math.hypot(x - cx, y - cy)) / EDGE + 0.5))
                px = [p * (1 - cover) + c * cover for p, c in zip(px, color)]
            row += bytes(max(0, min(255, round(v))) for v in px)
        rows.append(bytes(row))
    return rows


def png(width, height, rows):
    def chunk(kind, data):
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + r for r in rows)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, (w, h, seed) in SIZES.items():
        (out / (name + ".png")).write_bytes(png(w, h, draw(seed, w, h)))


main()
