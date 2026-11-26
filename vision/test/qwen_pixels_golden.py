"""Prints what Hugging Face's image processor of Qwen3.5 (`Qwen2VLImageProcessorPil`, which `AutoImageProcessor` gives
for `Qwen/Qwen3.5-0.8B`) makes of the drawn images of `test/fixtures/resnet` (and one of the 8-bit images that it makes, to grow it), for `test/qwen_pixels_test.agc` to compare:
for each case (an image and the pixel bounds `size` of the processor) the grid `image_grid_thw`, the mean of
`pixel_values` and one value of every `stride`th place of it (an odd number that gives about `SAMPLES` of them), and a table of `smart_resize` for the sizes that decide its
branches (a half that rounds to the even multiple, a side that rounds to 0, the shrink, the grow, the floor of one
factor, and the default bounds of the checkpoint). It also writes what the processor made of two images (the resized
image, 8 bits per channel) as a PNG into CROPS_DIR: `pixel_values` is exactly the patches of those samples, scaled and
normalized, so the test runs `Vision.patches` alone on the committed files. The script checks that.

Manual, not run by the tests or CI: it needs `torch` 2.15.0 and `transformers` 5.22.2 (in a venv, the script stops on
other versions) and `pillow` 11.3.0, from PyPI as the wheel `pillow-11.3.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`
with sha256 `13f87d581e71d9189ab21fe0efb5a23e9f28552d5be6979e84001d3b8505abe8` (the script stops on another version),
installed into a folder of its own, which the script puts first on `sys.path`:

    pip download pillow==11.3.0 --only-binary :all: --no-deps -d WHEEL_DIR
    sha256sum WHEEL_DIR/*.whl
    pip install --no-deps --only-binary :all: --target PILLOW_DIR WHEEL_DIR/*.whl

Run it from `vision/`, with Python's isolated mode:

    python -I test/qwen_pixels_golden.py test/fixtures/resnet test/fixtures/qwen/crops PILLOW_DIR CONFIG_DIR > test/fixtures/qwen/golden.json

CONFIG_DIR holds `preprocessor_config.json` of `Qwen/Qwen3.5-0.8B` (revision `2fc06364715b967f1860aea9cf38778875588b17`).
"""

import json
import sys
from pathlib import Path

PILLOW_DIR = sys.argv[3]
sys.path.insert(0, PILLOW_DIR)

import numpy as np
import PIL
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor
from transformers.image_utils import PILImageResampling
from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil, smart_resize

SAMPLES = 500
PATCH = 16
MERGE = 2
FRAMES = 2
FACTOR = PATCH * MERGE
DEFAULTS = {"shortest_edge": 65536, "longest_edge": 16777216}
# The id of the case, its image (in `test/fixtures`; `qwen/crops/tall-2.png` is made by the case `tall-2` of this script,
# so it comes after it), the bounds of `size` (none: the checkpoint's own), and whether the resized image is written (the
# small ones).
CASES = [
    ("default-1", "resnet/wide-1.png", None, None, False),
    ("wide-1", "resnet/wide-1.png", 4096, 16384, True),
    ("tall-1", "resnet/tall-1.png", 4096, 262144, False),
    ("tall-2", "resnet/tall-2.png", 1024, 4096, True),
    ("grow-1", "qwen/crops/wide-1.png", 24576, 16777216, False),
]
# `smart_resize` of height, width, min_pixels and max_pixels, at the factor 32.
SIZES = [
    (48, 48, 1, 1 << 30), (80, 80, 1, 1 << 30), (16, 700, 1, 1 << 30), (10, 10, 4096, 1 << 30),
    (360, 480, 4096, 16384), (333, 500, 262144, 16777216), (4000, 3000, 65536, 16777216),
    (6000, 8000, 65536, 16777216), (50, 3000, 65536, 16777216), (300, 400, 1024, 4096), (96, 128, 24576, 16777216), (20, 4000, 1, 3000),
    (1, 200, 1, 1 << 30), (224, 224, 65536, 16777216), (255, 257, 65536, 16777216), (1000, 1500, 65536, 16777216),
]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__
    assert PIL.__version__ == "11.3.0", PIL.__version__
    assert Path(PIL.__file__).resolve().is_relative_to(Path(PILLOW_DIR).resolve()), PIL.__file__


def unpatch(rows, grid_h, grid_w):
    """The 8-bit image whose `pixel_values` are `rows` (normalized with 0.5 and 0.5)."""
    channels = 3
    x = rows.reshape(grid_h // MERGE, grid_w // MERGE, MERGE, MERGE, channels, FRAMES, PATCH, PATCH)
    assert (x[:, :, :, :, :, 0] == x[:, :, :, :, :, 1]).all(), "the frames of a still image differ"
    x = x[:, :, :, :, :, 0].transpose(4, 0, 2, 5, 1, 3, 6)
    image = x.reshape(channels, grid_h * PATCH, grid_w * PATCH)
    samples = np.round((image * 0.5 + 0.5) * 255.0)
    assert np.abs((samples / 255.0 - 0.5) / 0.5 - image).max() < 1e-5, "pixel_values are not 8-bit samples"
    return samples.astype(np.uint8).transpose(1, 2, 0)


def main():
    check_environment()
    images_dir, crops_dir, config_dir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[4]
    fixtures = images_dir.parent
    crops_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, image, low, high, write in CASES:
        if low is None:
            # No override: the bounds are the checkpoint's own.
            low, high = DEFAULTS["shortest_edge"], DEFAULTS["longest_edge"]
            processor = AutoImageProcessor.from_pretrained(config_dir)
        else:
            processor = AutoImageProcessor.from_pretrained(config_dir, size={"shortest_edge": low, "longest_edge": high})
        assert isinstance(processor, Qwen2VLImageProcessorPil), type(processor)
        assert processor.size.shortest_edge == low and processor.size.longest_edge == high, processor.size
        assert tuple(processor.image_mean) == (0.5,) * 3 and tuple(processor.image_std) == (0.5,) * 3
        assert processor.do_rescale and processor.rescale_factor == 1 / 255, processor.rescale_factor
        assert processor.do_convert_rgb and processor.resample == PILImageResampling.BICUBIC, processor.resample
        assert (processor.patch_size, processor.merge_size, processor.temporal_patch_size) == (PATCH, MERGE, FRAMES)
        source = Image.open(fixtures / image).convert("RGB")
        out = processor(source, return_tensors="np")
        rows = out["pixel_values"]
        _, grid_h, grid_w = (int(v) for v in out["image_grid_thw"][0])
        if write:
            Image.fromarray(unpatch(rows, grid_h, grid_w)).save(crops_dir / f"{name}.png", optimize=True)
        flat = rows.reshape(-1)
        stride = max(1, flat.size // SAMPLES) | 1
        cases.append({
            "name": name, "image": image, "min_pixels": low, "max_pixels": high, "grid": [grid_h, grid_w], "crop": write,
            "stride": stride,
            "mean": round(float(rows.mean()), 6),
            "samples": [round(float(v), 6) for v in flat[::stride]],
        })
    sizes = [[h, w, low, high] + list(smart_resize(h, w, FACTOR, low, high)) for h, w, low, high in SIZES]
    json.dump({"cases": cases, "sizes": sizes}, sys.stdout, separators=(",", ":"))


main()
