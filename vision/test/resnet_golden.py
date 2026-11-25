"""Prints what Hugging Face answers for the images of `test/fixtures/resnet` with `microsoft/resnet-50` and
`microsoft/resnet-18`, for `tools/resnet_check.agc` to compare: for each image, what its image processor makes of it
(the mean of each channel and a grid of 8 by 8 pixels of `pixel_values`), and for each model the best 10 classes with
their logits, from `ResNetForImageClassification` in `float32`. It also writes `pixel_values` of every image to
`PIXELS_FILE` as little-endian `float32` (`[n, 3, 224, 224]`), which the check uses to test the model alone; that
file is not committed.

Manual, not run by the tests or CI: it needs `torch`, `transformers` (4.57.6 was used) and `pillow`. Run it in a
throwaway venv, from `vision/`, with Python's isolated mode:

    python -I test/resnet_golden.py test/fixtures/resnet PIXELS_FILE MODEL_DIR... > test/fixtures/resnet/golden.json

Each MODEL_DIR holds `model.safetensors`, `config.json` and `preprocessor_config.json` of one checkpoint (the revisions
that `transformers/DESIGN.md` pins), and its name in the output is the name of the folder.
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification

TOP = 10
GRID = 8


def main():
    images_dir, pixels_file, model_dirs = Path(sys.argv[1]), sys.argv[2], [Path(p) for p in sys.argv[3:]]
    names = sorted(p.name for p in images_dir.glob("*.png"))
    processor = AutoImageProcessor.from_pretrained(model_dirs[0], use_fast=False)
    pixels = processor([Image.open(images_dir / n).convert("RGB") for n in names], return_tensors="pt")["pixel_values"]
    with open(pixels_file, "wb") as f:
        f.write(pixels.numpy().astype("<f4").tobytes())

    steps = [round(i * (pixels.shape[2] - 1) / (GRID - 1)) for i in range(GRID)]
    out = {"images": [], "models": {}}
    for i, name in enumerate(names):
        grid = [[[round(float(pixels[i, c, y, x]), 6) for x in steps] for y in steps] for c in range(3)]
        out["images"].append({"name": name, "mean": [round(float(pixels[i, c].mean()), 6) for c in range(3)],
                              "grid": grid})
    for path in model_dirs:
        model = ResNetForImageClassification.from_pretrained(path, torch_dtype=torch.float32).eval()
        with torch.no_grad():
            logits = model(pixels).logits
        cases = []
        for row in logits:
            best = row.topk(TOP)
            cases.append([{"id": int(i), "label": model.config.id2label[int(i)], "logit": round(float(v), 5)}
                          for v, i in zip(best.values, best.indices)])
        out["models"][path.name] = cases
    json.dump(out, sys.stdout, separators=(",", ":"))


main()
