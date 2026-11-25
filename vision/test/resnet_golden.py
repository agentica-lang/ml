"""Prints what Hugging Face answers for the images of `test/fixtures/resnet` with `microsoft/resnet-50` and
`microsoft/resnet-18`, for `test/resnet_hf_test.agc` to compare: for each image, what its image processor makes of it
(the mean of each channel and a grid of 8 by 8 pixels of `pixel_values`), and for each model the best 10 classes with
their logits, from `ResNetForImageClassification` in `float32`. It also writes what the processor cut from each image
(the 224 by 224 crop, 8 bits per channel) as a PNG into CROPS_DIR: `pixel_values` is exactly those samples over 255,
minus the mean and over the std, so the test runs the model alone on the committed crops. The script checks that.

The slow image processor is used (`use_fast=False`): Pillow's bicubic resize of the shorter side to 256 (`size` 224
over `crop_pct` 0.875, `resample` 3 of the pinned `preprocessor_config.json`), a center crop that floors the offset, a
rescale by 1/255 and the ImageNet mean and std.

Manual, not run by the tests or CI: it needs `torch` (2.6.0 was used), `transformers` (4.57.6) and `pillow` (11.3.0). Run
it in a throwaway venv, from `vision/`, with Python's isolated mode:

    python -I test/resnet_golden.py test/fixtures/resnet test/fixtures/resnet/crops MODEL_DIR... > test/fixtures/resnet/golden.json

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


def write_crops(pixels, names, crops_dir, image_mean, image_std):
    crops_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(image_mean).view(1, 3, 1, 1)
    std = torch.tensor(image_std).view(1, 3, 1, 1)
    samples = torch.round((pixels * std + mean) * 255.0)
    rebuilt = (samples / 255.0 - mean) / std
    assert float((rebuilt - pixels).abs().max()) < 1e-5, "pixel_values are not 8-bit samples"
    for name, crop in zip(names, samples.to(torch.uint8)):
        Image.fromarray(crop.permute(1, 2, 0).numpy()).save(crops_dir / name, optimize=True)


def main():
    images_dir, crops_dir, model_dirs = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
    names = sorted(p.name for p in images_dir.glob("*.png"))
    processor = AutoImageProcessor.from_pretrained(model_dirs[0], use_fast=False)
    pixels = processor([Image.open(images_dir / n).convert("RGB") for n in names], return_tensors="pt")["pixel_values"]
    write_crops(pixels, names, crops_dir, processor.image_mean, processor.image_std)

    steps = [round(i * (pixels.shape[2] - 1) / (GRID - 1)) for i in range(GRID)]
    out = {"images": [], "models": {}}
    for i, name in enumerate(names):
        grid = [[[round(float(pixels[i, c, y, x]), 6) for x in steps] for y in steps] for c in range(3)]
        out["images"].append({"name": name, "mean": [round(float(pixels[i, c].mean()), 6) for c in range(3)],
                              "grid": grid})
    for path in model_dirs:
        model = ResNetForImageClassification.from_pretrained(path, dtype=torch.float32).eval()
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
