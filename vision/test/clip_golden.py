"""Prints what Hugging Face answers for the images of `test/fixtures/resnet` (the drawn images of `resnet_images.py`)
and the prompts below with `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, for `test/clip_hf_test.agc` to compare: for each
image, what its image processor makes of it (the mean of each channel and a grid of 8 by 8 pixels of `pixel_values`);
for each prompt, the ids of the slow `CLIPTokenizer`; and, from `CLIPModel` in `float32`, `get_image_features` and
`get_text_features` (not normalized), `logits_per_image` and its softmax over the prompts, and `exp(logit_scale)`. It
also writes what the processor cut from each image (the 224 by 224 crop, 8 bits per channel) as a PNG into CROPS_DIR:
`pixel_values` is exactly those samples over 255, minus the mean and over the std, so the test runs the model alone on
the committed crops. The script checks that.

The slow image processor is used (`use_fast=False`): Pillow's bicubic resize of the shorter side to 224 (`resample` 3
of the pinned `preprocessor_config.json`), a center crop that floors the offset, a rescale by 1/255 and the CLIP mean
and std. The text is padded with `<|endoftext|>` to the longest prompt, and the model reads the first of those (its
end token), so the padding changes nothing.

Manual, not run by the tests or CI: it needs `torch` (2.6.0 was used), `transformers` (4.57.6) and `pillow` (11.3.0).
Run it in a throwaway venv, from `vision/`, with Python's isolated mode:

    python -I test/clip_golden.py test/fixtures/resnet test/fixtures/clip/crops MODEL_DIR > test/fixtures/clip/golden.json

MODEL_DIR holds `model.safetensors`, `config.json`, `preprocessor_config.json`, `vocab.json` and `merges.txt` of the
checkpoint (the revision that `transformers/DESIGN.md` pins).
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, CLIPModel, CLIPTokenizer

GRID = 8
DIGITS = 6
PROMPTS = [
    "a photo of a cat",
    "a photo of a dog",
    "a drawing of colorful circles",
    "colored balloons floating in the sky",
    "an abstract painting with smooth gradients and round shapes",
    "A diagram of a neural network, with 3 layers.",
]


def write_crops(pixels, names, crops_dir, image_mean, image_std):
    crops_dir.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(image_mean).view(1, 3, 1, 1)
    std = torch.tensor(image_std).view(1, 3, 1, 1)
    samples = torch.round((pixels * std + mean) * 255.0)
    rebuilt = (samples / 255.0 - mean) / std
    assert float((rebuilt - pixels).abs().max()) < 1e-5, "pixel_values are not 8-bit samples"
    for name, crop in zip(names, samples.to(torch.uint8)):
        Image.fromarray(crop.permute(1, 2, 0).numpy()).save(crops_dir / name, optimize=True)


def rounded(t):
    return [[round(float(v), DIGITS) for v in row] for row in t]


def main():
    images_dir, crops_dir, model_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    names = sorted(p.name for p in images_dir.glob("*.png"))
    processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=False)
    pixels = processor([Image.open(images_dir / n).convert("RGB") for n in names], return_tensors="pt")["pixel_values"]
    write_crops(pixels, names, crops_dir, processor.image_mean, processor.image_std)

    steps = [round(i * (pixels.shape[2] - 1) / (GRID - 1)) for i in range(GRID)]
    out = {"images": []}
    for i, name in enumerate(names):
        grid = [[[round(float(pixels[i, c, y, x]), 6) for x in steps] for y in steps] for c in range(3)]
        out["images"].append({"name": name, "mean": [round(float(pixels[i, c].mean()), 6) for c in range(3)],
                              "grid": grid})

    tokenizer = CLIPTokenizer.from_pretrained(model_dir)
    ids = [tokenizer(p)["input_ids"] for p in PROMPTS]
    longest = max(len(r) for r in ids)
    padded = torch.tensor([r + [tokenizer.pad_token_id] * (longest - len(r)) for r in ids])
    model = CLIPModel.from_pretrained(model_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        image_embeds = model.get_image_features(pixel_values=pixels)
        text_embeds = model.get_text_features(input_ids=padded)
        scale = model.logit_scale.exp()
        logits = scale * torch.nn.functional.normalize(image_embeds, dim=-1) @ torch.nn.functional.normalize(
            text_embeds, dim=-1).T
        reference = model(input_ids=padded, pixel_values=pixels).logits_per_image
    assert float((logits - reference).abs().max()) < 1e-3, "logits_per_image is not the scaled cosine"
    out["texts"] = [{"text": p, "ids": r} for p, r in zip(PROMPTS, ids)]
    out["image_embeds"] = rounded(image_embeds)
    out["text_embeds"] = rounded(text_embeds)
    out["logits"] = rounded(reference)
    out["probabilities"] = rounded(reference.softmax(dim=-1))
    out["scale"] = round(float(scale), DIGITS)
    json.dump(out, sys.stdout, separators=(",", ":"))


main()
