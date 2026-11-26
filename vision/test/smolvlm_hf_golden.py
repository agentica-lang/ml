"""Prints what Hugging Face's `Idefics3ForConditionalGeneration` (`HuggingFaceTB/SmolVLM-256M-Instruct`, `float32`) gives
for two tiles, for `test/smolvlm_hf_test.agc` to compare: for each tile (a drawn image of `test/fixtures/resnet`
stretched to 512 by 512 with Pillow's bicubic and written as an 8-bit PNG into `test/fixtures/smolvlm/tiles`, whose
sha256 the golden holds) the rows of `get_image_features`: their mean, the mean of their absolute values and one value of
every `stride`th place of them (an odd number that gives about `SAMPLES` of them). The tower reads the pixels as the
checkpoint's image processor makes them for a tile of that size: rescaled by 1/255 and normalized with mean and std 0.5
(the script checks `preprocessor_config.json`), so no resize or split of the processor is in the golden.

Manual, not run by the tests or CI: it needs `torch` 2.15.0 and `transformers` 5.22.2 (in a venv, the script stops on
other versions), `pillow` 11.3.0 (the wheel and the install of `test/qwen_pixels_golden.py`, in a folder of its own that
this script puts first on `sys.path`), and the files of the official `HuggingFaceTB/SmolVLM-256M-Instruct` repository
(MODEL_DIR holds `model.safetensors`, `config.json` and `preprocessor_config.json`); the script checks their sha256. Run
it from `vision/`, with Python's isolated mode and the offline flags:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I test/smolvlm_hf_golden.py MODEL_DIR PILLOW_DIR \
        > test/fixtures/smolvlm/golden.json
"""

import hashlib
import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit("usage: python -I test/smolvlm_hf_golden.py MODEL_DIR PILLOW_DIR > test/fixtures/smolvlm/golden.json")
PILLOW_DIR = sys.argv[2]
sys.path.insert(0, PILLOW_DIR)

import numpy as np
import PIL
import torch
import transformers
from PIL import Image
from transformers import Idefics3ForConditionalGeneration

SIZE = 512
SAMPLES = 500
SHA256 = {
    "config.json": "b70fb4bfde88df9eeebc9d8ff523733b8bf70d6c9b06c610325960e06ae9db52",
    "preprocessor_config.json": "6cb6e36d6fcb88ca1502c4a26750715dc3e7dedddc9a8f17b27d8d167d1457e7",
    "model.safetensors": "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e",
}
# The id, the drawn image (in `test/fixtures`) that is stretched to a tile.
CASES = [("wide", "resnet/wide-1.png"), ("tall", "resnet/tall-1.png")]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__
    assert PIL.__version__ == "11.3.0", PIL.__version__
    assert Path(PIL.__file__).resolve().is_relative_to(Path(PILLOW_DIR).resolve()), PIL.__file__
    for name, digest in SHA256.items():
        with open(f"{sys.argv[1]}/{name}", "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, f"{name} is not the file of the golden"
    config = json.load(open(f"{sys.argv[1]}/preprocessor_config.json"))
    assert config["image_mean"] == [0.5] * 3 and config["image_std"] == [0.5] * 3, config
    assert config["do_rescale"] and config["rescale_factor"] == 1 / 255, config


def main():
    check_environment()
    fixtures = Path("test/fixtures")
    tiles = fixtures / "smolvlm" / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        sys.argv[1], dtype=torch.float32, attn_implementation="eager").eval()
    cases = []
    for name, image in CASES:
        path = tiles / f"{name}.png"
        Image.open(fixtures / image).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC).save(path, optimize=True)
        samples = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        pixels = torch.from_numpy(((samples / 255.0 - 0.5) / 0.5).transpose(2, 0, 1).copy())[None, None]
        with torch.no_grad():
            rows = model.get_image_features(pixel_values=pixels).pooler_output[0]
        flat = rows.reshape(-1)
        stride = max(1, flat.numel() // SAMPLES) | 1
        cases.append({
            "name": name, "tile": f"smolvlm/tiles/{name}.png", "tile_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "shape": list(rows.shape), "stride": stride, "mean": round(float(rows.mean()), 6),
            "abs_mean": round(float(rows.abs().mean()), 6), "samples": [round(float(v), 6) for v in flat[::stride]],
        })
    json.dump({"cases": cases}, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


main()
