"""Prints what Hugging Face's `Idefics3ForConditionalGeneration` (`HuggingFaceTB/SmolVLM-256M-Instruct`, `float32`) gives
for a prompt with one image, for `test/smolvlm_image_hf_test.agc` to compare: the ids of the prompt (a few text ids, the
`<fake_token_around_image>`, the 64 `<image>` tokens of one 512 by 512 tile, the fake token again and more text ids; no
tokenizer is in the golden) and, for each tile of `test/fixtures/smolvlm/tiles` (the ones of `smolvlm_hf_golden.py`,
whose sha256 are checked), the logits of the last position: their mean, the mean of their absolute values, one value
of every `stride`th place and the ten largest with their ids. The tile goes in as `smolvlm_hf_golden.py` makes its
pixels (rescaled by 1/255, mean and std 0.5), so no resize or split of the processor is in the golden.

Manual, not run by the tests or CI: it needs `torch` 2.15.0 and `transformers` 5.22.2 (in a venv, the script stops on
other versions), `pillow` 11.3.0 (in a folder of its own that this script puts first on `sys.path`) and the files of the
official `HuggingFaceTB/SmolVLM-256M-Instruct` repository (MODEL_DIR holds `model.safetensors` and `config.json`; the
script checks their sha256). Run it from `vision/`, with Python's isolated mode and the offline flags:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I test/smolvlm_image_hf_golden.py MODEL_DIR PILLOW_DIR \
        > test/fixtures/smolvlm/image_golden.json
"""

import hashlib
import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit("usage: python -I test/smolvlm_image_hf_golden.py MODEL_DIR PILLOW_DIR > test/fixtures/smolvlm/image_golden.json")
sys.path.insert(0, sys.argv[2])

import numpy as np
import PIL
import torch
import transformers
from PIL import Image
from transformers import Idefics3ForConditionalGeneration

SAMPLES = 500
FAKE = 49189
IMAGE = 49190
SHA256 = {
    "config.json": "b70fb4bfde88df9eeebc9d8ff523733b8bf70d6c9b06c610325960e06ae9db52",
    "model.safetensors": "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e",
}
IDS = [1, 9690, 198, 2683] + [FAKE] + [IMAGE] * 64 + [FAKE] + [3365, 198, 4119, 42, 7019, 282, 3365, 30]
TILES = ["wide", "tall"]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__
    assert PIL.__version__ == "11.3.0", PIL.__version__
    assert Path(PIL.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve()), PIL.__file__
    for name, digest in SHA256.items():
        with open(f"{sys.argv[1]}/{name}", "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, f"{name} is not the file of the golden"


def main():
    check_environment()
    tiles = Path("test/fixtures/smolvlm/tiles")
    model = Idefics3ForConditionalGeneration.from_pretrained(
        sys.argv[1], dtype=torch.float32, attn_implementation="eager").eval()
    cases = []
    for name in TILES:
        path = tiles / f"{name}.png"
        samples = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        pixels = torch.from_numpy(((samples / 255.0 - 0.5) / 0.5).transpose(2, 0, 1).copy())[None, None]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([IDS]), pixel_values=pixels).logits[0, -1]
        stride = max(1, logits.numel() // SAMPLES) | 1
        top = logits.topk(10)
        cases.append({
            "name": name, "tile": f"smolvlm/tiles/{name}.png", "tile_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "stride": stride, "mean": round(float(logits.mean()), 6), "abs_mean": round(float(logits.abs().mean()), 6),
            "samples": [round(float(v), 6) for v in logits[::stride]], "top_ids": top.indices.tolist(),
            "top_values": [round(float(v), 6) for v in top.values],
        })
    json.dump({"ids": IDS, "image_token": IMAGE, "cases": cases}, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


main()
