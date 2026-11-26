"""Prints what Hugging Face's `Qwen3_5VisionModel` (the vision tower of `Qwen/Qwen3.5-0.8B`) gives for the patches that
`tools/qwen35_vision_check.agc` makes by a formula: the merged features of a grid of 6 by 8 patches, in `float32`. Only
the tensors `model.visual.*` of the checkpoint are read.

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions)
and the checkpoint. Run it in a throwaway venv, from `transformers/`, with Python's isolated mode:

    python -I test/qwen35_vision_check_golden.py MODEL_DIR > GOLDEN.json

MODEL_DIR holds the files of the official `Qwen/Qwen3.5-0.8B` repository: the safetensors shard and its index, and
`config.json`.
"""

import json
import sys

import torch
import transformers
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

GRID = [6, 8]
PREFIX = "model.visual."


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__


def pixel(i):
    """The same integer formula as the check: a pseudo-random value in [-1, 1]."""
    return ((i * 7919 + 13) % 255) / 127.5 - 1.0


def main():
    check_environment()
    path = sys.argv[1]
    config = json.load(open(f"{path}/config.json"))["vision_config"]
    config.pop("model_type", None)
    model = Qwen3_5VisionModel(Qwen3_5VisionConfig(**config)).eval()
    index = json.load(open(f"{path}/model.safetensors.index.json"))["weight_map"]
    weights = {}
    for file in sorted({f for name, f in index.items() if name.startswith(PREFIX)}):
        with safe_open(f"{path}/{file}", framework="pt") as f:
            for name in f.keys():
                if name.startswith(PREFIX):
                    weights[name[len(PREFIX):]] = f.get_tensor(name).float()
    model.load_state_dict(weights)
    width = config["in_channels"] * config["temporal_patch_size"] * config["patch_size"] ** 2
    n = GRID[0] * GRID[1]
    pixels = torch.tensor([pixel(i) for i in range(n * width)], dtype=torch.float64).reshape(n, width).float()
    with torch.no_grad():
        out = model(pixels, grid_thw=torch.tensor([[1] + GRID]))
    json.dump({"grid": GRID, "features": out.pooler_output.reshape(-1).tolist()}, sys.stdout)
    sys.stdout.write("\n")


main()
