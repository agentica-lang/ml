"""Writes `test/fixtures/qwen35_vision_golden.json`: a tiny random Qwen3.5 vision tower (2 blocks, 16 wide, 2 heads,
patches of 4, a table of 4 by 4 learned positions, `rope_theta` 5000) as Hugging Face's `Qwen3_5VisionModel` runs it in
`float32`: the `vision_config` of a `config.json` (as an official `Qwen/Qwen3.5-*` repository writes it), the weights
under the names of the official checkpoint (`model.visual.*`), two images (a grid of patches and the flattened
`pixel_values` of each, in the order of the image processor) and the merged features `pooler_output` of each.
`test/qwen35_vision_test.agc` writes the weights as a safetensors file, opens the folder with `Qwen3_5Vision.open` and
compares the features. No model is stored, only this small file.

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions).
Run it in a throwaway venv, from `transformers/`, with Python's isolated mode:

    python -I test/qwen35_vision_golden.py > test/fixtures/qwen35_vision_golden.json
"""

import json
import sys

import torch
import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

SHAPE = dict(
    depth=2, hidden_size=16, hidden_act="gelu_pytorch_tanh", intermediate_size=24, num_heads=2, in_channels=3,
    patch_size=4, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=12, num_position_embeddings=16,
)
ROPE = {"rope_type": "axial", "rope_theta": 5000.0}
GRIDS = [[4, 6], [6, 2]]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def main():
    check_environment()
    torch.manual_seed(5)
    config = Qwen3_5VisionConfig(**SHAPE, rope_parameters=ROPE, attn_implementation="eager")
    model = Qwen3_5VisionModel(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = 0.2 if "norm" in name else 0.4
            p.copy_((torch.randn_like(p) * scale).double().round(decimals=8).float())
    width = SHAPE["in_channels"] * SHAPE["temporal_patch_size"] * SHAPE["patch_size"] ** 2
    images = []
    with torch.no_grad():
        for h, w in GRIDS:
            pixels = (torch.randn(h * w, width) * 0.5).double().round(decimals=8).float()
            out = model(pixels, grid_thw=torch.tensor([[1, h, w]]))
            images.append({"grid": [h, w], "pixels": flat(pixels), "features": flat(out.pooler_output)})
    weights = {}
    shapes = {}
    for name, p in model.state_dict().items():
        weights["model.visual." + name] = flat(p)
        shapes["model.visual." + name] = list(p.shape)
    document = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5",
                "vision_config": dict(SHAPE, model_type="qwen3_5", rope_parameters=ROPE), "image_token_id": 5}
    json.dump({"config": json.dumps(document), "weights": weights, "shapes": shapes, "images": images}, sys.stdout)
    sys.stdout.write("\n")


main()
