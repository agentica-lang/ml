"""Writes `test/fixtures/smolvlm_vision_golden.json`: a tiny random SmolVLM (`Idefics3Model`: a SigLIP tower of 2 blocks,
16 wide, 2 heads, patches of 4 over tiles of 16 by 16, a pixel shuffle by 2, and a connector to a text model 12 wide)
as Hugging Face runs it in `float32`: the `config.json` as an official `HuggingFaceTB/SmolVLM-*` repository writes its
fields (the text model is reduced to its width), the tower's and the connector's weights under the names of the
official checkpoint (`model.vision_model.*`, `model.connector.*`), three tiles (the normalized `pixel_values`) and the
rows that `get_image_features` gives for them. `test/smolvlm_vision_test.agc` writes the weights as a safetensors file,
opens the folder with `SmolVlmVision.open` and compares the features. No model is stored, only this small file.

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions).
Run it in a throwaway venv, from `transformers/`, with Python's isolated mode:

    python -I test/smolvlm_vision_golden.py > test/fixtures/smolvlm_vision_golden.json
"""

import json
import sys

import torch
import transformers
from transformers import Idefics3Config
from transformers.models.idefics3.modeling_idefics3 import Idefics3Model

VISION = dict(
    hidden_size=16, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2, num_channels=3, image_size=16,
    patch_size=4, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6,
)
TEXT = dict(
    model_type="llama", hidden_size=12, intermediate_size=20, num_hidden_layers=1, num_attention_heads=2,
    num_key_value_heads=1, vocab_size=40,
)
SCALE = 2
TILES = 3


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def main():
    check_environment()
    torch.manual_seed(7)
    config = Idefics3Config(text_config=TEXT, vision_config=dict(VISION, attn_implementation="eager"), scale_factor=SCALE)
    model = Idefics3Model(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.startswith(("vision_model.", "connector.")):
                scale = 0.2 if "norm" in name else 0.4
                p.copy_((torch.randn_like(p) * scale).double().round(decimals=8).float())
        shape = (1, TILES, VISION["num_channels"], VISION["image_size"], VISION["image_size"])
        pixels = (torch.randn(shape) * 0.8).double().round(decimals=8).float()
        features = model.get_image_features(pixel_values=pixels).pooler_output[0]
    weights = {}
    shapes = {}
    for name, p in model.state_dict().items():
        if name.startswith(("vision_model.", "connector.")):
            weights["model." + name] = flat(p)
            shapes["model." + name] = list(p.shape)
    document = {
        "architectures": ["Idefics3ForConditionalGeneration"], "model_type": "idefics3", "scale_factor": SCALE,
        "vision_config": dict(VISION, model_type="idefics3"), "text_config": {"hidden_size": TEXT["hidden_size"]},
    }
    json.dump(
        {"config": json.dumps(document), "weights": weights, "shapes": shapes, "pixels": flat(pixels),
         "tiles": TILES, "features": flat(features)},
        sys.stdout,
    )
    sys.stdout.write("\n")


main()
