"""Writes `test/fixtures/qwen35_image_golden.json`: a tiny random Qwen3.5 with a vision tower (the text model of 4 layers
and 16 wide, a head of 16 columns of which 12 turn, `mrope_section` [2, 2, 1]; the tower of `qwen35_vision_golden.py`)
as Hugging Face's `Qwen3_5ForConditionalGeneration` runs it in `float32` on a prompt with two images (grids of 4 by 6
and 6 by 2 patches, that is 6 and 3 tokens, the second one taller than wide): the `config.json` of its text model (as an
official `Qwen/Qwen3.5-*` repository writes it), the text weights under the names of the official checkpoint, the ids,
the merged features `pooler_output` of the images, the 3-axis positions and `rope_deltas` of `get_rope_index`, the
logits of every position of the prompt, and the 6 ids and logits of a greedy `generate` from it. The tower is not
stored: its features are, and `test/generate_qwen35_image_test.agc` splices them in. No model is stored, only this
small file.

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions) and
no `flash-linear-attention` or `causal-conv1d` in the venv: their kernels are not the reference. Run it in a throwaway
venv, from `transformers/`, with Python's isolated mode:

    python -I test/qwen35_image_golden.py > test/fixtures/qwen35_image_golden.json
"""

import importlib.util
import json
import sys

import torch
import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

TEXT = dict(
    hidden_size=16, intermediate_size=16, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
    head_dim=16, vocab_size=32, max_position_embeddings=128, rms_norm_eps=1e-6, hidden_act="silu",
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_conv_kernel_dim=4, linear_key_head_dim=4, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_value_head_dim=4, attention_bias=False, attn_output_gate=True, tie_word_embeddings=True,
)
ROPE = {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.75, "mrope_section": [2, 2, 1],
        "mrope_interleaved": True}
VISION = dict(
    depth=2, hidden_size=16, hidden_act="gelu_pytorch_tanh", intermediate_size=24, num_heads=2, in_channels=3,
    patch_size=4, spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=16, num_position_embeddings=16,
)
VISION_ROPE = {"rope_type": "axial", "rope_theta": 5000.0}
IMAGE, START, END = 28, 29, 30
GRIDS = [[4, 6], [6, 2]]
IDS = [5, 6, START] + [IMAGE] * 6 + [END, 7, START] + [IMAGE] * 3 + [END, 8, 9]
NEW = 6


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__
    for name in ("fla", "causal_conv1d"):
        assert importlib.util.find_spec(name) is None, f"{name} is installed: its kernel is not the reference"


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def main():
    check_environment()
    torch.manual_seed(11)
    config = Qwen3_5Config(
        text_config=dict(TEXT, rope_parameters=ROPE), vision_config=dict(VISION, rope_parameters=VISION_ROPE),
        image_token_id=IMAGE, vision_start_token_id=START, vision_end_token_id=END, tie_word_embeddings=True,
        attn_implementation="eager",
    )
    model = Qwen3_5ForConditionalGeneration(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = 0.2 if "norm" in name else 0.4
            p.copy_((torch.randn_like(p) * scale).double().round(decimals=8).float())
        for layer in model.model.language_model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.A_log.copy_((torch.randn_like(layer.linear_attn.A_log) * 0.5).double().round(decimals=8).float())
        width = VISION["in_channels"] * VISION["temporal_patch_size"] * VISION["patch_size"] ** 2
        pixels = torch.cat([(torch.randn(h * w, width) * 0.5).double().round(decimals=8).float() for h, w in GRIDS])
        grid = torch.tensor([[1, h, w] for h, w in GRIDS])
        ids = torch.tensor([IDS])
        types = (ids == IMAGE).int()
        features = torch.cat(model.model.get_image_features(pixels, grid).pooler_output)
        logits = model(input_ids=ids, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=types).logits
        positions, deltas = model.model.get_rope_index(ids, types, grid)
        model.generation_config.eos_token_id = None
        made = model.generate(input_ids=ids, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=types,
                              do_sample=False, max_new_tokens=NEW, output_scores=True, return_dict_in_generate=True,
                              pad_token_id=0)
    weights = {}
    shapes = {}
    for name, p in model.state_dict().items():
        if name.startswith("model.language_model."):
            weights[name] = flat(p)
            shapes[name] = list(p.shape)
    text = dict(TEXT, model_type="qwen3_5_text", rope_parameters=ROPE, mtp_num_hidden_layers=1, dtype="bfloat16")
    document = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": text,
                "image_token_id": IMAGE, "vision_start_token_id": START, "vision_end_token_id": END,
                "tie_word_embeddings": True}
    json.dump({"config": json.dumps(document), "weights": weights, "shapes": shapes, "ids": IDS, "grids": GRIDS,
               "features": flat(features), "time": positions[0, 0].tolist(), "down": positions[1, 0].tolist(),
               "across": positions[2, 0].tolist(), "delta": int(deltas[0, 0]), "logits": flat(logits),
               "generated": made.sequences[0, len(IDS):].tolist(), "scores": [flat(step[0]) for step in made.scores]},
              sys.stdout)
    sys.stdout.write("\n")


main()
