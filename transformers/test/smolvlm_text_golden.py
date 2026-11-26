"""Writes `test/fixtures/smolvlm_text_golden.json`: a tiny random SmolVLM (`Idefics3ForConditionalGeneration`) as Hugging
Face runs it on token ids alone, in `float32`: its `config.json` (the Llama text model of 2 blocks, 24 wide, 6 heads
over 2 key heads of 4 columns, `rope_theta` 100000, an untied head, in `text_config`; a tower of 1 block beside it),
the weights of the text model and the head under the names of the official checkpoint (`model.text_model.*`,
`lm_head.weight`; the tower's are left out, as a reader of the text model ignores them), two rows of 6 token ids and the
logits of every position. `test/smolvlm_text_test.agc` writes the weights as a safetensors file, opens the folder with
`Qwen3.open_idefics3` and compares the logits. No model is stored, only this small file.

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions).
Run it in a throwaway venv, from `transformers/`, with Python's isolated mode:

    python -I test/smolvlm_text_golden.py > test/fixtures/smolvlm_text_golden.json
"""

import json
import sys

import torch
import transformers
from transformers import Idefics3Config, Idefics3ForConditionalGeneration

TEXT = dict(
    model_type="llama", vocab_size=31, hidden_size=24, intermediate_size=40, num_hidden_layers=2,
    num_attention_heads=6, num_key_value_heads=2, head_dim=4, max_position_embeddings=64, rms_norm_eps=1e-5,
    rope_theta=100000.0, hidden_act="silu", tie_word_embeddings=False, attention_bias=False, mlp_bias=False,
    rope_scaling=None,
)
VISION = dict(
    hidden_size=16, intermediate_size=24, num_hidden_layers=1, num_attention_heads=2, num_channels=3, image_size=16,
    patch_size=4, hidden_act="gelu_pytorch_tanh", layer_norm_eps=1e-6,
)
ROWS = [[3, 17, 5, 30, 0, 12], [8, 8, 21, 2, 29, 6]]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def main():
    check_environment()
    torch.manual_seed(13)
    config = Idefics3Config(
        text_config=dict(TEXT, attn_implementation="eager"), vision_config=dict(VISION, attn_implementation="eager"),
        scale_factor=2, tie_word_embeddings=False)
    model = Idefics3ForConditionalGeneration(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = 0.3 if "norm" in name else 0.4
            p.copy_((torch.randn_like(p) * scale).double().round(decimals=8).float())
        logits = model(input_ids=torch.tensor(ROWS)).logits
    weights = {}
    shapes = {}
    for name, p in model.state_dict().items():
        if name.startswith(("model.text_model.", "lm_head.")) and not name.endswith("inv_freq"):
            weights[name] = flat(p)
            shapes[name] = list(p.shape)
    document = {
        "architectures": ["Idefics3ForConditionalGeneration"], "model_type": "idefics3", "scale_factor": 2,
        "tie_word_embeddings": False, "text_config": TEXT, "vision_config": dict(VISION, model_type="idefics3"),
    }
    json.dump(
        {"config": json.dumps(document), "weights": weights, "shapes": shapes, "rows": ROWS, "logits": flat(logits)},
        sys.stdout,
    )
    sys.stdout.write("\n")


main()
