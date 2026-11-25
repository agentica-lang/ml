"""Writes `test/fixtures/granite_moe_golden.json`: a tiny random Granite MoE (2 layers, 16 wide, 6 experts of which 2 are
picked, tied head, and multipliers that are not 1) as Hugging Face's `GraniteMoeForCausalLM` runs it in `float32`: its
`config.json` (as an official `ibm-granite` repository writes it), its weights under the names of the official checkpoint,
two sequences of token ids and the logits of every position. `test/granite_moe_test.agc` writes the weights as a
safetensors file, opens the folder with `GraniteMoe.open` and compares the logits. No model is stored, only this small
file.

Manual, not run by the tests: it needs `torch` 2.6.0 and `transformers` 4.57.6 (the script stops on other versions). Run
it in the venv of the repository (or a throwaway one), from `transformers/`, with Python's isolated mode:

    python -I test/granite_moe_golden.py > test/fixtures/granite_moe_golden.json
"""

import json
import sys

import torch
import transformers
from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeForCausalLM

assert torch.__version__.startswith("2.6.0"), torch.__version__
assert transformers.__version__ == "4.57.6", transformers.__version__

SHAPE = dict(
    hidden_size=16, intermediate_size=8, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    vocab_size=16, max_position_embeddings=64, rms_norm_eps=1e-6, hidden_act="silu", rope_theta=10000.0,
    num_local_experts=6, num_experts_per_tok=2, embedding_multiplier=3.0, attention_multiplier=0.3,
    residual_multiplier=0.5, logits_scaling=2.5, tie_word_embeddings=True, attention_bias=False, rope_scaling=None,
)
ROWS = [[3, 1, 4, 1, 5, 9, 2], [6, 5, 3, 5, 8, 9, 7]]


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def main():
    torch.manual_seed(3)
    config = GraniteMoeConfig(**SHAPE, attn_implementation="eager")
    model = GraniteMoeForCausalLM(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = 0.2 if "norm" in name else 0.5
            offset = 1.0 if "norm" in name else 0.0
            p.copy_((offset + torch.randn_like(p) * scale).double().round(decimals=8).float())
        logits = model(torch.tensor(ROWS)).logits
    weights = {}
    shapes = {}
    for name, p in model.state_dict().items():
        if name == "lm_head.weight":
            continue
        weights[name] = flat(p)
        shapes[name] = list(p.shape)
    document = dict(SHAPE, model_type="granitemoe", architectures=["GraniteMoeForCausalLM"], torch_dtype="bfloat16")
    json.dump({"config": json.dumps(document), "weights": weights, "shapes": shapes, "rows": ROWS,
               "logits": flat(logits)}, sys.stdout)
    sys.stdout.write("\n")


main()
