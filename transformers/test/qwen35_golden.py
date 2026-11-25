"""Writes `test/fixtures/qwen35_golden.json`: a tiny random Qwen3.5 text model (3 linear attention layers and 1 full
attention one, 16 wide, tied head) as Hugging Face's `Qwen3_5ForCausalLM` runs it in `float32`: its `config.json` (as an
official `Qwen/Qwen3.5-*` repository writes it), its weights under the names of the official checkpoint
(`model.language_model.*`, the convolution as `[channels, 1, taps]`, the norms as the zero-centered `weight`), two
sequences of token ids and the logits of every position. `test/qwen35_test.agc` writes the weights as a safetensors
file, opens the folder with `Qwen3_5.open` and compares the logits. No model is stored, only this small file.

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions)
and no `flash-linear-attention` or `causal-conv1d` in the venv: their kernels are not the reference. Run it in a
throwaway venv, from `transformers/`, with Python's isolated mode:

    python -I test/qwen35_golden.py > test/fixtures/qwen35_golden.json
"""

import importlib.util
import json
import sys

import torch
import transformers
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

SHAPE = dict(
    hidden_size=16, intermediate_size=16, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
    head_dim=8, vocab_size=16, max_position_embeddings=64, rms_norm_eps=1e-6, hidden_act="silu",
    layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    linear_conv_kernel_dim=4, linear_key_head_dim=4, linear_num_key_heads=2, linear_num_value_heads=4,
    linear_value_head_dim=4, attention_bias=False, attn_output_gate=True, tie_word_embeddings=True,
)
ROPE = {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5, "mrope_section": [11, 11, 10],
        "mrope_interleaved": True}
ROWS = [[3, 1, 4, 1, 5, 9, 2], [6, 5, 3, 5, 8, 9, 7]]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__
    for name in ("fla", "causal_conv1d"):
        assert importlib.util.find_spec(name) is None, f"{name} is installed: its kernel is not the reference"


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def main():
    check_environment()
    torch.manual_seed(3)
    config = Qwen3_5TextConfig(**SHAPE, rope_parameters=ROPE, attn_implementation="eager")
    model = Qwen3_5ForCausalLM(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = 0.2 if "norm" in name else 0.4
            p.copy_((torch.randn_like(p) * scale).double().round(decimals=8).float())
        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.A_log.copy_((torch.randn_like(layer.linear_attn.A_log) * 0.5).double().round(decimals=8).float())
        ids = torch.tensor(ROWS)
        logits = model(ids).logits
    weights = {}
    shapes = {}
    for name, p in model.state_dict().items():
        if name == "lm_head.weight":
            continue
        official = "model.language_model." + name[len("model."):]
        weights[official] = flat(p)
        shapes[official] = list(p.shape)
    text = dict(SHAPE, model_type="qwen3_5_text", rope_parameters=ROPE, mtp_num_hidden_layers=1, dtype="bfloat16")
    document = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": text,
                "tie_word_embeddings": True}
    json.dump({"config": json.dumps(document), "weights": weights, "shapes": shapes, "rows": ROWS,
               "logits": flat(logits)}, sys.stdout)
    sys.stdout.write("\n")


main()
