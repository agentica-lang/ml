"""Writes `test/fixtures/llama_golden.json`: a tiny random Llama (`LlamaForCausalLM`, the architecture of SmolLM2: 2
blocks, 24 wide, 6 heads over 2 key heads of 4 columns, `rope_theta` 100000, an untied head, no biases) as Hugging Face
runs it in `float32`: the `config.json`, the weights under the names of an official checkpoint (`model.layers.N...`,
`lm_head.weight`), two rows of 6 token ids and the logits of every position. `test/llama_test.agc` writes the weights as
a safetensors file, opens the folder with `Qwen3.open` and compares the logits. No model is stored, only this small file.

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions).
Run it in a throwaway venv, from `transformers/`, with Python's isolated mode:

    python -I test/llama_golden.py > test/fixtures/llama_golden.json
"""

import json
import sys

import torch
import transformers
from transformers import LlamaConfig, LlamaForCausalLM

SHAPE = dict(
    vocab_size=31, hidden_size=24, intermediate_size=40, num_hidden_layers=2, num_attention_heads=6,
    num_key_value_heads=2, head_dim=4, max_position_embeddings=64, rms_norm_eps=1e-5, rope_theta=100000.0,
    hidden_act="silu", tie_word_embeddings=False, attention_bias=False, mlp_bias=False,
)
ROWS = [[3, 17, 5, 30, 0, 12], [8, 8, 21, 2, 29, 6]]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def main():
    check_environment()
    torch.manual_seed(11)
    config = LlamaConfig(**SHAPE, attn_implementation="eager")
    model = LlamaForCausalLM(config).eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            scale = 0.3 if "norm" in name else 0.4
            p.copy_((torch.randn_like(p) * scale).double().round(decimals=8).float())
        logits = model(torch.tensor(ROWS)).logits
    weights = {}
    shapes = {}
    for name, p in model.state_dict().items():
        if name.endswith("rotary_emb.inv_freq"):
            continue
        weights[name] = flat(p)
        shapes[name] = list(p.shape)
    document = dict(SHAPE, model_type="llama", architectures=["LlamaForCausalLM"], rope_scaling=None)
    json.dump(
        {"config": json.dumps(document), "weights": weights, "shapes": shapes, "rows": ROWS, "logits": flat(logits)},
        sys.stdout,
    )
    sys.stdout.write("\n")


main()
