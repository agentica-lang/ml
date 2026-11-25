"""Writes `test/fixtures/attention_golden.json`: what Hugging Face's `Qwen3_5Attention` (`modeling_qwen3_5`, `float32`,
eager attention under a causal mask) answers for seeded weights and inputs, for `test/gated_attention_test.agc` to
compare with `Nn.attention` on a `gate: true` layer. A small layer: 4 heads of 8, 2 key-value heads, rotary on the first
4 columns of each head (`partial_rotary_factor` 0.5). The weights are the checkpoint's own: `q_proj` is
`[heads * 2 * head_dim, hidden]` (per head the query rows, then the gate rows) and the norms' `weight` is the
zero-centered one, so the layer takes `1 + weight` as its gamma.

Manual, not run by the tests: it needs `torch` and `transformers` (2.15.0 and 5.22.2 were used). Run it in a
throwaway venv, from `nn/`, with Python's isolated mode:

    python -I test/attention_golden.py > test/fixtures/attention_golden.json
"""

import json
import sys

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5TextRotaryEmbedding

TOKENS = 6
HIDDEN = 16
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
THETA = 10000.0
PARTIAL = 0.5


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def main():
    torch.manual_seed(5)
    config = Qwen3_5TextConfig(
        hidden_size=HIDDEN, num_hidden_layers=1, layer_types=["full_attention"], num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS, head_dim=HEAD_DIM, rms_norm_eps=1e-6, attn_implementation="eager",
        rope_parameters={"rope_type": "default", "rope_theta": THETA, "partial_rotary_factor": PARTIAL,
                         "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    )
    layer = Qwen3_5Attention(config, 0).eval()
    rotary = Qwen3_5TextRotaryEmbedding(config)
    with torch.no_grad():
        for p in layer.parameters():
            p.copy_((torch.randn_like(p) * 0.4).double().round(decimals=8).float())
        x = torch.randn(1, TOKENS, HIDDEN).double().round(decimals=8).float()
        positions = torch.arange(TOKENS)[None]
        cos, sin = rotary(x, positions)
        mask = torch.full((TOKENS, TOKENS), float("-inf")).triu(1)[None, None]
        out, _ = layer(x, (cos, sin), mask)
    json.dump({
        "tokens": TOKENS, "hidden": HIDDEN, "heads": HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
        "rotary_dims": int(HEAD_DIM * PARTIAL), "theta": THETA,
        "q_proj": flat(layer.q_proj.weight), "k_proj": flat(layer.k_proj.weight), "v_proj": flat(layer.v_proj.weight),
        "o_proj": flat(layer.o_proj.weight), "q_norm": flat(layer.q_norm.weight), "k_norm": flat(layer.k_norm.weight),
        "x": flat(x[0]), "out": flat(out[0]),
    }, sys.stdout)
    sys.stdout.write("\n")


main()
