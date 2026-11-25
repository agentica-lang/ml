"""Writes `test/fixtures/delta_golden.json`: what Hugging Face's `Qwen3_5GatedDeltaNet` (`modeling_qwen3_5`, `float32`,
no cache, so the chunked rule) answers for seeded weights and inputs, for `test/delta_test.agc` to compare with
`Nn.gated_delta_net`. Two small layers: as many value heads as key heads, and twice as many. The weights are the layer's
own names with the convolution as `[channels, taps]`.

Manual, not run by the tests: it needs `torch` and `transformers` (2.15.0 and 5.22.2 were used). Run it in a
throwaway venv, from `nn/`, with Python's isolated mode:

    python -I test/delta_golden.py > test/fixtures/delta_golden.json
"""

import json
import sys

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

TOKENS = 6
HIDDEN = 16
# key heads, value heads, head width
LAYERS = [(2, 2, 8), (2, 4, 8)]


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def main():
    torch.manual_seed(11)
    cases = []
    for key_heads, value_heads, width in LAYERS:
        config = Qwen3_5TextConfig(
            hidden_size=HIDDEN, num_hidden_layers=1, layer_types=["linear_attention"], linear_num_key_heads=key_heads,
            linear_num_value_heads=value_heads, linear_key_head_dim=width, linear_value_head_dim=width,
            linear_conv_kernel_dim=4, hidden_act="silu", rms_norm_eps=1e-6,
        )
        layer = Qwen3_5GatedDeltaNet(config, 0).eval()
        with torch.no_grad():
            for p in layer.parameters():
                p.copy_(torch.randn_like(p) * 0.4)
            layer.norm.weight.copy_(1.0 + 0.3 * torch.randn(width))
            layer.A_log.copy_(torch.randn(value_heads) * 0.5)
            for p in layer.parameters():
                p.copy_(p.double().round(decimals=8).float())
            x = torch.randn(1, TOKENS, HIDDEN)
            x = x.double().round(decimals=8).float()
            out = layer(x)
        cases.append({
            "key_heads": key_heads, "value_heads": value_heads, "width": width, "tokens": TOKENS, "hidden": HIDDEN,
            "qkv": flat(layer.in_proj_qkv.weight), "z": flat(layer.in_proj_z.weight),
            "b": flat(layer.in_proj_b.weight), "a": flat(layer.in_proj_a.weight),
            "conv": flat(layer.conv1d.weight.squeeze(1)), "dt_bias": flat(layer.dt_bias), "a_log": flat(layer.A_log),
            "norm": flat(layer.norm.weight), "output": flat(layer.out_proj.weight),
            "x": flat(x[0]), "out": flat(out[0]),
        })
    json.dump({"cases": cases}, sys.stdout)
    sys.stdout.write("\n")


main()
