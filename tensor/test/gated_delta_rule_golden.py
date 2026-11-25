"""Writes `test/fixtures/gated_delta_rule_golden.json`: what the recurrent gated delta rule of Hugging Face's
`modeling_qwen3_5` (`torch_recurrent_gated_delta_rule`, in `float32`) answers for a few seeded inputs, for
`test/gated_delta_rule_test.agc` to compare. `q` and `k` are normalized here, as the layer does before the rule, so the
rule gets them as the op takes them; `state` and `final` are in the layout of `Tensor.gated_delta_rule`
(`[heads, value, key]`, the transpose of Hugging Face's `[heads, key, value]`).

Manual, not run by the tests: it needs `torch` and `transformers` (2.15.0 and 5.22.2 were used). Run it in a
throwaway venv, from `tensor/`, with Python's isolated mode:

    python -I test/gated_delta_rule_golden.py > test/fixtures/gated_delta_rule_golden.json
"""

import json
import sys

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

CASES = [
    # tokens, heads, width, with an initial state
    (5, 2, 4, False),
    (3, 3, 8, True),
    (1, 2, 4, True),
]


def flat(t):
    return [round(x, 8) for x in t.reshape(-1).tolist()]


def main():
    torch.manual_seed(7)
    cases = []
    for tokens, heads, width, carried in CASES:
        q = torch.nn.functional.normalize(torch.randn(tokens, heads, width), dim=-1)
        k = torch.nn.functional.normalize(torch.randn(tokens, heads, width), dim=-1)
        v = torch.randn(tokens, heads, width)
        g = -torch.rand(tokens, heads) * 2.0
        beta = torch.rand(tokens, heads)
        state = torch.randn(heads, width, width) * 0.5 if carried else torch.zeros(heads, width, width)
        # Hugging Face: [batch, tokens, heads, width] and a state [batch, heads, key, value]
        out, final = torch_recurrent_gated_delta_rule(
            q[None], k[None], v[None], g[None], beta[None],
            initial_state=state.transpose(1, 2)[None], output_final_state=True,
        )
        cases.append({
            "tokens": tokens, "heads": heads, "width": width,
            "q": flat(q), "k": flat(k), "v": flat(v), "g": flat(g), "beta": flat(beta), "state": flat(state),
            "out": flat(out[0]), "final": flat(final[0].transpose(1, 2)),
        })
    json.dump({"cases": cases}, sys.stdout)
    sys.stdout.write("\n")


main()
