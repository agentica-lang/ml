"""Writes `test/fixtures/moe_golden.json`: what Hugging Face's `GraniteMoeMoE` (`modeling_granitemoe`) and
`Qwen3MoeSparseMoeBlock` (`modeling_qwen3_moe`, with `norm_topk_prob` off, the gates that do not add up to 1) answer in
`float32` for seeded weights and inputs, for `test/moe_test.agc` to compare with `Moe.moe`. The weights are in the
layout of `Moe.Moe`: the router `[experts, width]`, `gate_up` `[experts, 2 * inner, width]` (Granite's `input_linear`,
or the gate and up projections of Qwen3 stacked) and `down` `[experts, width, inner]`.

Seven cases: Granite with 2 of 6 experts, 1 of 6 and 6 of 6, Granite with a tie (an expert that always wins, and two
experts with the same router row and the same weights that tie for the second place, so the answer does not depend
on which one is picked), Qwen3, and Granite with 2 of 6 for one token and for 40 tokens (many more than experts).

Manual, not run by the tests: it needs `torch` 2.6.0 and `transformers` 4.57.6 (the script stops on other versions).
Run it in the venv of the repository (or a throwaway one), from `nn/`, with Python's isolated mode:

    python -I test/moe_golden.py > test/fixtures/moe_golden.json
"""

import json
import sys

import torch
import transformers
from transformers.models.granitemoe.configuration_granitemoe import GraniteMoeConfig
from transformers.models.granitemoe.modeling_granitemoe import GraniteMoeMoE
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

assert torch.__version__.startswith("2.6.0"), torch.__version__
assert transformers.__version__ == "4.57.6", transformers.__version__

WIDTH = 6
INNER = 4
EXPERTS = 6
TOKENS = 5


def flat(t):
    return [round(x, 8) for x in t.detach().reshape(-1).tolist()]


def rounded(t):
    return (t.double().round(decimals=8)).float()


def seeded(model):
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(rounded(torch.randn_like(p) * 0.5))


def case(name, top_k, normalize, router, gate_up, down, x, out):
    tokens = x.shape[0]
    return {
        "name": name, "experts": EXPERTS, "width": WIDTH, "inner": INNER, "tokens": tokens, "top_k": top_k,
        "normalize": normalize, "router": flat(router), "gate_up": flat(gate_up), "down": flat(down), "x": flat(x),
        "out": flat(out),
    }


def granite(name, top_k, tie=False, tokens=TOKENS):
    config = GraniteMoeConfig(
        hidden_size=WIDTH, intermediate_size=INNER, num_local_experts=EXPERTS, num_experts_per_tok=top_k,
        hidden_act="silu",
    )
    model = GraniteMoeMoE(config).eval()
    seeded(model)
    x = rounded(torch.randn(1, tokens, WIDTH))
    with torch.no_grad():
        if tie:
            # expert 0 wins for every token (the first input is 1 and its router weight is large), experts 1 and 2 are
            # the same expert and tie for the second place
            x[..., 0] = rounded(torch.rand(1, tokens) * 0.5 + 1.0)
            router = model.router.layer.weight
            router[0] = 0.0
            router[0, 0] = 20.0
            router[2] = router[1]
            model.input_linear.weight[2] = model.input_linear.weight[1]
            model.output_linear.weight[2] = model.output_linear.weight[1]
        out, _ = model(x)
    return case(
        name, top_k, True, model.router.layer.weight, model.input_linear.weight, model.output_linear.weight, x[0],
        out[0],
    )


def qwen3(name, top_k):
    config = Qwen3MoeConfig(
        hidden_size=WIDTH, moe_intermediate_size=INNER, num_experts=EXPERTS, num_experts_per_tok=top_k,
        norm_topk_prob=False, hidden_act="silu", intermediate_size=8,
    )
    model = Qwen3MoeSparseMoeBlock(config).eval()
    seeded(model)
    x = rounded(torch.randn(1, TOKENS, WIDTH))
    with torch.no_grad():
        out = model(x)
        out = out[0] if isinstance(out, tuple) else out
        gate_up = torch.stack([torch.cat([e.gate_proj.weight, e.up_proj.weight], dim=0) for e in model.experts])
        down = torch.stack([e.down_proj.weight for e in model.experts])
    return case(name, top_k, False, model.gate.weight, gate_up, down, x[0], out[0])


def main():
    torch.manual_seed(13)
    cases = [
        granite("granite, 2 of 6", 2),
        granite("granite, 1 of 6", 1),
        granite("granite, 6 of 6", 6),
        granite("granite, a tie for the second place", 2, tie=True),
        qwen3("qwen3, gates that do not add up to 1", 2),
        granite("granite, one token", 2, tokens=1),
        granite("granite, 40 tokens", 2, tokens=40),
    ]
    json.dump({"cases": cases}, sys.stdout)
    sys.stdout.write("\n")


main()
