"""Writes `test/fixtures/expert_matmul_golden.json`: what PyTorch's `torch.matmul` answers for a few seeded cases of
`Tensor.expert_matmul`, one matrix product per `(token, use)` with the expert that `ids` names, in `float32`
(`out[t, u] = weights[ids[t, u]] @ x[t, u]`, or `x[t, 0]` when `x` has one use). For `test/expert_matmul_test.agc` to
compare.

Manual, not run by the tests: it needs `torch` 2.6.0 (the script stops on another version). Run it in the venv of the
repository (or a throwaway one), from `tensor/`, with Python's isolated mode:

    python -I test/expert_matmul_golden.py > test/fixtures/expert_matmul_golden.json
"""

import json
import sys

import torch

assert torch.__version__.startswith("2.6.0"), torch.__version__

CASES = [
    # experts, rows, inner, tokens, uses, one x for all the uses of a token, the ids
    (5, 3, 4, 3, 2, False, [[0, 4], [2, 3], [1, 0]]),
    (6, 4, 3, 2, 3, True, [[5, 1, 2], [2, 2, 0]]),
    (4, 2, 5, 2, 1, True, [[3], [0]]),
    (3, 3, 3, 2, 3, False, [[0, 1, 2], [2, 1, 0]]),
]


def flat(t):
    return [round(x, 8) for x in t.reshape(-1).tolist()]


def main():
    torch.manual_seed(5)
    cases = []
    for experts, rows, inner, tokens, uses, shared, ids in CASES:
        weights = torch.randn(experts, rows, inner)
        x = torch.randn(tokens, 1 if shared else uses, inner)
        picks = torch.tensor(ids)
        out = torch.empty(tokens, uses, rows)
        for t in range(tokens):
            for u in range(uses):
                out[t, u] = weights[picks[t, u]] @ x[t, 0 if shared else u]
        cases.append({
            "experts": experts, "rows": rows, "inner": inner, "tokens": tokens, "uses": uses,
            "shared": shared, "weights": flat(weights), "x": flat(x),
            "ids": [i for row in ids for i in row], "out": flat(out),
        })
    json.dump({"cases": cases}, sys.stdout)
    sys.stdout.write("\n")


main()
