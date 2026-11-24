"""Writes `test/fixtures/sample_golden.json`: for a few logit vectors and sampling settings, the probability of every
id after Hugging Face's `TemperatureLogitsWarper`, `TopKLogitsWarper` and `TopPLogitsWarper` (in that order, which is
the order `generate` uses) and a softmax, in `float64`. `test/sample_test.agc` compares `Sample.probabilities` to it.

It also holds the ids drawn from fixed logits: a stdlib peer of `Sample.pick` (Park-Miller `x = 48271 x mod 2^31-1`
started from the seed with three states skipped, the ids sorted by logit, then cut by `top_k` and `top_p`, then a
cumulative scan against `u * total`), for `test/sample_test.agc` to compare with `Sample.pick` state after state.

Manual, not run by the tests: it needs `torch` and `transformers` (4.57.6 was used). Run it in a throwaway venv, from
`transformers/`, with Python's isolated mode:

    python -I test/sample_golden.py > test/fixtures/sample_golden.json
"""

import json
import math
import random

import torch
from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

VOCAB = 40


def logits_of(seed, spread, ties):
    rng = random.Random(seed)
    values = [rng.gauss(0.0, spread) for _ in range(VOCAB)]
    for i in range(ties):
        values[2 * i + 1] = values[2 * i]
    return values


LOGITS = [logits_of(1, 3.0, 0), logits_of(2, 1.0, 0), logits_of(3, 5.0, 4), logits_of(4, 0.2, 0)]
SETTINGS = [
    (1.0, 0, 1.0),
    (0.7, 0, 1.0),
    (1.0, 5, 1.0),
    (1.0, 0, 0.9),
    (0.8, 10, 0.95),
    (1.5, 3, 0.5),
    (1.0, 1, 1.0),
    (1.0, 0, 0.05),
]

cases = []
for logits in LOGITS:
    for temperature, top_k, top_p in SETTINGS:
        scores = torch.tensor([logits], dtype=torch.float64)
        ids = torch.zeros((1, 1), dtype=torch.long)
        scores = TemperatureLogitsWarper(temperature)(ids, scores)
        if top_k > 0:
            scores = TopKLogitsWarper(top_k)(ids, scores)
        if top_p < 1.0:
            scores = TopPLogitsWarper(top_p)(ids, scores)
        probs = torch.softmax(scores, dim=-1)[0].tolist()
        cases.append(
            {"logits": logits, "temperature": temperature, "top_k": top_k, "top_p": top_p, "probabilities": probs}
        )

MODULUS = 2147483647


def draw_ids(logits, temperature, top_k, top_p, seed, count):
    state = seed % (MODULUS - 1) + 1
    for _ in range(3):
        state = state * 48271 % MODULUS
    out = []
    for _ in range(count):
        state = state * 48271 % MODULUS
        u = state / MODULUS
        order = sorted(range(len(logits)), key=lambda i: (-logits[i], i))
        top = logits[order[0]]
        kept = [(i, logits[i] / temperature) for i in order if logits[i] >= top - 30.0 * temperature]
        if top_k > 0 and len(kept) > top_k:
            edge = kept[top_k - 1][1]
            kept = [c for c in kept if c[1] >= edge]
        if top_p < 1.0:
            weights = [math.exp(score - kept[0][1]) for _, score in kept]
            total = sum(weights)
            before, cut = 0.0, []
            for c, w in zip(kept, weights):
                if before / total >= top_p:
                    break
                cut.append(c)
                before += w
            kept = cut
        weights = [math.exp(score - kept[0][1]) for _, score in kept]
        target = u * sum(weights)
        seen, chosen = 0.0, kept[-1][0]
        for (i, _), w in zip(kept, weights):
            seen += w
            if seen > target:
                chosen = i
                break
        out.append(chosen)
    return out


draws = []
for logits in (LOGITS[0], LOGITS[2]):
    for temperature, top_k, top_p in ((1.0, 0, 1.0), (0.8, 10, 0.95), (1.5, 3, 0.5)):
        for seed in (0, 7, 123456):
            draws.append(
                {
                    "logits": logits,
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "seed": seed,
                    "ids": draw_ids(logits, temperature, top_k, top_p, seed, 16),
                }
            )

print(json.dumps({"cases": cases, "draws": draws}))
