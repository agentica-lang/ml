"""Writes `test/fixtures/sample_golden.json`: for a few logit vectors and sampling settings, the probability of every
id after Hugging Face's `TemperatureLogitsWarper`, `TopKLogitsWarper` and `TopPLogitsWarper` (in that order, which is
the order `generate` uses) and a softmax, in `float64`. `test/sample_test.agc` compares `Sample.probabilities` to it.

Manual, not run by the tests: it needs `torch` and `transformers` (4.57.6 was used). Run it in a throwaway venv, from
`transformers/`, with Python's isolated mode:

    python -I test/sample_golden.py > test/fixtures/sample_golden.json
"""

import json
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

print(json.dumps({"cases": cases}))
