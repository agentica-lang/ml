"""Prints what Hugging Face answers for `sentence-transformers/msmarco-distilbert-cos-v5` on a few sentences, for
`tools/distilbert_check.agc` to compare: the ids of each text from `tokenizers` (the file truncates at 300 tokens, and
one text is longer), the last hidden state of each sentence alone, and from one padded batch the mean of each
sentence's token vectors and that mean over its length (the checkpoint's `Pooling` and `Normalize` modules), in
`float32`.

Manual, not run by the tests: it needs the checkpoint and `torch` 2.6.0+cpu, `transformers` 4.57.6, `tokenizers`
0.22.1, `huggingface_hub` 0.36.0, `safetensors` 0.6.2, `numpy` 2.2.6 and `regex` 2025.9.18 (the versions used, on
Python 3.13). Run it in a throwaway venv, from `transformers/`, with Python's isolated mode:

    python -I test/distilbert_golden.py MODEL_DIR > GOLDEN.json

MODEL_DIR holds `model.safetensors`, `config.json` and `tokenizer.json` of the checkpoint (revision
c598d92e340e735b82e5b73d7db135209b57b4b0 was used). Only the safetensors file is read.
"""

import json
import sys

import torch
from tokenizers import Tokenizer
from transformers import DistilBertModel

TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "A man is playing a guitar.",
    "Naïve café owners can't wait for the 2026 Tokyo meet-up, said José.",
    "A",
    "Every token attends to every other token, so the model can tell a river bank from a savings bank. " * 20,
]


def main():
    path = sys.argv[1]
    tok = Tokenizer.from_file(f"{path}/tokenizer.json")
    model = DistilBertModel.from_pretrained(
        path, use_safetensors=True, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    pad = model.config.pad_token_id
    cases = []
    for text in TEXTS:
        ids = tok.encode(text).ids
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]))
        cases.append({"text": text, "ids": ids, "hidden": out.last_hidden_state[0].tolist()})
    width = max(len(c["ids"]) for c in cases)
    ids = torch.tensor([c["ids"] + [pad] * (width - len(c["ids"])) for c in cases])
    mask = torch.tensor([[1] * len(c["ids"]) + [0] * (width - len(c["ids"])) for c in cases])
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    for i, c in enumerate(cases):
        mean = out.last_hidden_state[i, : len(c["ids"])].mean(dim=0)
        c["mean"] = mean.tolist()
        c["unit"] = torch.nn.functional.normalize(mean, dim=0).tolist()
    json.dump({"cases": cases}, sys.stdout)
    sys.stdout.write("\n")


main()
