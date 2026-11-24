"""Prints what Hugging Face's `BertModel` answers for `bert-base-uncased` on a
batch of sentences and sentence pairs, for `tools/bert_check.agc` to compare:
the input ids and token types (the check does not need our tokenizer), the
last hidden state of each sentence (its real tokens only) and the pooler
output, from one padded batch in `float32`, with the mean of each sentence's
token vectors.

Manual, not run by the tests: it needs `torch` and `transformers` (4.57.6 was
used) and the checkpoint. Run it in a throwaway venv, from `transformers/`,
with Python's isolated mode:

    python -I test/bert_golden.py MODEL_DIR > GOLDEN.json

MODEL_DIR holds `model.safetensors`, `config.json` and `tokenizer.json` of
`google-bert/bert-base-uncased`.
"""

import json
import sys

import torch
from transformers import BertModel, BertTokenizerFast

SENTENCES = [
    ("Hello, world!", None),
    ("The quick brown fox jumps over the lazy dog.", None),
    ("Paris is the capital of France.", "It is also its largest city."),
    ("Naïve café owners can't wait for the 2026 Tokyo meet-up, said José.", None),
    (
        "Transformers read a whole sentence at once: every token attends to every other token, "
        "so the model can tell a river bank from a savings bank by what surrounds the word.",
        None,
    ),
    ("A", None),
]


def main():
    path = sys.argv[1]
    tok = BertTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")
    model = BertModel.from_pretrained(path, torch_dtype=torch.float32, attn_implementation="eager").eval()
    cases = []
    for text, pair in SENTENCES:
        enc = tok(text, pair, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        n = enc["input_ids"].shape[1]
        cases.append({
            "ids": enc["input_ids"][0].tolist(),
            "type_ids": enc["token_type_ids"][0].tolist(),
            "hidden": out.last_hidden_state[0].tolist(),
            "pooled": out.pooler_output[0].tolist(),
            "mean": out.last_hidden_state[0].mean(dim=0).tolist(),
        })
        assert len(cases[-1]["hidden"]) == n
    # The padded batch of all of them, as one forward, must give the same:
    # recorded as the padded run so the check covers the mask too.
    width = max(len(c["ids"]) for c in cases)
    ids = torch.tensor([c["ids"] + [0] * (width - len(c["ids"])) for c in cases])
    types = torch.tensor([c["type_ids"] + [0] * (width - len(c["ids"])) for c in cases])
    mask = torch.tensor([[1] * len(c["ids"]) + [0] * (width - len(c["ids"])) for c in cases])
    with torch.no_grad():
        out = model(input_ids=ids, token_type_ids=types, attention_mask=mask)
    for i, c in enumerate(cases):
        n = len(c["ids"])
        c["batch_hidden"] = out.last_hidden_state[i, :n].tolist()
        c["batch_pooled"] = out.pooler_output[i].tolist()
    json.dump({"cases": cases}, sys.stdout)
    sys.stdout.write("\n")


main()
