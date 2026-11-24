"""Prints what Hugging Face's `Qwen3ForCausalLM` answers for `Qwen/Qwen3-0.6B`
on a few prompts, for `tools/qwen_check.agc` to compare: per prompt, the input
ids (the check does not need our tokenizer), the logits of the last position
over the whole vocabulary, and for every position the ids of the five largest
logits, in `float32`.

Manual, not run by the tests: it needs `torch` and `transformers` (4.57.6 was
used) and the checkpoint. Run it in a throwaway venv, from `transformers/`,
with Python's isolated mode:

    python -I test/qwen_golden.py MODEL_DIR > GOLDEN.json

MODEL_DIR holds `model.safetensors`, `config.json` and `tokenizer.json` of
`Qwen/Qwen3-0.6B`.
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n<|im_start|>assistant\n",
    "Naïve café owners: 東京は日本の首都です。 1234567890",
    "Transformers read a whole sequence at once. " * 6,
]


def main():
    path = sys.argv[1]
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()
    cases = []
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(ids).logits[0]
        cases.append({
            "ids": ids[0].tolist(),
            "last": logits[-1].tolist(),
            "top5": logits.topk(5, dim=-1).indices.tolist(),
        })
    json.dump({"cases": cases}, sys.stdout)
    sys.stdout.write("\n")


main()
