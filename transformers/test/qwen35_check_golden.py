"""Prints what Hugging Face's `AutoModelForCausalLM` (`Qwen3_5ForCausalLM`, the text model of `Qwen/Qwen3.5-0.8B`)
answers on a few prompts, for `tools/qwen35_check.agc` to compare: per prompt, the input ids (the check does not need our
tokenizer), the logits of the last position over the whole vocabulary, and for every position the ids of the five
largest logits, in `float32`.

Manual, not run by the tests: it needs `torch` and `transformers` (2.15.0 and 5.22.2 were used) and the checkpoint. Run
it in a throwaway venv, from `transformers/`, with Python's isolated mode:

    python -I test/qwen35_check_golden.py MODEL_DIR > GOLDEN.json

MODEL_DIR holds the files of the official `Qwen/Qwen3.5-0.8B` repository: the safetensors shard and its index,
`config.json` and `tokenizer.json`.
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
