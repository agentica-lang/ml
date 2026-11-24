"""Prints what Hugging Face's `AutoModelForCausalLM` (`Qwen3ForCausalLM` for `Qwen/Qwen3-0.6B`, `Qwen2ForCausalLM`
for `Qwen/Qwen2.5-0.5B-Instruct`) answers on a few prompts, for `tools/qwen_check.agc` to compare: per prompt, the input
ids (the check does not need our tokenizer), the logits of the last position
over the whole vocabulary, for every position the ids of the five largest
logits, and the ids that greedy `generate` adds (at most `NEW`, stopping at an
end id, which it keeps) with the gap between the two largest logits of each
step, in `float32`.

Manual, not run by the tests: it needs `torch` and `transformers` (4.57.6 was
used) and the checkpoint. Run it in a throwaway venv, from `transformers/`,
with Python's isolated mode:

    python -I test/qwen_golden.py MODEL_DIR > GOLDEN.json

MODEL_DIR holds `model.safetensors`, `config.json` and `tokenizer.json` of
`Qwen/Qwen3-0.6B` or `Qwen/Qwen2.5-0.5B-Instruct`.
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
NEW = 24
# Qwen2.5-Instruct's generation_config.json sets repetition_penalty 1.1, which `generate` applies even when greedy.


def main():
    path = sys.argv[1]
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()
    cases = []
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(ids).logits[0]
        with torch.no_grad():
            made = model.generate(ids, do_sample=False, repetition_penalty=1.0, max_new_tokens=NEW, output_scores=True,
                                  return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        top2 = [step[0].topk(2).values for step in made.scores]
        cases.append({
            "ids": ids[0].tolist(),
            "last": logits[-1].tolist(),
            "top5": logits.topk(5, dim=-1).indices.tolist(),
            "generated": made.sequences[0, ids.shape[1]:].tolist(),
            "gaps": [float(t[0] - t[1]) for t in top2],
        })
    json.dump({"cases": cases}, sys.stdout)
    sys.stdout.write("\n")


main()
