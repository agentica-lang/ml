"""Prints what Hugging Face's `GraniteMoeForCausalLM` (`ibm-granite/granite-3.1-1b-a400m-instruct`) answers on a few
prompts, for `tools/granite_moe_check.agc` to compare: per prompt, the input ids (the check does not need our
tokenizer), the logits of the last position over the whole vocabulary and, for every position, the ids of the five
largest logits, in `float32`, and the ids that greedy `generate` adds (at most 24, stopping at the end id `<|end_of_text|>`,
which it keeps) with, for each step, the gap between the two largest logits and the five largest logits and their ids.
`tools/granite_moe_check.agc` compares the first, `tools/generate_granite_moe_check.agc` the second.

The `float32` model is 5.3 GB, more than a 12 GB machine shared with other work may give. So the checkpoint is loaded in
`bfloat16` (2.7 GB, the way it is stored: every weight is exactly a `float32` number) and each decoder layer, with the
embedding table and the final norm, is turned to `float32` only while it runs; the arithmetic is the one of a
`float32` model.

Manual, not run by the tests: it needs `torch` 2.6.0 and `transformers` 4.57.6 (the script stops on other versions)
and the checkpoint. Run it in the venv of the repository (or a throwaway one), from `transformers/`, with Python's
isolated mode:

    python -I test/granite_moe_check_golden.py MODEL_DIR > GOLDEN.json

MODEL_DIR holds the files of the official `ibm-granite/granite-3.1-1b-a400m-instruct` repository: `model.safetensors`,
`config.json` and `tokenizer.json`.
"""

import json
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

assert torch.__version__.startswith("2.6.0"), torch.__version__
assert transformers.__version__ == "4.57.6", transformers.__version__

NEW = 24
END = 0
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "<|start_of_role|>user<|end_of_role|>Why is the sky blue?<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
    "Naïve café owners: 東京は日本の首都です。 1234567890",
    # Over 100 tokens: only the last position is compared.
    "The quick brown fox jumps over the lazy dog, and the dog does not mind. " * 12,
]


def main():
    path = sys.argv[1]
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").eval()
    inner = model.model
    inner.embed_tokens.float()
    model.lm_head.weight = inner.embed_tokens.weight
    inner.norm.float()
    for layer in inner.layers:
        layer.register_forward_pre_hook(lambda module, args, kwargs: (module.float(), None)[1], with_kwargs=True)
        layer.register_forward_hook(lambda module, args, kwargs, out: (module.to(torch.bfloat16), None)[1], with_kwargs=True)
    cases = []
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(ids).logits[0]
        assert logits.dtype == torch.float32, logits.dtype
        with torch.no_grad():
            made = model.generate(ids, do_sample=False, repetition_penalty=1.0, max_new_tokens=NEW, eos_token_id=END,
                                  pad_token_id=END, output_scores=True, return_dict_in_generate=True)
        cases.append({
            "ids": ids[0].tolist(),
            "last": logits[-1].tolist(),
            "top5": logits.topk(5, dim=-1).indices.tolist(),
            "generated": made.sequences[0, ids.shape[1]:].tolist(),
            "gaps": [float(t[0] - t[1]) for t in (step[0].topk(2).values for step in made.scores)],
            "step_top_ids": [step[0].topk(5).indices.tolist() for step in made.scores],
            "step_top_logits": [step[0].topk(5).values.tolist() for step in made.scores],
        })
    json.dump({"cases": cases}, sys.stdout)
    sys.stdout.write("\n")


main()
