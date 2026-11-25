"""Prints what Hugging Face's `AutoModelForCausalLM` (`Qwen3_5ForCausalLM`, the text model of `Qwen/Qwen3.5-0.8B`)
answers on a few prompts, for `tools/qwen35_check.agc` and `tools/generate_qwen35_check.agc` to compare: per prompt,
the input ids (the check does not need our tokenizer), the logits of the last position over the whole vocabulary, for
every position the ids of the five largest logits, and the ids that greedy `generate` adds (at most 24, stopping at an
end id, which it keeps) with, for each step, the gap between the two largest logits and the five largest logits and
their ids, in `float32`.

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions),
no `flash-linear-attention` or `causal-conv1d` in the venv (their kernels are not the reference), and the checkpoint. Run
it in a throwaway venv, from `transformers/`, with Python's isolated mode:

    python -I test/qwen35_check_golden.py MODEL_DIR > GOLDEN.json

MODEL_DIR holds the files of the official `Qwen/Qwen3.5-0.8B` repository: the safetensors shard and its index,
`config.json` and `tokenizer.json`.
"""

import importlib.util
import json
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

NEW = 24
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n<|im_start|>assistant\n",
    "Naïve café owners: 東京は日本の首都です。 1234567890",
    "Transformers read a whole sequence at once. " * 6,
    # Over 300 tokens: a prompt padded to 512 rows, and a cache that grows past it while generating.
    "The quick brown fox jumps over the lazy dog, and the dog does not mind. " * 22,
]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__
    for name in ("fla", "causal_conv1d"):
        assert importlib.util.find_spec(name) is None, f"{name} is installed: its kernel is not the reference"


def main():
    check_environment()
    path = sys.argv[1]
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="eager").eval()
    cases = []
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(ids).logits[0]
        with torch.no_grad():
            # Greedy, with the end ids of the chat model (`<|im_end|>`, `<|endoftext|>`), which `generate` keeps.
            made = model.generate(ids, do_sample=False, repetition_penalty=1.0, max_new_tokens=NEW,
                                  eos_token_id=[248046, 248044], pad_token_id=248044, output_scores=True,
                                  return_dict_in_generate=True)
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
