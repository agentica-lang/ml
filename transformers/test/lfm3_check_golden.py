"""Prints what the official LFM3 code (`LiquidAI/LFM3-1.7B`) answers on a few prompts, for `tools/lfm3_check.agc` to
compare: per prompt, the input ids (the check does not need our tokenizer), the logits of the last position over the
whole vocabulary and, for every position, the ids of the five largest logits, in `float32`; and what `weights: F16`
rounds away in the file (`half`).

The `float32` model is 6.9 GB, more than a 12 GB machine shared with other work may give. So the checkpoint is loaded in
`bfloat16` (3.4 GB, the way it is stored: every weight is exactly a `float32` number) and each decoder layer, with the
embedding table and the final norm, is turned to `float32` only while it runs; the arithmetic is the one of a `float32`
model. The last prompt is over 300 tokens, past the sliding window of 256.

The code is `modeling_lfm3.py`, `configuration_lfm3.py` and `lfm3_conv.py` of the repository, revision
1da06bec6a819da10b54c67e5c90972b46d81871, under the LFM Open License v1.0, which the script imports from MODEL_DIR and
runs as code (`trust_remote_code`): read them first, run in a throwaway venv (the script sets `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE`), and it refuses files whose sha256 is not the one of that revision. MODEL_DIR holds the files
of the official `LiquidAI/LFM3-1.7B` repository at that revision: `model.safetensors`, `config.json`, `tokenizer.json`
and the three `.py` files. In `bfloat16` the process takes about 4 GB: on a shared machine run it as
`flock LOCK systemd-run --user --scope -q -p MemoryMax=5G python ...`.

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions)
and no `triton` or `causal-conv1d`. Run it from `transformers/`, with Python's isolated mode:

    python -I test/lfm3_check_golden.py MODEL_DIR > GOLDEN.json
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import sys

import torch
import transformers
from transformers import PreTrainedTokenizerFast

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n<|im_start|>assistant\n",
    "Naïve café owners: 東京は日本の首都です。 1234567890",
    # Over 300 tokens: the sliding layers drop the first tokens, and the convolution carries 6 rows over every chunk.
    "The quick brown fox jumps over the lazy dog, and the dog does not mind. " * 22,
]
SHA256 = {
    "configuration_lfm3.py": "176cced35657aedfea7d7002221360582d1c05fcaca882c21508bc3a73b282e5",
    "lfm3_conv.py": "80d4a985382a22774212759248a6b4c737bfbf9c2a47b3dd7f797dda748e4bb4",
    "modeling_lfm3.py": "e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab",
}


def check_environment(folder):
    if not torch.__version__.startswith("2.15.0"):
        raise SystemExit(f"torch 2.15.0 is needed, got {torch.__version__}")
    if transformers.__version__ != "5.22.2":
        raise SystemExit(f"transformers 5.22.2 is needed, got {transformers.__version__}")
    for name in ("triton", "causal_conv1d"):
        if importlib.util.find_spec(name) is not None:
            raise SystemExit(f"{name} is installed: its kernel is not the reference")
    for name, digest in SHA256.items():
        with open(f"{folder}/{name}", "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != digest:
                raise SystemExit(f"{name} is not the file of the pinned revision")


def load_remote(folder):
    spec = importlib.machinery.ModuleSpec("lfm3_code", None, is_package=True)
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [folder]
    sys.modules["lfm3_code"] = package
    return importlib.import_module("lfm3_code.modeling_lfm3")


def f16_loss(model):
    """What `weights: F16` rounds away: of the matrices it keeps in `F16` (rank 2, 100,000 values or more), how many
    are there, how many non-zero ones are under 6.1e-5 (an `F16` subnormal: it keeps fewer digits) and how many are
    under 2.98e-8, half the smallest subnormal (they become 0)."""
    total = small = zeroed = 0
    for p in model.parameters():
        if p.ndim == 2 and p.numel() >= 100000:
            a = p.detach().float().abs()
            nonzero = a > 0
            total += p.numel()
            small += int((nonzero & (a < 6.103515625e-05)).sum())
            zeroed += int((nonzero & (a < 2.98e-8)).sum())
    return {"total": total, "small": small, "zeroed": zeroed}


def main():
    path = sys.argv[1]
    check_environment(path)
    modeling = load_remote(path)
    tok = PreTrainedTokenizerFast(tokenizer_file=f"{path}/tokenizer.json")
    model = modeling.Lfm3ForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="eager").eval()
    inner = model.model
    inner.embed_tokens.float()
    model.lm_head.weight = inner.embed_tokens.weight
    inner.embedding_norm.float()
    for layer in inner.layers:
        layer.register_forward_pre_hook(lambda module, args, kwargs: (module.float(), None)[1], with_kwargs=True)
        layer.register_forward_hook(
            lambda module, args, kwargs, out: (module.to(torch.bfloat16), None)[1], with_kwargs=True)
    cases = []
    half = f16_loss(model)
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(ids).logits[0]
        if logits.dtype != torch.float32:
            raise SystemExit(f"the logits are {logits.dtype}, not float32")
        cases.append({
            "ids": ids[0].tolist(),
            "last": logits[-1].tolist(),
            "top5": logits.topk(5, dim=-1).indices.tolist(),
        })
    json.dump({"cases": cases, "half": half}, sys.stdout)
    sys.stdout.write("\n")


main()
