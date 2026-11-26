"""Prints what the official LFM3 code (`LiquidAI/LFM3-1.7B`) generates, for `tools/generate_lfm3_check.agc` to compare:
per prompt, the input ids, the 24 ids of the greedy `generate` (no stop id, so every prompt gives 24) with the gap of
the two largest logits of each step, and the five largest logits and their ids of each step. The last two prompts are
over
300 and over 600 tokens, past the sliding window of 256, and generate 24 tokens after it (the second decodes in a cache
of 1024 rows and reads 768 of them).

The `float32` model is 6.9 GB, more than a 12 GB machine shared with other work may give: the checkpoint is loaded in
`bfloat16` (3.4 GB, every weight is exactly a `float32` number) and each decoder layer, with the embedding table and
the
final norm, is turned to `float32` only while it runs, as `test/lfm3_check_golden.py` does; the arithmetic is the one
of a
`float32` model. In `bfloat16` the process takes about 4 GB: on a shared machine run it as
`flock LOCK systemd-run --user --scope -q -p MemoryMax=5G python ...`.

The code is `modeling_lfm3.py`, `configuration_lfm3.py` and `lfm3_conv.py` of the repository, revision
1da06bec6a819da10b54c67e5c90972b46d81871, under the LFM Open License v1.0, which the script imports from MODEL_DIR and
runs as code (`trust_remote_code`): read them first, run in a throwaway venv (the script sets `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE`), and it refuses files whose sha256 is not the one of that revision. MODEL_DIR holds the files
of the official `LiquidAI/LFM3-1.7B` repository at that revision: `model.safetensors`, `config.json`, `tokenizer.json`
and the three `.py` files.

Manual, not run by the tests: it needs `torch` 2.15.0 and `transformers` 5.22.2 (the script stops on other versions)
and no `triton` or `causal-conv1d`. Run it from `transformers/`, with Python's isolated mode:

    python -I test/generate_lfm3_check_golden.py MODEL_DIR > GOLDEN.json
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

if not sys.flags.isolated:
    raise SystemExit("run it with python -I")

NEW = 24
# An id no token has: the generation never stops early (the generation config would stop at `<|im_end|>`).
NO_STOP = 2**31 - 1
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "<|im_start|>user\nWhy is the sky blue?<|im_end|>\n<|im_start|>assistant\n",
    "Naïve café owners: 東京は日本の首都です。 1234567890",
    # Over 300 tokens: the sliding layers drop the first tokens, and the convolution carries 6 rows over every chunk.
    "The quick brown fox jumps over the lazy dog, and the dog does not mind. " * 22,
    # Over 600 tokens: the cache has 1024 rows and a decode step reads 768 of them (`Padded.reach`).
    "A hybrid model keeps a short convolution state in most layers and a cache of keys in a few. " * 30,
]
SHA256 = {
    "configuration_lfm3.py": "176cced35657aedfea7d7002221360582d1c05fcaca882c21508bc3a73b282e5",
    "lfm3_conv.py": "80d4a985382a22774212759248a6b4c737bfbf9c2a47b3dd7f797dda748e4bb4",
    "modeling_lfm3.py": "e8cd5254ea99ce6bd731b57b10908c2e8b5010ce8315341f895a38484885eeab",
    "model.safetensors": "135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330",
    "config.json": "8c6ae8b8f6d88c87cfb98cf4b5e34e943d0ee07e24215c4121b3159a46e1399d",
    "tokenizer.json": "7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def check_environment(folder):
    if not torch.__version__.startswith("2.15.0"):
        raise SystemExit(f"torch 2.15.0 is needed, got {torch.__version__}")
    if transformers.__version__ != "5.22.2":
        raise SystemExit(f"transformers 5.22.2 is needed, got {transformers.__version__}")
    for name in ("triton", "causal_conv1d"):
        if importlib.util.find_spec(name) is not None:
            raise SystemExit(f"{name} is installed: its kernel is not the reference")
    for name, digest in SHA256.items():
        if file_sha256(f"{folder}/{name}") != digest:
            raise SystemExit(f"{name} is not the file of the pinned revision")


def check_cache(cache):
    """The official cache must be `float32`, so that the scores are those of a `float32` run."""
    for i, layer in enumerate(cache.layers):
        keys = getattr(layer, "keys", None)
        tensor = keys if keys is not None else cache.get_conv_state(i)
        if tensor.dtype != torch.float32:
            raise SystemExit(f"the cache of layer {i} is {tensor.dtype}, not float32")


def load_remote(folder):
    spec = importlib.machinery.ModuleSpec("lfm3_code", None, is_package=True)
    package = importlib.util.module_from_spec(spec)
    package.__path__ = [folder]
    sys.modules["lfm3_code"] = package
    return importlib.import_module("lfm3_code.modeling_lfm3")


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
    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            made = model.generate(ids, do_sample=False, repetition_penalty=1.0, max_new_tokens=NEW,
                                  eos_token_id=NO_STOP, pad_token_id=0, output_scores=True,
                                  return_dict_in_generate=True)
        check_cache(made.past_key_values)
        if made.scores[0].dtype != torch.float32:
            raise SystemExit(f"the scores are {made.scores[0].dtype}, not float32")
        cases.append({
            "ids": ids[0].tolist(),
            "generated": made.sequences[0, ids.shape[1]:].tolist(),
            "gaps": [float(t[0] - t[1]) for t in (step[0].topk(2).values for step in made.scores)],
            "step_top_ids": [step[0].topk(5).indices.tolist() for step in made.scores],
            "step_top_logits": [step[0].topk(5).values.tolist() for step in made.scores],
        })
    json.dump({"cases": cases}, sys.stdout)
    sys.stdout.write("\n")


main()
