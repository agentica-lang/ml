"""Prints what Hugging Face's `Idefics3ForConditionalGeneration` (`HuggingFaceTB/SmolVLM-256M-Instruct`, `float32`) gives
for the token ids of a prompt without an image, for `test/smolvlm_text_hf_test.agc` to compare: the logits of every
position (their shape, their mean, the mean of their absolute values and one value of every `stride`th place of them, an
odd number that gives about `SAMPLES` of them), and the ten largest logits of the last position. The ids are fixed
numbers, so no tokenizer is in the golden.

Manual, not run by the tests or CI: it needs `torch` 2.15.0 and `transformers` 5.22.2 (in a venv, the script stops on
other versions) and the files of the official `HuggingFaceTB/SmolVLM-256M-Instruct` repository (MODEL_DIR holds
`model.safetensors` and `config.json`; the script checks their sha256). Run it from `vision/`, with Python's isolated
mode and the offline flags:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I test/smolvlm_text_hf_golden.py MODEL_DIR \
        > test/fixtures/smolvlm/text_golden.json
"""

import hashlib
import json
import sys

import torch
import transformers
from transformers import Idefics3ForConditionalGeneration

if len(sys.argv) != 2:
    sys.exit("usage: python -I test/smolvlm_text_hf_golden.py MODEL_DIR > test/fixtures/smolvlm/text_golden.json")

SAMPLES = 500
SHA256 = {
    "config.json": "b70fb4bfde88df9eeebc9d8ff523733b8bf70d6c9b06c610325960e06ae9db52",
    "model.safetensors": "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e",
}
IDS = [1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34175, 198, 4119, 42, 7019, 282, 3365, 30]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__
    for name, digest in SHA256.items():
        with open(f"{sys.argv[1]}/{name}", "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, f"{name} is not the file of the golden"


def main():
    check_environment()
    model = Idefics3ForConditionalGeneration.from_pretrained(
        sys.argv[1], dtype=torch.float32, attn_implementation="eager").eval()
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([IDS])).logits[0]
    flat = logits.reshape(-1)
    stride = max(1, flat.numel() // SAMPLES) | 1
    top = logits[-1].topk(10)
    json.dump({
        "ids": IDS, "shape": list(logits.shape), "stride": stride, "mean": round(float(logits.mean()), 6),
        "abs_mean": round(float(logits.abs().mean()), 6), "samples": [round(float(v), 6) for v in flat[::stride]],
        "top_ids": top.indices.tolist(),
        "top_values": [round(float(v), 6) for v in top.values],
    }, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


main()
