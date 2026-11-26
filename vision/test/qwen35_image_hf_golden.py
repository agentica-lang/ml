"""Prints what Hugging Face's `Qwen3_5ForConditionalGeneration` (`Qwen/Qwen3.5-0.8B`, `float32`, eager attention) answers
about a few images, for `test/qwen35_image_hf_test.agc` to compare: for each case (a drawn image of `test/fixtures`, the
pixel bounds `size` of the image processor and a question) the prompt that its chat template and image processor make
(the template text, each `<|image_pad|>` repeated for the tokens of the image as `Qwen3VLProcessor` does, then the
tokenizer) as ids, the grid `image_grid_thw`, the ids that greedy `generate` adds (at most `NEW`, stopping at an end id,
which it keeps), and for each step the gap between the two largest logits and the five largest logits with their ids.

Manual, not run by the tests or CI: it needs `torch` 2.15.0 and `transformers` 5.22.2 (in a venv, the script stops on
other versions), `pillow` 11.3.0 (the wheel and the install of `test/qwen_pixels_golden.py`, in a folder of its own that
this script puts first on `sys.path`), no `flash-linear-attention` or `causal-conv1d` (their kernels are not the
reference), and the files of the official `Qwen/Qwen3.5-0.8B` repository (revision `2fc06364715b967f1860aea9cf38778875588b17`;
MODEL_DIR holds the safetensors shard and its index, `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`, `vocab.json` and `merges.txt`, CONFIG_DIR `preprocessor_config.json`); the script checks the sha256
of those that decide the golden. Run it from `vision/`, with Python's isolated mode and the offline flags:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I test/qwen35_image_hf_golden.py MODEL_DIR CONFIG_DIR PILLOW_DIR \
        > test/fixtures/qwen/chat_golden.json
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

PILLOW_DIR = sys.argv[3]
sys.path.insert(0, PILLOW_DIR)

import PIL
import torch
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

NEW = 24
MERGE = 2
STOP = [248046, 248044]
IMAGE = "<|image_pad|>"
SHA256 = {
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "model.safetensors-00001-of-00001.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
}
CONFIG_SHA256 = {"preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"}
# The id, its image (in `test/fixtures`), the bounds of `size`, and the question. The first two are shrunk (they have more
# pixels than the upper bound), the last one is enlarged (the 8-bit crop has fewer than the lower bound).
CASES = [
    ("wide", "resnet/wide-1.png", 4096, 65536, "Describe this image in one sentence."),
    ("tall", "resnet/tall-1.png", 4096, 65536, "What shapes and colors do you see?"),
    ("grow", "qwen/crops/wide-1.png", 65536, 16777216, "What is in the picture?"),
]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__
    assert PIL.__version__ == "11.3.0", PIL.__version__
    assert Path(PIL.__file__).resolve().is_relative_to(Path(PILLOW_DIR).resolve()), PIL.__file__
    for name in ("fla", "causal_conv1d"):
        assert importlib.util.find_spec(name) is None, f"{name} is installed: its kernel is not the reference"
    for folder, pins in ((sys.argv[1], SHA256), (sys.argv[2], CONFIG_SHA256)):
        for name, digest in pins.items():
            with open(f"{folder}/{name}", "rb") as f:
                assert hashlib.sha256(f.read()).hexdigest() == digest, f"{name} is not the file of the golden"


def main():
    check_environment()
    model_dir, config_dir = sys.argv[1], sys.argv[2]
    fixtures = Path("test/fixtures")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    assert model.config.image_token_id == 248056, model.config.image_token_id
    cases = []
    for name, image, low, high, question in CASES:
        processor = AutoImageProcessor.from_pretrained(config_dir, size={"shortest_edge": low, "longest_edge": high})
        source = Image.open(fixtures / image).convert("RGB")
        seen = processor(source, return_tensors="pt")
        grid = seen["image_grid_thw"]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = int(grid.prod(-1)[0]) // MERGE ** 2
        assert text.count(IMAGE) == 1, text
        ids = tok(text.replace(IMAGE, IMAGE * tokens), return_tensors="pt", add_special_tokens=False)["input_ids"]
        types = (ids == model.config.image_token_id).int()
        assert int(types.sum()) == tokens, (int(types.sum()), tokens)
        with torch.no_grad():
            made = model.generate(
                input_ids=ids, pixel_values=seen["pixel_values"], image_grid_thw=grid, mm_token_type_ids=types,
                do_sample=False, repetition_penalty=1.0, max_new_tokens=NEW, eos_token_id=STOP, pad_token_id=248044,
                output_scores=True, return_dict_in_generate=True)
        cases.append({
            "name": name, "image": image, "min_pixels": low, "max_pixels": high, "question": question,
            "grid": grid[0, 1:].tolist(), "ids": ids[0].tolist(),
            "generated": made.sequences[0, ids.shape[1]:].tolist(),
            "gaps": [float(t[0] - t[1]) for t in (step[0].topk(2).values for step in made.scores)],
            "step_top_ids": [step[0].topk(5).indices.tolist() for step in made.scores],
            "step_top_logits": [step[0].topk(5).values.tolist() for step in made.scores],
        })
    json.dump({"cases": cases}, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


main()
