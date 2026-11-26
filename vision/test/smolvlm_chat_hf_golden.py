"""Prints what Hugging Face makes of a chat with one image on `HuggingFaceTB/SmolVLM-256M-Instruct` (`float32`), for
`test/smolvlm_chat_hf_test.agc` to compare: for each image of CASES, the text of the chat template, the ids of the
processor (its tiles and the global image expanded), and the 24 ids that `generate` gives greedily (with their text).
The whole path is Hugging Face's: the template of the tokenizer, the Pillow `Idefics3ImageProcessorPil`, the processor
and the model.

Manual, not run by the tests or CI: it needs `torch` 2.15.0 and `transformers` 5.22.2 (in a venv, the script stops on
other versions), `pillow` 11.3.0 (in a folder of its own that this script puts first on `sys.path`) and the files of the
official `HuggingFaceTB/SmolVLM-256M-Instruct` repository in MODEL_DIR (the script checks their sha256). Run it from
`vision/`, with Python's isolated mode and the offline flags:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I test/smolvlm_chat_hf_golden.py MODEL_DIR PILLOW_DIR \
        > test/fixtures/smolvlm/chat_golden.json
"""

import hashlib
import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit("usage: python -I test/smolvlm_chat_hf_golden.py MODEL_DIR PILLOW_DIR > test/fixtures/smolvlm/chat_golden.json")
sys.path.insert(0, sys.argv[2])

import PIL
import torch
import transformers
from PIL import Image
from transformers import AutoTokenizer, Idefics3ForConditionalGeneration
from transformers.models.idefics3.image_processing_pil_idefics3 import Idefics3ImageProcessorPil
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor

NEW = 24
SHA256 = {
    "config.json": "b70fb4bfde88df9eeebc9d8ff523733b8bf70d6c9b06c610325960e06ae9db52",
    "model.safetensors": "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e",
    "chat_template.json": "12462caec000542ca1bb73cd0ba25aaf320d90335fc46731fa6a8355ea34bd26",
    "tokenizer.json": "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a",
}
# The id of the case, its image (in `test/fixtures`) and the question.
CASES = [
    ("wide", "resnet/wide-1.png", "What is in the picture?"),
    ("tall", "resnet/tall-1.png", "Describe this image in one sentence."),
]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__
    assert PIL.__version__ == "11.3.0", PIL.__version__
    assert Path(PIL.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve()), PIL.__file__
    for name, digest in SHA256.items():
        with open(f"{sys.argv[1]}/{name}", "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, f"{name} is not the file of the golden"


def main():
    check_environment()
    model_dir = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = Idefics3Processor(
        image_processor=Idefics3ImageProcessorPil.from_pretrained(model_dir), tokenizer=tokenizer, image_seq_len=64)
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    cases = []
    for name, path, question in CASES:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(f"test/fixtures/{path}").convert("RGB")
        inputs = processor(text=text, images=[image], return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=NEW, do_sample=False)
        new = out[0, inputs["input_ids"].shape[1]:].tolist()
        cases.append({
            "name": name, "image": path, "image_sha256": hashlib.sha256(Path(f"test/fixtures/{path}").read_bytes()).hexdigest(),
            "question": question, "text": text, "prompt_ids": inputs["input_ids"][0].tolist(), "new_ids": new,
            "reply": tokenizer.decode(new),
        })
    json.dump({"cases": cases}, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


main()
