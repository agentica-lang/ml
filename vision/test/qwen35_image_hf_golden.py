"""Prints what Hugging Face's `Qwen3_5ForConditionalGeneration` (`Qwen/Qwen3.5-0.8B`, `float32`, eager attention) answers
about a few images, for `test/qwen35_image_hf_test.agc` to compare: for each case (a drawn image of `test/fixtures`, the
pixel bounds `size` of the image processor and a question) the prompt that the processor (`AutoProcessor`: its chat
template, its image processor and the tokenizer) makes as ids, the grid `image_grid_thw`, the ids that greedy `generate`
adds (at most `NEW`, stopping at an end id, which it keeps), and for each step the gap between the two largest logits and
the five largest logits with their ids. The resized image that the processor made, 8 bits per channel, is written as a
PNG into `test/fixtures/qwen/crops/chat-NAME.png`, whose sha256 goes in the golden (`pixel_values` are exactly its patches, scaled and normalized, as in
`test/qwen_pixels_golden.py`, which checks the resize), and the test gives the tower that crop: the gap of its logits is
then the model's and not the resize of libvips.

Manual, not run by the tests or CI: it needs `torch` 2.15.0 and `transformers` 5.22.2 (in a venv, the script stops on
other versions), `pillow` 11.3.0 (the wheel and the install of `test/qwen_pixels_golden.py`, in a folder of its own that
this script puts first on `sys.path`), no `flash-linear-attention` or `causal-conv1d` (their kernels are not the
reference), and the files of the official `Qwen/Qwen3.5-0.8B` repository (revision `2fc06364715b967f1860aea9cf38778875588b17`;
MODEL_DIR holds the safetensors shard and its index, `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`, `vocab.json` and `merges.txt`, CONFIG_DIR `preprocessor_config.json`; no
`generation_config.json`, the script passes every option of `generate`); the script checks the sha256 of those that
decide the golden. Run it from `vision/`, with Python's isolated mode and the offline flags:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I test/qwen35_image_hf_golden.py MODEL_DIR CONFIG_DIR PILLOW_DIR \
        TORCHVISION_DIR > test/fixtures/qwen/chat_golden.json

TORCHVISION_DIR holds `torchvision` 0.29.2 (the CPU wheel, `pip install --no-deps --target`): `AutoProcessor` loads the
video processor of the checkpoint's processor class and refuses to without it; no video goes through it.
"""

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

PILLOW_DIR, TORCHVISION_DIR = sys.argv[3], sys.argv[4]
sys.path.insert(0, PILLOW_DIR)
sys.path.append(TORCHVISION_DIR)

import numpy as np
import PIL
import torch
import torchvision
import transformers
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

NEW = 24
PATCH = 16
MERGE = 2
FRAMES = 2
STOP = [248046, 248044]
IMAGE = "<|image_pad|>"
SHA256 = {
    "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
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
    assert torchvision.__version__.startswith("0.29.2"), torchvision.__version__
    assert Path(PIL.__file__).resolve().is_relative_to(Path(PILLOW_DIR).resolve()), PIL.__file__
    for name in ("fla", "causal_conv1d"):
        assert importlib.util.find_spec(name) is None, f"{name} is installed: its kernel is not the reference"
    for folder, pins in ((sys.argv[1], SHA256), (sys.argv[2], CONFIG_SHA256)):
        for name, digest in pins.items():
            with open(f"{folder}/{name}", "rb") as f:
                assert hashlib.sha256(f.read()).hexdigest() == digest, f"{name} is not the file of the golden"


def unpatch(rows, grid_h, grid_w):
    """The 8-bit image whose `pixel_values` are `rows` (normalized with 0.5 and 0.5)."""
    channels = 3
    x = rows.reshape(grid_h // MERGE, grid_w // MERGE, MERGE, MERGE, channels, FRAMES, PATCH, PATCH)
    assert (x[:, :, :, :, :, 0] == x[:, :, :, :, :, 1]).all(), "the frames of a still image differ"
    x = x[:, :, :, :, :, 0].transpose(4, 0, 2, 5, 1, 3, 6)
    image = x.reshape(channels, grid_h * PATCH, grid_w * PATCH)
    samples = np.round((image * 0.5 + 0.5) * 255.0)
    assert np.abs((samples / 255.0 - 0.5) / 0.5 - image).max() < 1e-5, "pixel_values are not 8-bit samples"
    return samples.astype(np.uint8).transpose(1, 2, 0)


def make_processor(model_dir, config_dir, folder):
    """`AutoProcessor` over the tokenizer files of MODEL_DIR and the `preprocessor_config.json` of CONFIG_DIR."""
    names = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")
    for name in names:
        (Path(folder) / name).symlink_to(Path(model_dir).resolve() / name)
    (Path(folder) / "preprocessor_config.json").symlink_to(Path(config_dir).resolve() / "preprocessor_config.json")
    return AutoProcessor.from_pretrained(folder)


def main():
    check_environment()
    with tempfile.TemporaryDirectory() as folder:
        run(make_processor(sys.argv[1], sys.argv[2], folder))


def run(processor):
    model_dir, config_dir = sys.argv[1], sys.argv[2]
    fixtures = Path("test/fixtures")
    crops = fixtures / "qwen" / "crops"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="eager").eval()
    assert model.config.image_token_id == 248056, model.config.image_token_id
    cases = []
    for name, image, low, high, question in CASES:
        # The image processor of the case: the processor keeps the tokenizer and the template, and `size` is its bound.
        processor.image_processor = AutoImageProcessor.from_pretrained(
            config_dir, size={"shortest_edge": low, "longest_edge": high})
        source = Image.open(fixtures / image).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert text.count(IMAGE) == 1, text
        seen = processor(text=[text], images=[source], return_tensors="pt")
        grid = seen["image_grid_thw"]
        ids = seen["input_ids"]
        types = (ids == model.config.image_token_id).int()
        # The tokens of the image are the processor's own, and the grid has as many merged squares.
        assert int(types.sum()) == int(grid.prod(-1)[0]) // MERGE ** 2, (int(types.sum()), grid)
        _, grid_h, grid_w = (int(v) for v in grid[0])
        crop = crops / f"chat-{name}.png"
        Image.fromarray(unpatch(seen["pixel_values"].numpy(), grid_h, grid_w)).save(crop, optimize=True)
        with torch.no_grad():
            made = model.generate(
                input_ids=ids, pixel_values=seen["pixel_values"], image_grid_thw=grid, mm_token_type_ids=types,
                do_sample=False, repetition_penalty=1.0, max_new_tokens=NEW, eos_token_id=STOP, pad_token_id=248044,
                output_scores=True, return_dict_in_generate=True)
        cases.append({
            "name": name, "image": image, "crop": f"qwen/crops/chat-{name}.png",
            "crop_sha256": hashlib.sha256(crop.read_bytes()).hexdigest(), "min_pixels": low, "max_pixels": high,
            "question": question,
            "grid": grid[0, 1:].tolist(), "ids": ids[0].tolist(),
            "generated": made.sequences[0, ids.shape[1]:].tolist(),
            "gaps": [float(t[0] - t[1]) for t in (step[0].topk(2).values for step in made.scores)],
            "step_top_ids": [step[0].topk(5).indices.tolist() for step in made.scores],
            "step_top_logits": [step[0].topk(5).values.tolist() for step in made.scores],
        })
    json.dump({"cases": cases}, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


main()
