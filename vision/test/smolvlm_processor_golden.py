"""Prints what Hugging Face's `Idefics3Processor` (the Pillow image processor `Idefics3ImageProcessorPil` and the
tokenizer of `HuggingFaceTB/SmolVLM-256M-Instruct`, `image_seq_len` 64) makes of the drawn images of
`test/fixtures/resnet` and a text with `<image>` in it, for `test/smolvlm_processor_test.agc` to compare:

- the ids of the special tokens (`tokens`) and of `"\n"` and `"\n\n"`;
- a table of the sizes (`sizes`): of an image of height by width, the size after the resize to a longest edge of 2048
  (`resized`), the size that `resize_for_vision_encoder` makes of that (`grid`, multiples of 512) and the rows and
  columns of 512 by 512 tiles that it splits into (`rows`, `cols`);
- for each case, its images, its text, the ids of the text with one `<image>` for each image (`ids_one`, which the
  tokenizer gives and the test takes as they are), the ids of the processor (`ids`, with each `<image>` expanded to
  its tiles and its global image), the rows and columns of each image, and the `pixel_values` of every frame (the tiles
  of each image in order and then its global image, 512 by 512 each): the mean of each frame and one value of every
  `stride`th place of it (an odd number that gives about `SAMPLES` of them), its standard deviation and the mean of the
  absolute differences of neighbours across (`dx`) and down (`dy`); for the case `edges` also the values at the `EDGES`
  places of each frame where the neighbours differ most (`edge_at`, flat indexes, and `edge_values`).
- a synthetic image (`checker.png`, 1000 wide and 333 high, written into `test/fixtures/smolvlm`, which the case `edges`
  reads): a checkerboard of cells of 5, 6 and 7 pixels, 1-pixel diagonal lines and sharp steps, so that a resize that
  differs in its kernel, its phase or its aspect ratio shows (the drawn images of `resnet` are smooth, and these
  differences are under the gaps on them). The 333 by 1000 size is one whose grid (1024 by 2048) is not its resized size
  (682 by 2048), so the global image depends on whether it is taken before or after the second resize;
- a folder (`prompt`, written into `test/fixtures/smolvlm`) with only what `SmolVlmPrompt.open` reads: the added tokens
  of the prompt and the two newline entries of `tokenizer.json`, with the checkpoint's ids, and `processor_config.json`.

Manual, not run by the tests or CI: it needs `torch` 2.15.0 and `transformers` 5.22.2 (in a venv, the script stops on
other versions), `pillow` 11.3.0 (in a folder of its own, as `qwen_pixels_golden.py` says, which this script puts first
on `sys.path`) and the files of the official `HuggingFaceTB/SmolVLM-256M-Instruct` repository in MODEL_DIR
(`config.json`, `preprocessor_config.json`, `processor_config.json`, `tokenizer.json`, `tokenizer_config.json`,
`added_tokens.json`, `special_tokens_map.json`). Run it from `vision/`, with Python's isolated mode and the offline
flags:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -I test/smolvlm_processor_golden.py MODEL_DIR PILLOW_DIR \
        > test/fixtures/smolvlm/processor_golden.json
"""

import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit("usage: python -I test/smolvlm_processor_golden.py MODEL_DIR PILLOW_DIR > test/fixtures/smolvlm/processor_golden.json")
sys.path.insert(0, sys.argv[2])

import numpy as np
import PIL
import torch
import transformers
from PIL import Image
from transformers import AutoTokenizer
from transformers.models.idefics3.image_processing_pil_idefics3 import (
    Idefics3ImageProcessorPil, _resize_output_size_rescale_to_max_len, _resize_output_size_scale_below_upper_bound,
)
from transformers.models.idefics3.processing_idefics3 import Idefics3Processor

SAMPLES = 300
EDGES = 300
SEQ_LEN = 64
TILE = 512
LONGEST = 2048
# Heights and widths of the table: a square one, one below a tile, odd and even sides, a very long one, and the ones
# whose second resize (to multiples of 512) does not keep the aspect ratio.
SIZES = [
    (360, 480), (333, 500), (400, 300), (512, 512), (513, 100), (100, 513), (1, 1), (7, 3), (2048, 2048), (3000, 1000),
    (1000, 3000), (4096, 4096), (600, 601), (601, 600), (1, 5000), (333, 1000),
]
TEXT = "User:<image>What is in the picture?<end_of_utterance>\nAssistant:"
TEXT_TWO = "User:<image>Compare it with<image>.<end_of_utterance>\nAssistant:"
# The id of the case, its images (in `test/fixtures`) and its text.
CASES = [
    ("one", ["resnet/wide-1.png"], TEXT),
    ("two", ["resnet/tall-1.png", "resnet/wide-2.png"], TEXT_TWO),
    ("edges", ["smolvlm/checker.png"], TEXT),
]


def check_environment():
    assert torch.__version__.startswith("2.15.0"), torch.__version__
    assert transformers.__version__ == "5.22.2", transformers.__version__
    assert PIL.__version__ == "11.3.0", PIL.__version__


def make_checker(path):
    height, width = 333, 1000
    out = np.zeros((height, width, 3), dtype=np.uint8)
    # A checkerboard of cells of 5, 6 and 7 pixels (the size cycles across the columns and down the rows).
    cell = np.array([5, 6, 7])
    columns = np.repeat(np.arange(300), np.tile(cell, 100)[:300])[:600]
    rows = np.repeat(np.arange(300), np.tile(cell[::-1], 100)[:300])[:height]
    out[:, :600] = (((rows[:, None] + columns[None, :600]) % 2) * 255).astype(np.uint8)[:, :, None]
    yy, xx = np.mgrid[0:height, 0:width]
    # 1-pixel diagonals, white and coloured, over black.
    diagonal = (xx + yy) % 9 == 0
    out[:, 600:800][diagonal[:, 600:800]] = 255
    out[:, 600:800][((xx - yy) % 11 == 0)[:, 600:800]] = (255, 40, 0)
    # Sharp steps: bands of 8 pixels, then rows of 37 pixels with five levels.
    out[:, 800:900] = (((xx[:, 800:900] // 8) % 2) * 255).astype(np.uint8)[:, :, None]
    levels = np.array([0, 64, 128, 192, 255], dtype=np.uint8)
    out[:, 900:] = levels[(yy[:, 900:] // 37) % 5][:, :, None]
    Image.fromarray(out).save(path, optimize=True)


def prompt_folder(directory, tokenizer, processor, model_dir):
    directory.mkdir(parents=True, exist_ok=True)
    added = [
        a for a in json.load(open(f"{model_dir}/tokenizer.json", encoding="utf-8"))["added_tokens"]
        if a["content"] in {"<fake_token_around_image>", "<image>", "<global-img>"} or a["content"].startswith("<row_")
    ]
    newline, newlines = "\u010a", "\u010a\u010a"
    vocab = {newline: tokenizer.convert_tokens_to_ids(newline), newlines: tokenizer.convert_tokens_to_ids(newlines)}
    with open(directory / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump({"added_tokens": added, "model": {"type": "BPE", "vocab": vocab, "merges": []}}, f, indent=1)
        f.write("\n")
    with open(directory / "processor_config.json", "w", encoding="utf-8") as f:
        json.dump({"image_seq_len": processor.image_seq_len, "processor_class": "Idefics3Processor"}, f, indent=1)
        f.write("\n")


def edge_places(frame):
    """The flat indexes of `EDGES` places where the neighbours of the frame differ most, spread over the top ones."""
    gradient = torch.zeros_like(frame)
    gradient[:, :, :-1] += (frame[:, :, 1:] - frame[:, :, :-1]).abs()
    gradient[:, :-1, :] += (frame[:, 1:, :] - frame[:, :-1, :]).abs()
    order = torch.argsort(gradient.reshape(-1), descending=True, stable=True)[: EDGES * 10]
    return order[::10].tolist()


def sizes_of(processor, height, width):
    h, w = _resize_output_size_rescale_to_max_len(height, width, max_len=LONGEST)
    h, w = _resize_output_size_scale_below_upper_bound(h, w, max_len=4096)
    pixels = np.zeros((3, h, w), dtype=np.uint8)
    grid = processor.resize_for_vision_encoder(pixels, TILE).shape[-2:]
    rows, cols = -(-grid[0] // TILE), -(-grid[1] // TILE)
    return {"height": height, "width": width, "resized": [h, w], "grid": list(grid), "rows": rows, "cols": cols}


def main():
    check_environment()
    model_dir = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    image_processor = Idefics3ImageProcessorPil.from_pretrained(model_dir)
    processor = Idefics3Processor(image_processor=image_processor, tokenizer=tokenizer, image_seq_len=SEQ_LEN)
    sizes = [sizes_of(image_processor, h, w) for h, w in SIZES]
    make_checker("test/fixtures/smolvlm/checker.png")
    prompt_folder(Path("test/fixtures/smolvlm/prompt"), tokenizer, processor, model_dir)
    cases = []
    for name, files, text in CASES:
        images = [Image.open(f"test/fixtures/{f}").convert("RGB") for f in files]
        out = processor(text=[text], images=[images], return_tensors="pt")
        frames = out["pixel_values"][0]
        assert bool(out["pixel_attention_mask"].all()), "padding in a frame"
        rows_cols = image_processor(images, return_row_col_info=True, return_tensors="pt")
        rows = rows_cols["rows"][0]
        cols = rows_cols["cols"][0]
        flat = frames.reshape(frames.shape[0], -1)
        stride = max(1, flat.shape[1] // SAMPLES) | 1
        places = [edge_places(f) for f in frames] if name == "edges" else [[] for _ in frames]
        cases.append({
            "name": name, "images": files, "text": text,
            "ids_one": tokenizer(text)["input_ids"], "ids": out["input_ids"][0].tolist(),
            "rows": rows, "cols": cols, "stride": stride,
            "means": [round(float(f.mean()), 6) for f in flat],
            "stds": [round(float(f.std(correction=0)), 6) for f in flat],
            "dx": [round(float((f[:, :, 1:] - f[:, :, :-1]).abs().mean()), 6) for f in frames],
            "dy": [round(float((f[:, 1:, :] - f[:, :-1, :]).abs().mean()), 6) for f in frames],
            "edge_at": places,
            "edge_values": [[round(float(f.reshape(-1)[i]), 6) for i in at] for f, at in zip(frames, places)],
            "samples": [[round(float(v), 6) for v in f[::stride]] for f in flat],
        })
    tokens = {
        "fake": processor.fake_image_token_id, "image": processor.image_token_id, "global": processor.global_image_token_id,
        "row_cols": processor.row_col_ids, "seq_len": processor.image_seq_len,
        "newline": tokenizer("\n", add_special_tokens=False)["input_ids"],
        "newlines": tokenizer("\n\n", add_special_tokens=False)["input_ids"],
    }
    assert len(tokens["newline"]) == 1 and len(tokens["newlines"]) == 1, tokens
    tokens["newline"] = tokens["newline"][0]
    tokens["newlines"] = tokens["newlines"][0]
    json.dump({"tokens": tokens, "sizes": sizes, "cases": cases}, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


main()
