"""Adapter files that PEFT would not write, for `lora_peft_test.agc`, made with the standard library alone from
the adapter of `fixtures/lora_peft/`: `half/`, the same adapter stored as `F16`, and `integer/`, where the first
tensor is declared `I32`. It writes them into a new temporary folder and prints its path. Run from `train/`:

    python3 -I test/lora_files.py
"""

import json
import shutil
import struct
import tempfile
from pathlib import Path

PEFT = Path("test/fixtures/lora_peft")
WEIGHTS = "adapter_model.safetensors"


def read(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    return json.loads(raw[8 : 8 + size]), raw[8 + size :]


def write(folder: Path, header: dict, data: bytes) -> None:
    folder.mkdir()
    shutil.copy(PEFT / "adapter_config.json", folder / "adapter_config.json")
    text = json.dumps(header).encode()
    text += b" " * (-len(text) % 8)
    (folder / WEIGHTS).write_bytes(struct.pack("<Q", len(text)) + text + data)


def main() -> None:
    out = Path(tempfile.mkdtemp(prefix="agentica-lora-"))
    header, data = read(PEFT / WEIGHTS)
    names = sorted(name for name in header if name != "__metadata__")

    half, packed = {"__metadata__": header["__metadata__"]}, b""
    for name in names:
        start, stop = header[name]["data_offsets"]
        values = struct.unpack(f"<{(stop - start) // 4}f", data[start:stop])
        raw = struct.pack(f"<{len(values)}e", *values)
        offsets = [len(packed), len(packed) + len(raw)]
        half[name] = {"dtype": "F16", "shape": header[name]["shape"], "data_offsets": offsets}
        packed += raw
    write(out / "half", half, packed)

    integer = json.loads(json.dumps(header))
    integer[names[0]]["dtype"] = "I32"
    write(out / "integer", integer, data)
    print(out)


if __name__ == "__main__":
    main()
