"""The independent peer of the Q4_K and Q6_K reads: the official `gguf` package (gguf-py of llama.cpp) writes a GGUF
file of Q4_K and Q6_K tensors made of random blocks and dequantizes them itself.

Manual, not run by the tests (they read the committed fixtures). The fixtures change only with this script:

    python3 -m venv ~/.cache/agentica-ml/venv-gguf
    ~/.cache/agentica-ml/venv-gguf/bin/pip install --only-binary :all: gguf==0.19.0
    ~/.cache/agentica-ml/venv-gguf/bin/python -I hub/test/kquants_golden.py

It writes `hub/test/fixtures/kquants.gguf` and `kquants.json` (the dequantized values of each tensor as the bit patterns
of float32, with the shape), and `tensor/test/fixtures/kquants.json` (the blocks as hex, with the same values).
The blocks are random bytes, so every scale, every nibble and every high bit pair occurs; the scales `d` and `dmin` are
binary16 values small enough that a value stays under 16, which the tests' tolerance counts on. One block of each type
is all zero and one is all ones (0xFF) but for its scales.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf import GGUFWriter, quants

ROOT = Path(__file__).resolve().parents[2]
ROWS, COLS = 2, 512
BLOCKS = {T.Q4_K: 144, T.Q6_K: 210}
# where `d` (and `dmin`) sit in a block: Q4_K starts with d and dmin, Q6_K ends with d
SCALES = {T.Q4_K: [(0, 0.0005, 0.01), (2, 0.0005, 0.01)], T.Q6_K: [(208, 0.0001, 0.002)]}


def blocks(kind, seed):
    rng = np.random.default_rng(seed)
    count = ROWS * COLS // 256
    data = rng.integers(0, 256, size=(count, BLOCKS[kind]), dtype=np.uint8)
    for at, low, high in SCALES[kind]:
        d = rng.uniform(low, high, size=count) * rng.choice([-1.0, 1.0], size=count)
        data[:, at:at + 2] = d.astype("<f2").view(np.uint8).reshape(count, 2)
    data[0, :] = 0
    for at, low, high in SCALES[kind]:
        data[0, at:at + 2] = np.array([high], dtype="<f2").view(np.uint8)
    data[1, :] = 255
    for at, low, high in SCALES[kind]:
        data[1, at:at + 2] = np.array([-low], dtype="<f2").view(np.uint8)
    return data.reshape(ROWS, count // ROWS * BLOCKS[kind])


def bits(values):
    return [int(b) for b in np.asarray(values, dtype="<f4").reshape(-1).view("<u4")]


def main():
    out = ROOT / "hub/test/fixtures"
    out.mkdir(exist_ok=True)
    writer = GGUFWriter(str(out / "kquants.gguf"), "qwen3")
    names = {T.Q4_K: "blk.0.attn_q.weight", T.Q6_K: "blk.0.ffn_up.weight"}
    hub, tensor = {}, {}
    for seed, kind in enumerate([T.Q4_K, T.Q6_K]):
        raw = blocks(kind, seed)
        want = quants.dequantize(raw, kind)
        assert want.shape == (ROWS, COLS) and want.dtype == np.float32
        assert float(np.abs(want).max()) < 16, "a value is too large for the tolerance of the tests"
        writer.add_tensor(names[kind], raw, raw_dtype=kind)
        hub[names[kind]] = {"type": kind.name, "shape": [ROWS, COLS], "bits": bits(want)}
        tensor[kind.name.lower()] = {"shape": [ROWS, COLS], "hex": raw.tobytes().hex(), "bits": bits(want)}
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    (out / "kquants.json").write_text(json.dumps(hub))
    (ROOT / "tensor/test/fixtures/kquants.json").write_text(json.dumps(tensor))
    print("wrote", len(hub), "tensors")


if __name__ == "__main__":
    sys.exit(main())
