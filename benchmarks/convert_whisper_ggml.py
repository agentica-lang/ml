"""Converts a Hugging Face Whisper checkpoint to whisper.cpp's ggml format.

Same file layout, tensor names and dtypes as whisper.cpp's
models/convert-h5-to-ggml.py, but it needs only the standard library and numpy
(no torch, no transformers). It reads model.safetensors directly: 8 bytes of
header length, a JSON header, then the raw tensor bytes.

One difference: the original rounds every tensor through float16 first, so its
F32 files hold fp16-rounded weights. Here an F32 file keeps the checkpoint's
float32 values, so F32 against F32 is a fair row.

Usage: python3 -I convert_whisper_ggml.py MODEL_DIR MEL_FILTERS_NPZ OUT.bin f32|f16

MODEL_DIR holds model.safetensors, config.json and vocab.json.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

CONV_MAP = {
    "self_attn.k_proj": "attn.key",
    "self_attn.q_proj": "attn.query",
    "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out",
    "self_attn_layer_norm": "attn_ln",
    "encoder_attn.q_proj": "cross_attn.query",
    "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out",
    "encoder_attn_layer_norm": "cross_attn_ln",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
    "encoder.layer_norm.bias": "encoder.ln_post.bias",
    "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.layer_norm.bias": "decoder.ln.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
}
KEEP_F32 = {"encoder.conv1.bias", "encoder.conv2.bias", "encoder.positional_embedding", "decoder.positional_embedding"}


def bytes_to_unicode():
    """GPT-2's byte to unicode table (the BPE vocabulary is written in it)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def read_safetensors(path):
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
        base = 8 + size
        header.pop("__metadata__", None)
        tensors = {}
        for name, meta in header.items():
            if meta["dtype"] != "F32":
                raise SystemExit(f"{name}: dtype {meta['dtype']}, only F32 checkpoints are read")
            start, end = meta["data_offsets"]
            f.seek(base + start)
            tensors[name] = np.frombuffer(f.read(end - start), dtype="<f4").reshape(meta["shape"])
    return tensors


def ggml_name(name):
    parts = name.split(".")[1:]  # drops the leading "model"
    if parts[1] == "layers":
        parts[1] = "blocks"
        key = ".".join(parts[3:-1])
        if key == "encoder_attn.k_proj":
            mapped = "attn.key" if parts[0] == "encoder" else "cross_attn.key"
        else:
            mapped = CONV_MAP[key]
        return ".".join(parts[:3] + [mapped] + parts[-1:])
    joined = ".".join(parts)
    return CONV_MAP.get(joined, joined)


def convert(model_dir, mel_npz, out_path, use_f16):
    hp = json.loads((model_dir / "config.json").read_text(encoding="utf8"))
    max_length = hp.get("max_length") or hp.get("max_target_positions", 448)
    tokens = json.loads((model_dir / "vocab.json").read_text(encoding="utf8"))
    with np.load(mel_npz) as f:
        filters = f[f"mel_{hp['num_mel_bins']}"].astype("<f4")
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}
    tensors = read_safetensors(model_dir / "model.safetensors")

    with open(out_path, "wb") as out:
        out.write(struct.pack("i", 0x67676D6C))
        for v in (
            hp["vocab_size"],
            hp["max_source_positions"],
            hp["d_model"],
            hp["encoder_attention_heads"],
            hp["encoder_layers"],
            max_length,
            hp["d_model"],
            hp["decoder_attention_heads"],
            hp["decoder_layers"],
            hp["num_mel_bins"],
            int(use_f16),
        ):
            out.write(struct.pack("i", v))
        out.write(struct.pack("ii", *filters.shape))
        filters.tofile(out)
        out.write(struct.pack("i", len(tokens)))
        for text, _ in sorted(tokens.items(), key=lambda kv: kv[1]):
            raw = bytes(byte_decoder[c] for c in text)
            out.write(struct.pack("i", len(raw)))
            out.write(raw)

        for src in tensors:
            if src == "proj_out.weight":  # tied to the token embedding
                continue
            name = ggml_name(src)
            data = tensors[src].squeeze()
            if name in ("encoder.conv1.bias", "encoder.conv2.bias"):
                data = data.reshape(data.shape[0], 1)
            if use_f16 and not (data.ndim < 2 or name in KEEP_F32):
                data = data.astype("<f2")
                ftype = 1
            else:
                ftype = 0
            raw_name = name.encode()
            out.write(struct.pack("iii", data.ndim, len(raw_name), ftype))
            for dim in reversed(data.shape):
                out.write(struct.pack("i", dim))
            out.write(raw_name)
            data.tofile(out)


def main():
    if len(sys.argv) != 5 or sys.argv[4] not in ("f32", "f16"):
        raise SystemExit(__doc__)
    convert(Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == "f16")


if __name__ == "__main__":
    main()
