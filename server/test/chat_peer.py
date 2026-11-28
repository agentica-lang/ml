"""The chat-capable tiny Qwen3 fixture of the server tests: a small random tied model (the `plain` builder and config
of `transformers/test/qwen_peer.py`) with a vocabulary of every byte, so the Qwen3 chat template encodes, and the
added tokens `<|im_start|>`, `<|im_end|>` and `<|endoftext|>`. Standard library only.

Usage, from `server/`: python3 test/chat_peer.py. Writes into a new temporary folder, prints its path, and leaves:

- `chat/`: `float32` weights with tied embeddings and no `lm_head.weight` (as the `tiny` fixture of `qwen_peer.py`),
  with `vocab_size` 259 (256 byte tokens, then the three added ones);
- `chat/tokenizer.json`: Qwen's `tokenizer.json` form with no merges, so every byte is one token and any text
  encodes, and the template markers cut out as added tokens;
- `chat/config.json`: `model_type` qwen3 with room for 64 positions.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent / ".." / ".." / "transformers" / "test").resolve()))

import qwen_peer as q

# 256 byte tokens, then `<|im_start|>`, `<|im_end|>` and `<|endoftext|>`, all special.
CONFIG = dict(q.CONFIG, vocab_size=259, max_position_embeddings=64, bos_token_id=258, eos_token_id=257,
              tie_word_embeddings=True, num_hidden_layers=1)
IDS = {"<|im_start|>": 256, "<|im_end|>": 257, "<|endoftext|>": 258}


def bytes_to_unicode():
    """The byte-level alphabet of GPT-2 (as `tokenizers`' `ByteLevel`): printable bytes map to themselves, the rest to
    the characters 256 and up, in byte order."""
    kept = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    kept += list(range(ord("®"), ord("ÿ") + 1))
    text, chars = kept[:], kept[:]
    for byte in range(256):
        if byte not in kept:
            text.append(byte)
            chars.append(256 + len(chars) - len(kept))
    return {byte: chr(char) for byte, char in zip(text, chars)}


def write_tokenizer(folder):
    base = Path(__file__).parent / ".." / ".." / "transformers" / "test" / "fixtures" / "bpe.json"
    j = json.loads(base.resolve().read_text())
    table = bytes_to_unicode()
    j["model"]["vocab"] = {table[byte]: byte for byte in range(256)}
    j["model"]["merges"] = []
    template = j["added_tokens"][0]
    j["added_tokens"] = [dict(template, id=ids, content=text, special=True) for text, ids in IDS.items()]
    (folder / "tokenizer.json").write_text(json.dumps(j))
    (folder / "tokenizer_config.json").write_text(json.dumps({"chat_template": q.CHAT_TEMPLATE,
                                                              "pad_token": "<|endoftext|>", "bos_token": None}))


def write(out):
    (out / "chat").mkdir()
    q.plain(CONFIG, 7).save(out / "chat" / "model.safetensors")
    (out / "chat" / "config.json").write_text(json.dumps(CONFIG))
    write_tokenizer(out / "chat")


def main():
    out = Path(tempfile.mkdtemp(prefix="agentica-chat-"))
    try:
        write(out)
    except BaseException:
        import shutil

        shutil.rmtree(out)
        raise
    print(out)


if __name__ == "__main__":
    main()
