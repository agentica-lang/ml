"""The independent peer of `bpe_test.agc`'s GGUF tests: the tokenizer of
`fixtures/bpe.json` as llama.cpp's converter writes it into a GGUF (the tokens
in the order of their ids, their types and the merges as `left right`), plus
files that the reader must refuse. Standard library only.

Usage: python3 -I test/bpe_gguf_peer.py [BPE_JSON] (from `transformers/`, the
default `test/fixtures/bpe.json`). Writes into a new temporary
folder, prints its path, and leaves `bpe.gguf` (after the added tokens, two
unused ids as the padding of a vocabulary) and one file per refusal.
"""

import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

NORMAL, CONTROL, USER_DEFINED, UNUSED, BYTE = 1, 3, 4, 5, 6


def g_string(s):
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def g_array(kind, items):
    return struct.pack("<IQ", kind, len(items)) + b"".join(items)


def gguf(path, tokens, types, merges, model="gpt2", pre="qwen2", omit=(), text=()):
    """`omit` leaves keys out, `text` writes them as a string instead of an array."""
    kvs = [("general.architecture", 8, g_string("qwen3")),
           ("tokenizer.ggml.model", 8, g_string(model)),
           ("tokenizer.ggml.pre", 8, g_string(pre)),
           ("tokenizer.ggml.tokens", 9, g_array(8, [g_string(t) for t in tokens])),
           ("tokenizer.ggml.token_type", 9, g_array(5, [struct.pack("<i", t) for t in types])),
           ("tokenizer.ggml.merges", 9, g_array(8, [g_string(m) for m in merges]))]
    kvs = [(k, 8, g_string("x")) if k in text else (k, kind, body) for k, kind, body in kvs if k not in omit]
    head = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    for key, kind, body in kvs:
        head += g_string(key) + struct.pack("<I", kind) + body
    path.write_bytes(head + b"\0" * (-len(head) % 32))


def write(out, source):
    j = json.loads(Path(source).read_text())
    vocab = {i: t for t, i in j["model"]["vocab"].items()}
    added = {a["id"]: a for a in j["added_tokens"]}
    top = max(list(vocab) + list(added)) + 1
    tokens, types = [], []
    for i in range(top):
        if i in vocab:
            tokens.append(vocab[i])
            types.append(NORMAL)
        else:
            tokens.append(added[i]["content"])
            types.append(CONTROL if added[i]["special"] else USER_DEFINED)
    first = min(added)
    merges = [m if isinstance(m, str) else " ".join(m) for m in j["model"]["merges"]]
    gguf(out / "bpe.gguf", tokens + ["[PAD%d]" % top, "[PAD%d]" % (top + 1)], types + [UNUSED, UNUSED], merges)
    # an unused id between the normal tokens and the added ones: the added tokens move up by one
    gguf(out / "unused_middle.gguf", tokens[:first] + ["[PAD]"] + tokens[first:], types[:first] + [UNUSED] + types[first:],
         merges)
    # `<think>` as llama.cpp's control token: special for `decode`, which `tokenizer.json` does not say
    gguf(out / "think_control.gguf", tokens, types[:-1] + [CONTROL], merges)
    gguf(out / "other_model.gguf", tokens, types, merges, model="llama")
    gguf(out / "other_pre.gguf", tokens, types, merges, pre="llama3")
    gguf(out / "byte_token.gguf", tokens, types[:-1] + [BYTE], merges)
    gguf(out / "type_undefined.gguf", tokens, [0] + types[1:], merges)
    gguf(out / "type_unknown.gguf", tokens, [2] + types[1:], merges)
    gguf(out / "no_types.gguf", tokens, types, merges, omit=("tokenizer.ggml.token_type",))
    gguf(out / "no_merges.gguf", tokens, types, merges, omit=("tokenizer.ggml.merges",))
    gguf(out / "tokens_text.gguf", tokens, types, merges, text=("tokenizer.ggml.tokens",))
    gguf(out / "merge_empty_half.gguf", tokens, types, merges + [" b"])
    gguf(out / "types_short.gguf", tokens, types[:-1], merges)
    gguf(out / "token_twice.gguf", tokens + [tokens[0]], types + [NORMAL], merges)
    gguf(out / "merge_no_space.gguf", tokens, types, merges + ["ab"])
    gguf(out / "merge_three.gguf", tokens, types, merges + ["a b c"])


def main():
    out = Path(tempfile.mkdtemp(prefix="agentica-bpe-gguf-"))
    try:
        write(out, sys.argv[1] if len(sys.argv) > 1 else "test/fixtures/bpe.json")
    except BaseException:
        shutil.rmtree(out)
        raise
    print(out)


if __name__ == "__main__":
    main()
