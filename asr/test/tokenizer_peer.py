"""The independent peer of the tokenizer tests: decodes token ids with the
vocabulary of a Hugging Face `tokenizer.json`, byte-level BPE as GPT-2 writes
it, and names the special tokens by their text. Standard library only.

Usage: python3 tokenizer_peer.py TOKENIZER_JSON IDS_JSON, where IDS_JSON is a
JSON list of id lists. Prints a JSON object: `texts`, one text per id list,
and `special`, the ids of the special tokens and the language tokens.
"""

import json
import sys


def byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main():
    with open(sys.argv[1], encoding="utf-8") as f:
        tok = json.load(f)
    added = {a["content"]: a["id"] for a in tok["added_tokens"]}
    eot = added["<|endoftext|>"]
    by_id = {i: text for text, i in tok["model"]["vocab"].items()}
    decoder = byte_decoder()
    texts = []
    for ids in json.loads(sys.argv[2]):
        raw = bytes(decoder[c] for i in ids if i < eot for c in by_id[i])
        texts.append(raw.decode("utf-8", errors="replace"))
    sot, translate = added["<|startoftranscript|>"], added["<|translate|>"]
    languages = sorted((i, c[2:-2]) for c, i in added.items() if sot < i < translate)
    special = {
        "eot": eot,
        "sot": sot,
        "translate": translate,
        "transcribe": added["<|transcribe|>"],
        "no_timestamps": added["<|notimestamps|>"],
        "timestamp_begin": added["<|0.00|>"],
        "vocab_size": 1 + max(list(added.values()) + list(tok["model"]["vocab"].values())),
        "languages": [[c, i] for i, c in languages],
    }
    print(json.dumps({"texts": texts, "special": special}))


main()
