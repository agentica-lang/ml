"""Makes the CLIP tokenizer fixture and goldens of `transformers/` with Hugging Face's slow `CLIPTokenizer`, for
`tokenizer/clip.agc` to match: the ids that `tokenizer(text, truncation=True, max_length=77)` gives (a start and an end
token around the text, the text cut to 75 tokens), for the strings below.

The slow tokenizer is the reference because it is the one whose cleaning is written in Python (`ftfy` is not installed,
so it runs its `BasicTokenizer`: control characters dropped, a space around each CJK ideograph, NFC, lowercase and white
space made one space; no accent stripping, no punctuation splitting).

Manual, not run by the tests: it needs `transformers` (4.57.6 was used, with `regex`; no `ftfy`). Run it in a throwaway
venv, from `transformers/`, with Python's isolated mode.

    python -I test/clip_tokenizer_golden.py fixtures test/fixtures/clip

writes `vocab.json` and `merges.txt` (a tiny tokenizer in CLIP's two files, trained here on `CORPUS`) and
`golden.json` (the ids of `TEXTS` with it).

    python -I test/clip_tokenizer_golden.py golden VOCAB_JSON MERGES_TXT [N] > OUT.json

writes the golden of a real tokenizer (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`), which `tools/tokenizer_check.agc`
compares; it is not committed. `N` adds that many random strings (seeded) of the pieces in `FUZZ`.
"""

import json
import random
import sys
from collections import Counter

import regex
from transformers import CLIPTokenizer
from transformers.models.clip.tokenization_clip import bytes_to_unicode

PATTERN = r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"
MERGES = 150
BOS, EOS = "<|startoftext|>", "<|endoftext|>"

CORPUS = [
    "a photo of a cat", "a photo of a dog", "a drawing of a red square", "a picture of the sea at night",
    "the quick brown fox jumps over the lazy dog", "hello world, hello there", "don't stop, we've got it's all here",
    "a photo of two cats on a sofa", "a painting of a mountain and a lake", "a close up of a flower in the garden",
]

TEXTS = [
    "a photo of a cat", "A PHOTO OF A DOG", "a  photo\tof \n a   bird  ", "hello, world!", "don't", "DON'T", "we've it's I'll",
    "'s", "x'", "rock'n'roll", "2024", "a1b2", "3.14", "(foo) [bar] {baz}", "...", "?!", "a-b_c", "$100 and 50%",
    "café", "CAFÉ", "naïve", "Ünïcödé", "你好 world", "hello你好world", "日本語のテキスト", "한국어", "emoji 😀 test", "a\x00b",
    "a\x07b", "a­b", "a�b", "a　b", "a b", "\x85x", "İstanbul", "ǅ", "ß", "ΑΣ", "",
    " ", "   ", "\n", "<|startoftext|>", "<|endoftext|>", "a <|startoftext|> b", "a<|endoftext|>b", "<|STARTOFTEXT|>",
    "<|EndOfText|> x", "<|endofte", "x<|", "&amp; &lt;", "<html>tom &amp; jerry</html>", "snake_case_name", "word " * 100,
    "x" * 200, "the quick brown fox " * 30,
]
NATURAL = [
    "The capital of France is Paris, and its population is about 2.1 million.",
    "El niño comió piñas en la mañana; ¿quieres más?",
    "Der schnelle braune Fuchs springt über den faulen Hund. Größe: 42cm.",
    "東京は日本の首都です。人口は約1400万人。",
    "Привет, мир! Это тест токенизатора.",
    "a photo of a cat", "a photo of a dog", "a photo of a person riding a horse on the beach at sunset",
    "O'Neil's e-mail: john.doe@example.com, price $1,299.99 (-15%).",
    "Emoji 👩‍👩‍👧‍👦 and 🇪🇸 flags; naïve façade coöperate.",
]
FUZZ = ["a", "b", "cat", "Dog", " ", "  ", "\n", "'s", "'", ",", ".", "!", "1", "23", "é", "ñ", "你", "好", "😀",
        "<|", "|>", "x", "photo", "of", "\t", "\x00", "-", "_", " "]


def tiny_tokenizer(directory):
    """Trains BPE with CLIP's `</w>` word ending on CORPUS and writes the two files."""
    byte_chars = list(bytes_to_unicode().values())
    words = Counter()
    for text in CORPUS:
        for piece in regex.findall(PATTERN, text.lower()):
            word = "".join(bytes_to_unicode()[b] for b in piece.encode("utf-8"))
            words[tuple(word[:-1]) + (word[-1] + "</w>",)] += 1
    merges = []
    for _ in range(MERGES):
        pairs = Counter()
        for word, n in words.items():
            for pair in zip(word, word[1:]):
                pairs[pair] += n
        if not pairs:
            break
        best = max(sorted(pairs), key=lambda p: pairs[p])
        merges.append(best)
        joined = {}
        for word, n in words.items():
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            joined[tuple(out)] = joined.get(tuple(out), 0) + n
        words = Counter(joined)
    tokens = byte_chars + [c + "</w>" for c in byte_chars] + ["".join(m) for m in merges] + [BOS, EOS]
    with open(f"{directory}/vocab.json", "w", encoding="utf-8") as f:
        json.dump({t: i for i, t in enumerate(tokens)}, f, ensure_ascii=False, separators=(",", ":"))
    with open(f"{directory}/merges.txt", "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n" + "\n".join(" ".join(m) for m in merges) + "\n")


def golden(vocab, merges, texts):
    tok = CLIPTokenizer(vocab, merges)
    return [{"text": t, "ids": tok(t, truncation=True, max_length=77)["input_ids"]} for t in texts]


def main():
    if sys.argv[1] == "fixtures":
        directory = sys.argv[2]
        tiny_tokenizer(directory)
        out = golden(f"{directory}/vocab.json", f"{directory}/merges.txt", TEXTS + NATURAL)
        with open(f"{directory}/golden.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    else:
        texts = TEXTS + NATURAL
        rng = random.Random(7)
        for _ in range(int(sys.argv[4]) if len(sys.argv) > 4 else 0):
            texts.append("".join(rng.choice(FUZZ) for _ in range(rng.randint(1, 40))))
        json.dump(golden(sys.argv[2], sys.argv[3], texts), sys.stdout, ensure_ascii=False, separators=(",", ":"))


main()
