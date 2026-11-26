"""Makes the tokenizer fixtures and goldens of `transformers/` with Hugging Face
`tokenizers`: the ids that its `encode` gives and the text that its `decode`
gives, for `tokenizer/wordpiece.agc` and `tokenizer/bpe.agc` to match.

Manual, not run by the tests: it needs `tokenizers` (0.22.2 was used). Run it
in a throwaway venv, from `transformers/`, with Python's isolated mode.

    python -I test/tokenizer_golden.py fixtures test/fixtures

writes `wordpiece.json`, `bpe.json` and `bpe_lines.json` (`bpe.json` with its merges as "a b" lines) (tiny hand-made tokenizers, in
`tokenizer.json` form) and the goldens `wordpiece_golden.json` and
`bpe_golden.json` (the ids of the strings below and their decoded text).

    python -I test/tokenizer_golden.py marks test/fixtures

writes `bpe_marks.json` and `bpe_marks_golden.json`: the same tiny tokenizer with the split of Qwen3.5 (a mark counts
as a letter) and the ids of the strings below plus those of `MARK_TEXTS` (it needs `tokenizers` 0.23.3; the script
stops on another version).

    python -I test/tokenizer_golden.py groups test/fixtures

writes `bpe_groups.json` and `bpe_groups_golden.json`: the same tiny tokenizer with the split of LFM3 (up to three digits
in a piece, `\\p{N}{1,3}`), no normalizer, `ignore_merges` on and its last merges cut off (so a piece that is a token
and that the merges no longer reach is still one token), and the ids of the strings below plus those of `GROUP_TEXTS`
and 20 random ones (`tokenizers` 0.23.3, as above).

    python -I test/tokenizer_golden.py gpt2 test/fixtures

writes `bpe_gpt2.json` and `bpe_gpt2_golden.json`: the same tiny tokenizer with the pre-tokenizer of SmolLM2 and SmolVLM
(`Digits` with `individual_digits`, then ByteLevel with its own GPT-2 split `use_regex`), no normalizer and no
post-processor, and the ids of the strings below plus those of `GPT2_TEXTS` and 60 random ones (`tokenizers` 0.23.3, as
above).

    python -I test/tokenizer_golden.py golden wordpiece TOKENIZER_JSON [N] > OUT.json
    python -I test/tokenizer_golden.py golden bpe TOKENIZER_JSON [N] > OUT.json
    python -I test/tokenizer_golden.py golden bpe_groups TOKENIZER_JSON [N] > OUT.json
    python -I test/tokenizer_golden.py golden bpe_gpt2 TOKENIZER_JSON [N] > OUT.json

writes the golden of a real tokenizer (`bert-base-uncased`, `Qwen3-0.6B`, `LFM3-1.7B` with `bpe_groups`, which adds `GROUP_TEXTS`, and `SmolVLM-256M-Instruct` with `bpe_gpt2`, which adds `GPT2_TEXTS`; both check the version of `tokenizers` and the sha256 of the file),
which `tools/tokenizer_check.agc` compares; it is not committed. `N` adds that
many random strings (seeded) of the pieces in `FUZZ`.
"""

import json
import random
import sys

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

QWEN_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

WORDPIECE_TEXTS = [
    "Hello, World!",
    "hello world",
    "unaffable",
    "unaffables",
    "playing plays",
    "CAFÉ café",
    "naïve",
    "don't",
    "中文hello",
    "hello中文",
    "[CLS] hello [SEP]",
    "[MASK]",
    "hello [MASK] world",
    "  spaced   out  ",
    "tab\tnew\nline",
    "a\x00b",
    "­soft",
    "Ǆ ǅ",
    "İstanbul",
    "ΑΣ",
    "x" * 150,
    "x" * 100,
    "xyz",
    "hello-world",
    "$%",
    " em",
    "'",
    "!!!",
    "",
    " ",
    "école",
    "a�b",
    "\U00020000 hello",
    "hello　world",
    "ǰ",
]
# Natural text, for the real tokenizers too (the tiny ones answer unknowns).
NATURAL = [
    "The capital of France is Paris, and its population is about 2.1 million.",
    "El niño comió piñas en la mañana; ¿quieres más?",
    "Der schnelle braune Fuchs springt über den faulen Hund. Größe: 42cm.",
    "東京は日本の首都です。人口は約1400万人。",
    "北京是中华人民共和国的首都，人口超过两千万。",
    "안녕하세요, 만나서 반갑습니다.",
    "Привет, мир! Это тест токенизатора.",
    "مرحبا بالعالم",
    "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
    "O'Neil's e-mail: john.doe@example.com, price $1,299.99 (-15%).",
    "Emoji 👩‍👩‍👧‍👦 and 🇪🇸 flags; naïve façade coöperate.",
    "<html><body class=\"x\">Tom &amp; Jerry</body></html>",
    "tokenization of unbelievably uncharacteristically long words like pneumonoultramicroscopicsilicovolcanoconiosis",
]
QWEN35_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Strings where a mark (Mn, Mc, Me) decides the split: after a letter, after a space, after a symbol, alone, doubled.
MARK_TEXTS = [
    "a\u0301b", "\u0301\u0301x", "$\u0301a", " \u0301a", "e\u0301\u0903 x", "\u0903", "1\u0301", "a\u20dd b",
    "हिन्दी भाषा", "ไทย ภาษา", "عَرَبِيّ", "שָׁלוֹם", "x.\u0301y", "\n\u0301a", "'s\u0301", "e\u0301\n\u0301",
]

LFM3_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

# Strings where the digit groups decide the cut: runs of every length, digits of other scripts, other numbers (No, Nl),
# digits next to letters, signs and separators, and text that NFC would change (LFM3 has no normalizer).
# The `tokenizer.json` of `LiquidAI/LFM3-1.7B` at revision 1da06bec6a819da10b54c67e5c90972b46d81871.
LFM3_TOKENIZER_SHA256 = "7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701"

GROUP_TEXTS = [
    "1", "12", "123", "1234", "12345", "123456", "1234567", "0001", "a1234b", "x 1234567", "-1234", " 12", "$1,000,000",
    "1.234.567,89", "3.14159", "a 123\n456", "2024-11-26", "٣٤٥٦٧", "２０２４年", "²³⁴⁵", "½¼¾", "Ⅷ Ⅻ", "①②③④",
    "e\u0301 é", "1\u0301234", "12\u030134", "'s123", "x'123", "999 9999 99999", "abc123def456ghi789jkl",
]

# The `tokenizer.json` of `HuggingFaceTB/SmolVLM-256M-Instruct`, as it is in the cache of the tests.
SMOLVLM_TOKENIZER_SHA256 = "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a"

# Strings where the split of the SmolLM2 pre-tokenizer decides the cut: a digit is its own piece and takes no space
# (`Digits` cuts before the GPT-2 regex runs), contractions are lower case only, a space leads one letter run or one
# symbol run, and the white space before a letter, a digit or the end.
GPT2_TEXTS = [
    "1", "12", "123", "a1", "1a", "a1b2", " 1", "  1", "a 12", "x  12", "\n1", "1\n", "'s1", "x'1", "3.14", "-1234",
    "2024-11-26", "$1,000,000", "a 123\n456", "١٢٣", "٣٤٥٦٧", "²³", "½¼", "Ⅷ Ⅻ", "①②", "１２３",
    "don't", "DON'T", "I'LL", "I'll", "we've", "It's", "'S", "'t'd", "x'", "'ſ",
    "  two  spaces", "trailing   ", "a  1", "a   b", "  a", " \n a", "a \n b", "\n\n", " \n", "\t1", "!?1", "!!! 2",
    "...x", " ...x", "(1)", "[2]", "e\u0301 x", "\u0301a", "a\u0301", " \u0301a", "1\u0301", "x \u0301 1",
    "Hello 1234 world", "a\u00a0b", "a\u2003 b", "a\u3000\u3000b", "a\u0085b", "\r\n1", "end. 1", "snake_case_1",
]

WORDPIECE_PAIRS = [("hello", "world"), ("", "world"), ("hello", ""), ("don't", "playing [SEP] x")]

BPE_TEXTS = [
    "hello world",
    " hello",
    "Hello, World!",
    "don't",
    "DON'T",
    "I'LL",
    "we've",
    "it's",
    "'s",
    "'S",
    "x'",
    "'ſ",
    "'ſt'\u212aS",
    "12345",
    "a1b2",
    "3.14",
    "  two  spaces",
    "trailing   ",
    "line\nbreak",
    "\n\n",
    " \n",
    "a \n b",
    "tab\there",
    "!!!\n\n",
    "?!",
    "...",
    "a  \n  b",
    " ",
    "",
    "\r\n",
    "\r",
    "\n \n",
    "x \n",
    "naïve café",
    "é",
    "Å",
    "日本語",
    "emoji 😀 test",
    "<|endoftext|>",
    "a<|im_start|>b",
    "<|im_start|><|im_end|>",
    "x<think>y",
    "<|endoft",
    "한국어",
    "한",
    "ǅ",
    "İ",
    "ß",
    "snake_case_name",
    "a-b",
    "$100",
    "(foo)",
    "[x]",
    " nbsp",
    " em",
    "a b",
    "\x0b",
    "\x85x",
    "word  \n\n  word",
    "end.\n",
    "if x == 1:\n    return y\n",
    "The quick brown fox jumps over the lazy dog.",
    "  \t  ",
    "é" * 5,
    "�",
    "\x00",
] + NATURAL

WORDPIECE_VOCAB = (
    "[PAD] [UNK] [CLS] [SEP] [MASK] . , ! ? ' - $ % hello world the un ##aff ##able play ##ing ##s cafe e i don t "
    "ab ##c a b c d x ##x 中 文 ##e ##a istanbul ##stanbul naive ##ve na ##i o ##l ##co ##ole ##cole em soft so ##ft "
    "[ ] ı ##ı α σ ς ασ ας \U00020000 sp ##aced out tab new line"
).split()

WORDPIECE_TEXTS += NATURAL


def make_wordpiece(directory):
    vocab = {t: i for i, t in enumerate(WORDPIECE_VOCAB)}
    tok = Tokenizer(models.WordPiece(vocab, unk_token="[UNK]", max_input_chars_per_word=100))
    tok.normalizer = normalizers.BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True
    )
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    tok.decoder = decoders.WordPiece(prefix="##", cleanup=True)
    tok.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in WORDPIECE_VOCAB[:5]])
    tok.save(f"{directory}/wordpiece.json")


def make_bpe(directory, name="bpe", split=QWEN_SPLIT, groups=False, gpt2=False):
    tok = Tokenizer(models.BPE())
    if not groups and not gpt2:
        tok.normalizer = normalizers.NFC()
    if gpt2:
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
            ]
        )
    else:
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(split), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
    tok.decoder = decoders.ByteLevel()
    if not gpt2:
        tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=420, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    extra = GROUP_TEXTS if groups else GPT2_TEXTS if gpt2 else []
    corpus = [t for t in BPE_TEXTS + extra if "<" not in t] * 3 + [
        "the quick brown fox jumps over the lazy dog, and the dog does not mind the fox at all.",
        "hello world, hello there, hello again; the world says hello to everyone in the world.",
        "def add(a, b):\n    return a + b\n\nprint(add(1, 2))\n",
        "1234567890 0987654321 42 42 42 100 100 100",
    ] * 20
    tok.train_from_iterator(corpus, trainer)
    tok.add_special_tokens(
        [AddedToken(t, special=True, normalized=False) for t in ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]]
    )
    tok.add_tokens([AddedToken("<think>", special=False, normalized=False)])
    tok.save(f"{directory}/{name}.json")
    if groups:
        with open(f"{directory}/{name}.json", encoding="utf-8") as f:
            written = json.load(f)
        written["model"]["ignore_merges"] = True
        del written["model"]["merges"][-40:]
        with open(f"{directory}/{name}.json", "w", encoding="utf-8") as f:
            json.dump(written, f, ensure_ascii=False, indent=2)
    if name != "bpe":
        return
    # Qwen2.5's tokenizer.json writes a merge as "a b" where Qwen3's writes ["a", "b"]; `tokenizers` reads both.
    with open(f"{directory}/bpe.json", encoding="utf-8") as f:
        written = json.load(f)
    written["model"]["merges"] = [" ".join(pair) if isinstance(pair, list) else pair for pair in written["model"]["merges"]]
    with open(f"{directory}/bpe_lines.json", "w", encoding="utf-8") as f:
        json.dump(written, f, ensure_ascii=False)


# Pieces for the random strings: letters of several scripts, digits, marks,
# punctuation, every kind of white space, symbols, special-token text.
FUZZ = (
    list("abcxyzABCXYZ")
    + list("éÉñßıİ\u0131ǅΣσς中文日本한국어Привет")
    + list("0123456789٣")
    + ["ſ", "\u212a", "\u0301", "\u0323", "\u1161", "e\u0301", "'", "'s", "'LL", "n't"]
    + list(" \t\n\r\u00a0\u2003\u3000\u0085\u2028 ")
    + list(".,!?;:-_()[]{}<>/\\|@#$%^&*+=~`\"")
    + ["😀", "👩\u200d👩", "\ufffd", "\x00", "\x1f", "\u00ad", "<|im_start|>", "<|endoftext|>", "[CLS]", "[MASK]", "##"]
)


def fuzz(count, seed=7):
    rng = random.Random(seed)
    return ["".join(rng.choice(FUZZ) for _ in range(rng.randint(0, 24))) for _ in range(count)]


def golden(kind, path, count=0, extra=()):
    tok = Tokenizer.from_file(path)
    if kind == "wordpiece":
        cases = [(t, None) for t in WORDPIECE_TEXTS + fuzz(count)] + list(WORDPIECE_PAIRS)
    else:
        cases = [(t, None) for t in BPE_TEXTS + list(extra) + fuzz(count)]
    encode = []
    for text, pair in cases:
        for special in ([True, False] if kind == "wordpiece" else [True]):
            e = tok.encode(text, pair, add_special_tokens=special)
            encode.append({"text": text, "pair": pair, "special": special, "ids": e.ids, "type_ids": e.type_ids})
    decode = []
    for case in encode:
        if not case["special"] or not case["ids"]:
            continue
        # The whole ids, and without the last one (a character cut in two).
        for ids in (case["ids"], case["ids"][:-1]):
            for skip in (True, False):
                decode.append({"ids": ids, "skip": skip, "text": tok.decode(ids, skip_special_tokens=skip)})
    return {"encode": encode, "decode": decode}


def main():
    if sys.argv[1] == "marks":
        import tokenizers

        assert tokenizers.__version__ == "0.23.3", tokenizers.__version__
        directory = sys.argv[2]
        make_bpe(directory, "bpe_marks", QWEN35_SPLIT)
        with open(f"{directory}/bpe_marks_golden.json", "w", encoding="utf-8") as f:
            json.dump(golden("bpe", f"{directory}/bpe_marks.json", 60, MARK_TEXTS), f, ensure_ascii=False, indent=None)
            f.write("\n")
    elif sys.argv[1] == "groups":
        import tokenizers

        assert tokenizers.__version__ == "0.23.3", tokenizers.__version__
        directory = sys.argv[2]
        make_bpe(directory, "bpe_groups", LFM3_SPLIT, groups=True)
        with open(f"{directory}/bpe_groups_golden.json", "w", encoding="utf-8") as f:
            json.dump(golden("bpe", f"{directory}/bpe_groups.json", 20, GROUP_TEXTS), f, ensure_ascii=False, indent=None)
            f.write("\n")
    elif sys.argv[1] == "gpt2":
        import tokenizers

        assert tokenizers.__version__ == "0.23.3", tokenizers.__version__
        directory = sys.argv[2]
        make_bpe(directory, "bpe_gpt2", gpt2=True)
        with open(f"{directory}/bpe_gpt2_golden.json", "w", encoding="utf-8") as f:
            json.dump(golden("bpe", f"{directory}/bpe_gpt2.json", 60, GPT2_TEXTS), f, ensure_ascii=False, indent=None)
            f.write("\n")
    elif sys.argv[1] == "fixtures":
        directory = sys.argv[2]
        make_wordpiece(directory)
        make_bpe(directory)
        for kind in ("wordpiece", "bpe"):
            with open(f"{directory}/{kind}_golden.json", "w", encoding="utf-8") as f:
                json.dump(golden(kind, f"{directory}/{kind}.json"), f, ensure_ascii=False, indent=None)
                f.write("\n")
    else:
        kind = "bpe" if sys.argv[2] in ("bpe_groups", "bpe_gpt2") else sys.argv[2]
        pinned = {"bpe_groups": LFM3_TOKENIZER_SHA256, "bpe_gpt2": SMOLVLM_TOKENIZER_SHA256}
        if sys.argv[2] in pinned:
            import hashlib

            import tokenizers

            if tokenizers.__version__ != "0.23.3":
                raise SystemExit(f"tokenizers 0.23.3 is needed, got {tokenizers.__version__}")
            with open(sys.argv[3], "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() != pinned[sys.argv[2]]:
                    raise SystemExit("the file is not the tokenizer.json at the pinned revision")
        extra = {"bpe_groups": GROUP_TEXTS, "bpe_gpt2": GPT2_TEXTS}.get(sys.argv[2], [])
        count = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        json.dump(golden(kind, sys.argv[3], count, extra), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")


main()
