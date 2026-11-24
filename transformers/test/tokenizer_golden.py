"""Makes the tokenizer fixtures and goldens of `transformers/` with Hugging Face
`tokenizers`: the ids that its `encode` gives and the text that its `decode`
gives, for `tokenizer/wordpiece.agc` and `tokenizer/bpe.agc` to match.

Manual, not run by the tests: it needs `tokenizers` (0.22.2 was used). Run it
in a throwaway venv, from `transformers/`, with Python's isolated mode.

    python -I test/tokenizer_golden.py fixtures test/fixtures

writes `wordpiece.json` and `bpe.json` (tiny hand-made tokenizers, in
`tokenizer.json` form) and the goldens `wordpiece_golden.json` and
`bpe_golden.json` (the ids of the strings below and their decoded text).

    python -I test/tokenizer_golden.py golden wordpiece TOKENIZER_JSON [N] > OUT.json
    python -I test/tokenizer_golden.py golden bpe TOKENIZER_JSON [N] > OUT.json

writes the golden of a real tokenizer (`bert-base-uncased`, `Qwen3-0.6B`),
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


def make_bpe(directory):
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=420, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False
    )
    corpus = [t for t in BPE_TEXTS if "<" not in t] * 3 + [
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
    tok.save(f"{directory}/bpe.json")


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


def golden(kind, path, count=0):
    tok = Tokenizer.from_file(path)
    if kind == "wordpiece":
        cases = [(t, None) for t in WORDPIECE_TEXTS + fuzz(count)] + list(WORDPIECE_PAIRS)
    else:
        cases = [(t, None) for t in BPE_TEXTS + fuzz(count)]
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
    if sys.argv[1] == "fixtures":
        directory = sys.argv[2]
        make_wordpiece(directory)
        make_bpe(directory)
        for kind in ("wordpiece", "bpe"):
            with open(f"{directory}/{kind}_golden.json", "w", encoding="utf-8") as f:
                json.dump(golden(kind, f"{directory}/{kind}.json"), f, ensure_ascii=False, indent=None)
                f.write("\n")
    else:
        json.dump(golden(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0), sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")


main()
