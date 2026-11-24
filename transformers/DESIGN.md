# transformers — design

BERT and Qwen3 for inference, from Hugging Face checkpoints, over `tensor/`,
`nn/` and `hub/`. This folder has only the tokenizers so far; the models come
in later PRs of ml#78.

## Tokenizers

`tokenizer/wordpiece.agc` (BERT) and `tokenizer/bpe.agc` (Qwen) read a
`tokenizer.json` and match Hugging Face `tokenizers`: the same ids from
`encode` and the same text from `decode`. They read the file's own flags and
refuse what they do not implement (another normalizer, pre-tokenizer, decoder
or model, an added token with `lstrip`, `rstrip`, `single_word` or
`normalized`), so a different tokenizer is an error, never a quiet difference.

```text
tokenizer/unicode_data.agc   the tables (generated, frozen)
tokenizer/unicode.agc        letter?/number?/mark?/punctuation?/other?/space?, lower, nfd, nfc
tokenizer/added.agc          the added tokens, cut out of the text first (longest first)
tokenizer/source.agc         reading a file in chunks (File.read_text stops at 1 MiB)
tokenizer/wordpiece.agc      BertNormalizer, BertPreTokenizer, WordPiece, [CLS] A [SEP]
tokenizer/bpe.agc            NFC, the Qwen split by hand, byte-level BPE
```

- **The split of Qwen** is the regular expression of its `tokenizer.json`
  (`PATTERN` in `bpe.agc`, compared with the file's) written by hand,
  alternative by alternative, with the backtracking spelled out: Agentica has
  no `\p{L}`. `\s` is White_Space; the categories are Python's.
- **The tables** come from Python's `unicodedata` (15.1) through
  `tools/unicode_tables.py`, standard library only, checked in and frozen like
  other data. agentica#1952 asks for these in the standard library; when it
  lands, `unicode.agc` shrinks to calls and the table goes.
- **Case** is Hugging Face's: lowercase one character at a time, no word-final
  sigma. U+0130 lowercases to two characters.
- **WordPiece decoding** cleans each piece, as `tokenizers` does (a piece
  `" '"` stays apart from `"t"`), not the joined text, and the result is what
  `tokenizers` prints, not what `transformers` prints.

## Tests

- `test/unicode_test.agc` against Python's `unicodedata` (`unicode_peer.py`,
  live), on sample strings.
- `test/wordpiece_test.agc` and `test/bpe_test.agc` against ids recorded from
  `tokenizers` for tiny hand-made tokenizers (`test/fixtures/`), made by
  `test/tokenizer_golden.py fixtures` (manual: it needs `tokenizers`).
- `tools/tokenizer_check.agc` against the real `bert-base-uncased` and
  `Qwen3-0.6B` files, from a golden of `test/tokenizer_golden.py golden`, with
  `N` random strings (seeded). It matched on 48312 cases (BERT) and 39149
  (Qwen) with `N = 8000`, and has to match on a new run before a change to the
  tokenizers merges.
