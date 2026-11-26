---
title: tokenizers
description: The WordPiece, byte-level BPE and CLIP tokenizers, the tokenizer example and the Unicode tables they use.
weight: 3
---

## Tokenizer example

Encode and decode with `bert-base-uncased` and with `Qwen/Qwen3-0.6B`. Each folder holds the `tokenizer.json` of the model:

```agentica
import WordPiece from "pkg/transformers/tokenizer/wordpiece"
import Bpe from "pkg/transformers/tokenizer/bpe"

def main() -> Void
  bert = WordPiece.open("bert-base-uncased/tokenizer.json").unwrap!
  encoded = WordPiece.encode(bert, "Hello, world!").unwrap!
  puts(encoded.ids)
  puts(WordPiece.decode(bert, encoded.ids).unwrap!)

  qwen = Bpe.open("Qwen3-0.6B/tokenizer.json").unwrap!
  ids = Bpe.encode(qwen, "<|im_start|>user\nHello, world!<|im_end|>").unwrap!
  puts(ids)
  puts(Bpe.decode(qwen, ids, false).unwrap!)
end
```

```output
[101, 7592, 1010, 2088, 999, 102]
hello, world!
[151644, 872, 198, 9707, 11, 1879, 0, 151645]
<|im_start|>user
Hello, world!<|im_end|>
```

## WordPiece

`WordPiece.open(path)` reads a BERT tokenizer: the BERT normalizer (control characters dropped, spaces around CJK ideographs, accents stripped, lowercase), the BERT pre-tokenizer (white space and punctuation) and a WordPiece model with the `[CLS] A [SEP]` template.

- `WordPiece.parse(text)` reads the tokenizer from the text of a `tokenizer.json`, and `WordPiece.open(path)` from its file.
- `WordPiece.encode(tok, text, pair, add_special, max_length)` gives `Encoding(ids, type_ids)`. `pair` is `none` or `some(text)`, and `add_special` (default `true`) adds `[CLS]` and `[SEP]`. Added tokens in the text (`[MASK]`) are one id each.
- With `max_length: some(n)`, or else the `truncation` of the file (`tok.max_length`), an encoding of one text has `n` ids at most, `[CLS]` and `[SEP]` counted: the tokens past it are dropped, as `tokenizers` does. A pair over the limit is an error. Only the plain truncation is read (cut at the right, `LongestFirst`, no stride). A limit that leaves no room for a token is an error, and a call can replace the file's limit but not turn it off. A file that pads to the longest of a batch is accepted, and `encode` of one text adds no padding (`Bert.run` pads a batch); a fixed-length padding and `pad_to_multiple_of` above 1, which would pad one text too, are errors. `tok.pad_id` is the `pad_id` of the file's padding, none when it has none.
- `WordPiece.decode(tok, ids, skip_special)` joins the pieces back, with the spaces before punctuation cleaned as `tokenizers` does. `skip_special` is `true` by default.
- A word longer than the file's `max_input_chars_per_word`, or one the vocabulary cannot cover, is `[UNK]`.

## Byte-level BPE

`Bpe.open(path)` reads a Qwen tokenizer: NFC normalization, the Qwen split pattern (that of Qwen3 and Qwen2.5, or that of Qwen3.5, which counts the marks `\p{M}` as letters), bytes mapped to the byte-level alphabet, and BPE merges. A file with another pattern is an error. The tokenizer of `Qwen/Qwen3.5-0.8B` (248,070 tokens; the embedding has 248,320 rows) gives Hugging Face's ids on 1,827 encode and decode cases.

- `Bpe.parse(text)` reads the tokenizer from the text of a `tokenizer.json`, and `Bpe.open(path)` from its file.
- `Bpe.encode(tok, text)` gives the ids. It adds no tokens; added tokens in the text (`<|im_start|>`) are one id each.
- `Bpe.decode(tok, ids, skip_special)` gives the text. A character split across tokens joins back, and a broken byte sequence becomes U+FFFD.

## CLIP BPE

`Clip.open(dir)` of `pkg/transformers/tokenizer/clip` reads `vocab.json` and `merges.txt` of a CLIP checkpoint and tokenizes as the slow `CLIPTokenizer` of Hugging Face does without `ftfy`: control characters dropped, white space made one space, a space around each CJK ideograph, NFC, lowercase (accents are kept), then CLIP's split (the two special tokens, the endings `'s 't 're 've 'm 'll 'd`, a run of letters, one digit, a run of anything else), the UTF-8 bytes of each piece in GPT-2's alphabet with `</w>` on the last one, and the BPE merges. `Clip.encode(tok, text, max_length)` gives the ids with the start and end tokens around them, the text cut to `max_length - 2` tokens (77 by default); `<|startoftext|>` and `<|endoftext|>` in the text are one id each. A symbol the vocabulary lacks is `<|endoftext|>`. The fast tokenizer (`tokenizer.json`) differs from the slow one on control characters and CJK ideographs; the slow one is what this follows. Python's lowercase also skips an apostrophe or a full stop when it decides that a sigma ends a word, which this does not.

## Unicode

Agentica has no Unicode properties, so the tokenizers carry their own tables for Unicode 15.1 (`tokenizer/unicode_data.agc`, written by `tools/unicode_tables.py`): letters, numbers, marks, punctuation, white space, lowercase, and NFD and NFC. `tokenizers` has tables of its own, so a character that a newer Unicode version classifies differently can split differently; U+1FAE0 and U+2EBF0 give the same ids. The case-insensitive contractions (`'S`, `'RE`) are ASCII only, which agrees with `tokenizers` on `'ſ`.
