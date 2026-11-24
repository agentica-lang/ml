---
title: transformers
description: Hugging Face tokenizers in Agentica, from a tokenizer.json to ids and back, for BERT and Qwen.
weight: 7
---

`transformers` will hold BERT and Qwen3 in Agentica. For now it has their tokenizers: it reads the `tokenizer.json` that sits beside a Hugging Face checkpoint and gives the same ids as Hugging Face `tokenizers`.

## Example

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

- `WordPiece.encode(tok, text, pair, add_special)` gives `Encoding(ids, type_ids)`. `pair` is `none` or `some(text)`, and `add_special` (default `true`) adds `[CLS]` and `[SEP]`. Added tokens in the text (`[MASK]`) are one id each.
- `WordPiece.decode(tok, ids, skip_special)` joins the pieces back, with the spaces before punctuation cleaned as `tokenizers` does. `skip_special` is `true` by default.
- A word longer than the file's `max_input_chars_per_word`, or one the vocabulary cannot cover, is `[UNK]`.

## Byte-level BPE

`Bpe.open(path)` reads a Qwen tokenizer: NFC normalization, the Qwen split pattern, bytes mapped to the byte-level alphabet, and BPE merges. A file with another pattern is an error.

- `Bpe.encode(tok, text)` gives the ids. It adds no tokens; added tokens in the text (`<|im_start|>`) are one id each.
- `Bpe.decode(tok, ids, skip_special)` gives the text. A character split across tokens joins back, and a broken byte sequence becomes U+FFFD.

## Errors

A file with another normalizer, pre-tokenizer, post-processor or template, decoder or model, with `truncation`, `padding` or BPE `dropout`, or with an added token that matches normalized text or by words, is an error from `open` and `parse`, not a silent difference. Ids out of range are an error from `decode`.

## Unicode

Agentica has no Unicode properties, so the tokenizers carry their own tables for Unicode 15.1 (`tokenizer/unicode_data.agc`, written by `tools/unicode_tables.py`): letters, numbers, marks, punctuation, white space, lowercase, and NFD and NFC. `tokenizers` has tables of its own, so a character that a newer Unicode version classifies differently can split differently; U+1FAE0 and U+2EBF0 give the same ids. The case-insensitive contractions (`'S`, `'RE`) are ASCII only, which agrees with `tokenizers` on `'ſ`.
