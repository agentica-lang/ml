## A tokenizer against a golden made by Hugging Face `tokenizers`
## (`tokenizer_golden.py`): the ids of `encode` and the text of `decode`. Used
## by the tests, on the fixtures, and by `tools/tokenizer_check.agc`, on the
## real tokenizers.

import WordPiece from "../tokenizer/wordpiece"
import Bpe from "../tokenizer/bpe"
import Source from "../tokenizer/source"

struct Encode
  text: String
  pair: Optional[String]
  special: Bool
  ids: List[Int]
  type_ids: List[Int]
end

struct Decode
  ids: List[Int]
  skip: Bool
  text: String
end

struct Golden
  encode: List[Encode]
  decode: List[Decode]
end

## The golden in the file at `path`.
def read(path: String) -> Result[Golden] = JSON.decode(try Source.read_text(path), Golden)

## Checks the WordPiece tokenizer of `tokenizer_path` against the golden at
## `golden_path`: how many cases agree, or the first that does not.
def wordpiece(tokenizer_path: String, golden_path: String) -> Result[Int]
  tok = try WordPiece.open(tokenizer_path)
  golden = try read(golden_path)
  for c in golden.encode
    got = try WordPiece.encode(tok, c.text, c.pair, c.special)
    try agree(
      "encode #{c.text.inspect} #{c.pair} #{c.special}",
      "#{[got.ids, got.type_ids]}",
      "#{[c.ids, c.type_ids]}",
    )
  end

  for c in golden.decode
    try agree("decode #{c.ids} #{c.skip}", try WordPiece.decode(tok, c.ids, c.skip), c.text)
  end

  ok(golden.encode.size + golden.decode.size)
end

## The same for a byte-level BPE tokenizer.
def bpe(tokenizer_path: String, golden_path: String) -> Result[Int] = bpe_of(try Bpe.open(tokenizer_path), golden_path)

## The same goldens for a tokenizer that is already loaded.
def bpe_of(tok: Bpe.ByteLevelBpe, golden_path: String) -> Result[Int]
  golden = try read(golden_path)
  for c in golden.encode
    try agree("encode #{c.text.inspect}", "#{try Bpe.encode(tok, c.text)}", "#{c.ids}")
  end

  for c in golden.decode
    try agree("decode #{c.ids} #{c.skip}", try Bpe.decode(tok, c.ids, c.skip), c.text)
  end

  ok(golden.encode.size + golden.decode.size)
end

private

def agree(what: String, got: String, want: String) -> Result[Void] = Source.within(what, Test.eq(got, want))
