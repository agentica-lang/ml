## `tokenizer/wordpiece.agc` against Hugging Face `tokenizers`: the ids and
## decoded text recorded in `fixtures/wordpiece_golden.json` for the tiny
## tokenizer `fixtures/wordpiece.json` (`tokenizer_golden.py`). Run from
## `transformers/`.

import WordPiece from "../tokenizer/wordpiece"
import Golden from "golden"

def test_encode_and_decode_agree_with_tokenizers() -> Result[Void]
  checked = try Golden.wordpiece("test/fixtures/wordpiece.json", "test/fixtures/wordpiece_golden.json")
  Test.assert(checked > 100, "only #{checked} cases ran")
end

def test_a_pair_has_two_segments() -> Result[Void]
  tok = try WordPiece.open("test/fixtures/wordpiece.json")
  e = try WordPiece.encode(tok, "hello", some("world"))
  try Test.eq(e.ids, [tok.cls, 13, tok.sep, 14, tok.sep])
  try Test.eq(e.type_ids, [0, 0, 0, 1, 1])
  Test.eq(try WordPiece.decode(tok, e.ids, false), "[CLS] hello [SEP] world [SEP]")
end

def test_other_pipelines_are_errors() -> Result[Void]
  text = try File.read_text("test/fixtures/wordpiece.json")
  try Test.ok?(WordPiece.parse(text))
  try Test.error?(WordPiece.parse(text.replace("\"BertNormalizer\"", "\"Lowercase\"")))
  try Test.error?(WordPiece.parse(text.replace("\"BertPreTokenizer\"", "\"Whitespace\"")))
  try Test.error?(WordPiece.parse(text.replace("\"normalized\": false", "\"normalized\": true")))
  try Test.error?(WordPiece.parse("{"))
  tok = try WordPiece.parse(text)
  try Test.error?(WordPiece.decode(tok, [9999]))
  Test.error?(WordPiece.decode(tok, [-1]))
end
