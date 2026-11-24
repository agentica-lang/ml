## `tokenizer/wordpiece.agc` against Hugging Face `tokenizers`: the ids and
## decoded text recorded in `fixtures/wordpiece_golden.json` for the tiny
## tokenizer `fixtures/wordpiece.json` (`tokenizer_golden.py`). Run from
## `transformers/`.

import WordPiece from "../tokenizer/wordpiece"
import Golden from "golden"
import S from "support"

def test_encode_and_decode_agree_with_tokenizers() -> Result[Void]
  checked = try Golden.wordpiece("test/fixtures/wordpiece.json", "test/fixtures/wordpiece_golden.json")
  Test.assert(checked > 100, "only #{checked} cases ran")
end

def test_an_added_token_matched_by_words_or_normalized_text_is_refused() -> Result[Void]
  text = try File.read_text("test/fixtures/wordpiece.json")
  for flag in ["lstrip", "rstrip", "single_word"]
    try Test.error?(WordPiece.parse(try S.replaced(text, "\"#{flag}\": false", "\"#{flag}\": true")))
  end

  Test.error?(WordPiece.parse(try S.replaced(text, "\"normalized\": false", "\"normalized\": true")))
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
  edits = [
    try S.replaced(text, "\"BertNormalizer\"", "\"Lowercase\""),
    try S.replaced(text, "\"BertPreTokenizer\"", "\"Whitespace\""),
    try S.replaced(text, "\"truncation\": null", "\"truncation\": {\"max_length\": 8}"),
    try S.replaced(text, "\"padding\": null", "\"padding\": {\"strategy\": \"BatchLongest\"}"),
    # The second "WordPiece" is the model's, after the decoder's.
    try S.edited(text, "\"WordPiece\"", "\"Unigram\"", 1),
    try S.replaced(text, "\"prefix\": \"##\"", "\"prefix\": \"@@\""),
    # The pair template's second text, and the id of [SEP] in special_tokens.
    try S.edited(text, "\"type_id\": 1", "\"type_id\": 0", 0),
    try S.edited(text, "3\n        ]", "2\n        ]", 0),
  ]

  for wrong in edits
    try Test.error?(WordPiece.parse(wrong))
  end

  try Test.error?(WordPiece.parse("{"))
  tok = try WordPiece.parse(text)
  try Test.error?(WordPiece.decode(tok, [9999]))
  Test.error?(WordPiece.decode(tok, [-1]))
end
