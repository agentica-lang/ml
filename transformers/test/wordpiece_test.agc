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

## The `truncation` of a `tokenizer.json`, as `tokenizers` writes it.
def truncation(max_length: Int, strategy: String, direction: String, stride: Int) -> String
  fields = [
    "\"direction\": \"#{direction}\"",
    "\"max_length\": #{max_length}",
    "\"strategy\": \"#{strategy}\"",
    "\"stride\": #{stride}",
  ]

  "\"truncation\": {#{fields.join(", ")}}"
end

## A file that truncates and pads to the longest of a batch (sentence-transformers' checkpoints): an encoding of one
## text has `max_length` ids at most, the special tokens counted, and the padding changes nothing.
def test_a_fixed_padding_of_the_file_changes_nothing() -> Result[Void]
  text = try File.read_text("test/fixtures/wordpiece.json")
  fixed = "\"padding\": {\"strategy\": {\"Fixed\": 128}, \"direction\": \"Right\", \"pad_id\": 0}"
  tok = try WordPiece.parse(try S.replaced(text, "\"padding\": null", fixed))
  plain = try WordPiece.parse(text)
  try Test.eq(tok.pad_id, some(0))
  Test.eq(try WordPiece.encode(tok, "hello world"), try WordPiece.encode(plain, "hello world"))
end

def test_the_truncation_of_the_file_cuts_one_text() -> Result[Void]
  text = try File.read_text("test/fixtures/wordpiece.json")
  limited = try S.replaced(text, "\"truncation\": null", truncation(6, "LongestFirst", "Right", 0))
  padding = "\"padding\": {\"strategy\": \"BatchLongest\", \"pad_to_multiple_of\": null, \"pad_id\": 0}"
  tok = try WordPiece.parse(try S.replaced(limited, "\"padding\": null", padding))
  try Test.eq(tok.max_length, some(6))
  try Test.eq(tok.pad_id, some(0))
  long = "hello world hello world hello"
  cut = try WordPiece.encode(tok, long)
  try Test.eq(cut.ids, [tok.cls, 13, 14, 13, 14, tok.sep])
  try Test.eq(cut.type_ids, [0, 0, 0, 0, 0, 0])
  # `unaffable` is `un ##aff ##able`: the cut falls inside the word.
  try Test.eq((try WordPiece.encode(tok, "unaffable", none, true, some(4))).ids, [tok.cls, 16, 17, tok.sep])
  try Test.eq((try WordPiece.encode(tok, "hello world")).ids, [tok.cls, 13, 14, tok.sep])
  try Test.eq((try WordPiece.encode(tok, long, none, false)).ids, [13, 14, 13, 14, 13])
  try Test.eq((try WordPiece.encode(tok, long, none, true, some(3))).ids, [tok.cls, 13, tok.sep])
  try Test.eq((try WordPiece.encode(tok, "hello", some("world"))).ids.size, 5)
  try Test.error?(WordPiece.encode(tok, "hello world", some("hello world")))
  try Test.error?(WordPiece.encode(tok, long, none, true, some(2)))
  plain = try WordPiece.parse(text)
  try Test.eq(plain.max_length, none)
  try Test.eq(plain.pad_id, none)
  try Test.eq((try WordPiece.encode(plain, long)).ids.size, 7)
  Test.eq((try WordPiece.encode(plain, long, none, true, some(4))).ids, [tok.cls, 13, 14, tok.sep])
end

def test_other_pipelines_are_errors() -> Result[Void]
  text = try File.read_text("test/fixtures/wordpiece.json")
  try Test.ok?(WordPiece.parse(text))
  edits = [
    try S.replaced(text, "\"BertNormalizer\"", "\"Lowercase\""),
    try S.replaced(text, "\"BertPreTokenizer\"", "\"Whitespace\""),
    try S.replaced(text, "\"truncation\": null", truncation(8, "OnlyFirst", "Right", 0)),
    try S.replaced(text, "\"truncation\": null", truncation(8, "LongestFirst", "Left", 0)),
    try S.replaced(text, "\"truncation\": null", truncation(8, "LongestFirst", "Right", 2)),
    try S.replaced(
      text,
      "\"padding\": null",
      "\"padding\": {\"strategy\": {\"Fixed\": 8, \"Other\": 1}, \"pad_id\": 0}",
    ),
    try S.replaced(text, "\"padding\": null", "\"padding\": {\"strategy\": {\"Other\": 8}, \"pad_id\": 0}"),
    try S.replaced(
      text,
      "\"padding\": null",
      "\"padding\": {\"strategy\": \"BatchLongest\", \"pad_to_multiple_of\": 8}",
    ),
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
