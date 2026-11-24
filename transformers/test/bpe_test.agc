## `tokenizer/bpe.agc` against Hugging Face `tokenizers`: the ids and decoded
## text recorded in `fixtures/bpe_golden.json` for the tiny tokenizer
## `fixtures/bpe.json` (`tokenizer_golden.py`), with the Qwen split pattern.
## Run from `transformers/`.

import Bpe from "../tokenizer/bpe"
import Unicode from "../tokenizer/unicode"
import Golden from "golden"
import S from "support"

FIXTURE = "test/fixtures/bpe.json"

def test_encode_and_decode_agree_with_tokenizers() -> Result[Void]
  checked = try Golden.bpe(FIXTURE, "test/fixtures/bpe_golden.json")
  Test.assert(checked > 100, "only #{checked} cases ran")
end

def test_the_split_follows_the_pattern() -> Result[Void]
  try Test.eq(pieces("Hello, world!"), ["Hello", ",", " world", "!"])
  try Test.eq(pieces("it's DON'T"), ["it", "'s", " DON", "'T"])
  try Test.eq(pieces("a 12\n\n b"), ["a", " ", "1", "2", "\n\n", " b"])
  Test.eq(pieces("x  y \n"), ["x", " ", " y", " \n"])
end

def test_other_pipelines_are_errors() -> Result[Void]
  text = try File.read_text(FIXTURE)
  try Test.ok?(Bpe.parse(text))
  edits = [
    try S.replaced(text, "\"NFC\"", "\"NFD\""),
    try S.replaced(text, "\\\\p{L}+|", "\\\\p{L}|"),
    try S.edited(text, "\"use_regex\": false", "\"use_regex\": true", 0),
    try S.replaced(text, "\"type\": \"BPE\"", "\"type\": \"Unigram\""),
    try S.replaced(text, "\"truncation\": null", "\"truncation\": {\"max_length\": 8}"),
    try S.replaced(text, "\"padding\": null", "\"padding\": {\"strategy\": \"BatchLongest\"}"),
    try S.replaced(text, "\"dropout\": null", "\"dropout\": 0.1"),
    # The second "ByteLevel" is the post-processor's, between the split's and the decoder's.
    try S.edited(text, "\"ByteLevel\"", "\"TemplateProcessing\"", 1),
    try S.edited(text, "\"ByteLevel\"", "\"Metaspace\"", 2),
  ]

  for wrong in edits
    try Test.error?(Bpe.parse(wrong))
  end

  tok = try Bpe.parse(text)
  try Test.error?(Bpe.decode(tok, [100000]))
  Test.error?(Bpe.decode(tok, [-1]))
end

def test_an_added_token_matched_by_words_or_normalized_text_is_refused() -> Result[Void]
  text = try File.read_text(FIXTURE)
  for flag in ["lstrip", "rstrip", "single_word"]
    try Test.error?(Bpe.parse(try S.replaced(text, "\"#{flag}\": false", "\"#{flag}\": true")))
  end

  Test.error?(Bpe.parse(try S.replaced(text, "\"normalized\": false", "\"normalized\": true")))
end

private

def pieces(text: String) -> List[String] = Bpe.split(Unicode.codepoints(text)).map { |w| Unicode.text_of(w) }
