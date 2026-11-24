## `tokenizer/bpe.agc` against Hugging Face `tokenizers`: the ids and decoded
## text recorded in `fixtures/bpe_golden.json` for the tiny tokenizer
## `fixtures/bpe.json` (`tokenizer_golden.py`), with the Qwen split pattern.
## Run from `transformers/`.

import Bpe from "../tokenizer/bpe"
import Unicode from "../tokenizer/unicode"
import Golden from "golden"

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
  try Test.error?(Bpe.parse(text.replace("\"NFC\"", "\"NFD\"")))
  try Test.error?(Bpe.parse(text.replace("\\\\p{L}+|", "\\\\p{L}|")))
  try Test.error?(Bpe.parse(text.replace("\"use_regex\": false", "\"use_regex\": true")))
  try Test.error?(Bpe.parse(text.replace("\"type\": \"BPE\"", "\"type\": \"Unigram\"")))
  tok = try Bpe.parse(text)
  try Test.error?(Bpe.decode(tok, [100000]))
  Test.error?(Bpe.decode(tok, [-1]))
end

private

def pieces(text: String) -> List[String] = Bpe.split(Unicode.codepoints(text)).map { |w| Unicode.text_of(w) }
