## Long inputs take time in the size of the input, not its square: 200k
## characters of mixed text and 50k CJK characters through the Unicode
## functions and both tokenizers, 20k letters in one BPE piece (base64 in a
## code prompt), each under a time bound that a quadratic loop misses by far
## (they run in about a second). They spend 1.33 G steps (580 M for the heaviest),
## far over a test file's usual budget, so they live in `long/` and CI runs
## them in a step of their own with `--max-steps 400000000`. Run from
## `transformers/`: `agentica test --max-steps 400000000 long/`.

import Bpe from "../tokenizer/bpe"
import Unicode from "../tokenizer/unicode"
import WordPiece from "../tokenizer/wordpiece"

## The bound of each test, in milliseconds.
LIMIT_MS = 10000

def test_normalization_of_200k_characters_is_linear() -> Result[Void]
  chars = Unicode.codepoints(mixed(7000))
  try Test.assert(chars.size > 200000, "only #{chars.size} characters")
  started = Time.monotonic_ms
  composed = Unicode.nfc(Unicode.nfd(chars))
  lowered = Unicode.lower(chars)
  try within(started, "normalization")
  try Test.eq(composed, chars)
  Test.eq(lowered.size, chars.size)
end

def test_wordpiece_of_200k_characters_and_50k_cjk_is_linear() -> Result[Void]
  tok = try WordPiece.open("test/fixtures/wordpiece.json")
  started = Time.monotonic_ms
  text = try WordPiece.encode(tok, mixed(7000))
  try within(started, "mixed text")
  try Test.assert(text.ids.size > 50000, "only #{text.ids.size} ids")
  cjk = (0...50000).to_a.map { |i| String.from_code(0x4E00 + i % 2000) }.join("")
  started = Time.monotonic_ms
  ideographs = try WordPiece.encode(tok, cjk)
  try within(started, "cjk")
  Test.eq(ideographs.ids.size, 50002)
end

def test_bpe_of_200k_characters_and_a_20k_letter_piece_is_linear() -> Result[Void]
  tok = try Bpe.open("test/fixtures/bpe.json")
  started = Time.monotonic_ms
  text = try Bpe.encode(tok, mixed(7000))
  try within(started, "mixed text")
  try Test.assert(text.size > 50000, "only #{text.size} ids")
  letters = (0...20000).to_a.map { |i| String.from_code(97 + (i * 7 + i / 3) % 26) }.join("")
  started = Time.monotonic_ms
  piece = try Bpe.encode(tok, letters)
  try within(started, "20k letters")
  Test.eq(try Bpe.decode(tok, piece), letters)
end

private

## Lines of ASCII, accents, a digit run, a contraction and blank lines.
def mixed(lines: Int) -> String = (0...lines).to_a.map { |i| "Hello, wörld #{i}! don't  go\n\n" }.join("")

def within(started: Int, what: String) -> Result[Void]
  took = Time.monotonic_ms - started
  return error("#{what} took #{took} ms, over #{LIMIT_MS}") if took > LIMIT_MS
  ok()
end
