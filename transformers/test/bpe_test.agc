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

## Qwen2.5's `tokenizer.json` writes a merge as "a b", Qwen3's as ["a", "b"]: both give the ids of `tokenizers`.
def test_merges_written_as_lines_are_read_too() -> Result[Void]
  checked = try Golden.bpe("test/fixtures/bpe_lines.json", "test/fixtures/bpe_golden.json")
  Test.assert(checked > 100, "only #{checked} cases ran")
end

## A broken file in the "a b" lines form is reported against that form.
def test_a_broken_lines_file_reports_the_lines_form() -> Result[Void]
  text = try File.read_text("test/fixtures/bpe_lines.json")
  case Bpe.parse(try S.replaced(text, "\"merges\": [\"h e\", ", "\"merges\": [\"h e\", 7, "))
    ok(_) -> error("a merge that is a number was accepted")
    error(e) -> Test.assert(e.message.include?("model.merges[1]: expected String"), e.message)
  end
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

## The queue-and-linked-list merge against the plain loop (the lowest rank
## anywhere, joined everywhere it stands, until no pair is a merge) on words
## made of pieces of the trained vocabulary, so that many merges fire and
## overlap. Lowercase letters are their own byte-level characters.
def test_the_merge_queue_agrees_with_the_plain_loop() -> Result[Void]
  tok = try Bpe.open(FIXTURE)
  pieces = tok.vocab.keys.select { |k| k.size >= 2 and k.split("").all? { |c| c.code >= 97 and c.code <= 122 } }
  try Test.assert(pieces.size > 40, "only #{pieces.size} pieces to build words from")
  seed = 12345
  checked = 0
  for _ in 0...300
    word = ""
    seed = (seed * 1103515245 + 12345) % 2147483648
    count = 1 + (seed / 65536) % 6
    for _ in 0...count
      seed = (seed * 1103515245 + 12345) % 2147483648
      word = word + pieces[(seed / 65536) % pieces.size]
    end

    want = plain_merge(tok, word.split(""))
    got = try Bpe.encode(tok, word)
    try Test.eq(got, try ids_of(tok, want))
    if want.size < word.size
      checked += 1
    end
  end

  try Test.assert(checked > 250, "only #{checked} words had a merge")
  # A run of one letter whose pair is a merge: overlapping pairs, joined left to right.
  doubled = tok.ranks.keys.select { |k| same_letters?(k) }
  try Test.assert(not doubled.empty?, "the fixture has no merge of a letter with itself")
  for pair in doubled
    for n in 2...13
      word = pair.split("")[0] * n
      try Test.eq(try Bpe.encode(tok, word), try ids_of(tok, plain_merge(tok, word.split(""))))
    end
  end

  ok()
end

private

def pieces(text: String) -> List[String] = Bpe.split(Unicode.codepoints(text)).map { |w| Unicode.text_of(w) }

def ids_of(tok: Bpe.ByteLevelBpe, symbols: List[String]) -> Result[List[Int]]
  out: List[Int] = []
  for sym in symbols
    out = out.push(try tok.vocab.get(sym).to_result("no id for #{sym}"))
  end

  ok(out)
end

def plain_merge(tok: Bpe.ByteLevelBpe, symbols: List[String]) -> List[String]
  current = symbols
  while current.size > 1
    best = -1
    for i in 0...(current.size - 1)
      rank = tok.ranks.get(current[i] + " " + current[i + 1]).unwrap_or(-1)
      if rank >= 0 and (best < 0 or rank < best)
        best = rank
      end
    end

    break if best < 0
    out: List[String] = []
    i = 0
    while i < current.size
      if i + 1 < current.size and tok.ranks.get(current[i] + " " + current[i + 1]).unwrap_or(-1) == best
        out = out.push(current[i] + current[i + 1])
        i += 2
      else
        out = out.push(current[i])
        i += 1
      end
    end

    current = out
  end

  current
end

## A merge key `x x` of one lowercase letter twice.
def same_letters?(key: String) -> Bool
  parts = key.split(" ")
  parts.size == 2 and parts[0] == parts[1] and parts[0].size == 1 and parts[0].code >= 97 and parts[0].code <= 122
end
