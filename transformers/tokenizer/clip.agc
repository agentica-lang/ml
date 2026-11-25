## CLIP's tokenizer, from the `vocab.json` and `merges.txt` of a Hugging Face checkpoint, as the slow `CLIPTokenizer`
## of `transformers` runs without `ftfy`: the cleaning of its `BasicTokenizer` (control characters dropped, white space
## made one space, a space around each CJK ideograph, NFC and lowercase; accents are kept and punctuation is not split
## off), CLIP's split of the text (the two special tokens, the endings `'s 't 're 've 'm 'll 'd`, a run of letters, one
## digit, a run of anything else), the UTF-8 bytes of each piece in GPT-2's byte alphabet, and BPE with the word ending
## `</w>` on the last byte. A symbol the vocabulary lacks is the unknown token, which is `<|endoftext|>`.
##
## Case is Python's `str.lower`: one character at a time (`Unicode.lower`), and a capital sigma at the end of a word
## (after a letter, before no letter, combining marks skipped) is the final sigma. Python also skips the other
## case-ignorable characters (an apostrophe, a full stop) there, which this does not.

import Added from "added"
import Bpe from "bpe"
import Source from "source"
import Unicode from "unicode"
import WordPiece from "wordpiece"

## The `</w>` that ends the last symbol of a word.
WORD_END = "</w>"
BOS = "<|startoftext|>"
EOS = "<|endoftext|>"
## The merges that CLIP's vocabulary of 49408 tokens has: the lines after the header, up to this many.
MAX_MERGES = 48894
SIGMA = 0x3A3
FINAL_SIGMA = 0x3C2

struct Clip
  vocab: Map[String, Int]
  ranks: Map[String, Int]
  bos: Int
  eos: Int
end

## The tokenizer of the folder `dir`, which holds `vocab.json` and `merges.txt`.
def open(dir: String) -> Result[Clip]
  vocab_path = "#{dir}/vocab.json"
  merges_path = "#{dir}/merges.txt"
  vocab = try Source.within(
    vocab_path,
    JSON.decode(try Source.within(vocab_path, Source.read_text(vocab_path)), Map[String, Int]),
  )

  merges = try Source.within(merges_path, Source.read_text(merges_path))
  Source.within(dir, from_parts(vocab, merges))
end

## The tokenizer of the text of a `vocab.json` and of a `merges.txt`.
def from_parts(vocab: Map[String, Int], merges: String) -> Result[Clip]
  bos = try vocab.get(BOS).to_result("the vocabulary has no #{BOS}")
  eos = try vocab.get(EOS).to_result("the vocabulary has no #{EOS}")
  lines = merges.strip.split("\n")
  ranks: Map[String, Int] = {}
  for i in 1...[lines.size, MAX_MERGES + 1].min.unwrap!
    pair = lines[i].split(" ").select { |part| part != "" }
    return error("merge #{i} is not a pair of tokens") if pair.size != 2
    ranks = ranks.put(pair[0] + " " + pair[1], i - 1)
  end

  ok(Clip(vocab: vocab, ranks: ranks, bos: bos, eos: eos))
end

## The ids of `text` as `tokenizer(text, truncation=True, max_length=max_length)` gives them: the start token, the
## tokens of the text cut to `max_length - 2`, and the end token. A special token in the text is one id.
def encode(tok: Clip, text: String, max_length: Int = 77) -> Result[List[Int]]
  return error("max_length needs room for the start and the end token, got #{max_length}") if max_length < 2
  body: List[Int] = []
  for part in Added.split(specials(tok), Unicode.codepoints(text))
    case part
      Added.Part.Token(id) -> body = body.push(id)
      Added.Part.Text(chars) ->
        for piece in pieces(tok, normalized(chars))
          case piece
            Added.Part.Token(id) -> body = body.push(id)
            Added.Part.Text(word) -> body = body + word_ids(tok, word)
          end
        end
    end
  end

  ok([tok.bos] + body.take(max_length - 2) + [tok.eos])
end

private

def specials(tok: Clip) -> List[Added.Added] = [Added.of(tok.bos, BOS, true), Added.of(tok.eos, EOS, true)]

## `BasicTokenizer`: clean, space the CJK ideographs, NFC, then each white-space-separated word in lowercase, joined
## by one space.
def normalized(chars: List[Int]) -> List[Int]
  words: List[List[Int]] = []
  word: List[Int] = []
  for c in Unicode.nfc(WordPiece.spaced_chinese(WordPiece.cleaned_chars(chars)))
    if Unicode.space?(c)
      if !word.empty?
        words = words.push(lowered(word))
        word = []
      end
    else
      word = word.push(c)
    end
  end

  if !word.empty?
    words = words.push(lowered(word))
  end

  out: List[Int] = []
  for i in 0...words.size
    if i > 0
      out = out.push(32)
    end

    out = Unicode.append(out, words[i])
  end

  out
end

## `word` in lowercase.
def lowered(word: List[Int]) -> List[Int]
  out: List[Int] = []
  for i in 0...word.size
    if word[i] == SIGMA and final_sigma?(word, i)
      out = out.push(FINAL_SIGMA)
    else
      out = Unicode.append(out, Unicode.lower([word[i]]))
    end
  end

  out
end

## Whether the capital sigma at `at` ends a word: a letter before it and none after it, combining marks skipped.
def final_sigma?(word: List[Int], at: Int) -> Bool
  before = at - 1
  while before >= 0 and Unicode.mark?(word[before])
    before -= 1
  end

  after = at + 1
  while after < word.size and Unicode.mark?(word[after])
    after += 1
  end

  before >= 0 and Unicode.letter?(word[before]) and (after >= word.size or !Unicode.letter?(word[after]))
end

## The pieces of `chars` that CLIP's pattern cuts, in its order of alternatives: `<|startoftext|>` and `<|endoftext|>`
## (one id each), the endings of a contraction, a run of letters, one digit, a run of anything but white space, letters
## and digits. The spaces between them are dropped.
def pieces(tok: Clip, chars: List[Int]) -> List[Added.Part]
  out: List[Added.Part] = []
  tokens = specials(tok)
  at = 0
  while at < chars.size
    c = chars[at]
    size = 1
    if Unicode.space?(c)
      at += 1
      next
    end

    special = tokens.find { |s| starts?(chars, at, s.chars) }
    case special
      ok(s) ->
        out = out.push(Added.Part.Token(s.id))
        at += s.chars.size
        next
      error(_) -> nil
    end

    size = contraction(chars, at)
    if size == 0 and Unicode.letter?(c)
      size = run(chars, at, ->(x: Int) { Unicode.letter?(x) })
    elsif size == 0 and Unicode.number?(c)
      size = 1
    elsif size == 0
      size = run(chars, at, ->(x: Int) { !Unicode.space?(x) and !Unicode.letter?(x) and !Unicode.number?(x) })
    end

    out = out.push(Added.Part.Text(Unicode.span(chars, at, at + size)))
    at += size
  end

  out
end

def starts?(chars: List[Int], at: Int, prefix: List[Int]) -> Bool
  return false if at + prefix.size > chars.size
  for i in 0...prefix.size
    return false if chars[at + i] != prefix[i]
  end

  true
end

## The length of the run that starts at `at` and holds only characters of `kind`.
def run(chars: List[Int], at: Int, kind: Fn[Int, Bool]) -> Int
  stop = at
  while stop < chars.size and kind.call(chars[stop])
    stop += 1
  end

  stop - at
end

## The length of the ending `'s`, `'t`, `'m`, `'d`, `'re`, `'ve` or `'ll` that starts at `at`, or 0.
def contraction(chars: List[Int], at: Int) -> Int
  return 0 if chars[at] != 39 or at + 1 >= chars.size
  a = chars[at + 1]
  return 2 if a == 115 or a == 116 or a == 109 or a == 100
  return 0 if at + 2 >= chars.size
  b = chars[at + 2]
  return 3 if (a == 114 and b == 101) or (a == 118 and b == 101) or (a == 108 and b == 108)
  0
end

## The ids of one piece: its bytes as symbols, the last with `</w>`, merged by rank.
def word_ids(tok: Clip, word: List[Int]) -> List[Int]
  symbols = Bpe.byte_chars(word)
  symbols = symbols.set(symbols.size - 1, symbols.last + WORD_END)
  Bpe.merge(tok.ranks, symbols).map { |s| tok.vocab.get(s).unwrap_or(tok.eos) }
end
