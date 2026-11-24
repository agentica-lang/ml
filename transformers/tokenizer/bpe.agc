## Qwen's tokenizer: the `tokenizer.json` of a byte-level BPE model with an NFC
## normalizer and the GPT-4 style split (apostrophe endings, one digit at a
## time, a run of letters with one lead character, punctuation, new lines,
## white space), which `Qwen/Qwen3-0.6B` ships. A file with another
## normalizer, split pattern or model is an error, not a silent difference.
##
## The split is Hugging Face's regular expression written by hand over
## `unicode.agc`'s tables (`PATTERN`, with `\s` as White_Space and `\p{L}`,
## `\p{N}` as the L and N categories), alternative by alternative in its order.

import Unicode from "unicode"
import Added from "added"
import Source from "source"

## The split pattern of the Qwen family.
PATTERN = (
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)

struct ByteLevelBpe
  vocab: Map[String, Int]
  ## The byte-level text of each id, "" where the vocabulary has none.
  pieces: List[String]
  ## The rank of the merge of `left` and `right`, keyed `left + " " + right`.
  ranks: Map[String, Int]
  added: List[Added.Added]
  nfc: Bool
end

## The tokenizer in the `tokenizer.json` at `path`.
def open(path: String) -> Result[ByteLevelBpe]
  Source.within(path, parse(try Source.within(path, Source.read_text(path))))
end

## The tokenizer of the text of a `tokenizer.json`.
def parse(text: String) -> Result[ByteLevelBpe]
  file = try JSON.decode(text, Json)
  nfc = false
  case file.normalizer
    some(n) ->
      return error("the normalizer is #{n.type}, not NFC") if n.type != "NFC"
      nfc = true
    none -> 0
  end

  try check_pre_tokenizer(file.pre_tokenizer)
  return error("the decoder is #{file.decoder.type}, not ByteLevel") if file.decoder.type != "ByteLevel"
  m = file.model
  return error("the model is #{m.type}, not BPE") if m.type != "BPE"
  if (
    m.ignore_merges or
    m.byte_fallback or
    m.continuing_subword_prefix.unwrap_or("") != "" or
    m.end_of_word_suffix.unwrap_or("") != ""
  )
    return error("the BPE model has a prefix, a suffix, byte fallback or ignore_merges, which are not supported")
  end

  added = try Added.parse(file.added_tokens)
  top = 0
  for id in m.vocab.values
    return error("the vocabulary has a negative id #{id}") if id < 0
    top = [top, id + 1].max.unwrap!
  end

  for a in added
    top = [top, a.id + 1].max.unwrap!
  end

  pieces = List.repeat("", top)
  for name in m.vocab.keys
    pieces = pieces.set(m.vocab[name], name)
  end

  ranks: Map[String, Int] = {}
  for i in 0...m.merges.size
    pair = m.merges[i]
    return error("merge #{i} is not a pair of tokens") if pair.size != 2
    ranks = ranks.put(pair[0] + " " + pair[1], i)
  end

  ok(ByteLevelBpe(vocab: m.vocab, pieces: pieces, ranks: ranks, added: added, nfc: nfc))
end

## The ids of `text`, as `tokenizer.encode` gives them (no tokens are added
## around it). Added tokens in the text (`<|im_start|>`) are one id each.
def encode(tok: ByteLevelBpe, text: String) -> Result[List[Int]]
  out: List[Int] = []
  for part in Added.split(tok.added, Unicode.codepoints(text))
    case part
      Added.Part.Token(id) -> out = out.push(id)
      Added.Part.Text(chars) ->
        normal = chars
        if tok.nfc
          normal = Unicode.nfc(chars)
        end

        for word in split(normal)
          for symbol in merge(tok, byte_chars(word))
            case tok.vocab.get(symbol)
              some(id) -> out = out.push(id)
              none -> return error("the vocabulary has no #{symbol}")
            end
          end
        end
    end
  end

  ok(out)
end

## The text of `ids`: the bytes of the pieces (and of added tokens), read as
## UTF-8 with U+FFFD for a broken sequence, as `tokenizer.decode` does. The
## special tokens are left out when `skip_special` is true.
def decode(tok: ByteLevelBpe, ids: List[Int], skip_special: Bool = true) -> Result[String]
  bytes: List[Int] = []
  for id in ids
    return error("no token has the id #{id}") if id < 0 or id >= tok.pieces.size
    found = tok.added.find { |a| a.id == id }
    case found
      ok(a) ->
        if not (skip_special and a.special)
          bytes = bytes + a.content.bytes.to_a
        end

      error(_) ->
        for c in tok.pieces[id].split("")
          code = c.code
          b = -1
          if code < BYTE_OF.size
            b = BYTE_OF[code]
          end

          return error("token #{id} has #{c}, which is no byte-level character") if b < 0
          bytes = bytes.push(b)
        end
    end
  end

  ok((try Bytes.from_list(bytes)).scrub)
end

## The pieces of `chars` that the pattern cuts, as code points.
def split(chars: List[Int]) -> List[List[Int]]
  out: List[List[Int]] = []
  at = 0
  while at < chars.size
    size = piece_size(chars, at)
    out = out.push(span(chars, at, at + size))
    at += size
  end

  out
end

private

## The bytes that GPT-2's byte-level alphabet hides (0..32, 127..160 and 173):
## the token text writes them as the characters 256, 257, ... in this order.
HIDDEN = (0..32).to_a + (127..160).to_a + [173]
## The byte each character code of the token text stands for, or -1.
BYTE_OF = (0...(256 + HIDDEN.size)).to_a.map do |c|
  return HIDDEN[c - 256] if c >= 256
  return -1 if HIDDEN.include?(c)
  c
end

## The character of each byte in the token text.
CHAR_OF = (0...256).to_a.map do |b|
  at = HIDDEN.index(b)
  return String.from_code(256 + at.unwrap!) if at.some?
  String.from_code(b)
end

CR = 13
LF = 10

struct Json
  added_tokens: List[Added.Entry]
  normalizer: Optional[KindJson]
  pre_tokenizer: PreJson
  decoder: KindJson
  model: ModelJson
end

struct KindJson
  type: String
end

struct PreJson
  type: String
  pretokenizers: List[StepJson]
end

struct StepJson
  type: String
  pattern: Optional[Map[String, String]]
  behavior: Optional[String]
  invert: Optional[Bool]
  add_prefix_space: Optional[Bool]
  use_regex: Optional[Bool]
end

struct ModelJson
  type: String
  vocab: Map[String, Int]
  merges: List[List[String]]
  continuing_subword_prefix: Optional[String]
  end_of_word_suffix: Optional[String]
  byte_fallback: Bool = false
  ignore_merges: Bool = false
end

## The pre-tokenizer must be the Qwen split (isolated, as written in
## `PATTERN`) followed by byte-level mapping without its own split.
def check_pre_tokenizer(p: PreJson) -> Result[Void]
  steps = p.pretokenizers
  ok_shape = p.type == "Sequence" and steps.size == 2 and steps[0].type == "Split" and steps[1].type == "ByteLevel"
  return error("the pre-tokenizer is not the Qwen split followed by ByteLevel") if not ok_shape
  split_step = steps[0]
  pattern = split_step.pattern.unwrap_or({}).get("Regex").unwrap_or("")
  if pattern != PATTERN or split_step.behavior != some("Isolated") or split_step.invert != some(false)
    return error("the split pattern of the pre-tokenizer is not the Qwen one")
  end

  if steps[1].add_prefix_space != some(false) or steps[1].use_regex != some(false)
    return error("the ByteLevel pre-tokenizer adds a prefix space or splits again, which is not supported")
  end

  ok()
end

def span(chars: List[Int], from: Int, to: Int) -> List[Int]
  out: List[Int] = []
  for i in from...to
    out = out.push(chars[i])
  end

  out
end

## The symbols of one piece: its UTF-8 bytes, each as its character.
def byte_chars(chars: List[Int]) -> List[String] = Unicode.text_of(chars).bytes.to_a.map { |b| CHAR_OF[b] }

## `symbols` with the lowest-ranked adjacent pair joined, everywhere it
## stands, until no pair is a merge.
def merge(tok: ByteLevelBpe, symbols: List[String]) -> List[String]
  current = symbols
  while current.size > 1
    best = -1
    best_at = -1
    for i in 0...(current.size - 1)
      rank = tok.ranks.get(current[i] + " " + current[i + 1]).unwrap_or(-1)
      if rank >= 0 and (best < 0 or rank < best)
        best = rank
        best_at = i
      end
    end

    break if best_at < 0
    left = current[best_at]
    right = current[best_at + 1]
    out: List[String] = []
    i = 0
    while i < current.size
      if i + 1 < current.size and current[i] == left and current[i + 1] == right
        out = out.push(left + right)
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

def newline?(c: Int) -> Bool = c == CR or c == LF

## The length of the piece that starts at `at`: the first alternative of the
## pattern that matches there (one always does).
def piece_size(chars: List[Int], at: Int) -> Int
  c = chars[at]
  n = chars.size
  ending = contraction(chars, at)
  return ending if ending > 0
  return letters(chars, at) if Unicode.letter?(c)

  if not newline?(c) and not Unicode.number?(c) and at + 1 < n and Unicode.letter?(chars[at + 1])
    return 1 + letters(chars, at + 1)
  end

  return 1 if Unicode.number?(c)
  start = at
  if c == 32
    start = at + 1
  end

  if start < n and symbol?(chars[start])
    stop = start
    while stop < n and symbol?(chars[stop])
      stop += 1
    end

    while stop < n and newline?(chars[stop])
      stop += 1
    end

    return stop - at
  end

  blanks(chars, at)
end

## `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll` or `'d` (any case) at `at`: 2, 3 or 0.
def contraction(chars: List[Int], at: Int) -> Int
  return 0 if chars[at] != 39 or at + 1 >= chars.size
  a = fold(chars[at + 1])
  return 2 if a == 115 or a == 116 or a == 109 or a == 100
  return 0 if at + 2 >= chars.size
  b = fold(chars[at + 2])
  return 3 if (a == 114 and b == 101) or (a == 118 and b == 101) or (a == 108 and b == 108)
  0
end

def fold(c: Int) -> Int
  return c + 32 if c >= 65 and c <= 90
  c
end

def letters(chars: List[Int], at: Int) -> Int
  stop = at
  while stop < chars.size and Unicode.letter?(chars[stop])
    stop += 1
  end

  stop - at
end

## Not white space, a letter or a number.
def symbol?(c: Int) -> Bool = not Unicode.space?(c) and not Unicode.letter?(c) and not Unicode.number?(c)

## The white space alternatives at `at` (a white space character): up to the
## last new line of the run, else the run without its last character when
## another character follows (so that it can lead the next piece), else the run.
def blanks(chars: List[Int], at: Int) -> Int
  stop = at
  while stop < chars.size and Unicode.space?(chars[stop])
    stop += 1
  end

  last = stop - 1
  while last >= at
    return last + 1 - at if newline?(chars[last])

    last -= 1
  end

  return stop - at - 1 if stop < chars.size and stop - at >= 2

  stop - at
end
