## Qwen's tokenizer: the `tokenizer.json` of a byte-level BPE model with an NFC
## normalizer and the GPT-4 style split (apostrophe endings, one digit at a
## time, a run of letters with one lead character, punctuation, new lines,
## white space), which `Qwen/Qwen3-0.6B` ships, or the one of `Qwen/Qwen3.5-0.8B`, where a mark counts as a
## letter (`PATTERN_MARKS`), or the one of `LiquidAI/LFM3-1.7B`, which has no normalizer, cuts digits in groups of up to
## three (`PATTERN_GROUPS`) and takes a whole piece that is in the vocabulary without merging it (`ignore_merges`).
## A file with another normalizer, split pattern or model is an error, not a silent difference.
##
## The split is Hugging Face's regular expression written by hand over
## `unicode.agc`'s tables (`PATTERN`, with `\s` as White_Space and `\p{L}`,
## `\p{N}` as the L and N categories, `\p{M}` as `Unicode.any_mark?`), alternative by alternative in its order.

import Gguf from "pkg/hub/gguf"
import Unicode from "unicode"
import Added from "added"
import Source from "source"
import Heap from "std/heap"

## The split pattern of Qwen3 and Qwen2.5.
PATTERN = (
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)

## The split pattern of Qwen3.5: a mark (`\p{M}`) is a letter in it.
PATTERN_MARKS = (
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)

## The pattern of LFM3: Qwen's, with up to three digits in a piece (`\p{N}{1,3}`).
PATTERN_GROUPS = (
  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)

struct ByteLevelBpe
  vocab: Map[String, Int]
  ## The byte-level text of each id, "" where the vocabulary has none.
  pieces: List[String]
  ## The rank of the merge of `left` and `right`, keyed `left + " " + right`.
  ranks: Map[String, Int]
  added: List[Added.Added]
  nfc: Bool
  ## The split of Qwen3.5 (`PATTERN_MARKS`): marks count as letters.
  marks: Bool = false
  ## The most digits in a piece: 1, or 3 for `PATTERN_GROUPS`.
  digits: Int = 1
  ## A piece that is a token of the vocabulary is that token, whatever the merges say.
  ignore_merges: Bool = false
end

## The tokenizer in the `tokenizer.json` at `path`.
def open(path: String) -> Result[ByteLevelBpe]
  Source.within(path, parse(try Source.within(path, Source.read_text(path))))
end

## The tokenizer of the text of a `tokenizer.json`.
def parse(text: String) -> Result[ByteLevelBpe]
  file = try decoded(text)
  try Added.refuse_limits(file.truncation, file.padding)
  case file.post_processor
    some(p) -> return error("the post-processor is #{p.type}, not ByteLevel") if p.type != "ByteLevel"
    none -> 0
  end

  nfc = false
  case file.normalizer
    some(n) ->
      return error("the normalizer is #{n.type}, not NFC") if n.type != "NFC"
      nfc = true
    none -> 0
  end

  pattern = try check_pre_tokenizer(file.pre_tokenizer)
  return error("the decoder is #{file.decoder.type}, not ByteLevel") if file.decoder.type != "ByteLevel"
  m = file.model
  return error("the model is #{m.type}, not BPE") if m.type != "BPE"
  return error("the BPE model has dropout, which is not supported") if m.dropout.some?
  if m.byte_fallback or m.continuing_subword_prefix.unwrap_or("") != "" or m.end_of_word_suffix.unwrap_or("") != ""
    return error("the BPE model has a prefix, a suffix or byte fallback, which are not supported")
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

  ok(
    ByteLevelBpe(
      vocab: m.vocab,
      pieces: pieces,
      ranks: ranks,
      added: added,
      nfc: nfc,
      marks: pattern.marks,
      digits: pattern.digits,
      ignore_merges: m.ignore_merges,
    ),
  )
end

## The tokenizer in the GGUF at `path`: `from_gguf` of its header.
def open_gguf(path: String) -> Result[ByteLevelBpe] = from_gguf(try Gguf.open(path))

## The tokenizer that llama.cpp writes into a GGUF of Qwen (`tokenizer.ggml.model` gpt2, `pre` qwen2): the tokens in the
## order of their ids, their types and the merges as `left right` strings. A normal token (type 1) is a piece, a
## control (3) or user defined (4) token is an added token, and an unused one (5, the padding of the vocabulary) is no
## token. The NFC normalization is on, as in Qwen's `tokenizer.json`: the file does not say, and llama.cpp does not
## normalize. A control token is special: llama.cpp calls some tokens control that `tokenizer.json` does not mark so
## (`<|fim_prefix|>`), so `decode` leaves those out where the JSON's tokenizer keeps them.
def from_gguf(h: Gguf.Header) -> Result[ByteLevelBpe]
  model = try Gguf.string(h, "tokenizer.ggml.model")
  return error("#{h.path}: the tokenizer model is #{model}, not gpt2") if model != "gpt2"
  pre = try Gguf.string(h, "tokenizer.ggml.pre")
  return error("#{h.path}: the pre-tokenizer is #{pre}, not qwen2") if pre != "qwen2"
  tokens = try Gguf.strings(h, "tokenizer.ggml.tokens")
  kinds = try Gguf.ints(h, "tokenizer.ggml.token_type")
  merges = try Gguf.strings(h, "tokenizer.ggml.merges")
  return error("#{h.path}: #{tokens.size} tokens and #{kinds.size} token types") if tokens.size != kinds.size

  vocab: Map[String, Int] = {}
  added: List[Added.Added] = []
  top = 0
  for id in 0...tokens.size
    token = tokens[id]
    case kinds[id]
      1 ->
        return error("#{h.path}: the token #{token.inspect} appears twice") if vocab.key?(token)
        vocab = vocab.put(token, id)
        top = id + 1
      3, 4 ->
        return error("#{h.path}: the token #{id} has no text") if token.empty?
        added = added.push(Added.of(id, token, kinds[id] == 3))
        top = id + 1
      5 -> 0
      _ -> return error("#{h.path}: the token #{id} has the type #{kinds[id]}, which is not supported")
    end
  end

  pieces = List.repeat("", top)
  for name in vocab.keys
    pieces = pieces.set(vocab[name], name)
  end

  ranks: Map[String, Int] = {}
  for i in 0...merges.size
    pair = merges[i].split(" ")
    return error("#{h.path}: merge #{i} is not a pair of tokens") if pair.size != 2 or pair[0].empty? or pair[1].empty?

    ranks = ranks.put(merges[i], i)
  end

  ok(ByteLevelBpe(vocab: vocab, pieces: pieces, ranks: ranks, added: added, nfc: true))
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

        for word in split(normal, tok.marks, tok.digits)
          symbols = byte_chars(word)
          whole = symbols.join("")
          if tok.ignore_merges and tok.vocab.key?(whole)
            out = out.push(tok.vocab[whole])
            next
          end

          for symbol in merge(tok.ranks, symbols)
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
          for b in a.content.bytes.to_a
            bytes = bytes.push(b)
          end
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
def split(chars: List[Int], marks: Bool = false, digits: Int = 1) -> List[List[Int]]
  out: List[List[Int]] = []
  at = 0
  while at < chars.size
    size = piece_size(chars, at, marks, digits)
    out = out.push(Unicode.span(chars, at, at + size))
    at += size
  end

  out
end

## The symbols of one piece: its UTF-8 bytes, each as its character in the byte-level alphabet.
def byte_chars(chars: List[Int]) -> List[String] = Unicode.text_of(chars).bytes.to_a.map { |b| CHAR_OF[b] }

## `symbols` with adjacent pairs joined by `ranks` (the rank of a pair, keyed `left + " " + right`), lowest first
## (leftmost first among equal ranks), until no pair is a merge: `tokenizers`' algorithm, a queue of candidate pairs
## over a linked list of symbols, so a long piece costs O(n log n), not O(n^2).
def merge(ranks: Map[String, Int], symbols: List[String]) -> List[String]
  n = symbols.size
  return symbols if n < 2
  text = symbols
  after = (0...n).to_a.map { |i| i + 1 }.set(n - 1, -1)
  before = (0...n).to_a.map { |i| i - 1 }
  alive = List.repeat(true, n)
  queue = Heap.new do |a: Candidate, b: Candidate| a.rank < b.rank or (a.rank == b.rank and a.at < b.at) end
  for i in 0...(n - 1)
    queue = offer(ranks, queue, text[i], text[i + 1], i)
  end

  while not Heap.empty?(queue)
    popped = Heap.pop(queue).unwrap!
    queue = popped.remaining
    at = popped.value.at
    right = after[at]
    if alive[at] and right >= 0 and rank_of(ranks, text[at], text[right]) == popped.value.rank
      text = text.set(at, text[at] + text[right])
      alive = alive.set(right, false)
      next_at = after[right]
      after = after.set(at, next_at)
      if next_at >= 0
        before = before.set(next_at, at)
        queue = offer(ranks, queue, text[at], text[next_at], at)
      end

      if before[at] >= 0
        queue = offer(ranks, queue, text[before[at]], text[at], before[at])
      end
    end
  end

  out: List[String] = []
  at = 0
  while at >= 0
    out = out.push(text[at])
    at = after[at]
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

## The shape of a split pattern: whether a mark is a letter in it, and the most digits in a piece.
struct Pattern
  marks: Bool
  digits: Int
end

## A pair of neighbours to join: the symbol at `at` and the one after it.
struct Candidate
  rank: Int
  at: Int
end

struct Json[M]
  truncation: Optional[Added.Setting]
  padding: Optional[Added.Setting]
  added_tokens: List[Added.Entry]
  normalizer: Optional[KindJson]
  post_processor: Optional[KindJson]
  pre_tokenizer: PreJson
  decoder: KindJson
  model: M
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

## The model of a `tokenizer.json` that writes a merge as `["a", "b"]` (Qwen3's).
struct ModelJson
  type: String
  vocab: Map[String, Int]
  merges: List[List[String]]
  continuing_subword_prefix: Optional[String]
  end_of_word_suffix: Optional[String]
  dropout: Optional[Float]
  byte_fallback: Bool = false
  ignore_merges: Bool = false
end

## The model of one that writes it as `"a b"` (Qwen2.5's, the older form); `tokenizers` reads both.
struct ModelLinesJson
  type: String
  vocab: Map[String, Int]
  merges: List[String]
  continuing_subword_prefix: Optional[String]
  end_of_word_suffix: Optional[String]
  dropout: Optional[Float]
  byte_fallback: Bool = false
  ignore_merges: Bool = false
end

## The file of `text` with its model in the first form, or in the second, turned into the first.
def decoded(text: String) -> Result[Json[ModelJson]]
  first = JSON.decode(text, Json[ModelJson])
  return first if first.ok?
  second = JSON.decode(text, Json[ModelLinesJson])
  if not second.ok?
    case second
      error(e) -> return error(e.message) if lines_shape?(text)
      ok(_) -> nil
    end

    return first
  end

  f = second.unwrap!
  m = f.model
  ok(
    Json(
      truncation: f.truncation,
      padding: f.padding,
      added_tokens: f.added_tokens,
      normalizer: f.normalizer,
      post_processor: f.post_processor,
      pre_tokenizer: f.pre_tokenizer,
      decoder: f.decoder,
      model: ModelJson(
        type: m.type,
        vocab: m.vocab,
        merges: m.merges.map { |line| line.split(" ") },
        continuing_subword_prefix: m.continuing_subword_prefix,
        end_of_word_suffix: m.end_of_word_suffix,
        dropout: m.dropout,
        byte_fallback: m.byte_fallback,
        ignore_merges: m.ignore_merges,
      ),
    ),
  )
end

## Whether the file writes its merges as "a b" lines, so that is the shape whose error to report.
def lines_shape?(text: String) -> Bool
  parsed = JSON.parse(text)
  return false if not parsed.ok?
  merges = JSON.get(JSON.get(parsed.unwrap!, "model").unwrap_or(JSON.null()), "merges")
  return false if not merges.ok?
  items: List[JSON.Value] = JSON.list(merges.unwrap!).unwrap_or([])
  return false if items.empty?
  JSON.string(items[0]).ok?
end

## The pre-tokenizer must be the Qwen split (isolated, as written in `PATTERN`, `PATTERN_MARKS` or `PATTERN_GROUPS`)
## followed by byte-level mapping without its own split. Its pattern.
def check_pre_tokenizer(p: PreJson) -> Result[Pattern]
  steps = p.pretokenizers
  ok_shape = p.type == "Sequence" and steps.size == 2 and steps[0].type == "Split" and steps[1].type == "ByteLevel"
  return error("the pre-tokenizer is not the Qwen split followed by ByteLevel") if not ok_shape
  split_step = steps[0]
  regex = split_step.pattern.unwrap_or({}).get("Regex").unwrap_or("")
  pattern = Pattern(marks: regex == PATTERN_MARKS, digits: 1)
  if regex == PATTERN_GROUPS
    pattern = Pattern(marks: false, digits: 3)
  end

  if (
    (regex != PATTERN and regex != PATTERN_MARKS and regex != PATTERN_GROUPS) or
    split_step.behavior != some("Isolated") or
    split_step.invert != some(false)
  )
    return error("the split pattern of the pre-tokenizer is not the Qwen one")
  end

  if steps[1].add_prefix_space != some(false) or steps[1].use_regex != some(false)
    return error("the ByteLevel pre-tokenizer adds a prefix space or splits again, which is not supported")
  end

  ok(pattern)
end

def rank_of(ranks: Map[String, Int], left: String, right: String) -> Int = ranks.get(left + " " + right).unwrap_or(-1)

## `queue` with the pair `left right` at `at`, when it is a merge.
def offer(
  ranks: Map[String, Int],
  queue: Heap.Heap[Candidate],
  left: String,
  right: String,
  at: Int,
) -> Heap.Heap[Candidate]
  rank = rank_of(ranks, left, right)
  return queue if rank < 0
  Heap.push(queue, Candidate(rank: rank, at: at))
end

def newline?(c: Int) -> Bool = c == CR or c == LF

## The length of the piece that starts at `at`: the first alternative of the
## pattern that matches there (one always does).
def piece_size(chars: List[Int], at: Int, marks: Bool, digits: Int) -> Int
  c = chars[at]
  n = chars.size
  ending = contraction(chars, at)
  return ending if ending > 0
  return letters(chars, at, marks) if word?(c, marks)

  if not newline?(c) and not Unicode.number?(c) and at + 1 < n and word?(chars[at + 1], marks)
    return 1 + letters(chars, at + 1, marks)
  end

  return numbers(chars, at, digits) if Unicode.number?(c)
  start = at
  if c == 32
    start = at + 1
  end

  if start < n and symbol?(chars[start], marks)
    stop = start
    while stop < n and symbol?(chars[stop], marks)
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

## A letter, or a mark when `marks`.
def word?(c: Int, marks: Bool) -> Bool = Unicode.letter?(c) or (marks and Unicode.any_mark?(c))

## The run of numbers at `at`, up to `digits` of them.
def numbers(chars: List[Int], at: Int, digits: Int) -> Int
  stop = at
  while stop < chars.size and stop - at < digits and Unicode.number?(chars[stop])
    stop += 1
  end

  stop - at
end

def letters(chars: List[Int], at: Int, marks: Bool) -> Int
  stop = at
  while stop < chars.size and word?(chars[stop], marks)
    stop += 1
  end

  stop - at
end

## Not white space, a letter (or a mark when `marks`) or a number.
def symbol?(c: Int, marks: Bool) -> Bool = not Unicode.space?(c) and not word?(c, marks) and not Unicode.number?(c)

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
