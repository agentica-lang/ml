## BERT's tokenizer: the `tokenizer.json` of a WordPiece model with the BERT
## normalizer, the BERT pre-tokenizer and the `[CLS] A [SEP]` template, which
## is what `bert-base-uncased` and its relatives ship. The pipeline reads the
## file's flags; a file with another normalizer, pre-tokenizer or model is an
## error, not a silent difference.
##
## Case follows Hugging Face `tokenizers`: lowercase one character at a time
## (no word-final sigma), accents stripped by NFD and dropping the nonspacing
## marks when `strip_accents` is null and `lowercase` is true.

import Unicode from "unicode"
import Added from "added"
import Source from "source"

struct WordPiece
  vocab: Map[String, Int]
  ## The text of each id, "" where the vocabulary has none.
  pieces: List[String]
  added: List[Added.Added]
  unk: Int
  prefix: String
  max_chars: Int
  clean: Bool
  chinese: Bool
  strip_accents: Bool
  lowercase: Bool
  cleanup: Bool
  cls: Int
  sep: Int
  ## The `max_length` of the file's `truncation`, none when it has none.
  max_length: Optional[Int] = none
end

struct Encoding
  ids: List[Int]
  ## 0 for the first text and its `[CLS]` and `[SEP]`, 1 for a pair's second.
  type_ids: List[Int]
end

## The tokenizer in the `tokenizer.json` at `path`.
def open(path: String) -> Result[WordPiece]
  Source.within(path, parse(try Source.within(path, Source.read_text(path))))
end

## The tokenizer of the text of a `tokenizer.json`.
def parse(text: String) -> Result[WordPiece]
  file = try JSON.decode(text, Json)
  max_length = try Added.limit(file.truncation, file.padding)
  kind = file.model.type.unwrap_or("WordPiece")
  return error("the model is #{kind}, not WordPiece") if kind != "WordPiece"
  if file.normalizer.type != "BertNormalizer"
    return error("the normalizer is #{file.normalizer.type}, not BertNormalizer")
  end

  if file.pre_tokenizer.type != "BertPreTokenizer"
    return error("the pre-tokenizer is #{file.pre_tokenizer.type}, not BertPreTokenizer")
  end

  return error("the decoder is #{file.decoder.type}, not WordPiece") if file.decoder.type != "WordPiece"
  if file.decoder.prefix != file.model.continuing_subword_prefix
    return error(
      "the decoder prefix #{file.decoder.prefix} is not the model's #{file.model.continuing_subword_prefix}",
    )
  end

  added = try Added.parse(file.added_tokens)
  vocab = file.model.vocab
  top = 0
  for id in vocab.values
    return error("the vocabulary has a negative id #{id}") if id < 0
    top = [top, id + 1].max.unwrap!
  end

  for a in added
    top = [top, a.id + 1].max.unwrap!
  end

  pieces = List.repeat("", top)
  for a in added
    pieces = pieces.set(a.id, a.content)
  end

  for name in vocab.keys
    pieces = pieces.set(vocab[name], name)
  end

  cls = try token(vocab, "[CLS]")
  sep = try token(vocab, "[SEP]")
  try check_template(text, cls, sep)
  n = file.normalizer
  ok(
    WordPiece(
      vocab: vocab,
      pieces: pieces,
      added: added,
      unk: try token(vocab, file.model.unk_token),
      prefix: file.model.continuing_subword_prefix,
      max_chars: file.model.max_input_chars_per_word,
      clean: n.clean_text,
      chinese: n.handle_chinese_chars,
      strip_accents: n.strip_accents.unwrap_or(n.lowercase),
      lowercase: n.lowercase,
      cleanup: file.decoder.cleanup,
      cls: cls,
      sep: sep,
      max_length: max_length,
    ),
  )
end

## The ids of `text` (and of `pair`, after a `[SEP]`), with `[CLS]` and
## `[SEP]` around them when `add_special` is true, as `tokenizer.encode` gives them.
## Added tokens in the text (`[MASK]`) are one id each. With `max_length`, or else the `truncation` of the file, an
## encoding of one text has that many ids at most, `[CLS]` and `[SEP]` counted: the tokens past it are dropped. A pair
## longer than the limit is an error (cutting a pair is not supported).
def encode(
  tok: WordPiece,
  text: String,
  pair: Optional[String] = none,
  add_special: Bool = true,
  max_length: Optional[Int] = none,
) -> Result[Encoding]
  first = tokens(tok, text)
  limit = tok.max_length
  if max_length.some?
    limit = max_length
  end

  case limit
    some(most) ->
      room = most
      if add_special
        room = most - 2
      end

      return error("max_length #{most} leaves no room for a token") if room < 1
      if pair.none?
        first = first.take(room)
      end

    none -> nil
  end

  ids = first
  types = List.repeat(0, first.size)
  if add_special
    ids = [tok.cls] + ids + [tok.sep]
    types = [0] + types + [0]
  end

  case pair
    some(second) ->
      rest = tokens(tok, second)
      ids = ids + rest
      types = types + List.repeat(1, rest.size)
      if add_special
        ids = ids.push(tok.sep)
        types = types.push(1)
      end

    none -> 0
  end

  case limit
    some(most) ->
      if ids.size > most
        return error("the pair has #{ids.size} ids, over max_length #{most}: cutting a pair is not supported")
      end

    none -> nil
  end

  ok(Encoding(ids: ids, type_ids: types))
end

## The text of `ids`, as `tokenizer.decode` gives it: the special tokens left
## out when `skip_special` is true, `##` pieces joined to the word before, the
## rest apart by a space, and (when the file says so) each piece cleaned of
## the space before punctuation and contractions (piece by piece, as `tokenizers`
## does, so `don ' t` stays apart).
def decode(tok: WordPiece, ids: List[Int], skip_special: Bool = true) -> Result[String]
  parts: List[String] = []
  for id in ids
    return error("no token has the id #{id}") if id < 0 or id >= tok.pieces.size
    next if (skip_special and tok.added.any? { |a| a.id == id and a.special })

    piece = tok.pieces[id]
    if not parts.empty?
      if piece.start_with?(tok.prefix)
        piece = piece.slice(tok.prefix.size, piece.size - tok.prefix.size)
      else
        piece = " " + piece
      end
    end

    if tok.cleanup
      piece = cleaned(piece)
    end

    parts = parts.push(piece)
  end

  ok(parts.join(""))
end

private

struct Json
  truncation: Optional[Added.Setting]
  padding: Optional[Added.Setting]
  added_tokens: List[Added.Entry]
  normalizer: NormalizerJson
  pre_tokenizer: KindJson
  decoder: DecoderJson
  model: ModelJson
end

struct KindJson
  type: String
end

struct NormalizerJson
  type: String
  clean_text: Bool
  handle_chinese_chars: Bool
  strip_accents: Optional[Bool]
  lowercase: Bool
end

struct DecoderJson
  type: String
  prefix: String
  cleanup: Bool
end

struct ModelJson
  ## Left out of files that older `tokenizers` wrote.
  type: Optional[String]
  unk_token: String
  continuing_subword_prefix: String
  max_input_chars_per_word: Int
  vocab: Map[String, Int]
end

## The post-processor must be BERT's template: `[CLS] A [SEP]` for one text
## and `[CLS] A [SEP] B [SEP]` for a pair (the second with type id 1), with
## the ids of `[CLS]` and `[SEP]` that `encode` uses.
def check_template(text: String, cls: Int, sep: Int) -> Result[Void]
  wanted = try JSON.parse(template(cls, sep))
  found = try JSON.get(try JSON.parse(text), "post_processor")
  if found != wanted
    return error("the post-processor is not the BERT template ([CLS] A [SEP] and [CLS] A [SEP] B [SEP])")
  end

  ok()
end

def template(cls: Int, sep: Int) -> String
  special = ->(name: String, type_id: Int) { "{\"SpecialToken\": {\"id\": \"#{name}\", \"type_id\": #{type_id}}}" }
  sequence = ->(name: String, type_id: Int) { "{\"Sequence\": {\"id\": \"#{name}\", \"type_id\": #{type_id}}}" }
  single = [special.call("[CLS]", 0), sequence.call("A", 0), special.call("[SEP]", 0)]
  pair = single + [sequence.call("B", 1), special.call("[SEP]", 1)]
  ids = (
    "{\"[CLS]\": {\"id\": \"[CLS]\", \"ids\": [#{cls}], \"tokens\": [\"[CLS]\"]}, \"[SEP]\": {\"id\": \"[SEP]\", \"ids\": [#{sep}], \"tokens\": [\"[SEP]\"]}}"
  )

  (
    "{\"type\": \"TemplateProcessing\", \"single\": [#{single.join(", ")}], \"pair\": [#{pair.join(", ")}], \"special_tokens\": #{ids}}"
  )
end

def token(vocab: Map[String, Int], text: String) -> Result[Int]
  case vocab.get(text)
    some(id) -> ok(id)
    none -> error("the vocabulary has no #{text}")
  end
end

## The ids of `text` without `[CLS]` and `[SEP]`.
def tokens(tok: WordPiece, text: String) -> List[Int]
  out: List[Int] = []
  for part in Added.split(tok.added, Unicode.codepoints(text))
    case part
      Added.Part.Token(id) -> out = out.push(id)
      Added.Part.Text(chars) ->
        for word in words(normalize(tok, chars))
          for id in wordpiece(tok, word)
            out = out.push(id)
          end
        end
    end
  end

  out
end

## The normalizer: control characters (but tab and new lines) dropped and white
## space made a space,
## spaces around CJK ideographs, accents stripped, lowercase.
def normalize(tok: WordPiece, chars: List[Int]) -> List[Int]
  out = chars
  if tok.clean
    out = cleaned_chars(out)
  end

  if tok.chinese
    out = spaced_chinese(out)
  end

  if tok.strip_accents
    out = Unicode.nfd(out).select { |c| not Unicode.mark?(c) }
  end

  if tok.lowercase
    out = Unicode.lower(out)
  end

  out
end

def cleaned_chars(chars: List[Int]) -> List[Int]
  out: List[Int] = []
  for c in chars
    next if c == 0 or c == 0xFFFD or (Unicode.other?(c) and c != 9 and c != 10 and c != 13)

    if Unicode.space?(c)
      out = out.push(32)
    else
      out = out.push(c)
    end
  end

  out
end

def spaced_chinese(chars: List[Int]) -> List[Int]
  out: List[Int] = []
  for c in chars
    if chinese?(c)
      out = out.push(32).push(c).push(32)
    else
      out = out.push(c)
    end
  end

  out
end

def chinese?(c: Int) -> Bool
  (
    (c >= 0x4E00 and c <= 0x9FFF) or
    (c >= 0x3400 and c <= 0x4DBF) or
    (c >= 0x20000 and c <= 0x2A6DF) or
    (c >= 0x2A700 and c <= 0x2B73F) or
    (c >= 0x2B740 and c <= 0x2B81F) or
    (c >= 0x2B820 and c <= 0x2CEAF) or
    (c >= 0xF900 and c <= 0xFAFF) or
    (c >= 0x2F800 and c <= 0x2FA1F)
  )
end

## ASCII punctuation (the symbols too) and the Unicode P categories.
def punctuation?(c: Int) -> Bool
  (
    (c >= 33 and c <= 47) or
    (c >= 58 and c <= 64) or
    (c >= 91 and c <= 96) or
    (c >= 123 and c <= 126) or
    Unicode.punctuation?(c)
  )
end

## The pre-tokenizer: cut at white space, and each punctuation mark alone.
def words(chars: List[Int]) -> List[List[Int]]
  out: List[List[Int]] = []
  word: List[Int] = []
  for c in chars
    if Unicode.space?(c) or punctuation?(c)
      if not word.empty?
        out = out.push(word)
        word = []
      end

      if not Unicode.space?(c)
        out = out.push([c])
      end
    else
      word = word.push(c)
    end
  end

  if not word.empty?
    out = out.push(word)
  end

  out
end

## The model: the longest piece of the vocabulary at each place, `##` before
## every piece but the first; one `[UNK]` for the whole word when a place has
## none or the word is too long.
def wordpiece(tok: WordPiece, word: List[Int]) -> List[Int]
  return [tok.unk] if word.size > tok.max_chars
  out: List[Int] = []
  start = 0
  while start < word.size
    stop = word.size
    found = -1
    while start < stop
      piece = Unicode.text_of(word.drop(start).take(stop - start))
      if start > 0
        piece = tok.prefix + piece
      end

      found = tok.vocab.get(piece).unwrap_or(-1)
      break if found >= 0

      stop -= 1
    end

    return [tok.unk] if found < 0
    out = out.push(found)
    start = stop
  end

  out
end

## `tokenizers`' WordPiece cleanup, in its order.
def cleaned(text: String) -> String
  (
    text.replace(" .", ".")
      .replace(" ?", "?")
      .replace(" !", "!")
      .replace(" ,", ",")
      .replace(" ' ", "'")
      .replace(" n't", "n't")
      .replace(" 'm", "'m")
      .replace(" do not", " don't")
      .replace(" 's", "'s")
      .replace(" 've", "'ve")
      .replace(" 're", "'re")
  )
end
