## Whisper's tokenizer, for decoding: token ids to text, the special tokens by
## id, and the prompt that starts the decoder (`asr/DESIGN.md`). It loads the
## Hugging Face `tokenizer.json` beside a checkpoint. Encoding text is not here.

struct Language
  ## The code, such as `es`, from the token `<|es|>`.
  code: String
  id: Int
end

## A Whisper vocabulary: the ids below `eot` are text, the others special.
struct Tokenizer
  ## The byte-level text of each text token, by id.
  pieces: List[String]
  eot: Int
  sot: Int
  ## The language tokens, in id order: the ids between `sot` and `translate`.
  languages: List[Language]
  translate: Int
  transcribe: Int
  no_timestamps: Int
  ## `<|0.00|>`; the id `timestamp_begin + n` is `n * 0.02` seconds.
  timestamp_begin: Int
end

enum Job
  Transcribe
  Translate
end

## The tokenizer in the `tokenizer.json` at `path`.
def open(path: String) -> Result[Tokenizer]
  text = try within(path, read_text(path))
  within(path, parse(text))
end

## The tokenizer of the text of a `tokenizer.json`: the vocabulary of its BPE
## model and its added tokens, which must name the special tokens.
def parse(text: String) -> Result[Tokenizer]
  file = try JSON.decode(text, Json)
  added: Map[String, Int] = {}
  for t in file.added_tokens
    added = added.put(t.content, t.id)
  end

  eot = try special(added, "<|endoftext|>")
  sot = try special(added, "<|startoftranscript|>")
  translate = try special(added, "<|translate|>")
  no_timestamps = try special(added, "<|notimestamps|>")
  pieces = List.repeat("", eot)
  ids = file.model.vocab.values
  texts = file.model.vocab.keys
  for i in 0...ids.size
    id = ids[i]
    return error("the vocabulary has a negative id #{id}") if id < 0
    if id < eot
      pieces = pieces.set(id, texts[i])
    end
  end

  languages: List[Language] = []
  for t in (file.added_tokens.sort_by { |a| a.id })
    if t.id > sot and t.id < translate
      code = t.content.replace("<|", "").replace("|>", "")
      languages = languages.push(Language(code: code, id: t.id))
    end
  end

  ok(
    Tokenizer(
      pieces: pieces,
      eot: eot,
      sot: sot,
      languages: languages,
      translate: translate,
      transcribe: try special(added, "<|transcribe|>"),
      no_timestamps: no_timestamps,
      timestamp_begin: no_timestamps + 1,
    ),
  )
end

## The text of `ids`: the text tokens' bytes, joined, read as UTF-8 (a
## character split across tokens joins back; a broken one becomes U+FFFD).
## Special and timestamp tokens (ids from `eot` on) give no text.
def decode(tok: Tokenizer, ids: List[Int]) -> Result[String]
  bytes: List[Int] = []
  for id in ids
    return error("no token has the id #{id}") if id < 0
    next if id >= tok.eot
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

  ok((try Bytes.from_list(bytes)).scrub)
end

## The prompt that starts the decoder: `<|startoftranscript|>`, the language
## token (none leaves it to the language detection), the job token, and
## `<|notimestamps|>` when `timestamps` is false.
def prompt(tok: Tokenizer, language: Optional[String], job: Job, timestamps: Bool) -> Result[List[Int]]
  out = case language
    some(code) -> [tok.sot, try language_id(tok, code)]
    none -> [tok.sot]
  end

  out = case job
    Job.Transcribe -> out.push(tok.transcribe)
    Job.Translate -> out.push(tok.translate)
  end

  return ok(out) if timestamps
  ok(out.push(tok.no_timestamps))
end

## The id of the language token of `code` (`es` is `<|es|>`).
def language_id(tok: Tokenizer, code: String) -> Result[Int]
  case (tok.languages.find { |l| l.code == code })
    ok(l) -> ok(l.id)
    error(_) -> error("the tokenizer has no language #{code}")
  end
end

## The time of a timestamp token, in seconds, or none for another token.
def seconds(tok: Tokenizer, id: Int) -> Optional[Float]
  return none if id < tok.timestamp_begin
  some((id - tok.timestamp_begin).to_f * TIME_STEP)
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

## Seconds per timestamp step.
TIME_STEP = 0.02

## The fields of `tokenizer.json` that decoding needs.
struct Json
  added_tokens: List[Added]
  model: Model
end

struct Added
  id: Int
  content: String
end

struct Model
  vocab: Map[String, Int]
end

def special(added: Map[String, Int], content: String) -> Result[Int]
  case added.get(content)
    some(id) -> ok(id)
    none -> error("the added tokens have no #{content}")
  end
end

## The whole file at `path`, read in chunks (`File.read_text` stops at 1 MiB).
def read_text(path: String) -> Result[String]
  bytes: Result[Bytes] = File.open(path) do |f|
    parts: List[Bytes] = []
    while true
      chunk = try File.read_bytes(f, 1048576)
      break if chunk.empty?
      parts = parts.push(chunk)
    end

    ok(Bytes.concat(parts))
  end

  String.from_bytes(try bytes)
end

## `r`, with `prefix: ` before its error.
def within[T](prefix: String, r: Result[T]) -> Result[T]
  case r
    ok(v) -> ok(v)
    error(e) -> error("#{prefix}: #{e.message}")
  end
end
