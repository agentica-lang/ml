## The tokenizer on a tiny vocabulary made for the tests
## (`fixtures/tokenizer.json`), against `tokenizer_peer.py`. Run from `asr/`.

import Tokenizer from "../tokenizer"

FIXTURE = "test/fixtures/tokenizer.json"
## Text, specials and timestamps mixed; `ñ` and an emoji split across tokens;
## broken characters at the end and in the middle; the hidden bytes `\n` and
## `\t`; no ids; only specials.
CASES = [
  [4, 6, 16],
  [22, 24, 27, 32, 7, 13, 14, 17, 33, 21],
  [3, 9, 10, 1],
  [11, 12],
  [3, 9],
  [11, 16],
  [18, 19, 20],
  [],
  [22, 23, 26, 31, 21],
]
## The special ids of Hugging Face's `tokenizer.json` for `openai/whisper-tiny`,
## `openai/whisper-large-v3` and `openai/whisper-tiny.en` (read on 2026-11-23):
## eot, sot, translate, transcribe, notimestamps, `<|30.00|>`, then the number of
## language tokens between sot and translate (`<|es|>` is the fourth).
TINY = [50257, 50258, 50358, 50359, 50363, 51864, 99]
LARGE_V3 = [50257, 50258, 50359, 50360, 50364, 51865, 100]
TINY_EN = [50256, 50257, 50357, 50358, 50362, 51863, 99]

private

## A `tokenizer.json` that holds only the added tokens the parser reads.
struct FileJson
  added_tokens: List[AddedJson]
  model: ModelJson
end

struct AddedJson
  id: Int
  content: String
end

struct ModelJson
  vocab: Map[String, Int]
end

def test_ids_decode_as_the_peer_decodes_them() -> Result[Void]
  tok = try Tokenizer.open(FIXTURE)
  want = try JSON.parse(try peer())
  texts = try JSON.list(try JSON.get(want, "texts"))
  try Test.eq(texts.size, CASES.size)
  for i in 0...CASES.size
    try Test.eq(try Tokenizer.decode(tok, CASES[i]), try JSON.string(texts[i]))
  end

  try Test.eq(try Tokenizer.decode(tok, [3, 9, 10, 1]), "año")
  special = try JSON.get(want, "special")
  names = ["eot", "sot", "translate", "transcribe", "no_timestamps", "timestamp_begin", "vocab_size"]
  peer_ids: List[Int] = []
  for name in names
    peer_ids = peer_ids.push(try JSON.int(try JSON.get(special, name)))
  end

  ids = [tok.eot, tok.sot, tok.translate, tok.transcribe, tok.no_timestamps, tok.timestamp_begin, tok.vocab_size]
  try Test.eq(ids, peer_ids)
  languages: List[Tokenizer.Language] = []
  for pair in try JSON.list(try JSON.get(special, "languages"))
    languages = languages.push(
      Tokenizer.Language(code: try JSON.string(try JSON.at(pair, 0)), id: try JSON.int(try JSON.at(pair, 1))),
    )
  end

  try Test.eq(tok.languages, languages)
  Test.assert(tok.multilingual, "the fixture is multilingual")
end

def test_the_prompt_follows_language_job_and_timestamps() -> Result[Void]
  tok = try Tokenizer.open(FIXTURE)
  try Test.eq(try Tokenizer.prompt(tok, some("es"), Tokenizer.Job.Transcribe, false), [22, 24, 27, 31])
  try Test.eq(try Tokenizer.prompt(tok, some("yue"), Tokenizer.Job.Translate, true), [22, 25, 26])
  try Test.eq(try Tokenizer.prompt(tok, none, Tokenizer.Job.Transcribe, false), [22])
  try Test.error?(Tokenizer.prompt(tok, some("xx"), Tokenizer.Job.Transcribe, true))
  seconds = [31, 32, 35, 36].map { |id| Tokenizer.seconds(tok, id) }
  Test.eq(seconds, [none, some(0.0), some(0.06), none])
end

## Files with the special ids of real checkpoints: multilingual with 99 and
## 100 languages, and English-only, whose prompt has no language or job.
def test_real_layouts_give_whisper_prompts() -> Result[Void]
  tiny = try Tokenizer.parse(layout(TINY))
  try Test.eq([tiny.vocab_size, tiny.timestamp_begin, tiny.languages.size], [51865, 50364, 99])
  try Test.eq(try Tokenizer.prompt(tiny, some("es"), Tokenizer.Job.Transcribe, false), [50258, 50262, 50359, 50363])
  large = try Tokenizer.parse(layout(LARGE_V3))
  try Test.eq([large.vocab_size, large.timestamp_begin, large.languages.size], [51866, 50365, 100])
  try Test.eq(try Tokenizer.prompt(large, some("es"), Tokenizer.Job.Translate, true), [50258, 50262, 50359])
  english = try Tokenizer.parse(layout(TINY_EN))
  try Test.eq([english.vocab_size, english.timestamp_begin, english.languages.size], [51864, 50363, 0])
  try Test.assert(not english.multilingual, "tiny.en is English-only")
  try Test.eq(try Tokenizer.prompt(english, some("en"), Tokenizer.Job.Transcribe, false), [50257, 50362])
  try Test.eq(try Tokenizer.prompt(english, none, Tokenizer.Job.Transcribe, true), [50257])
  try Test.error?(Tokenizer.prompt(english, some("es"), Tokenizer.Job.Transcribe, true))
  Test.error?(Tokenizer.prompt(english, none, Tokenizer.Job.Translate, true))
end

## A file over 1 MiB, read in chunks, with a two-byte character across the
## first chunk's end.
def test_a_large_file_is_read_whole() -> Result[Void]
  fixture = (try File.read_text(FIXTURE)).strip
  head = "{\"pad\": \""
  pad = "a" * (1048575 - head.bytesize)
  rest = "ñ\", #{fixture.slice(1, fixture.length - 1)}"
  try Test.eq("#{head}#{pad}#{rest}".bytes.slice(1048575, 2), "ñ".bytes)
  dir = "#{System.env("TMPDIR").unwrap_or("/tmp")}/asr-tokenizer-#{System.pid}"
  try Dir.mkdir_private(dir)
  path = "#{dir}/tokenizer.json"
  written: Result[Void] = File.open(path, "create_private") do |f|
    try File.write(f, "#{head}#{pad}")
    File.write(f, rest)
  end

  result = case written
    ok(_) -> big_file_decodes(path)
    error(e) -> error(e.message)
  end

  _ = Path.remove(path)
  _ = Path.remove(dir)
  result
end

def test_bad_input_is_an_error() -> Result[Void]
  tok = try Tokenizer.open(FIXTURE)
  try Test.error?(Tokenizer.decode(tok, [4, -1]))
  try Test.error?(Tokenizer.decode(tok, [4, 36]))
  try Test.error?(Tokenizer.open("test/fixtures/missing.json"))
  try Test.error?(Tokenizer.parse("{\"added_tokens\": ["))
  no_vocab = "{\"added_tokens\": #{JSON.encode(added(TINY))}, \"model\": {}}"
  try Test.error?(Tokenizer.parse(no_vocab))
  try Test.ok?(Tokenizer.parse(layout(TINY)))
  try Test.error?(Tokenizer.parse(layout([1000000, 1000001, 1000101, 1000102, 1000106, 1001607, 99])))
  disordered = TINY.set(2, 50360).set(3, 50359)
  try Test.error?(Tokenizer.parse(layout(disordered)))
  euro = JSON.encode(FileJson(added_tokens: added(TINY), model: ModelJson(vocab: {"€": 0})))
  try Test.error?(Tokenizer.decode(try Tokenizer.parse(euro), [0]))
  no_eot = JSON.encode(FileJson(added_tokens: added(TINY).drop(1), model: ModelJson(vocab: {"a": 0})))
  case Tokenizer.parse(no_eot)
    ok(_) -> error("a tokenizer without <|endoftext|> loaded")
    error(e) -> Test.assert(e.message.include?("<|endoftext|>"), "the error does not name <|endoftext|>: #{e.message}")
  end
end

def big_file_decodes(path: String) -> Result[Void]
  tok = try Tokenizer.open(path)
  Test.eq(try Tokenizer.decode(tok, [3, 9, 10, 1]), "año")
end

## A `tokenizer.json` with the special ids of `ids` (as `TINY`) and one text
## token.
def layout(ids: List[Int]) -> String
  JSON.encode(FileJson(added_tokens: added(ids), model: ModelJson(vocab: {"a": 0})))
end

def added(ids: List[Int]) -> List[AddedJson]
  codes = ["en", "zh", "de", "es"]
  out = [AddedJson(id: ids[0], content: "<|endoftext|>"), AddedJson(id: ids[1], content: "<|startoftranscript|>")]
  for i in 0...ids[6]
    code = "x#{i}"
    if i < codes.size
      code = codes[i]
    end

    out = out.push(AddedJson(id: ids[1] + 1 + i, content: "<|#{code}|>"))
  end

  (
    out +
    [
      AddedJson(id: ids[2], content: "<|translate|>"),
      AddedJson(id: ids[3], content: "<|transcribe|>"),
      AddedJson(id: ids[4], content: "<|notimestamps|>"),
      AddedJson(id: ids[4] + 1, content: "<|0.00|>"),
      AddedJson(id: ids[5], content: "<|30.00|>"),
    ]
  )
end

def read(path: String) -> Result[String] = File.read_text(path)

def peer() -> Result[String]
  p = try Process.spawn_captured("python3", ["-I", "test/tokenizer_peer.py", FIXTURE, JSON.encode(CASES)])
  out = try Process.output(p, 30000)
  code = try Process.wait(p, 30000)
  return error("tokenizer_peer.py exited with #{code}: #{out.stderr}") if code != 0
  ok(out.stdout)
end
