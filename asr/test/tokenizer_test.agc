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

private

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
  try Test.eq(
    [tok.eot, tok.sot, tok.translate, tok.transcribe, tok.no_timestamps, tok.timestamp_begin],
    [
      try int(special, "eot"),
      try int(special, "sot"),
      try int(special, "translate"),
      try int(special, "transcribe"),
      try int(special, "no_timestamps"),
      try int(special, "timestamp_begin"),
    ],
  )

  languages: List[Tokenizer.Language] = []
  for pair in try JSON.list(try JSON.get(special, "languages"))
    languages = languages.push(
      Tokenizer.Language(code: try JSON.string(try JSON.at(pair, 0)), id: try JSON.int(try JSON.at(pair, 1))),
    )
  end

  Test.eq(tok.languages, languages)
end

def test_the_prompt_follows_language_job_and_timestamps() -> Result[Void]
  tok = try Tokenizer.open(FIXTURE)
  try Test.eq(try Tokenizer.prompt(tok, some("es"), Tokenizer.Job.Transcribe, false), [22, 24, 27, 31])
  try Test.eq(try Tokenizer.prompt(tok, some("yue"), Tokenizer.Job.Translate, true), [22, 25, 26])
  try Test.eq(try Tokenizer.prompt(tok, none, Tokenizer.Job.Transcribe, true), [22, 27])
  try Test.error?(Tokenizer.prompt(tok, some("xx"), Tokenizer.Job.Transcribe, true))
  Test.eq(
    [Tokenizer.seconds(tok, 32), Tokenizer.seconds(tok, 35), Tokenizer.seconds(tok, 31)],
    [some(0.0), some(0.06), none],
  )
end

def test_bad_input_is_an_error() -> Result[Void]
  tok = try Tokenizer.open(FIXTURE)
  try Test.error?(Tokenizer.decode(tok, [4, -1]))
  try Test.error?(Tokenizer.open("test/fixtures/missing.json"))
  no_eot = (
    "{\"added_tokens\": [{\"id\": 1, \"content\": \"<|startoftranscript|>\"}], \"model\": {\"vocab\": {\"a\": 0}}}"
  )

  case Tokenizer.parse(no_eot)
    ok(_) -> error("a tokenizer without <|endoftext|> loaded")
    error(e) -> Test.assert(e.message.include?("<|endoftext|>"), "the error does not name <|endoftext|>: #{e.message}")
  end
end

def peer() -> Result[String]
  p = try Process.spawn_captured("python3", ["-I", "test/tokenizer_peer.py", FIXTURE, JSON.encode(CASES)])
  out = try Process.output(p, 30000)
  code = try Process.wait(p, 30000)
  return error("tokenizer_peer.py exited with #{code}: #{out.stderr}") if code != 0
  ok(out.stdout)
end

def int(json: JSON.Value, key: String) -> Result[Int] = JSON.int(try JSON.get(json, key))
