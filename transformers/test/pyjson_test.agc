## `Pyjson.dumps` against Python's `json.dumps(value, ensure_ascii=False)` (`pyjson_golden.py`,
## `fixtures/pyjson_golden.json`): each text of the fixture, parsed and written again, is the same text. Run from
## `transformers/`.

import Pyjson from "../pyjson"

def test_dumps_writes_the_text_of_python() -> Result[Void]
  texts = try JSON.decode(try File.read_text("test/fixtures/pyjson_golden.json"), List[String])
  try Test.assert(texts.size > 300, "only #{texts.size} texts")
  for text in texts
    try Test.eq(Pyjson.dumps(try JSON.parse(text)), text)
  end

  ok()
end
