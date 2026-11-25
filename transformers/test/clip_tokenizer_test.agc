## `tokenizer/clip.agc` against Hugging Face's slow `CLIPTokenizer` (`clip_tokenizer_golden.py`): the ids of 66 strings
## with the tiny tokenizer of `fixtures/clip` (cleaning, case, contractions, digits, CJK, control characters, the
## special tokens, truncation at 77). Run from `transformers/`.

import Clip from "../tokenizer/clip"
import Golden from "golden"

FIXTURE = "test/fixtures/clip"

def test_the_ids_agree_with_clip_tokenizer() -> Result[Void]
  checked = try Golden.clip(FIXTURE, "#{FIXTURE}/golden.json")
  Test.eq(checked, 66)
end

def test_a_folder_that_is_not_a_tokenizer_and_a_short_limit_are_errors() -> Result[Void]
  try Test.error?(Clip.open("test/fixtures"))
  tok = try Clip.open(FIXTURE)
  try Test.error?(Clip.encode(tok, "a cat", 1))
  try Test.eq(try Clip.encode(tok, "a cat", 2), [tok.bos, tok.eos])
  Test.eq(try Clip.encode(tok, "a cat", 3), [tok.bos, (try Clip.encode(tok, "a cat"))[1], tok.eos])
end
