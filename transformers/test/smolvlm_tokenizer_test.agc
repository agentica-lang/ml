## The tokenizer of SmolVLM against Hugging Face `tokenizers` (`test/tokenizer_golden.py golden bpe_gpt2`,
## `fixtures/smolvlm_tokenizer_golden.json`): the ids of 165 strings (the ones of `bpe_test`, natural text in several
## scripts, the strings where the digit split decides and 20 random ones; the `<image>` and `<end_of_utterance>` tokens
## are one id each) and the text of 656 ids. The file is the real `tokenizer.json` of `HuggingFaceTB/SmolVLM-256M-Instruct`,
## which is not in the repository, so it runs only when `SMOLVLM_MODEL` names its folder. Run from `transformers/`:
## `SMOLVLM_MODEL=SmolVLM-256M-Instruct agentica test test/smolvlm_tokenizer_test.agc`.

import Bpe from "../tokenizer/bpe"
import Golden from "golden"

def test_the_ids_and_the_text_are_those_of_tokenizers() -> Result[Void]
  listed = System.env("SMOLVLM_MODEL")
  if !listed.ok? or try listed == ""
    puts("smolvlm_tokenizer_test: skipped, SMOLVLM_MODEL names no checkpoint folder")
    return ok()
  end

  tok = try Bpe.open("#{try listed}/tokenizer.json")
  checked = try Golden.bpe_of(tok, "test/fixtures/smolvlm_tokenizer_golden.json")
  try Test.assert(checked > 800, "only #{checked} cases ran")
  # The text of the processor golden of `vision/` (`ids_one` there): `<image>` and `<end_of_utterance>` are one id each.
  ids = try Bpe.encode(tok, "User:<image>What is in the picture?<end_of_utterance>\nAssistant:")
  try Test.eq(ids, [11126, 42, 49190, 1780, 314, 281, 260, 4177, 47, 49279, 198, 9519, 9531, 42])
  ok()
end
