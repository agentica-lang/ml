## Checks a real tokenizer against Hugging Face `tokenizers`: the ids of
## `encode` and the text of `decode` on `test/tokenizer_golden.py`'s strings.
## Manual (it needs the checkpoint's `tokenizer.json`); run from `transformers/`:
##
##   python -I test/tokenizer_golden.py golden wordpiece TOKENIZER_JSON > GOLDEN.json
##   agentica tools/tokenizer_check.agc wordpiece TOKENIZER_JSON GOLDEN.json
##
## `wordpiece` is `bert-base-uncased`, `bpe` is `Qwen/Qwen3-0.6B`, `Qwen/Qwen2.5-0.5B-Instruct` (they share the tokenizer) or `Qwen/Qwen3.5-0.8B` (marks count as letters).
##
## `clip` is the folder of a CLIP checkpoint (`vocab.json`, `merges.txt`) and its golden is
## `python -I test/clip_tokenizer_golden.py golden VOCAB_JSON MERGES_TXT [N] > GOLDEN.json`:
##
##   agentica tools/tokenizer_check.agc clip CHECKPOINT_DIR GOLDEN.json

import Golden from "../test/golden"

def main() -> Void
  args = System.args
  if args.size != 3 or not ["wordpiece", "bpe", "clip"].include?(args[0])
    puts("usage: tokenizer_check.agc wordpiece|bpe|clip TOKENIZER_JSON_OR_DIR GOLDEN_JSON")
    System.exit(2)
  end

  started = Time.monotonic_ms
  result: Result[Int] = ok(0)
  if args[0] == "wordpiece"
    result = Golden.wordpiece(args[1], args[2])
  elsif args[0] == "clip"
    result = Golden.clip(args[1], args[2])
  else
    result = Golden.bpe(args[1], args[2])
  end

  case result
    ok(n) -> puts("#{args[0]}: #{n} cases agree with Hugging Face (#{Time.monotonic_ms - started} ms)")
    error(e) ->
      puts("#{args[0]}: #{e.message}")
      System.exit(1)
  end
end
