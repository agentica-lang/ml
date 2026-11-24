# transformers — design

BERT and Qwen3 for inference, from Hugging Face checkpoints, over `tensor/`,
`nn/` and `hub/`. BERT, the Qwen3 forward and the tokenizers are here; the
KV cache and generation come in later PRs of ml#78.

## Qwen3

`qwen3.agc`: `Qwen3.open(dir, backend)` (`config.json` and `model.safetensors`),
`Qwen3.run(m, rows, last)` over rows of ids and `Qwen3.logits(m, ids, last)`
over an `I32` `[batch, seq]` tensor, `Qwen3.close_model(m)`.

- Blocks are `Nn.EncoderBlock`s with `norm_first: true` under a causal mask
  (`Nn.encoder_stack(..., causal: true)`): `input_layernorm` and
  `post_attention_layernorm` are `LayerNorm.rms` (the `beta` is zeros and
  unused), `Nn.Attention` has `kv_heads`, `Rope(base: rope_theta)` in the
  `rotate_half` layout, and `q_norm` and `k_norm` per head before the
  rotation; the head width is `head_dim`, not `hidden_size / heads`
  (`q_proj.weight` is `[heads * head_dim, hidden]`, as stored). The MLP is the gated `Nn.Mlp`
  with `Silu`.
- The head is the embedding table itself (`out_in`) when
  `tie_word_embeddings` is true, with no copy of it, else `lm_head`. The `bfloat16` weights come as `F32` through `hub`.
- The forward is one graph on ggml (`Lift.staged`). Every call traces and
  plans the graph again: seconds on the CPU, a second on Vulkan for a 50 token
  prompt. The next PR keeps a padded KV cache, compiles one step per capacity
  and runs the pick on the device, as `asr/` does.
- Only `qwen3`, `silu`, no attention bias, no `rope_scaling` and
  `use_sliding_window: false` are read; the rest is an error. `tie_word_embeddings`
  defaults to `true`, as in Hugging Face, and then the file's `lm_head.weight`
  is ignored. On the reference backend attention needs `q * k` of at most
  1,048,576 (about 1024 tokens, causal). A model of Qwen2.5 (bias, no norms) or GPT-2
  is a later config.

## BERT

`bert.agc`: `Bert.open(dir, backend)` reads `model.safetensors` and
`config.json` of a folder (`bert-base-uncased`, `bert.`-prefixed or not,
`gamma`/`beta` or `weight`/`bias` norms; the `cls.*` heads are ignored).

```text
Bert.run(m, encodings)   [WordPiece.Encoding] padded to the longest -> Output(hidden, pooled)
Bert.embed(m, encodings) mean of each sentence's token vectors (padding left out), [batch, hidden]
Bert.forward(m, ids, type_ids, mask)   [batch, seq] tensors: I32, I32, F32 (1.0 token, 0.0 padding)
```

- Post-norm blocks (`norm_first: false`, set with `.with` on what `Load`
  builds), the exact GELU (`hidden_act: gelu` is `GeluErf`; `gelu_new` is the
  tanh form), `layer_norm_eps` from the config (1e-12), the positions
  `0...seq`, and a pooler of `tanh(dense(first token))` when the file has one.
- Padding is an additive bias on the keys (`Nn.MASKED`), as Hugging Face does,
  so a padded batch gives each sentence what it gives alone.
- On ggml (which computes only graphs) the forward is one compiled graph of the
  model with its weights as trace constants, as in `asr/`.
- Other `model_type`s, `position_embedding_type`s and activations are errors;
  so is an encoding with no token. Only `model.safetensors` is read (no
  sharded checkpoint yet).
- Memory (as `asr/`): `run` closes its input tensors, `embed` closes its
  hidden state and mask and skips the pooler, `Bert.close(out)` and
  `Bert.close_model(m)` free the rest, and off ggml the forward runs in
  `Native.arena`. Reference and ggml are tested, libtorch is not.
- `lift.agc` holds what a model needs on ggml (the lifted weights, `staged`,
  `close`, the weight lists), shared by the models of this folder.
- The pins in `agentica.toml` are those of `asr/`: outside this repository
  `transformers` gets the `hub` without the `gamma`/`beta` fallback until the
  next release moves them (the workspace reads the folders).

## Tokenizers

`tokenizer/wordpiece.agc` (BERT) and `tokenizer/bpe.agc` (Qwen) read a
`tokenizer.json` and match Hugging Face `tokenizers`: the same ids from
`encode` and the same text from `decode`. They read the file's own flags and
refuse what they do not implement (another normalizer, pre-tokenizer,
post-processor or template, decoder or model, `truncation`, `padding`, BPE
`dropout`, an added token with `lstrip`, `rstrip`, `single_word` or
`normalized`), so a different tokenizer is an error, never a quiet difference.

```text
tokenizer/unicode_data.agc   the tables (generated, frozen)
tokenizer/unicode.agc        letter?/number?/mark?/punctuation?/other?/space?, lower, nfd, nfc
tokenizer/added.agc          the added tokens, cut out of the text first (longest first)
tokenizer/source.agc         reading a file in chunks (File.read_text stops at 1 MiB)
tokenizer/wordpiece.agc      BertNormalizer, BertPreTokenizer, WordPiece, [CLS] A [SEP]
tokenizer/bpe.agc            NFC, the Qwen split by hand, byte-level BPE
```

- **The split of Qwen** is the regular expression of its `tokenizer.json`
  (`PATTERN` in `bpe.agc`, compared with the file's) written by hand,
  alternative by alternative, with the backtracking spelled out: Agentica has
  no `\p{L}`. `\s` is White_Space; the categories are Python's.
- **`(?i)` in the contractions** is ASCII only here. It agrees with `tokenizers`
  on `'ſ` (U+017F, which folds to `s`), which is not a contraction there either.
- **The BPE merge** is `tokenizers`' queue over a linked list of symbols
  (`std/heap`), O(n log n) in the length of a piece; a 20k-letter run takes
  about 50 ms. The lists grow with `push`, not `+`, which copies both sides.
- **The tables** come from Python's `unicodedata` (15.1) through
  `tools/unicode_tables.py`, standard library only, checked in and frozen like
  other data. agentica#1952 asks for these in the standard library; when it
  lands, `unicode.agc` shrinks to calls and the table goes.
- **Unicode version.** The tables are 15.1's and `tools/unicode_tables.py`
  refuses another Python. `tokenizers` has its own tables: U+1FAE0 (Unicode 14)
  and U+2EBF0 (15.1) give the same ids as `tokenizers` 0.22.2 on both
  tokenizers, but a character that a newer Unicode classifies differently can
  split differently. The live peer test leaves out the 15.1 characters, so it
  passes on a Python with 15.0 too.
- **Case** is Hugging Face's: lowercase one character at a time, no word-final
  sigma. U+0130 lowercases to two characters.
- **WordPiece decoding** cleans each piece, as `tokenizers` does (a piece
  `" '"` stays apart from `"t"`), not the joined text, and the result is what
  `tokenizers` prints, not what `transformers` prints.

## Tests

- `test/qwen3_test.agc` against `test/qwen_peer.py` (standard library, doubles):
  a tiny random Qwen3 (4 heads, 2 key heads, `head_dim` 4 with hidden 8, so the
  projection is wider than the model) in `bfloat16` with tied embeddings and
  in `float32` untied, the logits of every position of two rows, on the
  reference and on ggml; the last row alone; refusals; bad inputs; a closed
  model.
- `tools/qwen_check.agc` against Hugging Face's `Qwen3ForCausalLM`
  (`test/qwen_golden.py`, manual): five prompts of 5 to 50 tokens, the logits
  of the last position over the vocabulary and the five largest logits of every
  position, in `float32`. Largest gap of a logit: 4.9e-5 on ggml CPU and
  8.6e-5 on Vulkan; the top five ids are the same at every position.
- `test/bert_test.agc` against `test/bert_peer.py` (standard library, doubles):
  a tiny random BERT in both namings, a padded batch, the hidden state, the
  pooler and the mean embedding, on the reference and on ggml (CPU, and Vulkan
  where there is one); and the refusals.
- `tools/bert_check.agc` against Hugging Face's `BertModel`
  (`test/bert_golden.py`, manual): six sentences and pairs alone and one padded
  batch, in `float32` on ggml CPU and Vulkan; the largest gap is 5.0e-5 (CPU)
  and 2.8e-5 (Vulkan) on `bert-base-uncased`.
- `test/unicode_test.agc` against Python's `unicodedata` (`unicode_peer.py`,
  live), on sample strings.
- `test/wordpiece_test.agc` and `test/bpe_test.agc` against ids recorded from
  `tokenizers` for tiny hand-made tokenizers (`test/fixtures/`), made by
  `test/tokenizer_golden.py fixtures` (manual: it needs `tokenizers`).
- `long/long_input_test.agc`: 200k characters and 50k CJK through the Unicode
  functions and both tokenizers, and a 20k-letter BPE piece, each under 10 s
  (about a second); `long/` runs in its own CI step with `--max-steps 400000000` (the heaviest test spends about 580 M steps, the file 1.33 G).
- `tools/tokenizer_check.agc` against the real `bert-base-uncased` and
  `Qwen3-0.6B` files, from a golden of `test/tokenizer_golden.py golden`, with
  `N` random strings (seeded). It matched on 48312 cases (BERT) and 39139
  (Qwen) with `N = 8000`, and has to match on a new run before a change to the
  tokenizers merges.
