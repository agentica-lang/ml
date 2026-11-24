# transformers — design

BERT and Qwen3 for inference, from Hugging Face checkpoints, over `tensor/`,
`nn/` and `hub/`. BERT, the Qwen3 forward, generation with a KV cache (greedy or
sampled) and the tokenizers are here; more models come in later PRs of ml#128.

## Generation

`generate.agc`, the design of `asr/`'s decoder copied and adapted (ml#117 tracks
sharing it): `Generate.open(m)` is the generator of a model, which holds its
compiled steps in a one-slot channel (as `Asr.Model.plans`), and
`Generate.generate(g, ids, max_new, stop, sampling)` decodes greedily, or by
`sample.agc` (below); `start`, then `step` (the id with the largest logit) or
`step_logits` per call, then `finish`, are the same by hand. `Generate.close(g)` ends the compiled steps, before
`Qwen3.close_model`. Batch of one, one task at a time.

- The cache of a layer is `[1, kv_heads, rows, head_dim]`: the keys already
  normalized and rotated, so a step never rotates the past. Rows are padded
  with zeros to a capacity of 8, 16, 32 and so on (the prompt too: a prompt of
  9 tokens runs as 16), so a graph changes shape only when the capacity grows. A step writes its rows with a product by a 0/1
  matrix (`written`; exact, since a row is one value plus zeros) and a mask
  hides the later positions and the padding, as `asr/` does.
- The prompt is one step: padded to a capacity, its padding gets the position
  `capacity` (which no cache row has, so it writes nothing), and a one-hot
  product takes the last real hidden row alone to the final norm and the head.
  Every later step is one token.
- On ggml each (tokens, rows, capacity) is one compiled graph (`Trace.compile`,
  kept in the generator at once, so an error path loses none, and a second call
  compiles none: `Generate.compiles(g)` counts them), and the pick
  of the next id is in the graph, so one number leaves the device, not 152k
  logits. The pick is `asr/`'s (`max` and `eq`): a graph cannot answer an `I64`
  (what `argmax` makes). Elsewhere the same graph runs eagerly in an arena; on
  the reference backend attention needs `q * k` of at most 1,048,576, which the
  padded prompt counts. libtorch is not tested.
- `step` closes the cache it is given, and the new keys and values are fresh
  tensors every step (`asr/` can write into those of a cache it is done with;
  not done here). A cache of 1024 tokens is 0.23 GB in `F32`.
- Chunked prefill is not done: a prompt is one step, to the model's
  `max_position_embeddings`.

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
  plans the graph again: under two seconds on the CPU, 0.6 s on Vulkan for a 50 token
  prompt. Generation (below) keeps a cache and compiles its steps.
- `model_type` `qwen3` or `qwen2` (Qwen2.5), `silu`, no `attention_bias`, no
  `rope_scaling` and `use_sliding_window: false` are read; the rest is an
  error. `qwen2` has no `q_norm` and `k_norm` and a bias on `q_proj`, `k_proj`
  and `v_proj` (`Load.dense` takes a `.bias` when the file has one), and no
  `head_dim` in its config (`hidden_size / num_attention_heads`, read once in
  `load`). `tie_word_embeddings` defaults to `true`, as in Hugging Face, and
  then the file's `lm_head.weight` is ignored. On the reference backend
  attention needs `q * k` of at most 1,048,576 (about 1024 tokens, causal). GPT-2
  is a later config.

`chat.agc`: `Chat.apply(model_type, messages, add_generation_prompt,
enable_thinking)` is the ChatML text of `apply_chat_template` for the two
families, with the system turn of Qwen2.5 by default, the empty think block of
Qwen3 (`enable_thinking: false`, and before an assistant message that ends the
conversation), the `</think>` split of an assistant message (Qwen3) and no
tools (a `tool` role is an error). The golden is
Hugging Face's own output (`test/chat_golden.py`). The `tokenizer.json` of
Qwen2.5 writes a merge as `"a b"`, where Qwen3's writes `["a", "b"]`;
`tokenizer/bpe.agc` reads both.

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

## Qwen3 as a GGUF

`qwen3_gguf.agc` writes a model folder as a GGUF with the writer of `hub/`
(`hub/DESIGN.md`, "Writing"): the config as llama.cpp's `qwen3.*` keys, the
tokenizer as `tokenizer.ggml.*` (tokens by id with the types of llama.cpp's
converter, padding ids up to `vocab_size`, the merges by rank), the eos, bos
and padding ids, the chat template, and the weights as `F32`, `F16` or
`BF16` (a norm stays `F32`). A tied model writes no `output.weight`, which
llama.cpp and `Qwen3.open` read as tied. `tools/to_gguf.agc` runs it:
Qwen3-0.6B writes in 4 s as `F16` and 22 s as `F32` on the CPU of the
benchmark machine.

## Tokenizers

`tokenizer/wordpiece.agc` (BERT) and `tokenizer/bpe.agc` (Qwen) read a
`tokenizer.json` and match Hugging Face `tokenizers`: the same ids from
`encode` and the same text from `decode`. They read the file's own flags and
refuse what they do not implement (another normalizer, pre-tokenizer,
post-processor or template, decoder or model, `truncation`, `padding`, BPE
`dropout`, an added token with `lstrip`, `rstrip`, `single_word` or
`normalized`), so a different tokenizer is an error, never a quiet difference.

`Bpe.open_gguf` reads the same tokenizer from a GGUF (`tokenizer.ggml.tokens`,
`token_type` and `merges`, written by llama.cpp for `model` gpt2, `pre`
qwen2): a normal token is a piece, a control or user-defined one is an added
token, an unused one (the padding of the vocabulary) is no token. The tables
of Qwen3-0.6B's file equal those of its `tokenizer.json`. Two differences,
both from the file: it does not say whether the text is NFC-normalized (it is
here, as in Qwen's `tokenizer.json`; llama.cpp does not normalize), and a
control token is special, so `decode` leaves out `<|fim_prefix|>` and its five
neighbours by default (`skip_special`), which `tokenizer.json` does not mark
special. An unused id after the last token does not decode; one between tokens
decodes to the empty text.

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
- `test/generate_test.agc` against the same peer: the logits of every step of
  a row of 20 ids (a prompt of 5, 8 or 9 padded to a bucket, then one id at a
  time, the cache growing to 16 and 32 rows) match the peer's forward over the whole
  row on the reference (1e-9) and ggml (1e-4); `generate` picks what the
  forward without a cache picks, with a stop id, and a second call compiles
  none; the pick of a tie is the first index; bad steps are errors.
- `tools/generate_check.agc` against Hugging Face's greedy `generate`
  (`test/qwen_golden.py`): 24 new ids for each of the five prompts, equal on
  ggml CPU and on Vulkan (the smallest gap between the two largest logits of
  a step in them is 0.024, far above the 1e-4 of the `float32` noise), with
  the speed and the memory.
- `tools/qwen_check.agc` against Hugging Face's `Qwen3ForCausalLM`
  (`test/qwen_golden.py`, manual): five prompts of 5 to 50 tokens, the logits
  of the last position over the vocabulary and the five largest logits of every
  position, in `float32`. Largest gap of a logit: 4.9e-5 on ggml CPU and
  8.6e-5 on Vulkan; the top five ids are the same at every position.
- `test/qwen3_test.agc` on a Q8_0 GGUF of the peer (width 32): the dequantized
  model gives the peer's logits (doubles), and with `weights: Q8_0` the matrices stay
  blocks and the logits stay within 0.1 of them (the largest gap measured on the CPU
  is 0.08: ggml rounds every activation of a product to Q8_0 blocks).
- `tools/llama_golden.c` and `tools/qwen_check.agc` with `QUANTIZED=1` (manual): the
  logits of llama.cpp for the same Q8_0 file (made by `llama-quantize` from the F16
  one of Qwen3-0.6B), five prompts of 5 to 50 tokens, on logits of up to 30. The
  largest gap of a logit is 0.32 to 0.59 on ggml CPU with `weights: Q8_0`, and 0.29
  to 0.67 with `F32` weights (the file dequantized, exact activations). That is the
  noise of the Q8_0 rounding of the activations, and it is chaotic: llama.cpp against
  itself with an `F32` KV cache instead of its `F16` one moves by 0.35 to 0.65, and its
  Q8_0 file against its own F16 one by 0.1 on average and 0.84 at most. The graph is
  the same: ours with `F32` weights against llama.cpp on the same file dequantized to
  `F32` (`llama-quantize --allow-requantize`) differs by 0.018 at most, which is its
  `F16` KV cache. The Q8_0 run takes 2.0 GB at its peak against 4.1 GB with `F32`
  weights, and its prompts 15% to 25% less time (one run).
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

## Sampling

`sample.agc`: `Sample.Sampling(temperature, top_k, top_p, seed)`, `Sample.pick`
over all the logits and `Sample.pick_top` over the largest ones.
`Generate.generate` given a `Sampling` (the greedy pick stays in the graph;
temperature 0 is greedy) runs `Generate.step_top`: the graph cuts the largest K
logits with `Tensor.top_ids` (`ggml_top_k`, in no order), gathers their values
and returns the K values and K ids as one vector of 2K floats, so K numbers
cross to the host and not 152k logits; K is `top_k` when set, else
`MAX_CANDIDATES` (256), and `pick_top` sorts them. Ids below the K are never
drawn: with `top_k` unset the draw is from the top 256, whose mass after a real
temperature is negligible, and a tie across the edge of the K keeps one id where
Hugging Face keeps all of them. The order is Hugging Face's warpers: the logits
over the temperature, the `top_k` largest, then the ids whose mass before them
is under `top_p`, a softmax over what is left and a draw by inverse CDF; with
the window of 256, `top_p` is measured against the mass of those 256, not of the
whole vocabulary. The generator is Park-Miller's minimal standard (as
`nn/init.agc`), started from the seed with three states skipped (a small seed
would give a first draw near 0, which is the largest id), so a seed gives the
same ids up to the float noise of each backend; a seed is from 0 to 2147483645
and a temperature is 0 or at least 1e-6. `generate` checks the `Sampling`
(`Sample.check`) before the first step. Only ids within 30 of the largest scaled
logit are candidates (a mass under 1e-13); a `-inf` logit is a masked id on
`Sample.pick`, and an error in `step_top` on ggml (the outputs of a graph are
checked to be finite); a NaN is an error. The probabilities after the three
steps equal Hugging Face's on fixed logits, and the ids drawn equal those of a
stdlib peer (`test/sample_golden.py`, `fixtures/sample_golden.json`, 1e-9 in
`float64`).
