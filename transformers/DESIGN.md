# transformers — design

BERT, DistilBERT, Qwen3, Qwen2.5, ResNet and ViT, from Hugging Face checkpoints (Qwen3 also
from a GGUF file), over `tensor/`, `nn/` and `hub/`: the forwards, generation
with a KV cache (greedy or sampled), the tokenizers, and the weights by name
for a fine-tune with `train/`. Qwen3.5 is planned (ml#150, ml#151).

## Generation

`generate.agc` follows `asr/`'s decoder (the padded-cache helpers are shared,
`nn/kv.agc`; each model keeps its own forward): `Generate.open(m)` is the generator of a model, which holds its
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
- On ggml a one-token step reads the rows of the cache in use, not the padding
  after them, which costs as much per row: the first `Padded.reach` rows, the
  rows in use rounded up to 256, or to an eighth of the capacity when that is
  more (llama.cpp rounds up to 256). At 2049 rows of a capacity of 4096 that
  is 2560 rows. Each reach is a compiled graph, and a generator keeps one per
  capacity: compiling the graph of a reach ends the graphs of the smaller
  reaches of its capacity, and a later sequence at a smaller reach takes the
  graph of the larger one that is there (its mask hides the rows after the
  length), so it reads more rows than it needs and compiles nothing. The 32
  slots of the graph cache then hold as many capacities as before, whatever
  the lengths of the prompts. Growing the capacity itself in steps of 256 was
  measured and dropped: a growth is two new graphs and a copy of the cache,
  0.4 s at 256 rows and 1.1 s at 2048. The field `whole` of a generator is for
  the tests only: they set it to read the whole capacity and compare every
  step with it; `open` gives `false`.
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
  not done here). A cache of 1024 tokens is 0.23 GB in `F32` and 0.12 GB in `F16`, which is what
  it is on ggml (`stepped` casts the keys and values it returns; the step that
  grows the capacity widens the old ones first, and the one-token step reads
  the `F16` cache in flash attention and `set_rows` rounds its new row into
  it). The logits of Qwen3-0.6B drift from the forward by 5e-3 to 7e-3 (2.5e-5
  with `F32`), no top-1 flip in 24 steps, and a decode step reads half the
  bytes of the cache. `Q8_0` was measured on llama.cpp (`-ctk q8_0
  -ctv q8_0`) and is no faster than `F16` on this iGPU, so it is not here.
- Chunked prefill is not done: a prompt is one step, to the model's
  `max_position_embeddings`.

## Qwen3

`qwen3.agc`: `Qwen3.open(dir, backend, weights = F32)` (`config.json` and
`model.safetensors` or its `.index.json`, or a `.gguf` file),
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
  plans the graph again. Generation (below) keeps a cache and compiles its steps.
- `model_type` `qwen3` or `qwen2` (Qwen2.5), `silu`, no `attention_bias`, no
  `rope_scaling` and `use_sliding_window: false` are read; the rest is an
  error. `qwen2` has no `q_norm` and `k_norm` and a bias on `q_proj`, `k_proj`
  and `v_proj` (`Load.dense` takes a `.bias` when the file has one), and no
  `head_dim` in its config (`hidden_size / num_attention_heads`, read once in
  `load`). `tie_word_embeddings` defaults to `true`, as in Hugging Face, and
  then the file's `lm_head.weight` is ignored. On the reference backend
  attention needs `q * k` of at most 1,048,576 (about 1024 tokens, causal).

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
  sharded checkpoint).
- Memory (as `asr/`): `run` closes its input tensors, `embed` closes its
  hidden state and mask and skips the pooler, `Bert.close(out)` and
  `Bert.close_model(m)` free the rest, and off ggml the forward runs in
  `Native.arena`. Reference and ggml are tested, libtorch is not.
- `lift.agc` holds what a model needs on ggml (the lifted weights, `staged`,
  `close`, the weight lists), shared by the models of this folder.
- `distilbert.agc` is BERT without token types and pooler, under DistilBERT's
  names (`transformer.layer.N.attention.q_lin`, `sa_layer_norm`, `ffn.lin1`,
  `output_layer_norm`; a `distilbert.` prefix or none), with the layer-norm
  `eps` of 1e-12 that Hugging Face has in its code. It is a file of its own,
  not optional fields of `Bert`: the config names differ and `Bert`'s struct
  is public. Both take `Nn.padding_bias`, and `embed` pools and normalizes
  in the same graph as the encoder.
- `DistilBert.weights` and `with_weights` name the weights as the checkpoint
  does, as `Qwen3`'s do, so a fine-tune is `Params.split`, a loss over
  `with_weights(m, Params.join(split, ps))` and `Train.fit`. The model keeps
  the `distilbert.` prefix of its file for that. `pooled` is the forward
  without `Lift.staged`: `run`, `embed` and `forward` lift the weights as
  constants and run one graph, which a loss must not do, since its weights
  are the inputs of the trace.
- LoRA needs nothing in a model but its names: `Lora` of `train` adds the
  adapter to the weights of `weights(m)` and `with_weights` takes the sums,
  so the names of PEFT's files are ours (`base_model.model.` and the
  checkpoint's name of the layer). What a model adds is `embeddings(m)`, the
  names of its embedding tables, which `Lora.save` refuses to write as the
  adapter of a linear layer (PEFT keeps those in another layout). The test
  reads adapters that PEFT wrote for the tiny models of the
  standard-library peers (`test/lora_golden.py`, manual).
- `sentence.agc` holds what a sentence-embedding model adds to an encoder
  (`mean_pooled`, `normalized`, `cosine_similarity_loss`). A loss lives with
  its domain: the cosine itself is generic math in `nn/`
  (`Nn.cosine_similarity`), and the loss on scored pairs is here, next to
  the pooling it is trained with.
- `train` is a plain dependency, not a test one: `tools/distilbert_stsb.agc`
  (the fine-tune of `benchmarks/stsb/`) calls `Train.fit_batches`, and a tool
  cannot import a test dependency. No module of the package imports `train`.

## Qwen3 as a GGUF

`qwen3_gguf.agc` writes a model folder as a GGUF with the writer of `hub/`
(`hub/DESIGN.md`, "Writing"): the config as llama.cpp's `qwen3.*` keys, the
tokenizer as `tokenizer.ggml.*` (tokens by id with the types of llama.cpp's
converter, padding ids up to `vocab_size`, the merges by rank), the eos, bos
and padding ids, the chat template, and the weights as `F32`, `F16` or
`BF16` (a norm stays `F32`). Every weight's shape is checked against the config
(a missing `head_dim` is the hidden size over the heads) and the ranks of the
merges must be 0 to n-1. A tied model writes no `output.weight`, which
llama.cpp (it falls back to `token_embd.weight`) and `Qwen3.open` read as
tied. `add_bos_token` and `add_eos_token` are the file's when
`tokenizer_config.json` gives them as booleans, else false, and the chat
template is the config's, else `chat_template.jinja`. `tools/to_gguf.agc` runs
it.

## Tokenizers

`tokenizer/wordpiece.agc` (BERT) and `tokenizer/bpe.agc` (Qwen) read a
`tokenizer.json` and match Hugging Face `tokenizers`: the same ids from
`encode` and the same text from `decode`. They read the file's own flags and
refuse what they do not implement (another normalizer, pre-tokenizer,
post-processor or template, decoder or model, `truncation`, `padding`, BPE
`dropout`, an added token with `lstrip`, `rstrip`, `single_word` or
`normalized`), so a different tokenizer is an error, never a quiet difference.
WordPiece reads the plain `truncation` of a file (cut at the right,
`LongestFirst`, no stride) as the `max_length` of an encoding of one text,
and accepts padding to the longest of a batch, which adds nothing to one
text: a sentence-transformers checkpoint sets both. Cutting a pair, another
truncation, a fixed-length padding and `pad_to_multiple_of` above 1 stay
errors. The `pad_id` of the file's padding is kept (`tok.pad_id`) and never
used by `encode`: a model pads a batch with its config's `pad_token_id`, and
`tools/distilbert_check.agc` fails when the two differ.

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

## ResNet

`resnet.agc` reads `ResNetForImageClassification` checkpoints (`microsoft/resnet-50`, `microsoft/resnet-18`): a 7x7
stem and a max pool, four stages of bottleneck or basic layers, a global average pool and a linear head. The input is
`[batch, 3, height, width]` in `F32`, already normalized as the checkpoint's image processor does; `vision/` makes it
(`Resize(256)`, `CenterCrop(224)`, `to_tensor` and `normalize` with the ImageNet mean and std), which is how
`crop_pct` 0.875 of 224 reads. The pinned `preprocessor_config.json` (the same in both checkpoints) says `resample` 3 (Pillow's bicubic), `size` 224, `crop_pct` 0.875, `do_normalize` with the ImageNet mean and std. The golden used the slow processor (`use_fast=False`). Two differences stay: libvips resizes with another kernel than Pillow (pixels up to 0.018 apart after normalizing), and `Vision.CenterCrop` rounds a half-pixel offset to the even pixel, as torchvision does, where Hugging Face floors it (`vision/` is not changed for it). `ResNet.run` gives the logits `[batch, classes]` and `ResNet.top(m, logits, k)` the `k`
best classes with their `id2label` names.

- The batch norm of every convolution is folded into it at load (`Nn.fold_batch_norm`, the idea of MNN): the weight of
  each output channel is scaled by `gamma / sqrt(var + eps)` and the bias is `beta + (bias - mean) * scale`. The fold
  runs as one small graph per convolution, then the checkpoint's five tensors are closed. The model is for inference
  only: the folded weights are not the checkpoint's, and the statistics do not update. `eps` is 1e-5, PyTorch's
  default, which Hugging Face's config does not hold.
- Where the stride sits follows `downsample_in_bottleneck` (the 3x3 convolution by default, the first 1x1 when true;
  the official checkpoints have it false) and `downsample_in_first_stage` (the first stage keeps the size when false).
  A shortcut is a 1x1 convolution (and its batch norm) when a layer changes the width or the stride. `hidden_act` is
  `relu` only. A `layer_type` is `bottleneck` or `basic`.
- `run` is one staged graph on ggml (`Lift.staged`), as BERT's. `forward` is the same as plain `Tensor` calls, for a
  loss: with the traced weights it is part of a trace (the bench takes the gradient of every weight and of the images
  that way). A convolution's bias gradient sums the batch and the plane, which `Trace.unbroadcast` splits in two sums so
  that ggml can lower it (a reduction there takes a leading or a trailing block of axes).
- Official checkpoints at pinned revisions, `model.safetensors` only (never a pickle or `pytorch_model.bin`):

  | Checkpoint | Revision | `model.safetensors` sha256 |
  |---|---|---|
  | `microsoft/resnet-50` | `34c2154c194f829b11125337b98c8f5f9965ff19` | `9c6061af1f450bb0847e529fd742aa5066017be379c71bbf5546b198e5b13a1e` |
  | `microsoft/resnet-18` | `65a5785d9156231087c481e0c7dd33a5ff6f7e3e` | `1cf00ee468998c23d084361b02cffadabacb5074105564c596b8079f77ecb126` |

  `config.json` sha256: `a1be0f56d516d0b55f9844ce0fbfe5fb359195cba6c6c102d41ae6c202c84d5e` (50) and
  `11a01bcc873444bb433140e5f30f3687356641ed923266d738eef1c3438f6d22` (18). `preprocessor_config.json` is the same
  file in both, sha256 `fd575b890da5a949493e1d1e7a70bfcb9e4b99fe444004d2dbfa253add254741`.

## ViT

`vit.agc` reads `ViTForImageClassification` checkpoints (`google/vit-base-patch16-224`): a patch embedding, a class token and learned positions, `Nn.EncoderStack` of pre-norm `Nn.EncoderBlock`s (`layernorm_before`, attention, `layernorm_after`, an MLP with the exact GELU, `hidden_act` `gelu`), the final `vit.layernorm`, and a linear head on the class token. The input is `[batch, 3, 224, 224]` in `F32`, already normalized as the checkpoint's image processor does; `vision/` makes it (`ResizeTo(224, 224)` and `normalize` with mean and std 0.5). `ViT.run` gives the logits `[batch, classes]` and `ViT.top(m, logits, k)` the `k` best classes with their `id2label` names (`Classify.top`, shared with ResNet).

- The patch embedding is `Nn.conv2d` with stride and kernel `patch_size` (`[width, 3, 16, 16]`, as stored), its `[batch, width, 14, 14]` output reshaped to `[batch, width, 196]` and permuted to tokens: a reshape and a linear layer would be the same product, but they need an unfold of the image that the convolution already is. The class token is broadcast over the batch and concatenated first, and the position table `[1, 197, width]` is added: a different image size needs another table, so an input that is not `image_size` by `image_size` is an error (no interpolation).
- `eps` of the layer norms is `layer_norm_eps` (1e-12). `qkv_bias` false drops the biases of the query, key and value. A `model_type` other than `vit` and an activation other than `gelu` are errors. The names and shapes of every tensor are checked against `config.json` before any is loaded, so a checkpoint that does not fit leaves nothing open.
- `run` is one staged graph on ggml (`Lift.staged`), `forward` the same as plain `Tensor` calls, as ResNet's.
- The image processor of the checkpoint (`preprocessor_config.json`: `size` 224, `resample` bilinear by default, `do_normalize` with mean and std 0.5) stretches the whole image to 224 by 224 with no crop and keeps no aspect ratio, which `Vision.Step.ResizeTo(224, 224)` does. `image` resizes with libvips' Lanczos 3 kernel, where Pillow's bilinear resize averages over the area it covers, so the pixels are up to 0.024 (normalized) apart (a kernel option for `Image.resize` is [agentica-lang/packages#238](https://github.com/agentica-lang/packages/issues/238), `image-0.2.0`).
- The official checkpoint at a pinned revision, `model.safetensors` only (never a pickle, `pytorch_model.bin`, or the flax and TensorFlow files):

  | Checkpoint | Revision | `model.safetensors` sha256 |
  |---|---|---|
  | `google/vit-base-patch16-224` | `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3` | `1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d` |

  `config.json` sha256: `abdf73cc4ce79495a83f32a8c9265daa7a1363f5a3efa4704c88a6e7f45ad19d`. `preprocessor_config.json` sha256: `90d1ae427c27d2b9dcb01f6a62ba96aa67aac2de1697c59cd212055b5035c4a2`.

## Tests

- `test/qwen3_test.agc` against `test/qwen_peer.py` (standard library, doubles):
  a tiny random Qwen3 (4 heads, 2 key heads, `head_dim` 4 with hidden 8, so the
  projection is wider than the model) in `bfloat16` with tied embeddings and
  in `float32` untied, on the reference and on ggml.
- `test/generate_test.agc` against the same peer: the logits of every step of
  a row of 20 ids (the prompt padded to a bucket, the cache growing to 16 and
  32 rows) match the peer's forward over the whole row on the reference (1e-9)
  and ggml (1e-4); the pick of a tie is the first index.
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
- `test/resnet_test.agc` against `test/resnet_peer.py` (standard library, doubles, the batch norms not folded): three
  tiny random ResNets (bottleneck, basic, and a bottleneck with `downsample_in_first_stage` and
  `downsample_in_bottleneck`) with batch norms that have statistics of their own, on the reference (1e-9) and on ggml
  (1e-4), plus the errors of a missing weight and an unsupported config.
- `vision/test/resnet_hf_test.agc` against Hugging Face (the golden script `vision/test/resnet_golden.py` is manual: it needs `torch` 2.6.0,
  `transformers` 4.57.6 and `pillow` 11.3.0, in a venv, run with `python -I`; the test does nothing unless `RESNET_MODELS` names the checkpoint folders): four drawn images (`vision/test/resnet_images.py`,
  140 KB in all, no license) through `Resize(256)`, `CenterCrop(224)`, `normalize` and the model, in `float32` on ggml
  CPU and Vulkan. The best 10 logits and the top 5 classes of each image are `float32` Hugging Face's, in the same
  order, for both models. On Hugging Face's own pixels the largest gap of a logit is 1.6e-5 (ResNet-50) and 1.1e-5
  (ResNet-18) on CPU, 2.1e-5 on Vulkan; with the pixels of our own pipeline 0.16 (ResNet-50) and 0.065 (ResNet-18),
  from the pixels, which are up to 0.018 (normalized units) from Pillow's: libvips and Pillow resize with different
  kernels. The 224 by 224 crops that the processor made are committed as 8-bit PNGs too (`vision/test/fixtures/resnet/crops`, 113 KB: `pixel_values` is exactly those samples over 255, normalized), so the model alone runs from committed files on its checkpoint, with a tolerance of 1e-3. The golden is committed (`vision/test/fixtures/resnet/golden.json`) and never made in CI. The tolerances are about twice the gaps measured: 0.035 for the pixels and 0.32 for the logits after the pipeline. Vulkan runs with `Tensor.vulkan_fp16(false)`.
- `test/vit_test.agc` against `test/vit_peer.py` (standard library, doubles): two tiny random ViTs (with and without the biases of the query, key and value) on the reference (1e-9) and on ggml (1e-4), plus the errors of a missing weight, an unsupported config and an input of another size.
- `vision/test/vit_hf_test.agc` against Hugging Face (the golden script `vision/test/vit_golden.py` is manual: it needs `torch` 2.6.0, `transformers` 4.57.6 and `pillow` 11.3.0, in a venv, run with `python -I`; the test does nothing unless `VIT_MODEL` names the checkpoint folder): the four drawn images of the ResNet test through `ResizeTo(224, 224)`, `normalize` and the model in `float32` on ggml CPU and Vulkan. The 224 by 224 images that the processor made are committed as 8-bit PNGs (`vision/test/fixtures/vit/crops`, 100 KB: `pixel_values` is exactly those samples over 255, normalized), so the model alone runs from committed files: the best 10 logits are Hugging Face's within 1e-4 (the largest gap measured is 1.2e-5 on CPU and 9.6e-6 on Vulkan) and the top 5 are identical, in the same order. Through our own resize the pixels are 0.0235 from Hugging Face's and the logits 0.156, from the pixels; the logits of these images lie within 0.03 to 0.3 of each other, so the resize reorders them and the pipeline check allows for the ties (each of our top 5 is in Hugging Face's best 10, and each class of its best 10 more than 0.32 above its sixth is in ours). The tolerances are about twice the gaps measured: 0.05 for the pixels, 0.32 for the pipeline. The golden is committed (`vision/test/fixtures/vit/golden.json`) and never made in CI. Vulkan runs with `Tensor.vulkan_fp16(false)`.
- `test/bert_test.agc` against `test/bert_peer.py` (standard library, doubles):
  a tiny random BERT in both namings and a padded batch, on the reference and
  on ggml (CPU, and Vulkan where there is one).
- `tools/bert_check.agc` against Hugging Face's `BertModel`
  (`test/bert_golden.py`, manual): six sentences and pairs alone and one padded
  batch, in `float32` on ggml CPU and Vulkan; the largest gap is 5.0e-5 (CPU)
  and 2.8e-5 (Vulkan) on `bert-base-uncased`.
- `test/unicode_test.agc` against Python's `unicodedata` (`unicode_peer.py`,
  live); `test/wordpiece_test.agc` and `test/bpe_test.agc` against ids
  recorded from `tokenizers` for tiny hand-made tokenizers
  (`test/tokenizer_golden.py fixtures`, manual: it needs `tokenizers`).
- `long/long_input_test.agc`: 200k characters and 50k CJK through the Unicode
  functions and both tokenizers, and a 20k-letter BPE piece; `long/` runs in
  its own CI step with `--max-steps 400000000`.
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
