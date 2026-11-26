# transformers — design

BERT, DistilBERT, Qwen3, Qwen2.5, ResNet, ViT and CLIP, from Hugging Face checkpoints (Qwen3 also
from a GGUF file), over `tensor/`, `nn/` and `hub/`: the forwards, generation
with a KV cache (greedy or sampled), the tokenizers, and the weights by name
for a fine-tune with `train/`. Qwen3.5 (text only) has its forward, its
tokenizer, its chat template and generation (ml#150).
Granite MoE (`granitemoe`) has its forward and generation (ml#274).
LFM3 (`lfm3`) has its forward, its tokenizer, its chat template and generation (ml#275).

## Generation

`kv_decoder.agc` follows `asr/`'s decoder (the padded-cache helpers are shared, `nn/kv.agc`; each model keeps its own
forward) and is the same for Qwen3, Granite MoE and LFM3: the model files (`generate.agc`, `generate_granite_moe.agc`, `generate_lfm3.agc`) give
it a `Stepper` (the sizes of the model, its token table, `traced` and the `stepped` graph of one step) and keep their
`Generator`, `stepped` and `layer`. `Request.compiled` tells `stepped` that the step is traced for ggml: Qwen3 keeps
its cache in `F16` then and Granite MoE in `F32` always. `Generate.open(m)` is the generator of a Qwen3 model, which
holds its compiled steps in a one-slot channel (as `Asr.Model.plans`), and `Generate.generate(g, ids, max_new, stop,
sampling)` decodes greedily, or by `sample.agc` (below); `start`, then `step` (the id with the largest logit) or
`step_logits` per call, then `finish`, are the same by hand. `Generate.close(g)` ends the compiled steps, before
`Qwen3.close_model`. Batch of one, one task at a time. `Generate.open(m, usable = 0)` is how many ids a step may
answer (below, "The ids the tokenizer cannot decode"): the embedding of Qwen3 has 151,936 rows and its tokenizer
151,669 tokens, so a caller with a tokenizer opens with `tok.pieces.size` and the sampler never gives an id that
`Bpe.decode` rejects. The default 0 is no mask, as in Hugging Face.

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

## Qwen3.5

`qwen3_5.agc`: `Qwen3_5.open(dir, backend, weights = F32)` (the folder of an
official `Qwen/Qwen3.5-*` repository: `config.json` and `model.safetensors` or
its `.index.json`), `Qwen3_5.run(m, rows, last)`, `Qwen3_5.logits(m, ids, last)`
and `Qwen3_5.close_model(m)`. The forward is over whole sequences, with no
cache; generation is `generate_qwen35.agc` (below).

- `config.json` is nested: the text model is `text_config` (`model_type`
  `qwen3_5_text`), the file is `qwen3_5`. The vision tower (`model.visual.*`)
  and the multi-token-prediction head (`mtp.*`) of the checkpoint are not read.
  Text weights are under `model.language_model.`.
- A layer is `Mixer.Linear` (`Delta.GatedDeltaNet`, `nn/DESIGN.md`) or
  `Mixer.Full` (`Nn.Attention` with `gate`, GQA, `q_norm` and `k_norm`, rope on
  the first `head_dim * partial_rotary_factor` columns), by `layer_types`
  (every fourth is full attention in 0.8B). Both are followed by the gated MLP
  with `Silu`. Rope positions are the token index: the three M-RoPE sections
  agree for text, so the plain `rotate_half` rope with `dims` is the same.
- Every norm of the decoder and the `q_norm` and `k_norm` scale by
  `1 + weight`, not `weight`: the loader adds the 1 on the host
  (`centered`). The gated norm of the linear layer keeps its weight as it is.
- `conv1d.weight` is stored `[channels, 1, taps]` and kept `[channels, taps]`.
  `q_proj.weight` is `[2 * heads * head_dim, hidden]`: per head the query rows,
  then the gate rows (`Nn.query_heads`).
- The head is the embedding table when `tie_word_embeddings` (the case of
  0.8B; the top level of `config.json` first, then `text_config`); with neither,
  the head is `lm_head.weight` when the file has one. A `model_type` (the
  mixture-of-experts `qwen3_5_moe` has its own message), `hidden_act`,
  `layer_types` entry or `rope_type` this loader does not run, `attention_bias`,
  a config without the output gate, linear key and value widths that differ
  (the delta rule needs one width), a bad head count or rotary width, and
  `weights` other than `F32` are errors.
- `graph` unrolls one forward per batch row, so a batch of N is N times the
  nodes of one row. There are no `weights`/`with_weights` yet, so a fine-tune
  with `train/` is not supported.

### Qwen3.5 generation

`generate_qwen35.agc` is the decoder of `kv_decoder.agc` with a hybrid
cache. What does not depend on the model is in `stepping.agc`, which a model
that keeps a cache (the next one is a mixture of experts) uses the same way: it
writes its `Generator`, its `Cache`, its `stepped` (the graph of a step, with
`Stepping.cached`, `widened` and `masked`) and its `step`, `step_logits` and
`step_top`, and gives `Stepping.generate` a `Stepping.Decoder` (`decoder(g)` in
each file: `usable`, `limit`, `empty`, and `pick`, `top` and `finish` as
closures over its generator). `Stepping.generate` is the loop of `generate`
(greedy or sampled, stop ids, the checks of `max_new` and the positions). The
rest is `Plans` with `reaches` and `planned` (the compiled steps by shape),
`Head` with `picked`, `top_candidates`, `chosen` and `ranked` (what a graph
answers and how it is read), `inputs`, `check_ids`, `in_place?` and
`usable_ids`. `open(m, usable = 0)`, `step`, `step_logits`, `step_top`,
`generate`, `finish`, `close` and `compiles` are the ones of `Generate`, and so
are the records `Stepping.Picked` and `Stepping.Candidates`.

- A full-attention layer keeps `[1, kv_heads, rows, head_dim]` keys and values
  as Qwen3's layers do (`F16` on ggml), padded to the capacity, written in place
  by a one-token step. A linear layer keeps the two `DeltaState` tensors in
  `F32`: their size does not grow, and a step makes new ones (the old ones are
  closed). A cache is the keys and values of the full layers then the two
  states of each linear layer; the graph's inputs are the ids, the positions,
  the one-hot of the last real token and the `Delta.Padding`.
- A prompt is padded to the capacity (8, 16, ...): the full layers mask the
  padding as before, the linear layers get `g` and `beta` of 0 on the padding
  rows and the convolution state picked at the last real token
  (`nn/DESIGN.md`). A decode step is one token with no padding: it gets no
  `Padding`, so no lists are built on the host, there is no extra product and
  the state is exact in every precision.
- **The ids the tokenizer cannot decode.** The embedding of Qwen3.5 has 248,320
  rows and the tokenizer 248,070 tokens; the 250 rows after them have no text.
  Hugging Face answers them when their logit is the largest or sampled. Here
  `Generator.usable` (`open(m, usable)`, the same in `Generate` of Qwen3, by
  `Stepping.masked`) is how many
  ids a step may answer: the logits of the others get `Nn.MASKED` in the graph,
  `step_top` and the sampler's window stop at `usable`, and `step`, `generate`
  never return one. The default 0 is no mask, so the ids are Hugging Face's
  exactly; a caller that has a tokenizer opens with its count
  (`Generate.open(m, tok.pieces.size)`, 248,070 for 0.8B). On the real
  checkpoint the ids with and without it are the same on the prompts.
- The reference backend runs a padded prompt and a decode step but is slow
  (the delta rule is the recurrence in doubles): the generation test runs on
  ggml, and the cached steps on both.

`chat.agc`: `Chat.apply(model_type, messages, add_generation_prompt,
enable_thinking)` is the ChatML text of `apply_chat_template` for the three
families, with the system turn of Qwen2.5 by default, the empty think block of
Qwen3 (`enable_thinking: false`, and before an assistant message that ends the
conversation), the `</think>` split of an assistant message (Qwen3) and no
tools (a `tool` role is an error). The golden is
Hugging Face's own output (`test/chat_golden.py`). `lfm3` is the template of
`LiquidAI/LFM3-1.7B` with tools (`test/chat_lfm3_golden.py`, 49 conversations of the
official `chat_template.jinja`); see LFM3 below. `qwen3_5` follows the
template of Qwen3.5 (`test/chat35_golden.py`, `fixtures/chat35_golden.json`):
every message is stripped like Python's `str.strip` (White_Space and
U+001C to U+001F), the system message must come first, a conversation needs a
user message (a `<tool_response>` one does not count), every assistant message
after the last user message gets a think block, empty or not, and the
generation prompt is `<think>\n` or, with `enable_thinking: some(false)`, the
empty block. `enable_thinking` is an `Optional[Bool]` that is `none` by
default, and `none` is what each template does when it is not told: Qwen3
thinks, Qwen3.5 does not (its template thinks only when the option is exactly
true), and Qwen2.5 refuses only `some(false)`. The `tokenizer.json` of
Qwen2.5 writes a merge as `"a b"`, where Qwen3's writes `["a", "b"]`;
`tokenizer/bpe.agc` reads both.

## Granite MoE

`granite_moe.agc`: `GraniteMoe.open(dir, backend, weights = F32)` (the folder of an official
`ibm-granite/granite-3.*-a*-instruct` repository: `config.json` and `model.safetensors` or its `.index.json`),
`GraniteMoe.run(m, rows, last)`, `GraniteMoe.logits(m, ids, last)` and `GraniteMoe.close_model(m)`. The forward is over
whole sequences, with no cache; generation is `generate_granite_moe.agc` (below).

- A layer is `Nn.Attention` (GQA, `Rope(base: rope_theta)`, no norms on the heads, no biases) and `Moe.Moe`
  (`nn/DESIGN.md`), under pre-norm RMS norms. `block_sparse_moe.router.layer.weight`, `input_linear.weight` (`[experts,
  2 * intermediate, hidden]`: the gate rows, then the value rows) and `output_linear.weight` are the router and the
  `gate_up` and `down` of the block as they are stored.
- Four multipliers of the config are not in a plain Llama forward: `embedding_multiplier` on the embeddings,
  `attention_multiplier` as the attention scale (`Nn.Attention.scale`, `1/64` in 1B-A400M, not `1/sqrt(64)`),
  `residual_multiplier` on each of the two branches before the add, and `logits_scaling` as a divisor of the logits.
- The head is the embedding table when `tie_word_embeddings` is true (its default is false, as in Hugging Face), else
  `lm_head`. `model_type`, `hidden_act` other than `silu`, `attention_bias` and `rope_scaling` that this loader does not
  run are errors, and so are `weights` other than `F32`, as `expert_matmul` is `F32` only.
- The batch goes through one graph (the block takes `[batch, seq, width]`; `Moe.moe` folds the tokens). The weights
  are 5.3 GB in `F32`: the real check runs at a peak of 5.6 GB on the CPU.

### Granite MoE generation

`generate_granite_moe.agc` gives `kv_decoder.agc` Granite's layer: the API, the compiled steps, the padded prompt, the
`usable` mask and the pick in the graph are those of [Qwen3.5 generation](#qwen35-generation) and
`Generate`. The cache is the keys and values of every layer, `[1, kv_heads, rows, head_dim]`, padded to a capacity, in
`F32` with no option: an `F16` cache moved the logits by up to 0.19, more than the gap of some greedy choices. A
prompt's padding rows go through the router and the experts too, which changes no real token (the mask keeps them off,
`Moe.moe` treats tokens alone). `GraniteMoe.times` and `GraniteMoe.scores` are what the forward and the steps share.

## LFM3

`lfm3.agc`: `Lfm3.open(dir, backend, weights = F32)` (the folder of an official `LiquidAI/LFM3-*` repository: `config.json`
and `model.safetensors` or its `.index.json`; `weights` is `F32` or, on ggml, `F16`), `Lfm3.run(m, rows, last)`,
`Lfm3.logits(m, ids, last)` and `Lfm3.close_model(m)`. The forward is over whole sequences, with no cache; generation is
`generate_lfm3.agc` (below).

- A layer is a mixer by `layer_types`: `ShortConv` (`nn/DESIGN.md`) for `conv`, and `Nn.Attention` for `full_attention`
  and `sliding_attention`, each followed by the SwiGLU MLP (`w1` the gate, `w3` the value, `w2` the output). Both branches
  have a norm before and one after (`operator_norm`, `post_operator_norm`, `ffn_norm`, `post_ffn_norm`), the output of the
  branch is normed before the residual add. `embedding_norm` is the final norm; there is no norm after the embeddings.
  Norms are plain `weight` scales (no `1 + weight`).
- The attention is GQA with a norm per head on the queries and keys before the rotation, and a gate of one number per
  head, `sigmoid(gate_proj(x))` (`Attention.head_gate`). A sliding layer sees its own position and the `sliding_window - 1`
  before it (`Nn.causal_mask` with a window; the whole prompt when the window covers it) and rotates by `rope_theta` of
  its `rope_parameters` entry; a global one rotates a `partial_rotary_factor` of the head (0.5 in 1.7B, `Rope.zero_tail`).
- The official code has no `hidden_act` and no bias in its config (its code fixes `silu` and `bias=False`), so the config
  fields the loader does not read (`use_cache`, `pad_token_id`, ...) change no result; `Load.dense` would read a `.bias`
  that a file had, which no official file does.
- The convolution of a `conv` layer is stored `[width, 1, conv_L_cache]` and kept `[width, taps]`. The head is the
  embedding table (`tie_word_embeddings` defaults to true), else `lm_head`.
- `model_type` other than `lfm3`, `conv_bias`, `use_sandwich_norm: false`, a `layer_types` entry other than `conv`,
  `full_attention` and `sliding_attention`, a `rope_type` other than `default`, a rotary width that is not an even number,
  a `sliding_attention` layer without `sliding_window`, and `weights` other than `F32` and `F16` are errors.
- The batch is unrolled, one sequence at a time, as in Qwen3.5 (the convolution is over one `[tokens, width]`). The model
  cannot be fine-tuned with `train/` yet (no `weights`/`with_weights`).
- The tokenizer is `Bpe` (see Tokenizers) and the template is `Chat.apply("lfm3", ...)`: `<|startoftext|>` first,
  `tools` (each a function, bare or under `function`) into the system turn as `<tools>` lines, calls as
  `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>` (a string as it is, the
  rest as `json.dumps(ensure_ascii=False)` writes it: `Pyjson.dumps`), `tool` messages as `<tool_response>` (consecutive ones with the same
  `tool_call_id` are one), and the reasoning of an assistant message (`thinking`) kept after the last user message, or
  everywhere with `preserve_thinking`. Not read: the aliases `reasoning` and `reasoning_content` of `thinking`, images and
  other content parts, `tool_reference`s, `additional_tools`,
  and a tool call whose arguments are a JSON string (the template refuses that one too).
- The tiny golden (`test/lfm3_golden.py`, rows of 9 tokens, window 4, the real rotary factor and thetas) is the official
  code's `float32` eager run. `tools/lfm3_check.agc` is the manual check on the real checkpoint (`test/lfm3_check_golden.py`).

### LFM3 generation

`generate_lfm3.agc` gives `kv_decoder.agc` LFM3's layer: the API, the compiled steps, the padded prompt, the `usable` mask
and the pick in the graph are those of [Qwen3.5 generation](#qwen35-generation) and Qwen3. The decoder learned two things
for it, both off for the models without them: a `Stepper` may have `states` (tensors of a size that does not depend on the
length, new at every step, kept in `Cache.states`) and `taps` (a prompt is then given the `Delta.Padding` of the
convolutions, as in Qwen3.5, so the padding rows leave the state as the real rows did).

- The cache holds the keys and values of the 11 attention layers, padded to a capacity of 8, 16, 32, ... (`F16` on ggml),
  and the state of the 13 convolution layers, `[6, width]` in `F32`: the last 6 rows of the input of the convolution.
- A sliding layer keeps every row and its mask hides the keys a window or more before the position
  (`padded_mask` with `window`), so its cache grows like a global one's and a decode step reads as many rows in a sliding
  layer as in a global one. A ring that drops the rows out of the window is not done: it would cut the reads of a sliding
  layer to 256 rows and halve the cache of a very long sequence (the five sliding layers hold 10 KB a token in `F16`, 4 kv
  heads of 128, against 12 KB for the six global ones), and nothing else.
- The prompt step takes its own `Delta.Padding`; every later step is one token with no padding. The masks of the
  prompt step are dense `[capacity, reach]` tensors and the attention scores of the whole prompt are too, so the cost of
  a prompt grows with the square of its length: the practical context is a few thousand tokens, not the 256K of the
  config (not measured above 601 tokens).
- The tiny golden of generation is the same file (`fixtures/lfm3_golden.json` has the 14 ids and scores of the official
  greedy `generate` from 7 ids, window 4): `test/generate_lfm3_test.agc` checks the logits of the cached steps against the
  whole forward on the reference (one row) and on ggml (two rows), and the greedy ids and scores against the official ones
  on ggml; a sliding layer that read the whole cache fails it.
- `tools/generate_lfm3_check.agc` is the manual check on the real checkpoint (`test/generate_lfm3_check_golden.py`, the
  official code's `float32` run layer by layer from the `bfloat16` file, which needed a 6 GB scope: 7.5 GB resident, mostly
  the mapped file; 5 GB was killed): the 24 greedy ids after each of six prompts of 5, 5, 14, 19, 375 and 601 tokens are equal
  with `F16` weights on ggml CPU and on Vulkan with `vulkan_fp16` off. The smallest gap between the
  two largest logits of a step is 0.116 (0.222 for the 601 tokens). The tokens past 256 test the window; the 601-token
  prompt leaves a cache of 1024 rows that a decode step reads 768 of (`Padded.reach`), the only test of a `reach` below the
  capacity: the tiny test has at most 32 rows, where they are equal. The largest gap of a logit at the five ids the
  official code found largest is 3.6e-3 on CPU with the `F16` cache and 2.9e-5 with an `F32` cache (the cache line of
  `generate_lfm3.agc` changed for the run, not kept): the `F16` cache is the whole gap, `F16` weights are not. Peak 4.0 GB
  on CPU at 601 tokens, nothing longer measured. `tools/bench_lfm3.agc` is the timing row of `benchmarks/README.md`.

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

`tokenizer/wordpiece.agc` (BERT) and `tokenizer/bpe.agc` (Qwen, Qwen3.5) read a
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

The tokenizer of LFM3 is `Bpe` too: `PATTERN_GROUPS` is Qwen3's with `\p{N}{1,3}` for `\p{N}` (the field `digits`, the
most numbers in a piece), it has no normalizer (the text is not NFC), and `ignore_merges` is on: a piece that is a token
of the vocabulary is that token and the merges are not run (`tokenizers` does the same). The 124,893 pieces and 140
added tokens (`<|startoftext|>` 124894, `<|im_start|>` 124899, `<|im_end|>` 124900) are read as Qwen's are; no added
token has a strip, single-word or normalized flag.

The split of Qwen3.5 is Qwen3's with `[\p{L}\p{M}]` where it has `\p{L}`, and
`\p{M}` also excluded from the symbol class: `Bpe` reads either pattern
(`PATTERN`, `PATTERN_MARKS`, the field `marks`) and anything else is an error (LFM3's: `PATTERN_GROUPS`, see below).
`\p{M}` is a table of its own (`Unicode.any_mark?`, `tools/unicode_tables.py`:
Mn, Mc and Me); the nonspacing table of BERT's accent stripping stays. The
vocabulary of Qwen3.5 has 248,070 tokens (248,044 pieces and 26 added tokens,
`<|im_start|>` is 248045); the embedding has 248,320 rows.

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
(`Resize(256, Cubic)`, `CenterCrop(224)`, `to_tensor` and `normalize` with the ImageNet mean and std), which is how
`crop_pct` 0.875 of 224 reads. The pinned `preprocessor_config.json` (the same in both checkpoints) says `resample` 3 (Pillow's bicubic), `size` 224, `crop_pct` 0.875, `do_normalize` with the ImageNet mean and std. The golden used the slow processor (`use_fast=False`). The pipeline resizes with `Image.Kernel.Cubic`, the kernel that `resample` 3 names; two differences stay: the pixels are up to 0.0175 from Pillow's after normalizing, which is one 8-bit level (1 / 255 / 0.224); of the 602,112 samples of the four drawn images 95.5% are Pillow's exactly, 4.5% are one level off and 0.05% two (none more), so the kernels agree to within rounding (the committed 8-bit crops of the processor against ours, counted for this note); and `Vision.CenterCrop` rounds a half-pixel offset to the even pixel, as torchvision does, where Hugging Face floors it (`vision/` is not changed for it). `ResNet.run` gives the logits `[batch, classes]` and `ResNet.top(m, logits, k)` the `k`
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

`vit.agc` reads `ViTForImageClassification` checkpoints (`google/vit-base-patch16-224`): a patch embedding, a class token and learned positions, `Nn.EncoderStack` of pre-norm `Nn.EncoderBlock`s (`layernorm_before`, attention, `layernorm_after`, an MLP with the exact GELU, `hidden_act` `gelu`), the final `vit.layernorm`, and a linear head on the class token. The input is `[batch, 3, 224, 224]` in `F32`, already normalized as the checkpoint's image processor does; `vision/` makes it (`ResizeTo(224, 224, Linear)` and `normalize` with mean and std 0.5). `ViT.run` gives the logits `[batch, classes]` and `ViT.top(m, logits, k)` the `k` best classes with their `id2label` names (`Classify.top`, shared with ResNet).

- The patch embedding is `Nn.conv2d` with stride and kernel `patch_size` (`[width, 3, 16, 16]`, as stored), its `[batch, width, 14, 14]` output reshaped to `[batch, width, 196]` and permuted to tokens: a reshape and a linear layer would be the same product, but they need an unfold of the image that the convolution already is. The class token is broadcast over the batch and concatenated first, and the position table `[1, 197, width]` is added: a different image size needs another table, so an input that is not `image_size` by `image_size` is an error (no interpolation).
- `eps` of the layer norms is `layer_norm_eps` (1e-12). `qkv_bias` false drops the biases of the query, key and value. A `model_type` other than `vit` and an activation other than `gelu` are errors. The names and shapes of every tensor are checked against `config.json` before any is loaded, so a checkpoint that does not fit leaves nothing open.
- `run` is one staged graph on ggml (`Lift.staged`), `forward` the same as plain `Tensor` calls, as ResNet's.
- The image processor of the checkpoint (`preprocessor_config.json`: `size` 224, `resample` bilinear by default, `do_normalize` with mean and std 0.5) stretches the whole image to 224 by 224 with no crop and keeps no aspect ratio, which `Vision.Step.ResizeTo(224, 224, Image.Kernel.Linear)` does (`resample` 2 is Pillow's default, which the processor's config leaves out). With libvips' `Linear` the pixels are up to 0.0078 (normalized, one level of 1 / 255 / 0.5) from Pillow's: 92.1% of the samples of the four images are Pillow's exactly, 7.8% one level off, 0.08% two; with its default Lanczos 3 (before `image-0.2.0` had a kernel option) they were up to 0.0235.
- The official checkpoint at a pinned revision, `model.safetensors` only (never a pickle, `pytorch_model.bin`, or the flax and TensorFlow files):

  | Checkpoint | Revision | `model.safetensors` sha256 |
  |---|---|---|
  | `google/vit-base-patch16-224` | `3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3` | `1cea07110a4a47edc51420b2dda6f3b8b58e7256e8f44b4ea6aa9696162ccb5d` |

  `config.json` sha256: `abdf73cc4ce79495a83f32a8c9265daa7a1363f5a3efa4704c88a6e7f45ad19d`. `preprocessor_config.json` sha256: `90d1ae427c27d2b9dcb01f6a62ba96aa67aac2de1697c59cd212055b5035c4a2`.

## CLIP

`clip.agc` reads `CLIPModel` checkpoints (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`): a vision tower and a text tower into one space of `projection_dim`. `Clip.image_embeds(m, images)` and `Clip.text_embeds(m, rows)` give `get_image_features` and `get_text_features` of Hugging Face (not normalized); `Clip.logits` is `exp(logit_scale)` times the cosines of images and texts (`logits_per_image`) and `Clip.probabilities` its softmax over the texts, which is zero-shot classification with a prompt for each class. Each call is one staged graph on ggml (`Lift.staged`). The weights are read by name through `Weights` (the names and shapes are checked before any tensor is loaded, and a failure closes the ones loaded), which `ViT` uses too.

- The vision tower is a convolution of stride `patch_size` without a bias (`[width, 3, 32, 32]`), the class embedding in front of the patches, the position table `[50, width]`, `pre_layrnorm` (the spelling of the checkpoint), pre-norm `Nn.EncoderBlock`s, and `post_layernorm` and `visual_projection` (no bias) on the class token. An input that is not `image_size` by `image_size` is an error, as ViT's.
- The text tower is `token_embedding` and the first rows of `position_embedding`, the blocks under a causal mask (`causal` of `Nn.encoder_stack`), `final_layer_norm`, and `text_projection` (no bias) on the vector of the end token. Hugging Face reads the end token as the first largest id of a row (its config says `eos_token_id` 2, which makes it do that); the model reads the first of the largest id of each row, as `Sentence.mean_pooled` over a mask that is 1.0 there, so a literal `<|endoftext|>` in the text ends the vector at that one, as Hugging Face's does. Rows are padded to the longest, and with a causal mask the padding cannot change the vector at the end token, so no padding mask exists. A row longer than `max_position_embeddings` (77) or an empty one is an error.
- `hidden_act` is `gelu` (the exact one, `Nn.Activation.GeluErf`) in both towers of this checkpoint; any other activation (OpenAI's own checkpoints say `quick_gelu`) is an error, since nothing here needs it. `model_type` must be `clip`.
- The checkpoint is LAION's, not OpenAI's: `openai/clip-vit-base-patch32` has no `model.safetensors` on its main branch (a pickle, `pytorch_model.bin`, and TensorFlow and Flax files), and the only safetensors are the open "Adding safetensors variant" pull requests of a bot, none merged. The owner accepted one official alternative with safetensors: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K`, LAION's ViT-B/32 trained on LAION-2B (MIT license), which publishes `model.safetensors` and a `CLIPModel` `config.json` on its main branch. Its architecture is OpenAI's ViT-B/32 (a 32 by 32 patch, 12 blocks of 768 for the images, 12 of 512 for the text, 512 dimensions) but its weights, and so its answers, are not OpenAI's. Only `model.safetensors` is read, never `pytorch_model.bin`, `open_clip_pytorch_model.bin` or `open_clip_model.safetensors` (another naming).

  | Checkpoint | Revision | `model.safetensors` sha256 |
  |---|---|---|
  | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | `1a25a446712ba5ee05982a381eed697ef9b435cf` | `74813fbcdc750f235c9784c367ca1394d2a5c25eb0aac92761752ac239db7cff` |

  `config.json` sha256: `1284cbff35169abb23a1c5525a8b0f543c7bd191d4b9aed63880c1571bc4191c`. `preprocessor_config.json`: `910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd`. `vocab.json`: `5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363`. `merges.txt`: `f1cd414c2d113a28b296afc0fd66ae854c0e0c34d5de373a2f5b7f62e0c02344`.
- The image processor (`preprocessor_config.json`: `size` 224, `resample` 3 (bicubic), `crop_size` 224, `do_normalize` with mean `[0.48145466, 0.4578275, 0.40821073]` and std `[0.26862954, 0.26130258, 0.27577711]`) is the pipeline of ResNet with other constants: `Resize(224, Image.Kernel.Cubic)`, `CenterCrop(224, 224)` and `normalize`. The pixels are up to 0.015 from Pillow's with `Cubic` (one level, 1 / 255 / 0.2613; 95.7% of the samples exact, 4.3% one level off, 0.03% two; the same maximum as with Lanczos 3), and `CenterCrop` rounds a half-pixel offset to the even pixel where Hugging Face floors it.
- The tokenizer is `tokenizer/clip.agc`, from `vocab.json` and `merges.txt`. It follows the slow `CLIPTokenizer` of `transformers` without `ftfy` (not installed), since that is the one with the cleaning written down: the `BasicTokenizer` (control characters dropped, white space made a space, a space around each CJK ideograph, NFC, lowercase, no accent stripping, no punctuation split; `WordPiece.cleaned_chars` and `spaced_chinese` are BERT's functions, made public), CLIP's split (`<|startoftext|>`, `<|endoftext|>`, `'s 't 're 've 'm 'll 'd`, a run of letters, one digit, a run of anything else), UTF-8 bytes in GPT-2's alphabet with `</w>` on the last one, and `Bpe.merge` (the merge loop of the Qwen tokenizer, now over a table of ranks). The merges are the lines after the header, at most 48894 (what the slow tokenizer slices). A symbol that the vocabulary lacks is `<|endoftext|>`. Hugging Face caches the BPE of each word; this does not (a prompt is a few words, and the merge is fast enough: a cache is a Map threaded through `encode`, left out until a bench needs it). `encode` adds the start token and the end token and cuts the text to `max_length - 2` tokens (77 by default). Two known differences: Python's lowercase also skips the case-ignorable characters (an apostrophe, a full stop) when it decides on a final sigma, which this does not; and the fast tokenizer (`tokenizer.json`) does not drop control characters or space the CJK ideographs, so it differs from the slow one on those.

## Tests

A new op (`Tensor.gated_delta_rule`, `softplus`, `l2_norm`) gets its full tests
in its core package (`tensor/`, `nn/`): a Python or Hugging Face peer, and ggml
against the reference. A new architecture (a model that wires existing ops)
gets one end-to-end golden of a tiny random model against Hugging Face (it
catches the wiring: a norm `+1`, the projection order, the gate layout, the
template defaults), a manual check on the real checkpoint (not committed; the
command is in the header of its `tools/` file), and only the config refusals
that prevent a silently wrong result. It gets no per-layer golden when the
model golden exists, no node-by-node ggml test (the core covers it) and no
error row per config field.

- `test/qwen3_test.agc` against `test/qwen_peer.py` (standard library, doubles):
  a tiny random Qwen3 (4 heads, 2 key heads, `head_dim` 4 with hidden 8, so the
  projection is wider than the model) in `bfloat16` with tied embeddings and
  in `float32` untied, on the reference and on ggml.
- `test/granite_moe_test.agc` against `test/granite_moe_golden.py` (manual, Hugging Face's `GraniteMoeForCausalLM` in
  `float32`, `fixtures/granite_moe_golden.json`): a tiny random model (2 layers, hidden 16, 6 experts of which 2 are
  picked, the four multipliers set to values other than 1, tied head) written as a safetensors file under the official
  names; the logits of every position of two rows match on the reference (1e-4) and on ggml CPU (1e-3); the cached steps
  of `Generate` (a padded prompt of 3 ids, then one id at a time) give the logits of the whole forward, and a greedy
  `generate` of 14 ids from a prompt of 7, crossing the capacities 8, 16 and 32, gives Hugging Face's ids and logits, on
  the reference and on ggml; an untied head is the file's; a config the model does not run and weights other than `F32`
  are errors.
- `tools/generate_granite_moe_check.agc` against Hugging Face's greedy `generate` (`test/granite_moe_check_golden.py`,
  manual): 24 new ids for each of five prompts of 5 to 217 tokens of `ibm-granite/granite-3.1-1b-a400m-instruct`, equal on
  ggml CPU (the smallest gap between the two largest logits of a step is 0.115). It feeds the prompt and Hugging Face's
  ids to `step_logits` and compares the five largest logits of each step: F32 cache, 8.2e-5 from Hugging Face's. About 17
  tokens per second after the prompt, 5.6 GB at the peak.
- `tools/granite_moe_check.agc` against Hugging Face's `GraniteMoeForCausalLM` on `ibm-granite/granite-3.1-1b-a400m-instruct`
  (`test/granite_moe_check_golden.py`, manual: it loads the checkpoint in `bfloat16` and turns one layer at a time to
  `float32`, so it fits in 5 GB): five prompts of 5 to 217 tokens, the logits of the last position over the vocabulary
  and the five largest ids of every position (the long prompt at its last position only). Largest gap of a logit: 3.7e-5
  on ggml CPU (peak 5.6 GB), 9.3e-5 on Vulkan with `vulkan_fp16` off, the top five equal; with `vulkan_fp16` on, 0.34,
  and the fifth id of two positions differs (logits are 14 to 28 at the top). The cause is the router, not
  `Tensor.expert_matmul` (ml#279; Vulkan, five prompts, 6552 pairs of a token and a layer). fp16 puts the router logits
  1e-4 off in relative rms (0.003 at most, layer 0), and the 8th and 9th largest logit of a token are closer than
  0.003 in 223 pairs (median gap 0.068), so the set of experts differs from the `F32` run's in 67 pairs (1.0%): none
  on the prompts of 5 and 7 tokens (largest gap of a logit over the last positions 0.01), 3, 25 and 39 on the
  prompts of 14, 30 and 217 tokens (1.35, 0.87 and 0.45 over their last positions, up to 30, and 0.34, 0.25 and 0.025
  at the last position: 0.34 is the figure above). With the ids of the `F32` run fed to the fp16 run the gaps
  are 0.033, 0.046 and 0.021 (0.018, 0.010 and 0.012 at the last position), the order of dense Qwen3-0.6B. On
  layer 0's weights, 120 tokens, `expert_matmul` and a dense `matmul` of the same shape are the same product to
  three digits: relative rms 2e-4 against `F32` on the CPU in fp16, 5e-7 with it off (gate_up and down, 8 experts).
  So `mul_mat_id` is no worse per product than `mul_mat`, and a model with a hard top-k choice moves in steps
  that the fp16 noise can flip. For context,
  `tools/qwen_check.agc` on Qwen3-0.6B (dense, five prompts of 5 to 50 tokens, logits up to 30) with `VULKAN_FP16=1`
  gives 0.028 (0.8e-4 with it off), the fifth id of two positions differing. The checkpoint:

  | Checkpoint | Revision | `model.safetensors` sha256 |
  |---|---|---|
  | `ibm-granite/granite-3.1-1b-a400m-instruct` | `0da7a48b0276d500ce5922fd2b33944091fc6c09` | `ac02591061f1344027a7e7b11dbb4143f75f166c47dc09b742f5de3ab1dde1d1` |

- `test/qwen35_test.agc` against `test/qwen35_golden.py` (manual, Hugging Face's
  `Qwen3_5ForCausalLM` in `float32`, `fixtures/qwen35_golden.json`): a tiny
  random model (3 linear layers and 1 full layer, hidden 16, partial rotary
  half, tied head) written as a safetensors file under the official names; the
  logits of every position of two rows match on the reference (1e-4) and on
  ggml CPU (1e-3); an untied head is the file's; a mixture-of-experts config,
  linear key and value widths that differ and weights other than `F32` are
  errors. The tiny model's linear head width is 4, which ggml-vulkan's gated delta rule does not run (16, 32, 64 and 128
  only), so these tests and `generate_qwen35_test.agc` leave Vulkan out (`Support.ggml_backends_for_delta`); the real
  checkpoint checks below cover Vulkan.
- `test/bpe_test.agc` on `fixtures/bpe_marks.json`, a tiny tokenizer with the
  split of Qwen3.5 (`test/tokenizer_golden.py marks`, `tokenizers` 0.23.3):
  the ids and text of `tokenizers` on the strings of the BPE test, 16 where a
  mark decides the cut and 60 random ones. `tools/tokenizer_check.agc bpe` on the
  `tokenizer.json` of `Qwen/Qwen3.5-0.8B` (manual, 300 random strings more):
  1,827 cases agree.
- `test/bpe_test.agc` on `fixtures/bpe_groups.json`, a tiny tokenizer with the split of LFM3, no normalizer and
  `ignore_merges` with its last 40 merges cut off, so that flipping the flag changes ids (`test/tokenizer_golden.py
  groups`, `tokenizers` 0.23.3): the ids and text of `tokenizers` on the strings of the BPE test, 30 of digit groups,
  scripts and NFC and 20 random ones. `tools/tokenizer_check.agc bpe` on the real `tokenizer.json` of
  `LiquidAI/LFM3-1.7B` (revision `1da06bec6a819da10b54c67e5c90972b46d81871`, sha256
  `7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701`; manual, 500 random strings more): 2,941 cases agree,
  and 1,354 more on 1,184 chunks and lines of the repository's own files.
- `test/pyjson_test.agc` against `test/pyjson_golden.py` (standard library): 321 texts of `json.dumps(value, ensure_ascii=False)`
  (floats at the edges of `repr` and of random bit patterns, big integers, control characters, nested objects with their
  keys out of order), parsed and written again by `Pyjson.dumps`.
- `test/chat_test.agc` against `test/chat_lfm3_golden.py` (`transformers` 5.22.2, `jinja2` 3.1.6, the sha256 of the four files asserted):
  49 conversations of the official LFM3 template, the Jinja run by Hugging Face, with tools (bare and under `function`,
  `defer_loading`), calls with arguments of every JSON kind (`json.dumps(ensure_ascii=False)` text, floats as `repr`), one or
  several answers per call, reasoning kept or dropped, `preserve_thinking`, `CONTINUE_FINAL_MESSAGE_TAG`, other roles.
- `test/chat_test.agc` against `test/chat35_golden.py` (`transformers` 5.22.2,
  the sha256 of the template and of `tokenizer.json` asserted): 28
  conversations of the Qwen3.5 template (white space of every kind, one
  `</think>`, two, none, an unclosed `<think>`), 4 of them refused by it.
- `test/generate_qwen35_test.agc` against the same golden (it adds a greedy
  `generate` of Hugging Face from a prompt of 7 ids, 14 new ids and their
  logits): the cached steps (a padded prompt of 3 ids, then one id at a time)
  give the logits of the whole forward on the reference and ggml, and the
  greedy generation crossing the capacities 8, 16 and 32 gives Hugging Face's
  ids and logits on ggml. `usable` is tested on ggml.
- `tools/generate_qwen35_check.agc` against Hugging Face's greedy `generate`
  (`test/qwen35_check_golden.py`, manual): 24 new ids for each of six prompts
  of `Qwen/Qwen3.5-0.8B` (5 to 375 tokens, the last one padded to 512 rows),
  equal on ggml CPU and on Vulkan with `vulkan_fp16` off (the smallest gap
  between the two largest logits of a step is 0.034), also with
  `USABLE=248070`. It also feeds the prompt and Hugging Face's ids to
  `step_logits` and compares the five largest logits of each step: the largest
  gap is 2.2e-3 on CPU and 2.1e-3 on Vulkan, the cost of the `F16` KV cache.
  About 9 tokens per second after the prompt on both, 3.9 GB at the peak on the
  CPU and 1.3 GB on Vulkan.
- `tools/qwen35_check.agc` on the same prompts (a prompt of more than 100
  tokens at its last position only): largest gap of a logit 2.0e-5 on CPU and
  5.5e-5 on Vulkan, the top five ids equal at every position checked.
- The checkpoint of those checks:

  | Checkpoint | Revision | `model.safetensors-00001-of-00001.safetensors` sha256 |
  |---|---|---|
  | `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696` |

  `tokenizer.json` sha256: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`;
  `chat_template.jinja`: `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`; the committed
  `test/fixtures/qwen35_golden.json` (a tiny random model, no checkpoint): `07665f66d6f3a7be24c30b3b59ee45436cf7fcce143242d604523804cae22865`.
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
- The same two tools on a Q4_K_M file (`llama-quantize` from the F16 file: Q4_K and Q6_K
  matrices, `F32` norms; 456 MiB) with `WEIGHTS=q4_k` (manual, ggml CPU, the llama.cpp build of the Q8_0
  check). Ours with `F32` weights (the file dequantized by `float_file.h`)
  against llama.cpp on the same file dequantized to `F32` (`llama-quantize --allow-requantize`):
  the largest gap of a logit is 0.021, so the Q4_K and Q6_K dequantizers give ggml's values on a
  real model. With `weights: Q4_K` against llama.cpp's own run of the Q4_K_M file the largest gap
  is 1.66 over the five prompts (logits up to 30), within the noise of the rounding of the
  activations to Q8_K blocks: llama.cpp's Q4_K_M run against its own `F32` run of the dequantized
  file differs by 0.65 to 1.97. The run takes 1.6 GB at the peak of `qwen_check` against 3.2 GB
  with `F32` weights. Speed and memory against llama.cpp on the same file are in
  `benchmarks/README.md` (ml#291): on the CPU 1.12 times its prompt and 0.89 times its tokens, on Vulkan 0.84 times both.
- `test/resnet_test.agc` against `test/resnet_peer.py` (standard library, doubles, the batch norms not folded): three
  tiny random ResNets (bottleneck, basic, and a bottleneck with `downsample_in_first_stage` and
  `downsample_in_bottleneck`) with batch norms that have statistics of their own, on the reference (1e-9) and on ggml
  (1e-4), plus the errors of a missing weight and an unsupported config.
- `vision/test/resnet_hf_test.agc` against Hugging Face (the golden script `vision/test/resnet_golden.py` is manual: it needs `torch` 2.6.0,
  `transformers` 4.57.6 and `pillow` 11.3.0, in a venv, run with `python -I`; the test does nothing unless `RESNET_MODELS` names the checkpoint folders): four drawn images (`vision/test/resnet_images.py`,
  140 KB in all, no license) through `Resize(256, Cubic)`, `CenterCrop(224)`, `normalize` and the model, in `float32` on ggml
  CPU and Vulkan. The best 10 logits and the top 5 classes of each image are `float32` Hugging Face's, in the same
  order, for both models. On Hugging Face's own pixels the largest gap of a logit is 1.6e-5 (ResNet-50) and 1.1e-5
  (ResNet-18) on CPU, 2.1e-5 on Vulkan; with the pixels of our own pipeline (`Resize(256, Image.Kernel.Cubic)`) 0.092 (ResNet-50) and 0.048 (ResNet-18) on
  CPU and Vulkan (0.16 and 0.065 with libvips' Lanczos 3), from the pixels, which are one level (0.0175 normalized) from
  Pillow's at the most with either kernel; the test also reports the mean gap of the means and the 8 by 8 grid
  (0.0006) and the share of those values that are not Hugging Face's (3.3% for ResNet-50, 3.5% for CLIP, 7.6% for
  ViT), so a change below one level shows. The 224 by 224 crops that the processor made are committed as 8-bit PNGs too (`vision/test/fixtures/resnet/crops`, 113 KB: `pixel_values` is exactly those samples over 255, normalized), so the model alone runs from committed files on its checkpoint, with a tolerance of 1e-3. The golden is committed (`vision/test/fixtures/resnet/golden.json`) and never made in CI. The tolerances are about twice the gaps measured: 0.035 for the pixels and 0.2 for the logits after the pipeline. Vulkan runs with `Tensor.vulkan_fp16(false)`.
- `test/vit_test.agc` against `test/vit_peer.py` (standard library, doubles): two tiny random ViTs (with and without the biases of the query, key and value) on the reference (1e-9) and on ggml (1e-4), plus the errors of a missing weight, an unsupported config and an input of another size.
- `vision/test/vit_hf_test.agc` against Hugging Face (the golden script `vision/test/vit_golden.py` is manual: it needs `torch` 2.6.0, `transformers` 4.57.6 and `pillow` 11.3.0, in a venv, run with `python -I`; the test does nothing unless `VIT_MODEL` names the checkpoint folder): the four drawn images of the ResNet test through `ResizeTo(224, 224, Linear)`, `normalize` and the model in `float32` on ggml CPU and Vulkan. The 224 by 224 images that the processor made are committed as 8-bit PNGs (`vision/test/fixtures/vit/crops`, 100 KB: `pixel_values` is exactly those samples over 255, normalized), so the model alone runs from committed files: the best 10 logits are Hugging Face's within 1e-4 (the largest gap measured is 1.2e-5 on CPU and 9.6e-6 on Vulkan) and the top 5 are identical, in the same order. Through our own resize (`Linear`) the pixels are 0.0078 from Hugging Face's and the logits 0.103 (0.0235 and 0.156 with Lanczos 3), from the pixels, on CPU and Vulkan; and the top 5 of the pipeline is Hugging Face's in the same order, on CPU and Vulkan (with Lanczos 3 the logits of these images, within 0.03 to 0.3 of each other, were reordered, and the check allowed for the ties). The tolerances are about twice the gaps measured: 0.016 for the pixels, 0.21 for the pipeline. The golden is committed (`vision/test/fixtures/vit/golden.json`) and never made in CI. Vulkan runs with `Tensor.vulkan_fp16(false)`.
- `test/clip_test.agc` against `test/clip_peer.py` (standard library, doubles): a tiny random CLIP (vision tower of 8 wide, text tower of 6, 6 dimensions) on the reference (1e-9) and on ggml (1e-4): the embeddings of two images and of four rows of ids (one with the end token twice), the scaled cosines, the probabilities and the scale; plus the errors of a missing weight, an unsupported config (`quick_gelu`, a patch size that does not divide the image, another `projection_dim`), and an input that does not fit.
- `test/clip_tokenizer_test.agc` against `test/clip_tokenizer_golden.py` (`transformers` 4.57.6, the slow `CLIPTokenizer`, manual): the ids of 66 strings (case, contractions, digits, accents, CJK, control characters, the special tokens in either case, truncation at 77) with a tiny tokenizer trained on a few prompts (`test/fixtures/clip`, 15 KB). `tools/tokenizer_check.agc clip` checks the real tokenizer of the checkpoint the same way: 466 strings, 400 of them random, agree.
- `vision/test/clip_hf_test.agc` against Hugging Face (the golden script `vision/test/clip_golden.py` is manual: it needs `torch` 2.6.0, `transformers` 4.57.6 and `pillow` 11.3.0, in a venv, run with `python -I`; the test does nothing unless `CLIP_MODEL` names the checkpoint folder): the four drawn images of the ResNet test and six prompts. The ids of the prompts are Hugging Face's. On the 8-bit crops that the processor made (`vision/test/fixtures/clip/crops`, 112 KB), the largest gap is 5e-6 (text embeddings) and 2e-5 (image embeddings), 4e-5 for a logit (`logit_scale` makes the scale 100) and 2e-6 for a probability, on CPU and Vulkan, and the best prompt of each image is Hugging Face's. Through our own pipeline (`Resize(224, Cubic)`) the pixels are 0.015 from Pillow's and a logit 0.21 (0.43 with Lanczos 3; the pixel gap times the scale 100), on CPU and Vulkan; the best prompt is the same. The tolerances are about twice the gaps measured for the pipeline (0.03 and 0.43) and looser for the model alone (1e-4 for an embedding, 1e-3 for a logit). The golden is committed (`vision/test/fixtures/clip/golden.json`, 57 KB) and never made in CI. Vulkan runs with `Tensor.vulkan_fp16(false)`.
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
