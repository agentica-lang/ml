---
title: transformers
description: BERT and Qwen3 in Agentica from Hugging Face checkpoints, and the tokenizers of both that give Hugging Face's ids.
weight: 7
---

`transformers` is BERT and Qwen3 in Agentica, for inference. It has BERT (`bert-base-uncased` and its relatives), Qwen3 (the forward and greedy generation with a KV cache), and the tokenizers of BERT and Qwen: they read the `tokenizer.json` that sits beside a Hugging Face checkpoint and give the same ids as Hugging Face `tokenizers`.

## BERT

`Bert.open(dir, backend)` loads `model.safetensors` and `config.json` from a folder. `Bert.run(m, encodings)` takes encodings from the tokenizer, pads them to the longest, and gives `Output(hidden, pooled)`: the last hidden state `[batch, seq, hidden_size]` and the pooler output `[batch, hidden_size]` (none when the checkpoint has no pooler). `Bert.embed(m, encodings)` gives one vector per sentence, the mean of its token vectors without the padding.

```agentica
import Bert from "pkg/transformers/bert"
import Tensor from "pkg/tensor/tensor"
import WordPiece from "pkg/transformers/tokenizer/wordpiece"

def main() -> Void
  backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  tok = WordPiece.open("bert-base-uncased/tokenizer.json").unwrap!
  m = Bert.open("bert-base-uncased", backend).unwrap!
  sentences = ["Hello, world!", "The quick brown fox jumps over the lazy dog."]
  encodings = sentences.map { |s| WordPiece.encode(tok, s).unwrap! }
  out = Bert.run(m, encodings).unwrap!
  puts(Tensor.shape(out.hidden))
  puts(Tensor.shape(Bert.embed(m, encodings).unwrap!))
end
```

```output
[2, 12, 768]
[2, 768]
```

- A run's outputs, and the model itself, hold ggml memory that `Bert.close(out)` and `Bert.close_model(m)` free at once; `embed` frees what it made. Nothing may use a closed model: a call that does is an error.
- The padding is masked as Hugging Face does, so a sentence in a padded batch gets what it gets alone.
- The model matches `BertModel` in `float32` on ggml CPU and Vulkan: the largest gap is 5e-5 on `bert-base-uncased`, over six sentences and a padded batch.
- Checkpoints with the `bert.` prefix and `gamma`/`beta` layer norms (`bert-base-uncased`) and those without (`save_pretrained` today) both load; the `cls.*` head is ignored. A `model_type` other than `bert`, a position embedding other than `absolute` and an activation other than `gelu`, `gelu_new`, `gelu_pytorch_tanh` and `relu` are errors.
- On ggml the forward is one graph with the weights as constants; the other backends run it eagerly. The weights are `F32` in memory (about 0.4 GB for `bert-base-uncased`).
- The sequence length may not pass `max_position_embeddings` (512), and every encoding needs a token.
- Reference and ggml are tested; libtorch is not. A sharded checkpoint (`model.safetensors.index.json`) is not supported yet.
- Installed from outside this repository, `transformers` pins the `hub` of an earlier commit, which does not read `bert-base-uncased`'s `gamma` and `beta` layer norms: it needs the next ml release, which moves the pins.

## Qwen3

<<<<<<< HEAD
`Qwen3.open(dir, backend, weights)` loads `config.json` and `model.safetensors` from a folder (`Qwen/Qwen3-0.6B`, whose weights are `bfloat16` on disk and `F32` in memory). `Qwen3.run(m, rows)` takes rows of token ids of one length and gives the logits `[batch, seq, vocab_size]`: each position reads itself and the ones before it. With `last: true` it gives only the last position's, `[batch, 1, vocab_size]`, the row that picks the next token. With `weights: Dtype.F16` the big matrices (the projections, the token table and the output head) are kept as half weights on ggml: half the memory and a decode 39% to 60% faster on Vulkan, with a load close to `F32`'s (the CPU gain is not claimed: see the hub page). `F32` is the default and the exact path, and Vulkan multiplies in fp16 by default (`Tensor.vulkan_fp16`), which breaks the Hugging Face goldens with `F32` weights too: `Tensor.vulkan_fp16(false)` before the first ggml call makes it exact, as the checks of this package do.
=======
`Qwen3.open(dir, backend, weights)` loads `config.json` and `model.safetensors` from a folder (or one `.gguf` file, whose key-values give the config) (`Qwen/Qwen3-0.6B`, whose weights are `bfloat16` on disk and `F32` in memory). `Qwen3.run(m, rows)` takes rows of token ids of one length and gives the logits `[batch, seq, vocab_size]`: each position reads itself and the ones before it. With `last: true` it gives only the last position's, `[batch, 1, vocab_size]`, the row that picks the next token. With `weights: Dtype.F16` the big matrices (the projections, the token table and the output head) are kept as half weights on ggml: half the memory and a decode 39% to 60% faster on Vulkan, with a load close to `F32`'s (the CPU gain is not claimed: see the hub page). `F32` is the default and the exact path, and `Tensor.vulkan_fp16` stays off by default because it breaks the Hugging Face goldens, with `F32` weights too.
>>>>>>> origin/main

```agentica
import Bpe from "pkg/transformers/tokenizer/bpe"
import Qwen3 from "pkg/transformers/qwen3"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  backend = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
  tok = Bpe.open("Qwen3-0.6B/tokenizer.json").unwrap!
  m = Qwen3.open("Qwen3-0.6B", backend).unwrap!
  ids = Bpe.encode(tok, "The capital of France is").unwrap!
  puts(ids)
  puts(Tensor.shape(Qwen3.run(m, [ids], true).unwrap!))
end
```

```output
[785, 6722, 315, 9625, 374]
[1, 1, 151936]
```

- The model matches `Qwen3ForCausalLM` in `float32` on ggml CPU and Vulkan: over five prompts of 5 to 50 tokens the largest gap of a logit is 4.9e-5 on the CPU and 8.6e-5 on Vulkan, and the five largest logits name the same ids at every position.
- The layers are grouped-query attention with a normalized query and key per head, rotary positions with `rope_theta`, a SwiGLU MLP, RMS norms, and an output head tied to the embedding table when the config says so (the file's `lm_head.weight` is then ignored).
- `Qwen3.run` is the forward over a whole sequence, with no cache: on ggml every call traces and plans a graph (0.6 to 1.7 seconds on the CPU, 0.2 to 0.6 second on Vulkan for these prompts). To generate, use `Generate`.
- `Qwen3.close_model(m)` frees the weights; the logits belong to the caller (`Lift.close`). `qwen_check` peaks at 3.7 GB on ggml CPU and 1.5 GB on Vulkan, where the weights live in device memory.
- A `model_type` other than `qwen3`, an activation other than `silu`, attention bias, `rope_scaling`, `use_sliding_window: true` (a `sliding_window` size with it `false` is ignored, as Hugging Face does), a head count that the key heads do not divide, and a tensor of another shape are errors. Qwen2.5 and others come later. Reference and ggml are tested; libtorch is not, and a sharded checkpoint is not tested. On the reference backend attention needs `q * k` of at most 1,048,576, about 1024 tokens with the causal mask; ggml has no such limit.
- The pins of `transformers` in `agentica.toml` are those of `asr`, as for BERT.

## Generation

`Generate.open(m)` gives the generator of a model, which keeps the compiled steps for every sequence of that model. `Generate.generate(g, ids, max_new, stop, sampling)` adds at most `max_new` ids to the prompt `ids`, and stops before an id of `stop`. By default it takes the id with the largest logit (greedy), which runs in the graph. With `sampling: some(Sample.Sampling(temperature: 0.8, top_k: 40, top_p: 0.95, seed: 7))` each step draws among the largest logits with Hugging Face's order (the temperature, then `top_k`, then `top_p`): `temperature: 0.0` is greedy, `top_k: 0` and `top_p: 1.0` keep every id, and the same `seed` (0 to 2147483645) draws the same ids, up to the float noise of each backend (a generator of its own, so the ids are not those of Hugging Face's `generate(do_sample=True)`, only the distribution is). The graph keeps only the largest logits and reads back their ids and values, so a sampled step costs about what a greedy one does: `top_k` of them when it is set, else the largest 256 (`Sample.MAX_CANDIDATES`). Ids below those are never drawn, so with `top_k` unset the draw is from the top 256 and `top_p` is measured against their mass, not that of the whole vocabulary (the mass below them after a real temperature is negligible; at a very high temperature over a flat distribution it is not), and a tie at the edge of the `top_k` keeps one id here where Hugging Face keeps all. Ids more than 30 below the largest logit (a mass under 1e-13) are never drawn either. It keeps the keys and values of the tokens so far (a KV cache), so a new token costs one step, not a pass over the whole sequence. It gives the new ids; a second call finds the compiled steps of the first and traces none. `Generate.close(g)` ends the compiled steps, and goes before `Qwen3.close_model(m)`.

```agentica
import Bpe from "pkg/transformers/tokenizer/bpe"
import Generate from "pkg/transformers/generate"
import Qwen3 from "pkg/transformers/qwen3"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  backend = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
  tok = Bpe.open("Qwen3-0.6B/tokenizer.json").unwrap!
  m = Qwen3.open("Qwen3-0.6B", backend).unwrap!
  prompt = Bpe.encode(tok, "The capital of France is").unwrap!
  g = Generate.open(m).unwrap!
  made = Generate.generate(g, prompt, 8, [151645, 151643]).unwrap!
  puts(Bpe.decode(tok, made).unwrap!)
  Generate.close(g)
end
```

```output
 Paris. The capital of Italy is Rome
```

For one-step control, `Generate.start()`, then `Generate.step(g, cache, ids)` (a prompt first, then one id) gives `Picked(id, cache)`, `Generate.step_logits` gives the logits `[vocab_size]`, and `Generate.finish(cache)` closes the cache. The cache that `step` is given is dead afterwards, even when the step fails: on ggml a step of the same capacity writes its row into its tensors, which are the ones of the cache it returns, elsewhere they are closed. Use the cache it returns, and never step a cache twice or read one after it was stepped. A generator belongs to one task at a time.

- The ids match Hugging Face's greedy `generate` for `Qwen3-0.6B` in `float32` on ggml CPU and Vulkan: 24 new ids for each of five prompts (English, code, a chat template, accents and CJK, a long repeat).
- The cache is padded to 8, 16, 32 and so on rows, and the prompt too, to a power of 2 (a prompt of 9 tokens runs as 16). On ggml each (tokens, rows, capacity) is one compiled graph, so a generation compiles a handful and a second call none. The choice of the id runs on the device. On ggml a one-token step writes its new row into the tensors of the cache it is given (`Tensor.set_rows`, in place), while the capacity stays: the cache it returns holds those very tensors. A step that grows the capacity makes new ones and closes the old.
- On the reference backend attention needs `q * k` of at most 1,048,576, which the padded prompt counts: a prompt of 600 tokens is padded to 1024. libtorch is not tested. The prompt and the new ids must fit `max_position_embeddings` (40960).
- Speed on this machine (Ryzen 5 3550H, Radeon Vega iGPU), in `float32`, after a prompt of 128 tokens: 5.6 tokens per second on the CPU and 7.1 on Vulkan, and 56 and 208 tokens per second for the prompt itself (medians of 5 interleaved rounds). llama.cpp on the same weights in F32 is 1.9 to 2.3 times faster, and its F16 weights make the CPU decode 1.7 times faster again: the table and the method are in [`benchmarks/`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md). Memory: 2.8 GB on the CPU and 0.85 GB on Vulkan, whole process.
- A batch of one, the prompt in one step (no chunks).

## Tokenizer example

Encode and decode with `bert-base-uncased` and with `Qwen/Qwen3-0.6B`. Each folder holds the `tokenizer.json` of the model:

```agentica
import WordPiece from "pkg/transformers/tokenizer/wordpiece"
import Bpe from "pkg/transformers/tokenizer/bpe"

def main() -> Void
  bert = WordPiece.open("bert-base-uncased/tokenizer.json").unwrap!
  encoded = WordPiece.encode(bert, "Hello, world!").unwrap!
  puts(encoded.ids)
  puts(WordPiece.decode(bert, encoded.ids).unwrap!)

  qwen = Bpe.open("Qwen3-0.6B/tokenizer.json").unwrap!
  ids = Bpe.encode(qwen, "<|im_start|>user\nHello, world!<|im_end|>").unwrap!
  puts(ids)
  puts(Bpe.decode(qwen, ids, false).unwrap!)
end
```

```output
[101, 7592, 1010, 2088, 999, 102]
hello, world!
[151644, 872, 198, 9707, 11, 1879, 0, 151645]
<|im_start|>user
Hello, world!<|im_end|>
```

## WordPiece

`WordPiece.open(path)` reads a BERT tokenizer: the BERT normalizer (control characters dropped, spaces around CJK ideographs, accents stripped, lowercase), the BERT pre-tokenizer (white space and punctuation) and a WordPiece model with the `[CLS] A [SEP]` template.

- `WordPiece.encode(tok, text, pair, add_special)` gives `Encoding(ids, type_ids)`. `pair` is `none` or `some(text)`, and `add_special` (default `true`) adds `[CLS]` and `[SEP]`. Added tokens in the text (`[MASK]`) are one id each.
- `WordPiece.decode(tok, ids, skip_special)` joins the pieces back, with the spaces before punctuation cleaned as `tokenizers` does. `skip_special` is `true` by default.
- A word longer than the file's `max_input_chars_per_word`, or one the vocabulary cannot cover, is `[UNK]`.

## Byte-level BPE

`Bpe.open(path)` reads a Qwen tokenizer: NFC normalization, the Qwen split pattern, bytes mapped to the byte-level alphabet, and BPE merges. A file with another pattern is an error.

- `Bpe.encode(tok, text)` gives the ids. It adds no tokens; added tokens in the text (`<|im_start|>`) are one id each.
- `Bpe.decode(tok, ids, skip_special)` gives the text. A character split across tokens joins back, and a broken byte sequence becomes U+FFFD.

## Errors

A file with another normalizer, pre-tokenizer, post-processor or template, decoder or model, with `truncation`, `padding` or BPE `dropout`, or with an added token that matches normalized text or by words, is an error from `open` and `parse`, not a silent difference. Ids out of range are an error from `decode`.

## Unicode

Agentica has no Unicode properties, so the tokenizers carry their own tables for Unicode 15.1 (`tokenizer/unicode_data.agc`, written by `tools/unicode_tables.py`): letters, numbers, marks, punctuation, white space, lowercase, and NFD and NFC. `tokenizers` has tables of its own, so a character that a newer Unicode version classifies differently can split differently; U+1FAE0 and U+2EBF0 give the same ids. The case-insensitive contractions (`'S`, `'RE`) are ASCII only, which agrees with `tokenizers` on `'ſ`.
