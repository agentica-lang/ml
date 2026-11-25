# Changelog

Each package is released on its own tag, `<package>-X.Y.Z`. Release 0.2.0 is the first one and covers everything shipped so far (milestones 0.1.0 and 0.2.0).

## [Unreleased]

### tensor

- **Breaking:** a function that you compile with `Trace.compile` and that calls `Optim.step` (or `Tensor.adam_step`) for Adam or AdamW on ggml now updates the first and second moments in place when nothing else reads them: the tensors you pass are consumed, and `Trace.run` gives them back with the new values (`Trace.donated` says which outputs). `Optim.step` itself, `Trace.eval` and `Trace.eval_many` still change none of their arguments.
- `Tensor.adam_step(p, g, m, v, hyper)`, the update kernel of Adam and AdamW (call `Optim`, not this): one `ggml_opt_step_adamw` node on ggml, a composition elsewhere. In a function compiled by `Trace.compile` it updates the moments in place when nothing else reads them (`Trace.donated` says which outputs are inputs). It has no gradient: a gradient through the Adam update of `Optim.step` is an error.
- `Trace.stablehlo(specs, f)` writes a traced function as a StableHLO module, for XLA and IREE (the ops of an MLP forward and of a training step); its goldens come from IREE (`make iree-goldens`, local only).
- `gelu_erf` has a gradient, so a model with the exact GELU (BERT, DistilBERT) can be trained.
- ggml: a decode step runs fewer kernels (the positions converted and checked once for every cache, a long-row `max` in pieces, SwiGLU as one kernel): Qwen3-0.6B decode on Vulkan is 7 to 11% faster.
- ggml: a `Trace.compile` function keeps its constants bound to its graph between runs and binds only its inputs; the graph that ran last keeps its allocation.
- ggml: an `attention` reads a `slice` of an `F16` cache that starts at its first element where it is, with no copy.
- ggml: `scatter_add` (the gradient of an embedding lookup) on Vulkan in F32 mode adds the rows of an index with a matmul by a mask instead of `get_rows_back`, whose Vulkan kernel costs rows x k x width; `tools/bench-scatter.agc` times it.
- ggml: a graph returns up to 512 outputs (it was 256), so a training step takes 170 parameters with Adam.
- `Tensor.append_file` appends the values of an `F32` tensor to a file, written by C on libtorch and ggml.
- A gradient through a frozen Q8_0 weight (for the input of a `linear`) is an error naming it, as the weight itself already was.

### nn

- `Nn.cosine_similarity`: the cosine of each pair of rows of two tensors, with a gradient.
- `Nn.padding_bias`: the attention mask of a padded batch.
- `Padded.reach` and `Padded.reached`: the rows of a padded cache that an attention reads, the rows in use rounded up to 256 or to an eighth of the capacity.

### train

- `Optim` takes the Adam update through `Tensor.adam_step`, so on ggml it is one kernel for each parameter (`Optim.step_inputs` gives the 7 numbers of the step as one input; `Optim.step` accepts them or the rank-0 scalars as before).
- `Lora`: LoRA adapters for the linear layers of a model by name (`init`, `params`, `with_params`, `applied`), trained with `Train.fit` while the model stays frozen.
- `Train.fit_batches` trains over batches that a function makes, each with its own shape (sentences padded to the longest of the batch).
- `Train.fit` frees the parameters and the optimizer state a step replaces at once on libtorch and ggml, where they waited for the GC: a large model no longer holds several copies of its weights.
- `Data.free(tensor)` frees the native memory of one tensor now, and `Data.freed?(batch)` says whether a batch was freed.
- `Optim.AdamW`: Adam with decoupled weight decay, as PyTorch's `AdamW`, with a `decay` flag per parameter (biases and layer-norm weights do not decay in a fine-tune).
- `Optim.Schedule` (`Constant`, `WarmupLinear`, `WarmupCosine`) for the `schedule:` of `Sgd`, `Adam` and `AdamW`: the learning rate follows the step count, as in Hugging Face's `Trainer` up to `total` (after it the factor stays 0). SGD's learning rate is now an input of the step (`Optim.step_scalars`), like Adam's.
- **Breaking:** the StableHLO module of `Train.stablehlo` for SGD takes one more rank-0 argument after the moments, its learning rate (`Optim.step_scalars`); an SGD step of `Optim.step` takes one scalar or none.
- `Metrics`: scalars written to a run folder as JSON lines and as a TensorBoard event file (pure Agentica), and `Train.fit(..., metrics: some(run))` reports `train/loss`, `train/lr` and `train/step_ms` for every step.
- `Train.stablehlo` writes the step of `fit` as one StableHLO program for every step: the step's scalars are inputs (`Optim.step_scalars`: Adam's bias corrections, SGD's rate).
- `Params.split`, `join`, `named` and `update` train part of a model: the weights selected by name train, and the frozen rest has no gradient and no optimizer state.
- `Train.fit` compiles its step once for each batch shape and runs it on every batch of that shape, instead of tracing and planning every step; inside `loss` the batch is now a graph input, not a constant.

### hub

- `SafetensorsWrite.write` writes `F32` tensors by name to a safetensors file, and `Hub.tensors` reads every tensor of a checkpoint.
- The GGUF writer refuses a layer name that is not a number, removes its `.part` file when the rename fails, and `Tensor.convert_file` reports a failed close.

### transformers

- ggml: a decode step reads the rows of the cache in use, not its whole capacity: Qwen3-0.6B decode on Vulkan at 2048 rows is 5% faster.
- `tools/distilbert_stsb.agc` fine-tunes a sentence-transformers DistilBERT on STS-B on the CPU (`benchmarks/stsb/` has the same run in PyTorch); `train` is now a plain dependency of the package for it.
- `DistilBert.weights`, `with_weights` and `pooled`: the weights of a DistilBERT by name and its forward inside a trace, so `Train.fit` fine-tunes it.
- `DistilBert`: DistilBERT from a Hugging Face checkpoint, with the mean pooling and the normalization of a sentence-transformers model (`embed`).
- `Sentence`: `mean_pooled`, `normalized` and `cosine_similarity_loss`, the pieces of a sentence-embedding model.
- `WordPiece` reads a `tokenizer.json` that sets `truncation` or batch padding (sentence-transformers' checkpoints), and `WordPiece.encode` takes a `max_length`; `pad_id` is the pad id of the file.
- `Qwen3.weights` and `Qwen3.with_weights`: the weights of a model by their Hugging Face names, for training part of it.
- `Qwen3Gguf.convert` checks every weight's shape against the config (a missing `head_dim` is the hidden size over the heads), refuses merge ranks that are not 0 to n-1, types an added token of the base vocabulary as llama.cpp does, always writes `add_bos_token` and `add_eos_token`, and reads `chat_template.jinja`.

## [0.2.0] - 2026-11-24

### tensor

- The tensor API over three backends: a pure Agentica reference, libtorch (CPU) and ggml (CPU and Vulkan), with short constructors and defaults.
- Tracing through `Tensor.*`, `value_and_grad` and backward rules for most ops; `Trace.diff` finds the first node where two backends disagree.
- On ggml a traced function compiles to one graph with a structural cache; `Trace.compile` and `Trace.run` reuse it, and `set_rows` writes a KV cache in place.
- Attention primitives (grouped heads, causal masks, rope), lowered to flash attention on ggml, and norms and rope fused into ggml kernels.
- Weights in F16 and Q8_0 and an F16 KV cache on ggml; `Tensor.from_file` reads float weights from a file in C.
- Vulkan matrix products run in fp16 by default; `Tensor.vulkan_fp16` turns it off for exact fp32.
- Packed (F16 and Q8_0) operands fail closed, covered by property tests over op combinations.

### nn

- Functional layers, forward only: dense, layer norm, embeddings, convolution, attention, MLP and seeded initializers.
- Transformer encoder and decoder blocks and stacks, pre or post norm, with grouped-query attention, rope and a shared padded KV cache.

### train

- Losses (`mse`, binary and softmax cross-entropy), SGD and Adam, and `Train.fit`, which runs each step as one graph on ggml.
- A data loader with a seeded shuffle and background prefetch.

### hub

- Reads safetensors checkpoints, one file or sharded, and loads `nn` layers from them by name.
- Reads GGUF files (F32, F16, BF16 and Q8_0) with their tokenizer, and writes GGUF in F32, F16 and BF16.
- Loads weights as F16 and reads float weights from the file without boxed lists.

### asr

- Whisper from a Hugging Face checkpoint: log-mel, tokenizer and `transcribe` with greedy decoding, language detection, timestamps and translation.
- Long audio is read in 30 s windows that seek.
- Runs on the reference backend, libtorch and ggml (CPU and Vulkan), with Hugging Face's token ids on the test clips.
- `close_model` frees the weights of a model.

### transformers

- BERT and Qwen3 forward from Hugging Face checkpoints, and Qwen3 from GGUF.
- Greedy generation and sampling (temperature, top-k, top-p, seed) with a KV cache, in F32 or F16.
- Qwen2.5-0.5B-Instruct and its chat template.
- WordPiece and byte-level BPE tokenizers that give Hugging Face's ids.

### whisper

- A binding over the pinned whisper.cpp engine, on CPU and Vulkan.
- WAV reading, language detection and translation, segments with timestamps, microphone capture and long-audio chunking.
