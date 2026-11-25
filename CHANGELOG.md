# Changelog

Each package is released on its own tag, `<package>-X.Y.Z`. Release 0.2.0 is the first one and covers everything shipped so far (milestones 0.1.0 and 0.2.0).

## [Unreleased]

### tensor

- `Trace.stablehlo(specs, f)` writes a traced function as a StableHLO module, for XLA and IREE (the ops of an MLP forward and of a training step); its goldens come from IREE (`make iree-goldens`, local only).
- `gelu_erf` has a gradient, so a model with the exact GELU (BERT, DistilBERT) can be trained.
- ggml: a decode step runs fewer kernels (the positions converted and checked once for every cache, a long-row `max` in pieces, SwiGLU as one kernel): Qwen3-0.6B decode on Vulkan is 7 to 11% faster.
- ggml: a `Trace.compile` function keeps its constants bound to its graph between runs and binds only its inputs; the graph that ran last keeps its allocation.
- ggml: an `attention` reads a `slice` of an `F16` cache that starts at its first element where it is, with no copy.
- ggml: a graph returns up to 512 outputs (it was 256), so a training step takes 170 parameters with Adam.
- `Tensor.append_file` appends the values of an `F32` tensor to a file, written by C on libtorch and ggml.

### nn

- `Nn.cosine_similarity`: the cosine of each pair of rows of two tensors, with a gradient.
- `Padded.reach` and `Padded.reached`: the rows of a padded cache that an attention reads, the rows in use rounded up to 256 or to an eighth of the capacity.

### train

- `Optim.Schedule` (`WarmupLinear`, `WarmupCosine`) for the `schedule:` of `Sgd` and `Adam`: the learning rate follows the step count, as in Hugging Face's `Trainer`. SGD's learning rate is now an input of the step (`Optim.step_scalars`), like Adam's.
- `Metrics`: scalars written to a run folder as JSON lines and as a TensorBoard event file (pure Agentica), and `Train.fit(..., metrics: some(run))` reports `train/loss`, `train/lr` and `train/step_ms` for every step.
- `Train.stablehlo` writes the step of `fit` as one StableHLO program for every step: Adam's bias corrections are inputs (`Optim.step_scalars`).
- `Params.split`, `join`, `named` and `update` train part of a model: the weights selected by name train, and the frozen rest has no gradient and no optimizer state.

### hub

- `SafetensorsWrite.write` writes `F32` tensors by name to a safetensors file, and `Hub.tensors` reads every tensor of a checkpoint.

### transformers

- ggml: a decode step reads the rows of the cache in use, not its whole capacity: Qwen3-0.6B decode on Vulkan at 2048 rows is 5% faster.
- `WordPiece` reads a `tokenizer.json` that sets `truncation` or batch padding (sentence-transformers' checkpoints), and `WordPiece.encode` takes a `max_length`.
- `Qwen3.weights` and `Qwen3.with_weights`: the weights of a model by their Hugging Face names, for training part of it.

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
