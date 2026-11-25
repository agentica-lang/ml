# roadmap

Functional tensors in Agentica, in the style of Nx and JAX: immutable values,
swappable backends, and `grad` computed in Agentica over a traced graph, so a
backend only implements forward. The target that drives the work is running
Whisper, by two routes: a binding over whisper.cpp (Via A) and the model
written over `tensor`, `nn` and `hub` (Via B). The contract is
[`tensor/DESIGN.md`](tensor/DESIGN.md); the original design is in
[ml#1](https://github.com/agentica-lang/ml/issues/1).

## Done

Milestones 0.1.0 to 0.3.0.

| Piece | What is in |
|---|---|
| `tensor/` | the API, 49 backend primitives and the derived ops over them, the pure Agentica reference backend (the oracle), the libtorch CPU backend (47 of the 49), the ggml backend on CPU and Vulkan (a traced graph compiles to one ggml graph, with a structural cache; `Trace.compile` and `Trace.run`), tracing, `value_and_grad` and backward rules for most ops, `Trace.diff`, attention primitives (grouped heads, causal, rope, flash attention on ggml), `set_rows` KV writes, F16 and Q8_0 weights, F16 KV cache, Vulkan fp16 on by default, property tests (packed operands fail closed), the embedding gradient (`scatter_add`) on Vulkan by a matmul |
| `nn/` | dense, layer norm, embeddings, convolution, attention (grouped-query, rotary), MLP, encoder and decoder blocks and stacks (pre or post norm), the shared padded KV cache helpers, seeded initializers |
| `train/` | `mse`, binary and softmax cross-entropy, SGD, Adam and AdamW (one kernel per parameter on ggml), learning-rate schedules, `Train.fit` and `Train.fit_batches`, freezing layers (`Params.split`), metrics as JSONL and TensorBoard, a data loader (seeded shuffle, background prefetch) |
| `hub/` | safetensors (one file or sharded), `nn` layers loaded by name, float weights read by the backend from the file (`Tensor.from_file`), GGUF read (F32, F16, BF16, Q8_0, the tokenizer from its metadata) and a GGUF writer for F32, F16 and BF16 |
| `asr/` | Whisper end to end: log-mel as one graph, tokenizer, greedy decoding with a padded KV cache, timestamps, language detection, translation, long audio read in 30 s windows that seek, the step and the pick on the device. It runs on the reference backend, libtorch and ggml (CPU and Vulkan), and its tokens equal Hugging Face's on the test clips (`asr/test/fixtures/transcribe_golden.json`) |
| `transformers/` | BERT, Qwen3 and Qwen2.5 (forward, greedy generation and sampling with a KV cache, chat template) from Hugging Face checkpoints or GGUF, and the WordPiece and byte-level BPE tokenizers, matching Hugging Face on the test logits and ids; measured against llama.cpp in `benchmarks/` |
| `whisper/` | the whisper.cpp binding, WAV reader, microphone capture, long-audio chunking |
| `spike/` | the FFI boundary (`spike/m0/`) and ggml on Vulkan, measured (`spike/ggml-vulkan/`, `spike/ggml-graph/`) |
| tooling and CI | per-package CI matrix, the ggml build cache, cchecker over the native C, whisper-tiny and Qwen3 against whisper.cpp and llama.cpp, and DistilBERT on STS-B against PyTorch and training against PyTorch and JAX, in [`benchmarks/`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md), the Hugo site |

## Needs optimization

Works, but not yet fast. `asr/` on ggml: the tokens match, but whisper-tiny
runs at a real-time factor of 0.235 on jfk on the CPU against 0.08 for
whisper.cpp, and 0.213 against 0.041 on Vulkan; the decoder step is the gap.
The numbers and the machine are in
[`benchmarks/README.md`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md).

## Not yet

By milestone (`gh issue list -R agentica-lang/ml -m <version>`).

- 0.4.0: inference benchmarks
  ([ml#173](https://github.com/agentica-lang/ml/issues/173)), Qwen3.5 text and
  vision ([ml#150](https://github.com/agentica-lang/ml/issues/150),
  [ml#151](https://github.com/agentica-lang/ml/issues/151)), LoRA
  ([ml#137](https://github.com/agentica-lang/ml/issues/137)), the rest of GGUF
  (Q4_K, Q6_K, safetensors to GGUF,
  [ml#126](https://github.com/agentica-lang/ml/issues/126)), cheaper math
  ([ml#86](https://github.com/agentica-lang/ml/issues/86)), placement across
  GPU and RAM ([ml#85](https://github.com/agentica-lang/ml/issues/85)), vision
  models and multimodal models
  ([ml#82](https://github.com/agentica-lang/ml/issues/82),
  [ml#83](https://github.com/agentica-lang/ml/issues/83),
  [ml#84](https://github.com/agentica-lang/ml/issues/84)).
- 0.5.0: CUDA ([ml#89](https://github.com/agentica-lang/ml/issues/89)) and an
  AWS Neuron spike ([ml#80](https://github.com/agentica-lang/ml/issues/80)).
- 0.6.0: an LLM server, speculative decoding and a paged KV cache
  ([ml#90](https://github.com/agentica-lang/ml/issues/90),
  [ml#91](https://github.com/agentica-lang/ml/issues/91),
  [ml#92](https://github.com/agentica-lang/ml/issues/92)).
- 0.7.0: collectives, data-parallel training and distributed inference
  ([ml#94](https://github.com/agentica-lang/ml/issues/94),
  [ml#95](https://github.com/agentica-lang/ml/issues/95),
  [ml#96](https://github.com/agentica-lang/ml/issues/96)).
- No milestone: backward rules for `pow`, `clamp`, `batch_matmul`, float
  `to_dtype`, `conv1d` and `avg_pool1d`; downloading
  checkpoints from Hugging Face; a non-required CI job that builds libtorch
  and runs the comparison with the reference.
