# roadmap

Functional tensors in Agentica, in the style of Nx and JAX: immutable values,
swappable backends, and `grad` computed in Agentica over a traced graph, so a
backend only implements forward. The target that drives the work is running
Whisper, by two routes: a binding over whisper.cpp (Via A) and the model
written over `tensor`, `nn` and `hub` (Via B). The contract is
[`tensor/DESIGN.md`](tensor/DESIGN.md); the open decisions are in
[ml#1](https://github.com/agentica-lang/ml/issues/1).

## Done

Works today.

| Piece | What is in |
|---|---|
| `tensor/` | the API, 49 backend primitives and the derived ops over them, the pure Agentica reference backend (the oracle), the libtorch CPU backend (47 of the 49), the ggml backend on CPU and Vulkan (a traced graph compiles to one ggml graph, with a structural cache), tracing, `value_and_grad` and backward rules for most ops, property tests |
| `nn/` | dense, layer norm, embeddings, convolution, attention (grouped-query, rotary), MLP, encoder and decoder blocks and stacks (pre or post norm), seeded initializers, `gelu_tanh_approx` on libtorch |
| `train/` | `mse`, binary and softmax cross-entropy, SGD, Adam, `Train.fit`, a data loader (seeded shuffle, background prefetch) |
| `hub/` | safetensors (one file or sharded), `nn` layers loaded by name, float weights read by the backend from the file (`Tensor.from_file`, [PR #67](https://github.com/agentica-lang/ml/pull/67)) |
| `asr/` | Whisper end to end (Via B): log-mel, tokenizer, greedy decoding with a padded KV cache, timestamps, language detection, translation, long audio read in 30 s windows that seek. It runs on the reference backend, libtorch and ggml (CPU and Vulkan), and its tokens equal Hugging Face's on the test clips (`asr/test/fixtures/transcribe_golden.json`) |
| `transformers/` | BERT (`bert-base-uncased`, matching `BertModel` to 5e-5 in float32 on ggml CPU and Vulkan), the Qwen3 forward and greedy generation with a padded KV cache (`Qwen3-0.6B`, matching `Qwen3ForCausalLM`'s logits to 9e-5 in float32 on ggml CPU and Vulkan, and its greedy `generate` ids), and the tokenizers of BERT (WordPiece) and Qwen (byte-level BPE) with a Unicode table, matching Hugging Face `tokenizers` on 48k and 39k cases (ml#78; sampling and more models follow) |
| `whisper/` | the whisper.cpp binding, WAV reader, microphone capture, long-audio chunking (Via A) |
| `spike/` | the FFI boundary (`spike/m0/`) and ggml on Vulkan, measured (`spike/ggml-vulkan/`) |
| tooling and CI | the ggml build cache, shared by every worktree ([PR #68](https://github.com/agentica-lang/ml/pull/68)); whisper-tiny against whisper.cpp in [`benchmarks/`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md) ([PR #66](https://github.com/agentica-lang/ml/pull/66)); cchecker over the native C in CI ([PR #49](https://github.com/agentica-lang/ml/pull/49)); Agentica 0.37.0 (PRs #69 to #72): `AN_FRESH` on libtorch, `Math`, `Ref` ids, `TENSOR_LOAD=agentica` |

## Needs optimization

Works, but not yet fast.

- `asr/` on ggml: the tokens match, but the decoder step costs about 68 ms on
  CPU (whisper.cpp: 7 ms). Speed against
  whisper.cpp and libtorch is in [`benchmarks/README.md`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md).
  The work is [ml#64](https://github.com/agentica-lang/ml/issues/64): an
  explicit `Trace.compile` and KV caches on the device. The log-mel is one graph
  of tensor ops now (35 ms on the CPU, from 0.6 s).

## Not yet

- [ml#50](https://github.com/agentica-lang/ml/issues/50): training benchmarks
  against PyTorch and JAX. The Whisper inference half is in `benchmarks/`.
- [ml#46](https://github.com/agentica-lang/ml/issues/46): training metrics: a
  logger with JSONL and TensorBoard sinks.
- Backward rules for `pow`, `clamp`, `batch_matmul`, float `to_dtype`,
  `conv1d` and `avg_pool1d`.
- `train/`: learning-rate schedules.
- `hub/`: downloading checkpoints from Hugging Face.
- CI: a non-required job that builds libtorch and runs the comparison with the
  reference.
- CUDA, later. No issue yet.
