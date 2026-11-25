# roadmap

Functional tensors in Agentica, in the style of Nx and JAX: immutable values,
swappable backends, and `grad` computed in Agentica over a traced graph, so a
backend only implements forward. The target that drives the work is running
Whisper and Qwen3 written over `tensor`, `nn` and `hub`, next to a binding over
whisper.cpp. The contract is
[`tensor/DESIGN.md`](tensor/DESIGN.md); the original design is in
[ml#1](https://github.com/agentica-lang/ml/issues/1).

## Done

Milestones 0.1.0 to 0.4.0.

| Piece | What is in |
|---|---|
| `tensor/` | the API, 52 backend primitives and the derived ops over them, the pure Agentica reference backend (the oracle), the libtorch CPU backend (50 of the 52), the ggml backend on CPU (llamafile's sgemm, one thread per physical core) and Vulkan (a traced graph compiles to one ggml graph, with a structural cache; `Trace.compile` and `Trace.run`), the XLA backend on CPU through the PJRT plugin (`Backend.Xla`, `Trace.stablehlo`), tracing, `value_and_grad` and backward rules for most ops, `Trace.diff`, attention primitives (grouped heads, causal, rope, flash attention on ggml), `set_rows` KV writes, F16 and Q8_0 weights, F16 KV cache, Vulkan fp16 on by default, property tests (packed operands fail closed), the embedding gradient (`scatter_add`) on Vulkan by a matmul |
| `nn/` | dense, layer norm, embeddings, convolution, attention (grouped-query, rotary), MLP, encoder and decoder blocks and stacks (pre or post norm), the shared padded KV cache helpers, seeded initializers |
| `train/` | `mse`, binary and softmax cross-entropy, SGD, Adam and AdamW (one kernel per parameter on ggml), learning-rate schedules, `Train.fit` (also on XLA, one program per step) and `Train.fit_batches`, freezing layers (`Params.split`), LoRA adapters in PEFT's files (`Lora`), metrics as JSONL and TensorBoard, a data loader (seeded shuffle, background prefetch) |
| `hub/` | safetensors (one file or sharded), `nn` layers loaded by name, float weights read by the backend from the file (`Tensor.from_file`), GGUF read (F32, F16, BF16, Q8_0, the tokenizer from its metadata) and a GGUF writer for F32, F16 and BF16 |
| `asr/` | Whisper end to end: log-mel as one graph, tokenizer, greedy decoding with a padded KV cache, timestamps, language detection, translation, long audio read in 30 s windows that seek, the step and the pick on the device. It runs on the reference backend, libtorch and ggml (CPU and Vulkan), and its tokens equal Hugging Face's on the test clips (`asr/test/fixtures/transcribe_golden.json`) |
| `transformers/` | BERT, Qwen3 and Qwen2.5 (forward, greedy generation and sampling with a KV cache, chat template) from Hugging Face checkpoints or GGUF, and the WordPiece and byte-level BPE tokenizers, matching Hugging Face on the test logits and ids; measured against llama.cpp in `benchmarks/` |
| `whisper/` | the whisper.cpp binding, WAV reader, microphone capture, long-audio chunking |
| tooling and CI | the agent skills `ml-inference` and `ml-training` (`.agents/skills/`), install by tag with no `override`, per-package CI matrix, the ggml build cache, cchecker over the native C, whisper-tiny and Qwen3 against whisper.cpp and llama.cpp, and DistilBERT on STS-B against PyTorch and training against PyTorch and JAX, in [`benchmarks/`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md), the Hugo site |

## Needs optimization

Works, but not yet fast:

- `asr/` on ggml: the tokens match, but whisper-tiny runs at a real-time factor
  of 0.116 on jfk on the CPU against 0.08 for whisper.cpp, and 0.069 against
  0.041 on Vulkan; the decoder step is the gap.
- LoRA fine-tuning: 1.35 times PEFT's time per step on the CPU
  ([ml#236](https://github.com/agentica-lang/ml/issues/236)).

The numbers and the machine are in
[`benchmarks/README.md`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md).

## Not yet

By milestone (`gh issue list -R agentica-lang/ml -m <version>`).

- 0.5.0, vision: image loading and transforms
  ([ml#82](https://github.com/agentica-lang/ml/issues/82)), ResNet, ViT and
  CLIP ([ml#83](https://github.com/agentica-lang/ml/issues/83)), vision-language
  models ([ml#84](https://github.com/agentica-lang/ml/issues/84)).
- 0.6.0, models and formats: Qwen3 on XLA
  ([ml#243](https://github.com/agentica-lang/ml/issues/243)), Qwen3.5 text and
  vision ([ml#150](https://github.com/agentica-lang/ml/issues/150),
  [ml#151](https://github.com/agentica-lang/ml/issues/151)), the rest of GGUF
  (Q4_K, Q6_K, safetensors to GGUF,
  [ml#126](https://github.com/agentica-lang/ml/issues/126)), placement across
  GPU and RAM ([ml#85](https://github.com/agentica-lang/ml/issues/85)), LoRA
  speed against PEFT ([ml#236](https://github.com/agentica-lang/ml/issues/236)),
  inference benchmarks
  ([ml#173](https://github.com/agentica-lang/ml/issues/173)).
- 0.7.0, backends: XLA on AWS Neuron
  ([ml#241](https://github.com/agentica-lang/ml/issues/241)) and CUDA
  ([ml#89](https://github.com/agentica-lang/ml/issues/89)).
- 0.8.0, serving: an LLM server, speculative decoding and a paged KV cache
  ([ml#90](https://github.com/agentica-lang/ml/issues/90),
  [ml#91](https://github.com/agentica-lang/ml/issues/91),
  [ml#92](https://github.com/agentica-lang/ml/issues/92)).
- 0.9.0, distributed: collectives, data-parallel training and distributed
  inference ([ml#94](https://github.com/agentica-lang/ml/issues/94),
  [ml#95](https://github.com/agentica-lang/ml/issues/95),
  [ml#96](https://github.com/agentica-lang/ml/issues/96)).
- No milestone: cheaper math
  ([ml#86](https://github.com/agentica-lang/ml/issues/86)), XLA on macOS and
  Windows ([ml#232](https://github.com/agentica-lang/ml/issues/232)), the last
  ms of a Qwen3 decode step against llama.cpp
  ([ml#181](https://github.com/agentica-lang/ml/issues/181)), one weight layout
  for `Dense` ([ml#125](https://github.com/agentica-lang/ml/issues/125)), dropout on
  the LoRA path ([ml#209](https://github.com/agentica-lang/ml/issues/209)),
  `batch_matmul` on libtorch
  ([ml#203](https://github.com/agentica-lang/ml/issues/203)), the CPU gap of
  training against PyTorch
  ([ml#192](https://github.com/agentica-lang/ml/issues/192)), backward rules
  for the ops the site lists as having none, downloading checkpoints from
  Hugging Face, and a non-required CI job that builds libtorch and runs the
  comparison with the reference.
