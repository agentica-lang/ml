# roadmap

Functional tensors in Agentica, in the style of Nx and JAX: immutable values,
swappable backends, and `grad` computed in Agentica over a traced graph, so a
backend only implements forward. The target that drives the work is running
Whisper and Qwen3 written over `tensor`, `nn` and `hub`, next to a binding over
whisper.cpp. The contract is
[`tensor/DESIGN.md`](tensor/DESIGN.md); the original design is in
[ml#1](https://github.com/agentica-lang/ml/issues/1).

## Done

Milestones 0.1.0 to 0.7.0, except placement ([ml#85](https://github.com/agentica-lang/ml/issues/85)).

| Piece | What is in |
|---|---|
| `tensor/` | the API, 52 backend primitives and the derived ops over them, the pure Agentica reference backend (the oracle), the libtorch CPU backend (51 of the 52), the ggml backend on CPU (llamafile's sgemm, one thread per physical core) and Vulkan (a traced graph compiles to one ggml graph, with a structural cache; `Trace.compile` and `Trace.run`), the XLA backend on CPU through the PJRT plugin (`Backend.Xla`, `Trace.stablehlo`), tracing, `value_and_grad` and backward rules for most ops, `Trace.diff`, attention primitives (grouped heads, causal, rope, flash attention on ggml), `set_rows` KV writes, F16 and Q8_0 weights, F16 KV cache, Vulkan fp16 on by default, property tests (packed operands fail closed), the embedding gradient (`scatter_add`) on Vulkan by a matmul, `conv2d` (stride, padding, groups), `max_pool2d` and `avg_pool2d` with gradients (a native kernel on ggml; XLA has no lowering for them), `Tensor.from_bytes` (pixels as `U8` bytes), `gated_delta_rule` and `expert_matmul` (the linear attention and the expert products), `Dtype.Q4_K` and `Dtype.Q6_K` read from files, a graph across a Vulkan device and the CPU (ggml's backend scheduler), packed `List[Float32]` values, the XLA CPU plugin installed from our own release |
| `nn/` | dense, layer norm, embeddings, 1-D and 2-D convolution (`fold_batch_norm`, `global_avg_pool2d`), attention (grouped-query, rotary), MLP, encoder and decoder blocks and stacks (pre or post norm), the shared padded KV cache helpers, seeded initializers, the mixture-of-experts block and the Gated DeltaNet layer (`Delta`) |
| `train/` | `mse`, binary and softmax cross-entropy, SGD, Adam and AdamW (one kernel per parameter on ggml), learning-rate schedules, `Train.fit` (also on XLA, one program per step) and `Train.fit_batches` (with `prefetch`), freezing layers (`Params.split`), LoRA adapters in PEFT's files (`Lora`), metrics as JSONL and TensorBoard, a data loader (seeded shuffle, background prefetch) |
| `hub/` | safetensors (one file or sharded), `nn` layers loaded by name, float weights read by the backend from the file (`Tensor.from_file`), GGUF read (F32, F16, BF16, Q8_0, Q4_K, Q6_K, the tokenizer from its metadata) and a GGUF writer for F32, F16 and BF16 |
| `asr/` | Whisper end to end: log-mel as one graph, tokenizer, greedy decoding with a padded KV cache, timestamps, language detection, translation, long audio read in 30 s windows that seek, the step and the pick on the device. It runs on the reference backend, libtorch and ggml (CPU and Vulkan), and its tokens equal Hugging Face's on the test clips (`asr/test/fixtures/transcribe_golden.json`) |
| `transformers/` | BERT, Qwen3, Qwen2.5, Qwen3.5 (0.8B, text and images), LFM3-1.7B and IBM Granite MoE (forward, greedy generation and sampling with a KV cache, chat template) from Hugging Face checkpoints or GGUF, ResNet-18/50 and ViT-B/16 (image classification) and CLIP ViT-B/32 (image and text embeddings, zero-shot), SmolVLM-256M (a chat with an image, to Hugging Face's ids), per-layer placement of Qwen3, Qwen3.5, LFM3 and Granite MoE across Vulkan and the CPU (`layers_on_device`, and `Qwen3.layers_that_fit` for the free memory of the card) and the KV cache in host memory (`Placement.Kv.Host`), and the WordPiece, byte-level BPE and CLIP tokenizers, matching Hugging Face on the test logits and ids; measured against llama.cpp in `benchmarks/` |
| `vision/` | images for tensors on libvips: decode, `Resize` with the kernel of the model's processor, crops and flip with a seed, pixels to a tensor as one buffer (`U8` bytes widened on the backend), `normalize`, and a folder per class as batches for the loader of `train`, decoded on the prefetch task |
| `whisper/` | the whisper.cpp binding, WAV reader, microphone capture, long-audio chunking |
| tooling and CI | the agent skills `ml-inference` and `ml-training` (`.agents/skills/`), install by tag with no `override`, per-package CI matrix, the ggml build cache, cchecker over the native C, whisper-tiny and Qwen3 against whisper.cpp and llama.cpp, ResNet, ViT and CLIP timings, and DistilBERT on STS-B against PyTorch and training against PyTorch and JAX, in [`benchmarks/`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md), the Hugo site |

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

- 0.7.0, local models: the rest of placement, weights and KV cache across GPU and RAM
  ([ml#85](https://github.com/agentica-lang/ml/issues/85)).
- 0.8.0, accelerators: CUDA ([ml#89](https://github.com/agentica-lang/ml/issues/89)), the failing ggml-cuda tensor tests
  ([ml#248](https://github.com/agentica-lang/ml/issues/248)), XLA on CUDA ([ml#249](https://github.com/agentica-lang/ml/issues/249)), XLA on AWS Neuron ([ml#241](https://github.com/agentica-lang/ml/issues/241)), Qwen3 on XLA
  ([ml#243](https://github.com/agentica-lang/ml/issues/243)), one model over the devices of a machine ([ml#250](https://github.com/agentica-lang/ml/issues/250)) and LoRA speed
  against PEFT ([ml#236](https://github.com/agentica-lang/ml/issues/236)).
- 0.9.0, serving: an LLM server, speculative decoding and a paged KV cache
  ([ml#90](https://github.com/agentica-lang/ml/issues/90), [ml#91](https://github.com/agentica-lang/ml/issues/91), [ml#92](https://github.com/agentica-lang/ml/issues/92)).
- 0.10.0, distributed: collectives, data-parallel training and distributed
  inference ([ml#94](https://github.com/agentica-lang/ml/issues/94), [ml#95](https://github.com/agentica-lang/ml/issues/95), [ml#96](https://github.com/agentica-lang/ml/issues/96)).
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
