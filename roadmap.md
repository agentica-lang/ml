# roadmap

Functional tensors in Agentica, in the style of Nx and JAX: immutable values,
swappable backends, and `grad` computed in Agentica over a traced graph, so a
backend only implements forward. The target that drives the work is running
Whisper, by two routes: a binding over whisper.cpp (Via A) and the model
written over `tensor`, `nn` and `hub` (Via B). The contract is
[`tensor/DESIGN.md`](tensor/DESIGN.md); the open decisions are in
[ml#1](https://github.com/agentica-lang/ml/issues/1).

## Done

| Piece | What is in |
|---|---|
| `tensor/` | the API, 45 backend primitives and the derived ops over them, the pure Agentica reference backend (the oracle), the libtorch CPU backend (42 of the 45), tracing, `value_and_grad` and backward rules for most ops, property tests |
| `nn/` | dense, layer norm, embeddings, convolution, attention, MLP, encoder and decoder blocks, seeded initializers |
| `train/` | `mse`, binary and softmax cross-entropy, SGD, Adam, `Train.fit` |
| `hub/` | safetensors (one file or sharded), and `nn` layers loaded by name |
| `asr/` | Whisper's model forward on the reference backend (Via B) |
| `whisper/` | the whisper.cpp binding, WAV reader, microphone capture, long-audio chunking (Via A) |
| `spike/` | the FFI boundary (`spike/m0/`) and ggml on Vulkan, measured (`spike/ggml-vulkan/`) |

## Next

- Backward rules for `pow`, `clamp`, `batch_matmul`, float `to_dtype`,
  `conv1d`, `avg_pool1d`, and a scatter-add for `gather` (ml#25).
- `nn/`: a KV cache for the decoder, `gelu` on libtorch (`gelu_tanh_approx`).
- `asr/`: the log-mel frontend, the tokenizer and decoding, then a comparison
  with a real checkpoint.
- `train/`: shuffling and learning-rate schedules.
- `hub/`: downloading checkpoints from Hugging Face.
- CI: a non-required job that builds libtorch and runs the comparison with the
  reference.
- When [agentica#1918](https://github.com/agentica-lang/agentica/issues/1918)
  lands: move the libtorch backend out of `tensor/` into a `torch/` package.
- Backends are CPU only. The Vulkan spike measured 2 ms per dispatched graph,
  so a GPU backend needs graph batching, not op-by-op calls.
