# Changelog

All packages are released together at one version, each on its own tag, `<package>-X.Y.Z`. Release 0.2.0 is the first one and covers everything shipped so far (milestones 0.1.0 and 0.2.0).

## [Unreleased]

### train

- `Loop.trainer(loss, params, optimizer)` and `Loop.run(trainer, data, epochs:)` (`Loop.run_batches` for a batch function) run a training as a pipe over `Train.fit`, and `Optim.sgd`, `Optim.adam` and `Optim.adamw` build the optimizers with their usual knobs (ml#397).
- `Table.read(path)` reads a CSV file (the `csv` package) into named columns, and `Data.from_table(table, inputs, target, backend, target_dtype)` makes a `Dataset` of its numeric columns for a `Loader` (ml#318).
- `Table.read` reads a `.parquet` file too (the `parquet` package), and `Data.from_table` makes a `Dataset` of its columns (ml#318).
- A Parquet column with a NaN or an infinity is an error that names the column and the row, as a CSV cell that is not finite keeps its column text; `train/examples/` has a CSV and a Parquet example with goldens, and `train/data.md` the text-column path with `Train.fit_batches` (ml#318).
- **Breaking:** train, transformers, vision and tools/skills ask for Agentica 0.40.0-beta.2 (the `parquet` package needs it); CI already ran on it (ml#318).
- `Table.read` reads a CSV or Parquet file over 1 MiB, in chunks (ml#378).

### hub

- `Load.max_device_bytes_from_env` reads `AGENTICA_ML_MAX_DEVICE_BYTES` for the tools (ml#331).
- `Load.held_bytes(c, name, weights)` counts the bytes a weight takes on a ggml device under `weights`, without reading it (ml#331).

### tensor

- `Tensor.batch_matmul` runs on `Backend.Torch(Cpu)`, so the gradient of `Tensor.attention` (written with it) runs on libtorch too: 51 of the 52 primitives (ml#203).
- The StableHLO emitter (so `Backend.Xla`) compiles `sigmoid`, `pow`, `clamp`, `to_dtype`, `permute`, `concat`, `stack`, `squeeze`, `unsqueeze`, `batch_matmul` and `argmax` (ml#243).
- The StableHLO emitter compiles `layer_norm`, `rms_norm`, `gelu_erf` and `rope` too (ml#243).
- The StableHLO emitter compiles `attention` and `scatter_add` (the gradient of an embedding) too, and a program with a `scatter_add` checks its indices as one with a `gather` does (ml#243).
- The StableHLO emitter compiles `top_ids` too (ml#243).
- `Tensor.from_file` and `Tensor.from_bytes` on `Backend.Xla` read and widen the range in C, straight into the buffer, as ggml does (no list of floats): Qwen3-0.6B opens in 2.8 GB instead of over 6 GB (ml#243).
- `Tensor.device_memory(backend)` gives the free and total bytes of the device of a ggml backend (`ggml_backend_dev_memory`), for a memory budget of `layers_on_device` (ml#331).

### transformers

- `tools/minilm_finetune.agc` takes `torch`, which fine-tunes MiniLM on libtorch with the word table frozen (ml#203, ml#394).
- `WordPiece` reads a tokenizer file with a fixed padding (`"strategy": {"Fixed": 128}`) as the padding to the longest, so `sentence-transformers/all-MiniLM-L6-v2` opens; no tokenizer here pads (ml#377).
- `tools/fetch_minilm.sh` fetches all-MiniLM-L6-v2 at a pinned revision and sha256, and `tools/minilm_finetune.agc` loads it and fine-tunes it for a few steps on ggml CPU or Vulkan (ml#377).
- A BERT fine-tunes on `Backend.Xla(Device.Cpu)`: the gradients of the cosine loss match the reference and `Train.fit_batches` lowers the loss, and `tools/minilm_finetune.agc` takes `xla` (all-MiniLM-L6-v2: the loss falls from 0.01137 to 0.00236 in six steps, as on ggml) (ml#243).
- `tools/finetune_stsb.agc` fine-tunes all-MiniLM-L6-v2 on the scored pairs of STS-B on `cpu`, `vulkan`, `xla` or `torch`, with `tools/fetch_stsb.sh` for the pinned files; the guide is the page `train/finetune-minilm.md`.
- `Generate` of Qwen3 runs on `Backend.Xla(Device.Cpu)`: the steps of the cache are compiled by XLA, and `Lift.staged` compiles the forward there too (ml#243). Checked on the tiny model of the tests; the check against Hugging Face on Qwen3-0.6B and the speed are not done yet.
- `Bert.weights`, `Bert.with_weights` and `Bert.pooled` name the weights of a BERT as the checkpoint does and run its mean embedding inside a trace, so `Train.fit` and `Params` fine-tune it (`sentence-transformers/all-MiniLM-L6-v2` sized models); `DistilBert` shares the helpers (ml#377).
- `GraniteMoe.open(dir, backend, weights, layers_on_device)`: the layers from the `n`th on, with their experts and keys, run on the CPU under a Vulkan device, as for Qwen3 (ml#331).
- `Lfm3.open(dir, backend, weights, layers_on_device)`: the layers from the `n`th on, with their convolution states and keys, run on the CPU under a Vulkan device, as for Qwen3 (ml#331).
- `Qwen3_5.open(dir, backend, weights, layers_on_device)`: the layers from the `n`th on, with their keys and states, run on the CPU under a Vulkan device, with the table and the head placed as for Qwen3 (ml#331).
- `Qwen3.layers_that_fit` and `Qwen3.layers_within` take `kv: Placement.Kv.Host` and then count no KV cache on the card (ml#331).
- `Qwen3.layers_that_fit(dir, backend, weights, max_device_bytes)` gives the `layers_on_device` that fills the free memory of a Vulkan device (less a reserve, up to a cap), and `AGENTICA_ML_MAX_DEVICE_BYTES` sets the cap for `bench_qwen` (ml#331).
- `Placement.footprint(c, ...)` counts the `Footprint` of a model from its checkpoint (ml#331).
- `Placement.fit(footprint, budget)` gives the largest `layers_on_device` whose layout fits a budget in bytes, and `Placement.budget(free, reserve, cap)` the budget; both are pure, the first step of a memory budget for `layers_on_device` (ml#331).

## [0.7.0] - 2026-11-27

### tensor

- The XLA backend starts with a CUDA PJRT plugin: the client is made with the plugin's default options (the CPU-only `cpu_device_count` made the CUDA plugin refuse it) (ml#324).
- `Dtype.Q4_K`, `Dtype.Q6_K` and the formats of the same names: `Tensor.from_file` dequantizes ggml's K blocks to `F32` on every backend, or keeps them as a weight on ggml (ml#126).
- `Tensor.packed?(dtype)`: true for the dtypes that ggml keeps packed (`F16` and the quantized ones).
- A graph on a Vulkan device may read tensors on `Ggml(Cpu)`: ggml's backend scheduler runs each node where its operands are and copies the activations between the two, and an output of a CPU node is a CPU tensor (ml#310).
- `Tensor.floats32(values, shape, backend)` makes an `F32` tensor from a packed `List[Float32]` (4 bytes an item, since Agentica 0.40.0); `floats` keeps its `List[Float]` (ml#316).
- `tools/install_xla.agc` installs the XLA CPU plugin from our own build, a GitHub Release of this repository (Linux x86_64 and aarch64, with the licenses of what it links), instead of the PyPI wheel, and needs no `unzip`; the old `xla/<wheel-sha>/` folders of the cache can be deleted (ml#287).
- CI checks every package on Agentica 0.40.0-beta.2, which has the native `List[Float32]` API 1.6 (`an_float32_run`, `an_float32s`); the manifests keep `agentica = "0.38.0"`.

### asr

- **Breaking:** the samples of `Asr.transcribe`, `Asr.features` and `Mel.features`/`Mel.spectrogram`, and the samples and the result of `Mel.log_mel`, are a packed `List[Float32]`, half the memory of a `List[Float]` (5 minutes of audio: 19 MB, not 38). `Mel.log_mel` still computes in doubles; it no longer refuses a sample past 1e150, which a `Float32` cannot hold (ml#316).

### whisper

- **Breaking:** `Wav.load`, `Wav.decode`, `Mic.capture`, `Mic.read`, `Mic.decode_s16le`, `Mic.Source.Samples`, `Stream.Window.samples`, `Stream.windows`, `Stream.has_speech`, `Stream.is_speech`, `Stream.frame_rms` and `Whisper.transcribe` hold the samples as a packed `List[Float32]`. `transcribe_with` and `transcribe_detailed` keep a `List[Float]`, because a native module cannot take a `Float32`: `Whisper.widen(pcm)` makes one. `Whisper.transcribe` widens the whole clip for the length of the call, 8 bytes a sample, so a long clip should go through `Stream.transcribe_stream`, which widens one window at a time (ml#316).

### nn

- `ShortConv`, the gated short convolution of LFM3 (`pkg/nn/short_conv`), `Attention.head_gate` (a gate of one number per head) and a `window` in `Nn.causal_mask` and `Padded.padded_mask`, the mask of a sliding-window layer (ml#275).
- `Rope.zero_tail`, the rotary of LFM3's global layers (a partial rotation with the frequencies of the whole head) (ml#275).
- `Rope.sections`, the interleaved M-RoPE of Qwen3.5: positions of `[3, seq]` (time, height, width) turn each pair of columns by the row of its axis, with `Nn.axis_masks` for a decoder that makes the masks once; positions of `[seq]` are the plain rope (ml#151).
- `Nn.Activation.GeluTanh`: the tanh GELU as ops that ggml compiles (`Gelu` is not lowered to ggml yet, ml#301); `Qwen3_5Vision` uses it (ml#84).

### transformers

- `Lfm3`: LFM3-1.7B (`LiquidAI/LFM3-1.7B`) from its official safetensors, forward over whole sequences, `F32` or `F16` weights (ml#275).
- SmolVLM-256M-Instruct describes an image (ml#84): `Chat.apply`, `Bpe`, `Vision.idefics3_tiles`, `SmolVlmVision`, `SmolVlmPrompt` and a greedy `Generate` give Hugging Face's `generate` ids for two images (checked by hand on Vulkan; fragment in the transformers models page).
- `Bpe` reads the `tokenizer.json` of SmolLM2 and SmolVLM (`Digits` with individual digits, then ByteLevel's GPT-2 split), to the ids and text of `tokenizers` on a tiny tokenizer and on the real SmolVLM-256M-Instruct one (ml#84).
- `Chat.apply("idefics3", ...)` is the chat template of SmolVLM (`Chat.SMOLVLM_IMAGE` is where an image goes), equal to Hugging Face's on 21 conversations (ml#84).
- `Bpe` reads LFM3's `tokenizer.json` (digits in groups of up to three, `ignore_merges`, no normalizer), and `Chat.apply("lfm3", ...)` is its chat template with tools (`tools`, `Message.tool_calls`, `thinking`, `tool_call_id`, `preserve_thinking`) (ml#275).
- `Generate` of `pkg/transformers/generate_lfm3`: greedy or sampled generation with LFM3's hybrid cache (the keys and values of the attention layers, the state of the convolutions); `KvDecoder` takes layers with a state (ml#275).
- `Qwen3.open(dir, backend, weights, layers_on_device)`: the blocks from the `n`th on run on the CPU, with their weights and KV cache in host memory, like `-ngl` of llama.cpp; `AGENTICA_ML_LAYERS_ON_DEVICE` for the tools; binding a tensor on the CPU to a graph on a Vulkan device now runs it through ggml's backend scheduler instead of failing (ml#310, ml#85).
- `layers_on_device`: with `n` below the number of blocks the embedding table is on the CPU, and the head and the final norm are on the device (a tied head is a copy of the table there: its size in card memory, about 128 MB for Qwen3-0.6B Q6_K, and its load time), as with `-ngl` of llama.cpp (which puts the output layer on the GPU for any `-ngl` of 1 or more and the first blocks on the CPU); with `n` of 0 they are on the CPU too, so a decode step with no block on the device is as fast as the CPU-only engine (ml#320, ml#327).
- `AGENTICA_ML_KV=host` (or `device`) for the tools (`bench_qwen`, `profile_qwen`, `layers_check`, `leak_check`): they open the generator with `kv: Placement.Kv.Host`; `benchmarks/bench_qwen.sh` has the rows `ours-vulkan-q4k-kvhost` and `llama-vulkan-q4km-nkvo` (ml#85).
- `Generate.open(m, usable, kv: Placement.Kv.Host)` (`import Placement from "pkg/transformers/placement"`): the keys and values of the cache are kept in host memory under a model on a Vulkan device, as llama.cpp's `--no-kv-offload`; the prompt step still makes the cache on the card first (ml#85). A step writes the keys and the values of a layer after making both, so with the cache in host memory a layer is two splits of the scheduler, not four (a decode step of Qwen3-0.6B Q4_K_M: 57 splits, not 113).
- `layers_on_device`: a decode step with the blocks on the CPU makes a few scheduler splits, not one or two per block: the residual stream stays on the CPU and the checks of the graph are copied together, about a third of the time of a step (ml#312).
- `tools/profile_qwen.agc` reads `WEIGHTS` (`f16`, `q8_0`, `q4_k`), and `benchmarks/bench_qwen.sh` has the Q4_K_M rows against llama.cpp, at equal threads, with the peak GPU memory (ml#291).
- Qwen3.5 chat with images: `Chat.IMAGE` is where a picture goes in a user or assistant message of `Chat.apply("qwen3_5", ...)` (a system message refuses it), `Mrope.expand(ids, pad, grids, merge)` gives the prompt its image tokens, `Qwen3_5.image_token(dir)` reads their id and `Qwen3_5Vision.merge(v)` the merge of the tower. The prompt ids and the greedy ids of `Qwen/Qwen3.5-0.8B` with a picture are Hugging Face's on CPU and Vulkan, with the five largest logits within 0.005 (ml#151).
- `Qwen3_5Vision` (`pkg/transformers/qwen3_5_vision`): the vision tower of Qwen3.5 (`Qwen/Qwen3.5-0.8B`), `open` and `encode` of one image's patches to the features that the text model reads at the image tokens (ml#151).
- Images in the Qwen3.5 text model: `Mrope.layout` (the 3-axis positions and `rope_deltas` of an image prompt), `Generate.start(sight)` and the `sight` of `Generate.generate` splice the features of `Qwen3_5Vision` in at the image tokens and turn the attention by the three axes, to Hugging Face's logits on a tiny model (ml#151).
- `SmolVlmVision` (`pkg/transformers/smolvlm_vision`): the vision tower and the connector of SmolVLM (`HuggingFaceTB/SmolVLM-256M-Instruct`), `open` and `encode` of normalized 512 by 512 tiles to their 64 rows each for the text model, to Hugging Face's `get_image_features` on a tiny model and on the real checkpoint (ml#84).
- `Qwen3.open` reads `model_type` `llama` (SmolLM2, the text model of SmolVLM): the Qwen2 block with no biases, to Hugging Face's `LlamaForCausalLM` logits on a tiny model; `mlp_bias: true` is an error (ml#84).
- `Qwen3.open_idefics3` reads the Llama text model of a SmolVLM folder (`model.text_model.*`, `text_config`), to Hugging Face's logits on the real SmolVLM-256M-Instruct (ml#84).
- SmolVLM text with images: `Generate` over a `KvDecoder.plain_sight` (a stepper with `rotated: false`, `Qwen3.image_token`) splices the features of `SmolVlmVision` into the prompt, to Hugging Face's logits on the real SmolVLM-256M-Instruct (ml#84).
- `SmolVlmPrompt` (`pkg/transformers/smolvlm_prompt`): `expand` writes the ids of a prompt with images as `Idefics3Processor` does (tiles, row and column ids, global image) from the ids with one `<image>` per image, and `open` reads the special ids of a SmolVLM folder (ml#84).

### hub

- GGUF files with `Q4_K` and `Q6_K` tensors are read; `weights: Dtype.Q4_K` (`WEIGHTS=q4_k`) keeps each matrix as blocks at the bit width the file stores it in (ml#126).

### vision

- `vision` needs `image-0.3.0`: an enlarged image has the pixel centres of Pillow's bicubic, so the pixels of an image under `min_pixels` are within one 8-bit level of Hugging Face's processor, as a shrunk one is (ml#306).
- `Vision.idefics3_tiling` and `Vision.idefics3_tiles`: the sizes and the tiles (512 by 512, row by row, then the global image) that SmolVLM's image processor makes of an sRGB image; as `[3, 512, 512]` tensors with Pillow's Lanczos resize (matrices of its weights; libvips enlarges with bicubic whatever the kernel), within 1.5 levels of 8 bits of Hugging Face's on every frame, with hard edges too (ml#84, ml#339).
- `Vision.smart_resize` (it gives a `Vision.Size`) and `Vision.Step.SmartResize(factor, min_pixels, max_pixels, kernel)`: the size that Hugging Face's `smart_resize` gives (Qwen2-VL's image processor); `Vision.patches`: the patches of a normalized 3-channel image in the order of the processor's `pixel_values` (ml#151).

### train

- No user-visible change in 0.7.0.

## [0.6.0] - 2026-11-26

### tensor

- `Tensor.expert_matmul(weights, x, ids)`: the product of each token by the experts it was routed to (`F32`, no gradient; ggml `ggml_mul_mat_id`), and `Tensor.top_ids` takes a matrix: the ids of each row (ml#274).
- `Tensor.gated_delta_rule`: the recurrence of Qwen3.5's linear attention (`F32`, no gradient; ggml `ggml_gated_delta_net`) (ml#150).
- `Tensor.softplus` and `Tensor.l2_norm(x, eps)`, derived ops of that layer (ml#150).

### nn

- `Nn.Attention` has `scale`, the factor of `q k^T` when the model sets one (Granite MoE's `attention_multiplier`) (ml#274).
- `Moe.moe`, the mixture-of-experts block (router, top-k, gated experts), in `pkg/nn/moe` (ml#274).
- `Nn.Attention` has `gate`: Qwen3.5's output-gated full attention (ml#150).
- `Delta.gated_delta_net`, the linear attention layer of Qwen3.5 with its conv and recurrent state, in `pkg/nn/delta` (ml#150).
- `Delta.padding`: a call over padded rows (a prompt padded to a compiled shape) leaves the state as the real tokens left it (ml#150).

### transformers

- `pkg/transformers/kv_decoder` is the KV-cache decoder that `generate` (Qwen3) and `generate_granite_moe` now share; it is internal, and the API of those two modules is unchanged (ml#281).
- `Generate` of `pkg/transformers/generate_granite_moe`: greedy and sampled generation with Granite MoE and a KV cache (ml#274).
- `GraniteMoe`: IBM Granite MoE (`ibm-granite/granite-3.1-1b-a400m-instruct`) from its official safetensors, forward over whole sequences (ml#274).
- `Qwen3_5`: the text model of `Qwen/Qwen3.5-0.8B` (a hybrid of Gated DeltaNet and gated attention layers) from its official safetensors, forward over whole sequences (ml#150).
- `Bpe` reads the tokenizer of Qwen3.5 (its split counts marks as letters), and `Chat.apply("qwen3_5", ...)` makes its chat prompt (ml#150).
- `Generate` of `pkg/transformers/generate_qwen35`: greedy and sampled generation with Qwen3.5's hybrid cache (the keys and values of the full layers, the states of the linear ones), with `usable` to keep the sampler off the ids the tokenizer cannot decode (ml#150). The shared parts of `Generate` (`picked`, `Plans`, `Candidate`, the `Head` modes) moved to `pkg/transformers/stepping`, with the loop of `generate`; they were not documented API, and the records of `step` and `step_top` are now `Stepping.Picked` and `Stepping.Candidates`.
- `Generate.open(m, usable)` of Qwen3 (and Qwen2.5) has the same `usable` as Qwen3.5's: pass `tok.pieces.size` and the sampler never gives an id that `Bpe.decode` rejects; the default 0 is no mask (ml#150).
- **Breaking:** the `enable_thinking` of `Chat.apply` is an `Optional[Bool]` (`none` by default: what each template does when it is not told, Qwen3 thinks, Qwen3.5 does not); pass `some(false)` where you passed `false` (ml#150).

### hub

- No user-visible change in 0.6.0; its readers now use `std/binary`.

### train

- The TensorBoard writer's CRC runs on `UInt32`; the files it writes are the same.

### asr

- A WAV file whose chunk runs past its end is an error, not a read past the buffer.

### vision

- No user-visible change since 0.5.1; released with the other members.

## [0.5.1] - 2026-11-25

### vision

- `Vision.Kernel` is `Image.Kernel`, re-exported: `Vision.Step.Resize(256, Vision.Kernel.Cubic)` needs no direct dependency on `image`.

## [0.5.0] - 2026-11-25

- `tensor` has `conv2d` (stride, padding and groups), `max_pool2d` and `avg_pool2d` for `[N, C, H, W]` images (ml#83): on the reference, libtorch and ggml (a native kernel on the CPU and on Vulkan), with gradients on every backend. XLA has no lowering for them, as for `conv1d`. `nn` has `Conv2d`, `conv2d` and `global_avg_pool2d`.
- The documentation site has a version selector: the pages of each release (`0.4.0`, `0.3.0`, `0.2.0`) and of `dev`.

### tensor

- The bias gradient of a convolution (a sum over the batch and the plane) runs on ggml at a batch above 1: the gradient of `broadcast_to` sums a leading and a trailing block of axes in two steps.
- `Tensor.from_bytes(data, shape, format, backend)`: an `F32` tensor from bytes that are in memory, as `from_file` makes one from a range of a file. libtorch and ggml read the bytes in C, so no value becomes a list.
- `FloatFormat.U8` for `Tensor.from_bytes` (ml#252): one byte for each value: the samples of a `U8` image cross into the backend as bytes, a quarter of the size of `F32` ones, and the backend widens them to `0.0..255.0` as an `F32` tensor. `from_file` and `convert_file` refuse it.

### nn

- `Nn.fold_batch_norm`: a convolution and the inference batch norm after it as one convolution.

### hub

- Writing a GGUF file refuses a `U8` tensor with an error: `U8` is the format of `Tensor.from_bytes`, not of a file, and only `F32`, `F16`, `BF16` and `Q8_0` are written.

### train

- `Train.fit_batches(..., prefetch: K)` (ml#252): a task calls the batch function for the next batches, `K` ahead, while the steps run, so decoding image files overlaps the device. `Data.ahead`, `Data.next` and `Data.finish` are the stream behind it.

### transformers

- `Clip` (ml#83): CLIP ViT-B/32 from `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (LAION's weights: OpenAI's checkpoint publishes no safetensors file): image and text embeddings, `logit_scale`, the scaled cosines and zero-shot probabilities, matching Hugging Face's. `tokenizer/clip` is CLIP's tokenizer (the slow `CLIPTokenizer`: `<|startoftext|>` and `<|endoftext|>`, `</w>` word ends, lowercase, 77 ids at most).
- `Bpe.merge(ranks, symbols)` and `Bpe.byte_chars` are public (the merge loop of the Qwen tokenizer, for CLIP's), and `WordPiece.cleaned_chars` and `spaced_chinese` too.
- `ViT` (ml#83): ViT-B/16 from the official `google/vit-base-patch16-224` checkpoint. Its logits match Hugging Face's, and the top-5 classes are the same. `ResNet.Prediction` is now `Classify.Prediction`, shared by both.
- `ResNet` (ml#83): ResNet-18 and ResNet-50 from the official `microsoft/resnet-*` checkpoints, with the batch norms folded into the convolutions at load. Their logits and top-5 classes match Hugging Face's on images through `vision`'s resize, crop and normalize.

### vision

- A `U8` image is handed to the backend as bytes (ml#252), not cast to `F32` in the image library first: a quarter of the memory up to the backend, which widens it to `F32`; the scale to `0..1` runs on the device. `Vision.samples` inside a `Train.fit_batches` batch function decodes on its `prefetch` task.
- `Vision.Step.Resize(n, kernel)` and `ResizeTo(width, height, kernel)` take the `Image.Kernel` that libvips resizes with (ml#258, `image-0.2.0`). The recipes of ResNet and CLIP use `Cubic` and ViT's `Linear`, as their Hugging Face processors do: the logits after the pipeline are 0.092 (ResNet-50), 0.048 (ResNet-18), 0.103 (ViT) and 0.21 (CLIP) from Hugging Face's, from 0.16, 0.065, 0.156 and 0.43 with Lanczos 3.
- `Vision.Step.ResizeTo(width, height, kernel)`: a resize to exactly that size, without keeping the aspect ratio.
- New package `vision` (ml#82): images for tensors, on the `image` package (libvips). It needs libvips.
- `Vision.to_tensor`: the pixels of an image go straight into a tensor, as `F32` in `0..1` or as `U8`, in `Chw` or `Hwc`.
- `Vision.apply`: `Resize` (shorter side), `CenterCrop`, `RandomCrop` and `HorizontalFlip`, with a seed.
- `Vision.normalize`: mean and std per channel, as tensor ops.
- `Vision.folder` and `Vision.samples`: a folder per class becomes batches for the loader of `train`.

### asr

- No changes of its own: released at 0.5.0 with the others, on the same `tensor`, `nn` and `hub`.

### whisper

- No changes of its own: released at 0.5.0 with the others.

## [0.4.0] - 2026-11-25

- **Breaking:** ml needs Agentica 0.38.0.
- Install by tag with no `override = true` (ml#240): the packages ask for each other by the tag of their own release (`tensor-0.4.0`), not by commit, so an app that declares `tensor` and `nn` by their 0.4.0 tags gets one source for each package.
- `nn`, `hub` and `asr` are released at 0.4.0 with the others; they have no changes of their own beyond the lines above.

### Agent skills

- `ml-inference` and `ml-training` (`.agents/skills/`): skills that guide an agent through running and training models with the packages.

### tensor

- `tools/build-torch.sh` finds Debian's `libtorch-dev` (system headers and multiarch libraries) when no libtorch release folder is found.
- `Backend.Xla(Device.Cpu)`: a traced function compiled and run by XLA on the CPU, through the PJRT plugin loaded at run time; `agentica tools/install_xla.agc` installs the plugin (Linux x86_64 and aarch64).
- ggml: the CPU build uses llamafile's sgemm for the F32 matmul (`-DGGML_LLAMAFILE=ON` in `build-engine.sh`, so the cache entry is built once more): a training step of an MLP is 1.2 to 1.7 times faster on the CPU, and the prompt of Qwen3-0.6B 1.9 times.
- ggml: the CPU backend runs on one thread per physical core instead of one per hardware thread (4 instead of 8 on a 4-core machine with SMT). The median step is about the same at batch 1024 and 4096 (3 to 9 percent less at 1024) and up to 20 percent slower in some small-batch `fit` cells (Adam, batch 64: 9.1 ms against 7.4 ms); the slowest step of a run is shorter (62 ms at most, against up to 123 ms). `TENSOR_THREADS` sets the number (up to 512, ggml's limit).
- ggml: an output of a compiled graph that an op makes is written where the caller holds it, with no copy, and its finiteness check is one row sum instead of a scaled copy: a step at batch 64 takes about 7 to 9 percent less on the CPU and on Vulkan (measured under load), and at batch 1024 3 to 4 percent less on Vulkan and the same on the CPU. A finite output whose sum overflows `F32` is refused too.

### train

- `Train.fit` runs on `Backend.Xla(Device.Cpu)`: each step is one XLA program, with the batch gathered in it and Adam's moments updated in place.
- `Lora.merged`: the weights of a model with an adapter merged in, for inference, on any backend.
- `Train.fit` on ggml gathers the rows of a batch inside the step instead of in a graph of its own, with the same batches: a step at batch 16 to 64 on Vulkan takes up to half the time (14 ms to 7 to 9 ms), and a CPU step 4 to 17 percent less. The loader's prefetch has nothing to do there, so `fit` starts no task.
- `Lora.save` and `Lora.open` keep an adapter in PEFT's files (`adapter_config.json`, `adapter_model.safetensors`): an adapter trained with PEFT runs here, and one trained here loads in PEFT.

### transformers

- `Qwen3.embeddings` and `DistilBert.embeddings`: the names of a model's embedding tables, for `Lora.save`; a PEFT LoRA adapter of a Qwen3 or a DistilBERT loads with `Lora.open` and runs merged.
- `tools/distilbert_stsb.agc` takes an adapter folder as its last argument and fine-tunes with LoRA (`benchmarks/stsb/run.sh ours-lora`, against PEFT): 1.35 times PEFT's time per step, the same loss to 1.4e-7, the same memory (`benchmarks/README.md`). The speed goal of ml#137 (equal to or better than PyTorch) is not met: ml#236.

### whisper

- Checks on Agentica 0.38.0: the WAV reader and writer use the conversions of its checker. Uses `native-gen` 0.5.0, like `tensor`.

## [0.3.0] - 2026-11-25

### tensor

- **Breaking:** a function that you compile with `Trace.compile` and that calls `Optim.step` (or `Tensor.adam_step`) for Adam or AdamW on ggml now updates the first and second moments in place when nothing else reads them: the tensors you pass are consumed, and `Trace.run` gives them back with the new values (`Trace.donated` says which outputs). `Optim.step` itself, `Trace.eval` and `Trace.eval_many` still change none of their arguments.
- `Tensor.adam_step(p, g, m, v, hyper)`, the update kernel of Adam and AdamW (call `Optim`, not this): one `ggml_opt_step_adamw` node on ggml, a composition elsewhere. In a function compiled by `Trace.compile` it updates the moments in place when nothing else reads them (`Trace.donated` says which outputs are inputs). It has no gradient: a gradient through the Adam update of `Optim.step` is an error.
- `Trace.stablehlo(specs, f)` writes a traced function as a StableHLO module, for XLA and IREE (the ops of an MLP forward and of a training step, with `slice`; an `adam_step` is written as the ops it composes); its goldens come from IREE (`make iree-goldens`, local only).
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

- `Optim` takes the Adam update through `Tensor.adam_step`, so on ggml it is one kernel for each parameter (`Optim.step_inputs` gives the 7 numbers of the step as one input, which `Optim.step` accepts).
- `Lora`: LoRA adapters for the linear layers of a model by name (`init`, `params`, `with_params`, `applied`), trained with `Train.fit` while the model stays frozen.
- `Train.fit_batches` trains over batches that a function makes, each with its own shape (sentences padded to the longest of the batch).
- `Train.fit` frees the parameters and the optimizer state a step replaces at once on libtorch and ggml, where they waited for the GC: a large model no longer holds several copies of its weights.
- `Data.free(tensor)` frees the native memory of one tensor now, and `Data.freed?(batch)` says whether a batch was freed.
- `Optim.AdamW`: Adam with decoupled weight decay, as PyTorch's `AdamW`, with a `decay` flag per parameter (biases and layer-norm weights do not decay in a fine-tune).
- `Optim.Schedule` (`Constant`, `WarmupLinear`, `WarmupCosine`) for the `schedule:` of `Sgd`, `Adam` and `AdamW`: the learning rate follows the step count, as in Hugging Face's `Trainer` up to `total` (after it the factor stays 0). SGD's learning rate is now an input of the step (`Optim.step_scalars`), like Adam's.
- `Optim.step` takes the inputs of `Optim.step_inputs` as its `scalars` (or none): SGD one rank-0 input, its learning rate; Adam one input of 7 numbers, AdamW two.
- `Metrics`: scalars written to a run folder as JSON lines and as a TensorBoard event file (pure Agentica), and `Train.fit(..., metrics: some(run))` reports `train/loss`, `train/lr` and `train/step_ms` for every step.
- `Train.stablehlo` writes the step of `fit` as one StableHLO program for every step: the inputs of the step (`Optim.step_inputs`: SGD's rate, Adam's 7 numbers) follow the moments, and the Adam update is the composition of `Tensor.adam_step`, as IREE verifies.
- `Params.split`, `join`, `named` and `update` train part of a model: the weights selected by name train, and the frozen rest has no gradient and no optimizer state.
- `Train.fit` compiles its step once for each batch shape and runs it on every batch of that shape, instead of tracing and planning every step; inside `loss` the batch is now a graph input, not a constant.

### hub

- `SafetensorsWrite.write` writes `F32` tensors by name to a safetensors file, and `Hub.tensors` reads every tensor of a checkpoint.
- The GGUF writer refuses a layer name that is not a number, removes its `.part` file when the rename fails, and `Tensor.convert_file` reports a failed close.

### asr

- Pins `tensor`, `nn` and `hub` at the commits of this release, so it installs from git with the code it calls.

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
