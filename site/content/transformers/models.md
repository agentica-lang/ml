---
title: models
description: BERT, DistilBERT, ResNet, ViT, Qwen3, Qwen3.5, Granite MoE and LFM3 in Agentica, with their loading, forward and checks against Hugging Face.
weight: 1
---

## BERT

`Bert.open(dir, backend)` loads `model.safetensors` and `config.json` from a folder. `Bert.run(m, encodings)` takes encodings from the tokenizer, pads them to the longest, and gives `Output(hidden, pooled)`: the last hidden state `[batch, seq, hidden_size]` and the pooler output `[batch, hidden_size]` (none when the checkpoint has no pooler). `Bert.embed(m, encodings)` gives one vector per sentence, the mean of its token vectors without the padding. `Bert.load(c, config, backend)` builds the model from an open `Hub.Checkpoint` and a `Config` (the fields of `config.json`). `Bert.forward(m, ids, type_ids, mask)` runs it on tensors: `ids` and `type_ids` are `I32` and `mask` is `F32` (1.0 for a token, 0.0 for padding), all `[batch, seq]`. `run` builds these from the encodings. Close it with `Bert.close(out)`.

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
- Reference and ggml are tested; libtorch is not. `Bert.open` reads `model.safetensors` only, not a sharded checkpoint.

## DistilBERT

`DistilBert.open(dir, backend)` loads `model.safetensors` and `config.json` of a DistilBERT (`sentence-transformers/msmarco-distilbert-cos-v5`, `distilbert-base-uncased`): BERT's blocks with word and position embeddings only, no token types and no pooler. `DistilBert.run(m, encodings)` pads the encodings to the longest and gives the last hidden state `[batch, seq, dim]`. `DistilBert.embed(m, encodings, normalize)` gives one vector per sentence, `[batch, dim]`: the mean of its token vectors, and with `normalize: true` that mean over its length, which is what a sentence-transformers model with `Pooling` and `Normalize` modules answers. `msmarco-distilbert-cos-v5` lists `Normalize` in its `modules.json`, so its embeddings need `normalize: true`.

```agentica
import DistilBert from "pkg/transformers/distilbert"
import Tensor from "pkg/tensor/tensor"
import WordPiece from "pkg/transformers/tokenizer/wordpiece"

def main() -> Void
  backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
  dir = "msmarco-distilbert-cos-v5"
  tok = WordPiece.open("#{dir}/tokenizer.json").unwrap!
  m = DistilBert.open(dir, backend).unwrap!
  sentences = ["How big is London?", "London has 9 million people."]
  encodings = sentences.map { |s| WordPiece.encode(tok, s).unwrap! }
  puts(Tensor.shape(DistilBert.embed(m, encodings, true).unwrap!))
end
```

```output
[2, 768]
```

- `DistilBert.forward(m, ids, mask)` is `run` on tensors: `I32` ids and an `F32` mask (1.0 for a token, 0.0 for padding), both `[batch, seq]`.
- `DistilBert.weights(m)` gives the weights by their names in the checkpoint (`transformer.layer.0.attention.q_lin.weight`; under `distilbert.` when the file has that prefix), and `DistilBert.with_weights(m, weights)` gives the model with those tensors in their place: the names of `weights(m)`, no more and no fewer, with the same shapes. The result shares those tensors, with no copy, so `close_model` of it closes them. With `Params` of the train package they train the model or part of it.
- `DistilBert.pooled(m, ids, mask, normalize)` is the mean embedding as plain tensor calls, for a loss: with the traced weights of `with_weights` it is part of the trace, so `Train.fit` takes its gradient. `Sentence.cosine_similarity_loss` of two of them is the loss of a model trained on scored pairs. Outside a trace it runs eagerly, which is an error on ggml ("does not run eagerly on the ggml backend"): `embed` is the call for inference. The gradient runs on the reference and ggml; libtorch has no backward rule for `batch_matmul`.
- A LoRA adapter (`Lora` of the [train page](../train/_index.md#lora)) goes on these names (`targets: ["q_lin", "v_lin"]`), and a PEFT adapter of a `DistilBertModel` loads as it is; for a checkpoint with the `distilbert.` prefix the names carry it. `DistilBert.embeddings(m)` names the two embedding tables, for `Lora.save`.
- The results belong to the caller (`Lift.close`), and `DistilBert.close_model(m)` frees the weights. Nothing may use a closed model.
- A checkpoint with the `distilbert.` prefix (`DistilBertFor...`) loads too, and its head is ignored. The layer-norm `eps` is 1e-12, as in Hugging Face's code. A `model_type` other than `distilbert`, an activation other than `gelu` and sinusoidal positions are errors.
- Only `model.safetensors` is read, never `pytorch_model.bin`. Reference and ggml are tested; libtorch is not.

`Sentence` (`pkg/transformers/sentence`) holds the pieces a sentence-embedding model adds to an encoder, each usable inside a trace:

- `Sentence.mean_pooled(hidden, mask)` averages `[batch, seq, width]` over the tokens the `[batch, seq]` mask keeps. A row with no token gives zeros.
- `Sentence.normalized(x)` gives each row of `[batch, width]` over its length. A row of zeros stays zeros.
- `Sentence.cosine_similarity_loss(a, b, scores)` is the mean of `(cosine - score)^2` over pairs of rows (`Nn.cosine_similarity`) with `[batch]` scores: sentence-transformers' `CosineSimilarityLoss`.

## ResNet

`ResNet.open(dir, backend)` loads `model.safetensors` and `config.json` of a `ResNetForImageClassification` (`microsoft/resnet-50`, `microsoft/resnet-18`). `ResNet.run(m, images)` takes the images as `F32` `[batch, 3, height, width]`, already normalized as the checkpoint's image processor does, and gives the logits `[batch, classes]`. `ResNet.top(m, logits, k)` gives the `k` best classes of each image as `Classify.Prediction(id, label, logit)`, best first, with the names of `config.json`. The [vision](../vision/_index.md) package makes the input: Hugging Face's image processor (`resample` 3, bicubic; `size` 224; `crop_pct` 0.875) of these checkpoints resizes the shorter side to 256 (224 over `crop_pct` 0.875), cuts the center 224 by 224, scales to `0..1` and normalizes with the ImageNet mean and std. `Vision.Step.Resize(256, Vision.Kernel.Cubic)`, `CenterCrop(224, 224)` and `normalize` do the same up to two differences: the pixels are at most one 8-bit level (0.0175 normalized) from Pillow's, 95% of them exactly, and a crop with an odd gap between the sizes (half a pixel) rounds the offset to the even pixel, as torchvision does, where Hugging Face floors it, so the crop can sit one pixel apart. To classify an image:

```agentica {mode="fragment"}
import ResNet from "pkg/transformers/resnet"
import Tensor from "pkg/tensor/tensor"
import Vision from "pkg/vision/vision"

backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
m = ResNet.open("resnet-50", backend).unwrap!
photo = Vision.Folder(classes: ["photo"], paths: ["cat.jpg"], labels: [0])
steps = [Vision.Step.Resize(256, Vision.Kernel.Cubic), Vision.Step.CenterCrop(224, 224)]
batch = Vision.samples(photo, [0], steps, 0, backend).unwrap!
pixels = Vision.normalize(batch.inputs, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unwrap!
logits = ResNet.run(m, pixels).unwrap!
for p in ResNet.top(m, logits, 5).unwrap![0]
  puts("#{p.label}: #{p.logit}")
end
```

- The batch norm after each convolution is folded into the convolution when the model loads, so the model is for inference and the weights it holds are not the checkpoint's. On ggml `run` is one graph; close its logits with `ResNet.close(logits)` and the weights with `ResNet.close_model(m)`.
- The ids and names are those of ImageNet-1k in the checkpoint's `config.json`. A `model_type` other than `resnet`, an activation other than `relu` and a `layer_type` other than `bottleneck` and `basic` are errors, and so is an input that is not `[batch, 3, height, width]`. Any height and width that the strides leave a pixel of are valid; the checkpoints were trained at 224.
- Against Hugging Face's `ResNetForImageClassification` in `float32`, on four images through the whole pipeline, the best 5 classes are the same and in the same order for both models on CPU and Vulkan. On Hugging Face's own pixels the largest gap of a logit is 1.6e-5 (ResNet-50) and 1.1e-5 (ResNet-18); with `vision`'s own resize (`Cubic`, the kernel of `resample` 3) it is 0.092 and 0.048, from the pixels, which are one level (0.0175) from Pillow's at the most after normalizing (0.16 and 0.065 with libvips' default Lanczos 3).
- Speed on the machine of [`benchmarks/`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md) (ResNet-50, 224 by 224): 7.6 images per second at batch 1 and 8.1 at batch 32 on the CPU, 17.8 and 27.8 on Vulkan; the forward and backward of the folded model takes 1.5 s at batch 1 and 35 s at batch 32 on the CPU.
- Reference and ggml are tested; libtorch is not. `ResNet.forward(m, images)` is the same model as plain tensor calls, for a loss.

## ViT

`ViT.open(dir, backend)` loads `model.safetensors` and `config.json` of a `ViTForImageClassification` (`google/vit-base-patch16-224`). `ViT.run(m, images)` takes the images as `F32` `[batch, 3, 224, 224]`, already normalized as the checkpoint's image processor does, and gives the logits `[batch, classes]`. `ViT.top(m, logits, k)` gives the `k` best classes of each image as `Classify.Prediction(id, label, logit)`, best first, with the names of `config.json`. The [vision](../vision/_index.md) package makes the input: Hugging Face's image processor of this checkpoint stretches the whole image to 224 by 224 (no crop, no aspect ratio kept; `resample` bilinear), scales to `0..1` and normalizes with mean and std 0.5. `Vision.Step.ResizeTo(224, 224, Vision.Kernel.Linear)` and `normalize` do the same: the pixels are one level (0.0078 normalized) from Pillow's at the most, 92% of them exactly (0.0235 with libvips' default Lanczos 3). To classify an image:

```agentica {mode="fragment"}
import Tensor from "pkg/tensor/tensor"
import Vision from "pkg/vision/vision"
import ViT from "pkg/transformers/vit"

backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
m = ViT.open("vit-base-patch16-224", backend).unwrap!
photo = Vision.Folder(classes: ["photo"], paths: ["cat.jpg"], labels: [0])
batch = Vision.samples(photo, [0], [Vision.Step.ResizeTo(224, 224, Vision.Kernel.Linear)], 0, backend).unwrap!
pixels = Vision.normalize(batch.inputs, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).unwrap!
logits = ViT.run(m, pixels).unwrap!
for p in ViT.top(m, logits, 5).unwrap![0]
  puts("#{p.label}: #{p.logit}")
end
```

- The model is a convolution of stride 16 (the patches), a class token, learned positions, twelve pre-norm blocks of `Nn.EncoderBlock` with the exact GELU, a final layer norm and a linear head on the class token. On ggml `run` is one graph; close its logits with `ViT.close(logits)` and the weights with `ViT.close_model(m)`.
- The ids and names are those of ImageNet-1k in the checkpoint's `config.json`. A `model_type` other than `vit` and an activation other than `gelu` are errors, and so is an input that is not `[batch, 3, image_size, image_size]`: the position table is for one size, and nothing interpolates it. The names and shapes of the checkpoint's tensors are checked before any is loaded.
- Against Hugging Face's `ViTForImageClassification` in `float32`, on four images: with Hugging Face's own 8-bit pixels the top 5 classes are the same and in the same order, and the largest gap of a logit is 1.2e-5 on CPU and 9.6e-6 on Vulkan. Through `vision`'s own resize (`Linear`) the largest gap is 0.103 (0.156 with Lanczos 3), from the pixels, and the top 5 is Hugging Face's in the same order, on CPU and Vulkan (with Lanczos 3 the best five of these drawn images, whose logits lie within 0.03 to 0.3 of each other, were reordered).
- Speed on the machine of [`benchmarks/`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md): ViT-B/16 forward at 224 by 224 runs at 2.6 images per second at batch 1 and 3.4 at batch 32 on the CPU, 5.3 and 7.4 on Vulkan (exact `float32`).
- Reference and ggml are tested; libtorch is not. `ViT.forward(m, images)` is the same model as plain tensor calls, for a loss.

## Qwen3

`Qwen3.open(dir, backend, weights)` loads `config.json` and `model.safetensors` from a folder (or one `.gguf` file, whose key-values give the config) (`Qwen/Qwen3-0.6B`, whose weights are `bfloat16` on disk and `F32` in memory). The same loader reads Qwen2.5 (`Qwen/Qwen2.5-0.5B-Instruct`): its `config.json` says `model_type` `qwen2`, which has no query and key norms and has a bias on the query, key and value projections, and `head_dim` is `hidden_size / num_attention_heads` when the config has none. `Qwen3.run(m, rows)` takes rows of token ids of one length and gives the logits `[batch, seq, vocab_size]`: each position reads itself and the ones before it. With `last: true` it gives only the last position's, `[batch, 1, vocab_size]`, the row that picks the next token. With `weights: Dtype.F16` the big matrices (the projections, the token table and the output head) are kept as half weights on ggml: half the memory and a decode 1.6 times faster on Vulkan, with a load close to `F32`'s (the CPU gain is not claimed: see the hub page). With `weights: Dtype.Q8_0` and a `.gguf` file that stores its matrices as `Q8_0`, or `weights: Dtype.Q4_K` and one that stores them as `Q4_K` and `Q6_K`, they stay as quantized blocks on ggml (see the hub page). `F32` is the default and the exact path, and Vulkan multiplies in fp16 by default (`Tensor.vulkan_fp16`), which breaks the Hugging Face goldens with `F32` weights too: `Tensor.vulkan_fp16(false)` before the first ggml call makes it exact, as the checks of this package do.

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
- Qwen2.5-0.5B-Instruct matches `Qwen2ForCausalLM` in `float32` too: the largest gap of a logit is 4.3e-5 on the CPU and 1.3e-4 on Vulkan over the same five prompts, the 24 greedy ids of each prompt are Hugging Face's on Vulkan (on the CPU for prompts 0 to 2), and its tokenizer agrees with `tokenizers` on 1827 cases. Hugging Face's own `generate` for Qwen2.5-Instruct applies the `repetition_penalty` 1.1 of its `generation_config.json` even when greedy, so its ids differ from ours (the goldens set 1.0).
- A Q8_0 GGUF of Qwen3-0.6B (made by llama.cpp's `llama-quantize` from the F16 file) runs with `weights: Dtype.Q8_0` in 2.0 GB at the peak of `qwen_check` against 4.1 GB with `F32` weights (the same GGUF read as `F32`), and its prompts take 15% to 25% less time on the CPU (one run). Against the logits of llama.cpp for the same file (`tools/llama_golden.c`; `QUANTIZED=1 qwen_check`) the largest gap of a logit is 0.32 to 0.59 over the five prompts, on logits of up to 30. Both round every activation of a product to Q8_0 blocks, and that rounding is chaotic: llama.cpp against itself with an `F32` KV cache instead of an `F16` one moves by up to 0.65. With `F32` weights from the same file dequantized the model matches llama.cpp's own `F32` run of it to 0.018.
- A Q4_K_M GGUF of Qwen3-0.6B (made by `llama-quantize`; Q4_K and Q6_K matrices) runs with `weights: Dtype.Q4_K` in 1.6 GB at the peak of `qwen_check` against 3.2 GB with `F32` weights. Against the logits of llama.cpp for the same file the largest gap of a logit is 1.66 over the five prompts, on logits of up to 30, which is the noise of rounding the activations to blocks: llama.cpp against its own `F32` run of the dequantized file differs by 0.65 to 1.97. With `F32` weights from the same file the gap against llama.cpp's `F32` run of it is 0.021. Against llama.cpp on the same file at equal threads (4), a prompt of 128 tokens runs about 1.1 times as fast as llama.cpp's on the CPU and 0.84 times on Vulkan, and 64 tokens after it 0.89 times and 0.84 times; the CPU memory is the same (0.74 GB) and, on Vulkan, the GPU memory a little lower (565 against 627 MB) and the host RSS 0.29 GB higher (`benchmarks/README.md`, ml#291).
- The layers are grouped-query attention (Qwen3 normalizes the query and key per head; Qwen2.5 has no such norms and has a bias on the query, key and value projections), rotary positions with `rope_theta`, a SwiGLU MLP, RMS norms, and an output head tied to the embedding table when the config says so (the file's `lm_head.weight` is then ignored).
- `Qwen3.logits(m, ids, last)` is `run` on a tensor of `I32` ids `[batch, seq]`; the caller keeps `ids`. `Qwen3.read_config(dir)` gives the `Config` of `config.json`, `Qwen3.checkpoint(dir)` the weights, and `Qwen3.load(c, config, backend, weights)` builds the model from both, as `open` does.
- `Qwen3.weights(m)` gives the weights by their Hugging Face names (`model.layers.0.self_attn.q_proj.weight`; a tied head is the table and has no name of its own), and `Qwen3.with_weights(m, weights)` gives the model with those tensors in place of its own; the names must be the model's, no more and no fewer, each with the model's shape. With `Params` of the [train page](../train/_index.md) they train part of a model: inside a loss the tensors are traced and `Qwen3.logits` of the rebuilt model is part of the trace. A half or quantized weight has no gradient, so train a model loaded as `F32`.
- A LoRA adapter (`Lora` of the [train page](../train/_index.md#lora)) goes on these names: `targets: ["q_proj", "v_proj"]` takes those projections of every layer, and a PEFT adapter of a `Qwen3ForCausalLM` loads as it is. `Qwen3.with_weights(m, try Lora.merged(Qwen3.weights(m), adapter))` is the model with the adapter merged in. `Qwen3.embeddings(m)` names the embedding table, for `Lora.save(dir, adapter, Qwen3.embeddings(m))`, which then refuses an adapter PEFT would misread.
- `Qwen3.run` is the forward over a whole sequence, with no cache: on ggml every call traces and plans a graph (0.6 to 1.7 seconds on the CPU, 0.2 to 0.6 second on Vulkan for these prompts). To generate, use `Generate`.
- `Qwen3.open(dir, backend, weights, layers_on_device)` with `layers_on_device: n` on a Vulkan device loads the weights of the blocks from the `n`th on on the CPU, so a model that does not fit in the memory of the card still runs, as with `-ngl` in llama.cpp. A step runs those blocks on the CPU, with their KV cache in host memory, and the first `n` blocks on the device; with `n` below the number of blocks the embedding table, the head and the final norm are on the CPU too, as with `-ngl` in llama.cpp (the head's product with the vocabulary is slower on an iGPU than on the CPU), and with every block on the device they are there as well. The activations cross the bus once at each boundary. The default, `none`, keeps everything on the device, exactly as before, and on a CPU backend, or with an `n` of the number of blocks or more, `n` changes nothing. The logits of a step are those of the model with every block on the device (`n` of 0, 1 or the number of blocks, in the tests), up to the rounding that the two backends differ by. The tools of `tools/` read the `AGENTICA_ML_LAYERS_ON_DEVICE` environment variable. On Qwen3-0.6B Q4_K_M the card holds 579 MB with every block, 386 MB with 14 and 209 MB with none; with the same `-ngl` of llama.cpp, the prompt runs at about 1.0 (14 blocks) and 1.4 (none) times its speed and a token at 0.8 and 0.45 times, so the knob is for a model that does not fit, not for speed. See the [tensor page](../tensor/_index.md#backends) for how the graph is split.
- `Qwen3.close_model(m)` frees the weights; the logits belong to the caller (`Lift.close`). `qwen_check` peaks at 3.7 GB on ggml CPU and 1.5 GB on Vulkan, where the weights live in device memory.
- A `model_type` other than `qwen3` or `qwen2`, an activation other than `silu`, `attention_bias: true`, `rope_scaling`, `use_sliding_window: true` (a `sliding_window` size with it `false` is ignored, as Hugging Face does), a head count that the key heads do not divide, and a tensor of another shape are errors. Reference and ggml are tested; libtorch is not, and a sharded checkpoint is not tested. On the reference backend attention needs `q * k` of at most 1,048,576, about 1024 tokens with the causal mask; ggml has no such limit.

## Qwen3.5

Its tokenizer is `Bpe` and its chat template `Chat.apply("qwen3_5", ...)`, below. `Qwen3_5.open(dir, backend, weights)` loads the text model of `Qwen/Qwen3.5-0.8B` from the folder of its official repository (`config.json`, and `model.safetensors` or its index; the weights are `bfloat16` on disk and `F32` in memory). `Qwen3_5.run(m, rows, last)` is the forward over whole sequences, as `Qwen3.run`, with no cache; `Generate` of `pkg/transformers/generate_qwen35` generates with a cache that holds the keys and values of the full-attention layers and the states of the linear ones.

```agentica
import Qwen3_5 from "pkg/transformers/qwen3_5"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  m = Qwen3_5.open("Qwen3.5-0.8B", Tensor.Backend.Ggml(Tensor.Device.Cpu)).unwrap!
  puts(Tensor.shape(Qwen3_5.run(m, [[760, 6511, 314, 9338, 369]], true).unwrap!))
  Qwen3_5.close_model(m)
end
```

- The model is hybrid: by `layer_types`, a layer is linear attention (a Gated DeltaNet, `Delta.gated_delta_net` of [nn](../nn/_index.md)) or full attention (grouped-query, with an output gate and rotary positions on part of each head, a quarter in 0.8B), followed by a gated MLP. The head is the embedding table. The vision tower and the multi-token-prediction head of the checkpoint are ignored.
- The norms of Qwen3.5 scale by `1 + weight`; the loader adds the 1. Only `F32` weights are loaded, a batch is unrolled row by row in the graph, and the model cannot be fine-tuned with `train/`.
- Against Hugging Face's `Qwen3_5ForCausalLM` in `float32` on ggml CPU, over five prompts of 5 to 50 tokens, the largest gap of a logit is 2.0e-5 (5.5e-5 on Vulkan with `vulkan_fp16` off) and the five largest logits of every position are the same ids. A tiny random model is tested on the reference and ggml.
- Generation: `Generate.open(m)` of `pkg/transformers/generate_qwen35` is the generator of a Qwen3.5 model, with `generate`, `step`, `step_logits`, `step_top`, `finish` and `close` as in [Generation](generation.md#generation). The cache of a full-attention layer is the padded KV cache described there; a linear-attention layer keeps its two states (the last rows of the convolution's input and the recurrent state), which do not grow.
- `Generate.open(m, usable)` is the one of Qwen3 above. The embedding of Qwen3.5 has more rows than its tokenizer has tokens (248,320 and 248,070 in 0.8B), and the ids in between have no text, so a caller with a tokenizer passes its token count (`Generate.open(m, tok.pieces.size)`) and the sampler never gives an id that `Bpe.decode` rejects. The default 0 is Hugging Face's, no mask.
- Against Hugging Face's greedy `generate` for `Qwen/Qwen3.5-0.8B` in `float32`, the 24 new ids of each of six prompts (5 to 375 tokens) are equal on ggml CPU and on Vulkan (with `vulkan_fp16` off), at 8 to 9 tokens per second after the prompt (7.8 on CPU, 8.7 on Vulkan, `benchmarks/README.md`).
- A `model_type` other than `qwen3_5`, an activation other than `silu`, `attention_bias: true`, a `rope_type` other than `default` and a layer type other than `linear_attention` and `full_attention` are errors.

## Granite MoE

`GraniteMoe.open(dir, backend, weights)` loads IBM Granite MoE (`granitemoe`, the 1B-A400M model `ibm-granite/granite-3.1-1b-a400m-instruct`) from the folder of its official repository (`config.json`, and `model.safetensors` or its index; the weights are `bfloat16` on disk and `F32` in memory, 5.3 GB for the 1B-A400M model). `GraniteMoe.run(m, rows, last)` is the forward over whole sequences, as `Qwen3.run`, with no cache; `Generate` of `pkg/transformers/generate_granite_moe` generates with a KV cache, as in [Generation](generation.md#generation). There is no tokenizer or chat template for it yet, so a caller passes token ids.

```agentica
import GraniteMoe from "pkg/transformers/granite_moe"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  m = GraniteMoe.open("granite-3.1-1b-a400m-instruct", Tensor.Backend.Ggml(Tensor.Device.Cpu)).unwrap!
  puts(Tensor.shape(GraniteMoe.run(m, [[1318, 203, 35]], true).unwrap!))
  GraniteMoe.close_model(m)
end
```

- Each block is attention (grouped-query, rotary positions with `rope_theta`, the scores scaled by `attention_multiplier`) and a mixture of experts (`Moe.moe` of [nn](../nn/_index.md)): the router picks `num_experts_per_tok` of `num_local_experts` experts for each token, and their gated MLPs are added with the softmax of the picked scores as gates. The embeddings are multiplied by `embedding_multiplier`, each of the two branches of a block by `residual_multiplier` before it is added, and the logits are divided by `logits_scaling`. The head is the embedding table when `tie_word_embeddings` is true, else `lm_head.weight`.
- Only `F32` weights are loaded, and the model cannot be fine-tuned with `train/`. The experts of a layer are two stacked tensors, so the model needs about 6 GB of memory on the CPU.
- Against Hugging Face's `GraniteMoeForCausalLM` in `float32` on the real checkpoint, over five prompts of 5 to 217 tokens, the largest gap of a logit is 3.7e-5 on ggml CPU and 9.3e-5 on Vulkan with `vulkan_fp16` off, and the five largest logits of every position are the same ids. With `vulkan_fp16` on (the default on Vulkan) the largest gap is 0.34, and the fifth largest id of two positions differs. Call `Tensor.vulkan_fp16(false)` before the first ggml call for exact logits. A tiny random model is tested on the reference and ggml.
- Against Hugging Face's greedy `generate` for the same checkpoint in `float32`, the 24 new ids of each of five prompts (5 to 217 tokens) are equal on ggml CPU, at about 17 tokens per second after the prompt and 5.6 GB at the peak. The KV cache is `F32`; a step's logits are 8.2e-5 from Hugging Face's.
- A `model_type` other than `granitemoe`, an activation other than `silu`, `attention_bias: true` and `rope_scaling` are errors.

## LFM3

`Lfm3.open(dir, backend, weights)` loads Liquid AI's LFM3 (`lfm3`, `LiquidAI/LFM3-1.7B`) from the folder of its official repository (`config.json`, and `model.safetensors` or its index; the weights are `bfloat16` on disk, 3.4 GB, and `F32` in memory, or `F16` with `weights: Dtype.F16` on ggml). `Lfm3.run(m, rows, last)` is the forward over whole sequences, as `Qwen3.run` is, so its cost grows with the square of the length; `Generate` of `pkg/transformers/generate_lfm3` generates with a cache, below. Its tokenizer is `Bpe` and its chat template `Chat.apply("lfm3", ...)`, both below: the tokenizer has no normalizer, cuts digits in groups of up to three and takes a piece that is a token as it is (`ignore_merges`), and gives the ids of Hugging Face `tokenizers` on `tokenizer.json` (`<|startoftext|>` is 124894, `<|im_end|>` 124900). The weights are Liquid AI's, under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM3-1.7B/blob/1da06bec6a819da10b54c67e5c90972b46d81871/LICENSE). Nothing of LFM3 is redistributed here: the tests use random weights, and the scripts that make their goldens import Liquid AI's code from a local download, whose license the person running them accepts.

```agentica
import Lfm3 from "pkg/transformers/lfm3"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  m = Lfm3.open("LFM3-1.7B", Tensor.Backend.Ggml(Tensor.Device.Cpu), Tensor.Dtype.F16).unwrap!
  puts(Tensor.shape(Lfm3.run(m, [[1, 2, 3]], true).unwrap!))
  Lfm3.close_model(m)
end
```

- A layer is a gated short convolution (`ShortConv` of [nn](../nn/_index.md): 7 taps, with no bias), a sliding-window attention (the last 256 tokens, rotated by theta 1e4) or a global one (theta 1e6, half of each head rotated), by `layer_types`, then a SwiGLU MLP. Each of the two branches is normed before and after, and the attention has a gate of one number per head.
- Against the official code (`trust_remote_code`, in `float32` layer by layer from the `bfloat16` file) on the real checkpoint, over five prompts of 5 to 375 tokens (the last is past the window), the largest gap of a logit with `F16` weights is 4.7e-5 on ggml CPU and 1.7e-4 on Vulkan with `vulkan_fp16` off, and the five largest logits of every position are the same ids. The CPU run holds 3.9 GB at the peak. `F16` rounds each weight once at load: a `bfloat16` value from 6.1e-5 to 65504 is exact, a larger one is an error that names the tensor, a smaller one keeps fewer digits and one under 3e-8 becomes 0 (in the checkpoint, 4.6 million of the 1.73 billion weights of the `F16` matrices are not 0 and under 6.1e-5, and 125,000 of them are under 3e-8).
- Generation: `Generate.open(m, usable)` of `pkg/transformers/generate_lfm3` is the generator of an LFM3 model, with `generate`, `step`, `step_logits`, `step_top`, `finish` and `close` as in [Generation](generation.md#generation). The cache holds the keys and values of the 11 attention layers (padded to a capacity of 8, 16, 32 and so on, `F16` on ggml; a sliding layer keeps every row and hides the ones past its window with its mask) and the last 6 rows of the input of each of the 13 convolution layers, which do not grow. The embedding has 128,000 rows and the tokenizer 125,033 tokens, so a caller with a tokenizer passes its token count (`Generate.open(m, tok.pieces.size)`).
- Against the official `generate` (`float32` layer by layer from the `bfloat16` file) with `F16` weights, the 24 new ids of each of six prompts (5 to 601 tokens, two past the window) are equal on ggml CPU and on Vulkan with `vulkan_fp16` off. The logits at the five largest ids are within 3.6e-3 with the `F16` cache and 2.9e-5 with an `F32` one, so the cache is the gap. A tiny random model with a window of 4 is tested on the reference and ggml. The cost of a prompt grows with the square of its length (its masks and scores are dense), so the practical context is a few thousand tokens and not the 256K of the config; nothing past 601 tokens was measured. Speed is in [`benchmarks/`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md).
- A `model_type` other than `lfm3`, `conv_bias: true`, `use_sandwich_norm: false`, a `rope_type` other than `default` and a layer type other than `conv`, `full_attention` and `sliding_attention` are errors, and so are `weights` other than `F32` and `F16`. The model cannot be fine-tuned with `train/`.
