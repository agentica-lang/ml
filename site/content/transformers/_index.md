---
title: transformers
description: BERT, Qwen3, Qwen2.5, ResNet, ViT and CLIP in Agentica from Hugging Face checkpoints, and the tokenizers that give Hugging Face's ids.
weight: 7
---

`transformers` is BERT, Qwen3, Qwen3.5 (text, forward only), Qwen2.5, ResNet, ViT and CLIP in Agentica, for inference. It has BERT (`bert-base-uncased` and its relatives), Qwen3 and Qwen2.5 (the forward, generation with a KV cache, and the chat template), ResNet-18 and ResNet-50 and ViT-B/16 (image classification), CLIP ViT-B/32 (image and text embeddings, zero-shot classification), and the tokenizers of BERT, Qwen and CLIP: they read the `tokenizer.json` that sits beside a Hugging Face checkpoint and give the same ids as Hugging Face `tokenizers`.

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

## Zero-shot with CLIP

`Clip.open(dir, backend)` loads `model.safetensors` and `config.json` of a `CLIPModel` (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`: LAION's ViT-B/32, with the architecture of OpenAI's but weights of its own; `openai/clip-vit-base-patch32` publishes no safetensors file). `Clip.image_embeds(m, images)` takes the images as `F32` `[batch, 3, 224, 224]`, normalized as the checkpoint's image processor does, and gives their embeddings `[batch, 512]`. `Clip.text_embeds(m, rows)` takes rows of ids from the tokenizer (`Clip.encode` of `pkg/transformers/tokenizer/clip`) and gives `[rows, 512]`. Neither is normalized, as `get_image_features` and `get_text_features` of Hugging Face give them. `Clip.logits(m, images, texts)` is the cosine of each image with each text times `exp(logit_scale)` (`m.scale`, 100 here), `[images, texts]`, and `Clip.probabilities` is the softmax of that over the texts: with a prompt for each class, the probability of each class. The [vision](../vision/_index.md) package makes the image input: Hugging Face's image processor of this checkpoint (`resample` 3, bicubic; `size` 224; `crop_size` 224) resizes the shorter side to 224, cuts the center 224 by 224, scales to `0..1` and normalizes with the mean `[0.48145466, 0.4578275, 0.40821073]` and the std `[0.26862954, 0.26130258, 0.27577711]`. `Vision.Step.Resize(224, Vision.Kernel.Cubic)`, `CenterCrop(224, 224)` and `normalize` do the same up to the half-pixel rounding of the crop, as for ResNet; the pixels are one level (0.015) from Pillow's at the most, 96% of them exactly. To classify an image with no training on its classes:

```agentica {mode="fragment"}
import Clip from "pkg/transformers/clip"
import ClipTokenizer from "pkg/transformers/tokenizer/clip"
import Tensor from "pkg/tensor/tensor"
import Vision from "pkg/vision/vision"

backend = Tensor.Backend.Ggml(Tensor.Device.Cpu)
m = Clip.open("clip-vit-b-32", backend).unwrap!
tok = ClipTokenizer.open("clip-vit-b-32").unwrap!
labels = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
rows = labels.map { |l| ClipTokenizer.encode(tok, l).unwrap! }
photo = Vision.Folder(classes: ["photo"], paths: ["pet.jpg"], labels: [0])
steps = [Vision.Step.Resize(224, Vision.Kernel.Cubic), Vision.Step.CenterCrop(224, 224)]
batch = Vision.samples(photo, [0], steps, 0, backend).unwrap!
mean = [0.48145466, 0.4578275, 0.40821073]
std = [0.26862954, 0.26130258, 0.27577711]
pixels = Vision.normalize(batch.inputs, mean, std).unwrap!
images = Clip.image_embeds(m, pixels).unwrap!
texts = Clip.text_embeds(m, rows).unwrap!
chances = Tensor.to_list(Clip.probabilities(m, images, texts).unwrap!).unwrap!
puts(chances)
```

- The tower of the images is a convolution of stride 32, a class embedding, learned positions, `pre_layrnorm`, twelve pre-norm blocks with the exact GELU, and the class token through `post_layernorm` and a projection; the tower of the texts is token and position embeddings, twelve pre-norm blocks under a causal mask, a final layer norm, and the vector of the end token through a projection. Each call is one graph on ggml: close what it gives with `Clip.close(t)` and the weights with `Clip.close_model(m)`. Rows of ids are padded to the longest, which changes nothing under the causal mask; a row is at most 77 ids, and an empty one is an error.
- A `model_type` other than `clip`, an activation other than `gelu` (OpenAI's own checkpoints use `quick_gelu`), and a checkpoint whose tensors do not have the names and shapes of its `config.json` are errors, checked before any tensor is loaded. An image of another size than `image_size` is an error, as ViT's.
- Against Hugging Face's `CLIPModel` in `float32`, on four images and six prompts: the ids of the prompts are the same, and on Hugging Face's own 8-bit crops the largest gap is 5e-6 for a text embedding, 2e-5 for an image embedding, 4e-5 for a logit (scale 100) and 2e-6 for a probability, on CPU and on Vulkan, with the same best prompt. Through `vision`'s own resize (`Cubic`) a logit is up to 0.21 away (0.43 with libvips' default Lanczos 3), from the pixels, and the best prompt is the same.
- Speed on the machine of [`benchmarks/`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md): image embeddings (224 by 224) run at 10.3 images per second at batch 1 and 13.5 at batch 32 on the CPU, 13.3 and 29.2 on Vulkan; the text embeddings of rows of 77 ids at 14.2 and 17.6 texts per second on the CPU, 14.4 and 39.7 on Vulkan (exact `float32`).
- Reference and ggml are tested; libtorch is not.

## Qwen3

`Qwen3.open(dir, backend, weights)` loads `config.json` and `model.safetensors` from a folder (or one `.gguf` file, whose key-values give the config) (`Qwen/Qwen3-0.6B`, whose weights are `bfloat16` on disk and `F32` in memory). The same loader reads Qwen2.5 (`Qwen/Qwen2.5-0.5B-Instruct`): its `config.json` says `model_type` `qwen2`, which has no query and key norms and has a bias on the query, key and value projections, and `head_dim` is `hidden_size / num_attention_heads` when the config has none. `Qwen3.run(m, rows)` takes rows of token ids of one length and gives the logits `[batch, seq, vocab_size]`: each position reads itself and the ones before it. With `last: true` it gives only the last position's, `[batch, 1, vocab_size]`, the row that picks the next token. With `weights: Dtype.F16` the big matrices (the projections, the token table and the output head) are kept as half weights on ggml: half the memory and a decode 1.6 times faster on Vulkan, with a load close to `F32`'s (the CPU gain is not claimed: see the hub page). With `weights: Dtype.Q8_0` and a `.gguf` file that stores its matrices as `Q8_0`, they stay as quantized blocks on ggml (see the hub page). `F32` is the default and the exact path, and Vulkan multiplies in fp16 by default (`Tensor.vulkan_fp16`), which breaks the Hugging Face goldens with `F32` weights too: `Tensor.vulkan_fp16(false)` before the first ggml call makes it exact, as the checks of this package do.

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
- The layers are grouped-query attention (Qwen3 normalizes the query and key per head; Qwen2.5 has no such norms and has a bias on the query, key and value projections), rotary positions with `rope_theta`, a SwiGLU MLP, RMS norms, and an output head tied to the embedding table when the config says so (the file's `lm_head.weight` is then ignored).
- `Qwen3.logits(m, ids, last)` is `run` on a tensor of `I32` ids `[batch, seq]`; the caller keeps `ids`. `Qwen3.read_config(dir)` gives the `Config` of `config.json`, `Qwen3.checkpoint(dir)` the weights, and `Qwen3.load(c, config, backend, weights)` builds the model from both, as `open` does.
- `Qwen3.weights(m)` gives the weights by their Hugging Face names (`model.layers.0.self_attn.q_proj.weight`; a tied head is the table and has no name of its own), and `Qwen3.with_weights(m, weights)` gives the model with those tensors in place of its own; the names must be the model's, no more and no fewer, each with the model's shape. With `Params` of the [train page](../train/_index.md) they train part of a model: inside a loss the tensors are traced and `Qwen3.logits` of the rebuilt model is part of the trace. A half or quantized weight has no gradient, so train a model loaded as `F32`.
- A LoRA adapter (`Lora` of the [train page](../train/_index.md#lora)) goes on these names: `targets: ["q_proj", "v_proj"]` takes those projections of every layer, and a PEFT adapter of a `Qwen3ForCausalLM` loads as it is. `Qwen3.with_weights(m, try Lora.merged(Qwen3.weights(m), adapter))` is the model with the adapter merged in. `Qwen3.embeddings(m)` names the embedding table, for `Lora.save(dir, adapter, Qwen3.embeddings(m))`, which then refuses an adapter PEFT would misread.
- `Qwen3.run` is the forward over a whole sequence, with no cache: on ggml every call traces and plans a graph (0.6 to 1.7 seconds on the CPU, 0.2 to 0.6 second on Vulkan for these prompts). To generate, use `Generate`.
- `Qwen3.close_model(m)` frees the weights; the logits belong to the caller (`Lift.close`). `qwen_check` peaks at 3.7 GB on ggml CPU and 1.5 GB on Vulkan, where the weights live in device memory.
- A `model_type` other than `qwen3` or `qwen2`, an activation other than `silu`, `attention_bias: true`, `rope_scaling`, `use_sliding_window: true` (a `sliding_window` size with it `false` is ignored, as Hugging Face does), a head count that the key heads do not divide, and a tensor of another shape are errors. Reference and ggml are tested; libtorch is not, and a sharded checkpoint is not tested. On the reference backend attention needs `q * k` of at most 1,048,576, about 1024 tokens with the causal mask; ggml has no such limit.

## Qwen3.5

`Qwen3_5.open(dir, backend, weights)` loads the text model of `Qwen/Qwen3.5-0.8B` from the folder of its official repository (`config.json`, and `model.safetensors` or its index; the weights are `bfloat16` on disk and `F32` in memory). `Qwen3_5.run(m, rows, last)` is the forward over whole sequences, as `Qwen3.run`, with no cache yet (generation and the tokenizer come next).

```agentica
import Qwen3_5 from "pkg/transformers/qwen3_5"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  m = Qwen3_5.open("Qwen3.5-0.8B", Tensor.Backend.Ggml(Tensor.Device.Cpu)).unwrap!
  puts(Tensor.shape(Qwen3_5.run(m, [[760, 6511, 314, 9338, 369]], true).unwrap!))
  Qwen3_5.close_model(m)
end
```

- The model is hybrid: by `layer_types`, a layer is linear attention (a Gated DeltaNet, `Delta.gated_delta_net` of [nn](../nn/_index.md)) or full attention (grouped-query, with an output gate and rotary positions on a quarter of each head), followed by a gated MLP. The head is the embedding table. The vision tower and the multi-token-prediction head of the checkpoint are ignored.
- The norms of Qwen3.5 scale by `1 + weight`; the loader adds the 1.
- Against Hugging Face's `Qwen3_5ForCausalLM` in `float32` on ggml CPU, over five prompts of 5 to 50 tokens, the largest gap of a logit is 2.0e-5 and the five largest logits of every position are the same ids. A tiny random model is tested on the reference and ggml. Vulkan is not measured yet.
- A `model_type` other than `qwen3_5`, an activation other than `silu`, `attention_bias: true`, a `rope_type` other than `default` and a layer type other than `linear_attention` and `full_attention` are errors.

## Generation

`Generate.open(m)` gives the generator of a model, which keeps the compiled steps for every sequence of that model. `Generate.generate(g, ids, max_new, stop, sampling)` adds at most `max_new` ids to the prompt `ids`, and stops before an id of `stop`. By default it takes the id with the largest logit (greedy), which runs in the graph. With `sampling: some(Sample.Sampling(temperature: 0.8, top_k: 40, top_p: 0.95, seed: 7))` each step draws among the largest logits with Hugging Face's order (the temperature, then `top_k`, then `top_p`), and the same `seed` (0 to 2147483645) draws the same ids, up to the float noise of each backend (a generator of its own, so the ids are not those of Hugging Face's `generate(do_sample=True)`, only the distribution is). The graph keeps only the largest logits and reads back their ids and values, so a sampled step costs about what a greedy one does: `top_k` of them when it is set, else the largest 256 (`Sample.MAX_CANDIDATES`). Ids below those are never drawn, so with `top_k` unset the draw is from the top 256 and `top_p` is measured against their mass, not that of the whole vocabulary (the mass below them after a real temperature is negligible; at a very high temperature over a flat distribution it is not), and a tie at the edge of the `top_k` keeps one id here where Hugging Face keeps all. Ids more than 30 below the largest logit (a mass under 1e-13) are never drawn either. It keeps the keys and values of the tokens so far (a KV cache), so a new token costs one step, not a pass over the whole sequence. It gives the new ids; a second call finds the compiled steps of the first and traces none. `Generate.close(g)` ends the compiled steps, and goes before `Qwen3.close_model(m)`.

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

For one-step control, `Generate.start()`, then `Generate.step(g, cache, ids)` (a prompt first, then one id) gives `Picked(id, cache)`, `Generate.step_logits` gives the logits `[vocab_size]`, and `Generate.finish(cache)` closes the cache. `Generate.step_top(g, cache, ids, k)` is `step_logits` for the `k` largest logits only: their `ids` and `logits`, largest first, and the cache. `Generate.compiles(g)` gives how many compiled steps `g` has made. The cache that `step` is given is dead afterwards, even when the step fails: on ggml a step of the same capacity writes its row into its tensors, which are the ones of the cache it returns, elsewhere they are closed. Use the cache it returns, and never step a cache twice or read one after it was stepped. A generator belongs to one task at a time.

- The ids match Hugging Face's greedy `generate` for `Qwen3-0.6B` in `float32` on ggml CPU and Vulkan: 24 new ids for each of five prompts (English, code, a chat template, accents and CJK, a long repeat).
- The cache is padded to 8, 16, 32 and so on rows, and the prompt too, to a power of 2 (a prompt of 9 tokens runs as 16). On ggml each (tokens, rows, capacity) is one compiled graph, so a generation compiles a handful and a second call none. A one-token step reads only the rows of the cache in use, rounded up to 256 rows (to an eighth of the capacity above 2048), not the padding after them: each such reach is a graph too, and a generator keeps one for a capacity: a sequence that moves to the next reach ends the graph of the one before, so a long cache compiles one more step every 256 rows or more, and a later, shorter sequence takes the graph that is there, which reads more rows than it needs. The choice of the id runs on the device. On ggml a one-token step writes its new row into the tensors of the cache it is given (`Tensor.set_rows`, in place), while the capacity stays: the cache it returns holds those very tensors. A step that grows the capacity makes new ones and closes the old. On ggml the cache is `F16`: half the memory of an `F32` one (0.12 GB per 1024 tokens of Qwen3-0.6B) and half the bytes a step reads, with the logits of a step moving by about 5e-3 on values of 10 to 30 (the greedy ids of `generate_check` are Hugging Face's). The other backends keep `F32`.
- On the reference backend attention needs `q * k` of at most 1,048,576, which the padded prompt counts: a prompt of 600 tokens is padded to 1024. libtorch is not tested. The prompt and the new ids must fit `max_position_embeddings` (40960).
- Speed on this machine (Ryzen 5 3550H, Radeon Vega iGPU with Mesa RADV), on Vulkan with `F32` weights, exact `float32` products (`Tensor.vulkan_fp16(false)`) and the `F16` cache: a token takes 72.7 ms after a prompt of 128 tokens (13.8 tokens per second) and 80.4 ms after one of 2048 (12.4), medians of 3 runs of 24 steps. llama.cpp on the same weights, with flash attention and an `F16` cache, takes 70.6 and 79.4 ms: ours needs 3% and 1.3% more time. The CPU, the prompt itself and the memory were not measured again, and their older numbers are with the tables and the method in [`benchmarks/`](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md).
- A batch of one, the prompt in one step (no chunks).

`Sample.Sampling` has defaults for every field: `temperature` 1.0, `top_k` 0, `top_p` 1.0 and `seed` 0. `top_k` must be 0 or more, `top_p` above 0 and at most 1, and `temperature` 0 (greedy) or at least `1e-6`; `Sample.probabilities(logits, s)` gives the probability of each id under the same settings, 0 for a cut id.

`tools/to_gguf.agc MODEL_DIR OUT.gguf [F32|F16|BF16]` writes a Qwen3 model folder (`config.json`, `model.safetensors` or its index, `tokenizer.json`, and `tokenizer_config.json` for the chat template) as a GGUF that llama.cpp and `Qwen3.open` read: the matrices in the format asked for (`F16` by default), the norms in `F32`, the config, the tokenizer and the chat template as key-values. A tied model writes no `output.weight`. The weights are converted in C from the safetensors file into the new one, so it needs the ggml module and takes a few seconds for Qwen3-0.6B. `Qwen3Gguf.convert(dir, out, format)` is the same call from Agentica code, and the tool runs it. Its F16 and F32 files hold the same key-values and the same bytes of every tensor as llama.cpp's own converter writes for that model.

A GGUF file carries its tokenizer: `Bpe.open_gguf("Qwen3-0.6B-F16.gguf")` gives the one of the `tokenizer.json` (the same ids and text), for a file of llama.cpp with `tokenizer.ggml.model` gpt2 and `pre` qwen2; another tokenizer is an error. The text is NFC-normalized, as `Qwen/Qwen3-0.6B`'s `tokenizer.json` says (llama.cpp does not), and `decode` leaves out control tokens by default (`skip_special`), which include `<|fim_prefix|>` and five like it that `tokenizer.json` does not mark special.

## Chat template

`Chat.apply(model_type, messages, add_generation_prompt, enable_thinking)` makes the prompt text of a conversation for the instruct models of the Qwen family, as Hugging Face's `apply_chat_template` does without tools. `model_type` is the one of the model's `config.json` (`qwen2` or `qwen3`), `messages` is a list of `Chat.Message(role, content)` with the roles `system`, `user` and `assistant`, and the answer is text for the tokenizer (`<|im_start|>` and `<|im_end|>` are special tokens it knows). With `add_generation_prompt: true` (the default) it ends with the opening of the assistant's turn; `qwen2` adds the system turn "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." when the first message is not a system one, and `enable_thinking: false` (Qwen3 only) opens the assistant's turn with an empty think block, so the model answers without thinking. For `qwen3`, an assistant message `<think>reasoning</think>answer` is read as the template does: after the last user message the reasoning is kept in a think block, before it only the answer stays, and the leading newlines of the answer are dropped; for `qwen2` the text is written as it is. A `tool` role, an unknown role and no messages are errors. The text equals Hugging Face's on 27 conversations for the two templates (`test/chat_golden.py`).

```agentica
import Chat from "pkg/transformers/chat"

def main() -> Void
  messages = [Chat.Message(role: "user", content: "Why is the sky blue?")]
  puts(Chat.apply("qwen2", messages).unwrap!)
end
```

```output
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Why is the sky blue?<|im_end|>
<|im_start|>assistant

```

To chat, put the text through the tokenizer and `Generate`, and stop at the end of the turn (`<|im_end|>`, id 151645) or the end of text (151643):

```agentica
import Bpe from "pkg/transformers/tokenizer/bpe"
import Chat from "pkg/transformers/chat"
import Generate from "pkg/transformers/generate"
import Qwen3 from "pkg/transformers/qwen3"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  backend = Tensor.Backend.Ggml(Tensor.Device.Vulkan(0))
  tok = Bpe.open("Qwen2.5-0.5B-Instruct/tokenizer.json").unwrap!
  m = Qwen3.open("Qwen2.5-0.5B-Instruct", backend).unwrap!
  messages = [Chat.Message(role: "user", content: "Name the capital of France in one word.")]
  prompt = Bpe.encode(tok, Chat.apply(m.config.model_type, messages).unwrap!).unwrap!
  g = Generate.open(m).unwrap!
  puts(Bpe.decode(tok, Generate.generate(g, prompt, 16, [151645, 151643]).unwrap!).unwrap!)
  Generate.close(g)
end
```

```output
Paris
```

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

- `WordPiece.parse(text)` reads the tokenizer from the text of a `tokenizer.json`, and `WordPiece.open(path)` from its file.
- `WordPiece.encode(tok, text, pair, add_special, max_length)` gives `Encoding(ids, type_ids)`. `pair` is `none` or `some(text)`, and `add_special` (default `true`) adds `[CLS]` and `[SEP]`. Added tokens in the text (`[MASK]`) are one id each.
- With `max_length: some(n)`, or else the `truncation` of the file (`tok.max_length`), an encoding of one text has `n` ids at most, `[CLS]` and `[SEP]` counted: the tokens past it are dropped, as `tokenizers` does. A pair over the limit is an error. Only the plain truncation is read (cut at the right, `LongestFirst`, no stride). A limit that leaves no room for a token is an error, and a call can replace the file's limit but not turn it off. A file that pads to the longest of a batch is accepted, and `encode` of one text adds no padding (`Bert.run` pads a batch); a fixed-length padding and `pad_to_multiple_of` above 1, which would pad one text too, are errors. `tok.pad_id` is the `pad_id` of the file's padding, none when it has none.
- `WordPiece.decode(tok, ids, skip_special)` joins the pieces back, with the spaces before punctuation cleaned as `tokenizers` does. `skip_special` is `true` by default.
- A word longer than the file's `max_input_chars_per_word`, or one the vocabulary cannot cover, is `[UNK]`.

## Byte-level BPE

`Bpe.open(path)` reads a Qwen tokenizer: NFC normalization, the Qwen split pattern, bytes mapped to the byte-level alphabet, and BPE merges. A file with another pattern is an error.

- `Bpe.parse(text)` reads the tokenizer from the text of a `tokenizer.json`, and `Bpe.open(path)` from its file.
- `Bpe.encode(tok, text)` gives the ids. It adds no tokens; added tokens in the text (`<|im_start|>`) are one id each.
- `Bpe.decode(tok, ids, skip_special)` gives the text. A character split across tokens joins back, and a broken byte sequence becomes U+FFFD.

## CLIP BPE

`Clip.open(dir)` of `pkg/transformers/tokenizer/clip` reads `vocab.json` and `merges.txt` of a CLIP checkpoint and tokenizes as the slow `CLIPTokenizer` of Hugging Face does without `ftfy`: control characters dropped, white space made one space, a space around each CJK ideograph, NFC, lowercase (accents are kept), then CLIP's split (the two special tokens, the endings `'s 't 're 've 'm 'll 'd`, a run of letters, one digit, a run of anything else), the UTF-8 bytes of each piece in GPT-2's alphabet with `</w>` on the last one, and the BPE merges. `Clip.encode(tok, text, max_length)` gives the ids with the start and end tokens around them, the text cut to `max_length - 2` tokens (77 by default); `<|startoftext|>` and `<|endoftext|>` in the text are one id each. A symbol the vocabulary lacks is `<|endoftext|>`. The fast tokenizer (`tokenizer.json`) differs from the slow one on control characters and CJK ideographs; the slow one is what this follows. Python's lowercase also skips an apostrophe or a full stop when it decides that a sigma ends a word, which this does not.

## Errors

A file with another normalizer, pre-tokenizer, post-processor or template, decoder or model, with `truncation`, `padding` or BPE `dropout`, or with an added token that matches normalized text or by words, is an error from `open` and `parse`, not a silent difference. Ids out of range are an error from `decode`.

## Unicode

Agentica has no Unicode properties, so the tokenizers carry their own tables for Unicode 15.1 (`tokenizer/unicode_data.agc`, written by `tools/unicode_tables.py`): letters, numbers, marks, punctuation, white space, lowercase, and NFD and NFC. `tokenizers` has tables of its own, so a character that a newer Unicode version classifies differently can split differently; U+1FAE0 and U+2EBF0 give the same ids. The case-insensitive contractions (`'S`, `'RE`) are ASCII only, which agrees with `tokenizers` on `'ſ`.
