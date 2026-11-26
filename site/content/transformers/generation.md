---
title: generation
description: Generation with a KV cache, the chat template, and zero-shot classification with CLIP.
weight: 2
---

## Generation

`Generate.open(m)` gives the generator of a model, which keeps the compiled steps for every sequence of that model. `Generate.generate(g, ids, max_new, stop, sampling)` adds at most `max_new` ids to the prompt `ids`, and stops before an id of `stop`. By default it takes the id with the largest logit (greedy), which runs in the graph. With `sampling: some(Sample.Sampling(temperature: 0.8, top_k: 40, top_p: 0.95, seed: 7))` each step draws among the largest logits with Hugging Face's order (the temperature, then `top_k`, then `top_p`), and the same `seed` (0 to 2147483645) draws the same ids, up to the float noise of each backend (a generator of its own, so the ids are not those of Hugging Face's `generate(do_sample=True)`, only the distribution is). The graph keeps only the largest logits and reads back their ids and values, so a sampled step costs about what a greedy one does: `top_k` of them when it is set, else the largest 256 (`Sample.MAX_CANDIDATES`). Ids below those are never drawn, so with `top_k` unset the draw is from the top 256 and `top_p` is measured against their mass, not that of the whole vocabulary (the mass below them after a real temperature is negligible; at a very high temperature over a flat distribution it is not), and a tie at the edge of the `top_k` keeps one id here where Hugging Face keeps all. Ids more than 30 below the largest logit (a mass under 1e-13) are never drawn either. It keeps the keys and values of the tokens so far (a KV cache), so a new token costs one step, not a pass over the whole sequence. It gives the new ids; a second call finds the compiled steps of the first and traces none. `Generate.close(g)` ends the compiled steps, and goes before `Qwen3.close_model(m)`.

`Generate.open(m, usable)` sets how many ids a step may answer. The embedding of Qwen3 has more rows than its tokenizer has tokens (151,936 and 151,669), and the ids in between have no text, so a caller with a tokenizer passes its token count (`Generate.open(m, tok.pieces.size)`) and the sampler never gives an id that `Bpe.decode` rejects. The default 0 is Hugging Face's, no mask.

`Generate.open(m, usable, kv: Placement.Kv.Host)` (`import Placement from "pkg/transformers/placement"`) keeps the keys and values of every layer in host memory when the model is on a Vulkan device, as llama.cpp's `--no-kv-offload` does: the attention reads them on the CPU and the cache grows in RAM. The step of the prompt makes the whole cache on the card first, at the capacity of the prompt, and copies it over, so the card's peak holds that cache once; the steps after it make no cache on the card. The default, `Placement.Kv.Device`, leaves each cache where its block is (the cache of a block that `layers_on_device` put on the CPU is in host memory already). The logits are the same. On Qwen3-0.6B Q4_K_M a token takes about 150 ms with `Host` against 20 ms with the cache on the card, 0.90 of llama.cpp's `--no-kv-offload` (134 ms): both pay about 2.3 ms for each of the two splits a layer makes between the card and the CPU (`benchmarks/README.md`). On the CPU engine, and for the blocks that are on the CPU already, `Host` changes nothing, and the recurrent states of Qwen3.5 and LFM3 stay where they are made (on the device).

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
  g = Generate.open(m, tok.pieces.size).unwrap!
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

`Chat.apply(model_type, messages, add_generation_prompt, enable_thinking)` makes the prompt text of a conversation for the instruct models of the Qwen family, as Hugging Face's `apply_chat_template` does without tools. `model_type` is the one of the model's `config.json` (`qwen2`, `qwen3` or `qwen3_5`), `messages` is a list of `Chat.Message(role, content)` with the roles `system`, `user` and `assistant`, and the answer is text for the tokenizer (`<|im_start|>` and `<|im_end|>` are special tokens it knows). With `add_generation_prompt: true` (the default) it ends with the opening of the assistant's turn; `qwen2` (which refuses `some(false)`) adds the system turn "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." when the first message is not a system one, and `enable_thinking` (an `Optional[Bool]`, Qwen3 and Qwen3.5) decides whether the assistant's turn opens with `<think>` (`some(true)`) or with an empty think block, so the model answers without thinking (`some(false)`). Left at `none` it does what each template does when it is not told: Qwen3 thinks and Qwen3.5 does not. For `qwen3`, an assistant message `<think>reasoning</think>answer` is read as the template does: after the last user message the reasoning is kept in a think block, before it only the answer stays, and the leading newlines of the answer are dropped; for `qwen2` the text is written as it is. `qwen3_5` (text only) trims every message, wants the system message first and at least one user message, puts a think block (empty or not) in every assistant message after the last user message and none before it, and ends the prompt with `<think>\n` under `some(true)` or with the empty think block otherwise; every message is stripped like Python's `str.strip`. A `tool` role, an unknown role and no messages are errors, except for `lfm3`. The text equals Hugging Face's on 27 conversations for the templates of `qwen2` and `qwen3` (`test/chat_golden.py`) and on 28 for `qwen3_5` (`test/chat35_golden.py`, 4 of them refused). `lfm3` (`LiquidAI/LFM3-1.7B`) is the one template that takes tools: `Chat.apply("lfm3", messages, add_generation_prompt, enable_thinking, tools, preserve_thinking)` starts with `<|startoftext|>`, writes `tools` (each a function `{"name": ..., "description": ..., "parameters": ...}` as a `Map[String, JSON.Value]`, or that under a `function` key) as `<tools>` lines in the system turn, reads `Chat.Message(role, content, thinking, tool_calls, tool_call_id)` with the roles `system`, `user`, `assistant` and `tool` (consecutive answers with one `tool_call_id` are written as one `<tool_response>`), and a `Chat.ToolCall(name, arguments)` as `<tool_call>` with one `<parameter=...>` per argument (a string as it is, anything else as `json.dumps(ensure_ascii=False)` writes it). It thinks unless `enable_thinking` is `some(false)`; an assistant message keeps its `thinking` (or an empty block) after the last user message, or everywhere with `preserve_thinking`, and before that only the text after its last `</think>`. The text equals the official template's on 49 conversations (`test/chat_lfm3_golden.py`); images, other content parts and `tool_reference`s are not read. Tools, calls, reasoning and `preserve_thinking` are errors for the other families.

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
  g = Generate.open(m, tok.pieces.size).unwrap!
  puts(Bpe.decode(tok, Generate.generate(g, prompt, 16, [151645, 151643]).unwrap!).unwrap!)
  Generate.close(g)
end
```

```output
Paris
```

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
