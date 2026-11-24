---
title: hub
description: Safetensors checkpoints read in Agentica, and nn layers loaded from them by tensor name.
weight: 4
---

`hub` loads trained weights. It opens a `.safetensors` file, a format that stores named tensors, and reads a tensor from it by name. It can also build `nn` layers from those tensors.

`hub` reads local files only.

## Example

Open a checkpoint, list its tensors, and load a dense layer. The file `model.safetensors` is in the folder you run the program from:

```agentica
import Hub from "pkg/hub/hub"
import Load from "pkg/hub/load"
import Tensor from "pkg/tensor/tensor"

def main() -> Void
  c = Hub.open("model.safetensors").unwrap!
  puts(Hub.names(c))
  layer = Load.dense(c, "fc", Tensor.Backend.Reference).unwrap!
  puts(Tensor.shape(layer.w))
end
```

```output
["fc.bias", "fc.weight"]
[2, 3]
```

The layer's weight is stored as `[out, in]` and comes back as it is, with `out_in` set on the `Nn.Dense`: `Nn.dense` multiplies by its transpose.

## Opening

- `Hub.open(path)` takes one `.safetensors` file, or the `.index.json` of a sharded checkpoint. A sharded checkpoint is one model split over several files. The index says which file holds each tensor. The shards must be file names in the index's folder.
- `Hub.names(c)` gives every tensor name, sorted. `Hub.has?(c, name)` tells whether one exists.
- `Hub.tensor(c, name, backend, dtype)` reads one tensor on `backend`; `Hub.shape(c, name)` gives its shape from the header without reading it. `F32`, `F16` and `BF16` come as `F32`, read by the backend from the file (`Tensor.from_file`); every other dtype keeps the dtype of the file. With `dtype: Dtype.F16` a float tensor is kept in 16 bits on the ggml backend (a half weight, see the tensor page).
- `Hub.metadata(c)` gives the `__metadata__` of each shard.

## Loading layers

Each `Load` function reads a layer by the prefix of its tensor names. It gives a `Result` of the `nn` struct.

- `Load.dense(c, prefix, backend, weights)` reads `{prefix}.weight` as `[out, in]` and keeps it so: the `Nn.Dense` has `out_in` set. It reads `{prefix}.bias` when the file has one.
- `Load.layer_norm(c, prefix, eps, backend)` reads `{prefix}.weight` and `{prefix}.bias`, or `{prefix}.gamma` and `{prefix}.beta` when the checkpoint has no `{prefix}.weight` (the names `bert-base-uncased` keeps).
- `Load.embedding(c, name, backend, weights)` reads one table by its full name.
- `Load.conv1d(c, prefix, stride, padding, backend)` reads a `[out, in, kernel]` weight and an optional bias.
- `Load.encoder_block` and `Load.decoder_block` take the names of one block (`BlockNames` or `DecoderBlockNames`), the head count, the activation, `eps` and the backend.

Float weights of 16 and 64 bits (`F16`, `BF16` and `F64`) become `F32`, the compute dtype. With `weights: Dtype.F16` (default `F32`, the exact path) `Load.dense`, `Load.embedding`, the blocks and the models built from them keep their big matrices, those of rank 2 with at least `Load.HALF_MIN_ELEMENTS` (100000) values, as half weights on the ggml backend: half the memory and the bandwidth, one rounding per value at load, and an error naming the tensor and the element for a value beyond 65504 (a BF16 checkpoint converts to F16 this way; its values under about 6e-8, F16's smallest, round to zero, and under 6e-5 they lose digits). An F64 file has no half form and stays `F32`. Vectors, convolutions and small matrices stay `F32`. Any backend but ggml answers an error naming the option. `Load.weights_from_env()` reads `WEIGHTS=f16` for the tools. Measured on one machine (Ryzen 5 3550H, Vega 8 iGPU): Qwen3-0.6B decodes 39% faster on Vulkan with half weights (its prompt 4%), and whisper-medium transcribes 9% faster on Vulkan (its decoder 5%); on the CPU the gain is inside the noise for Qwen3 and for the whisper-medium transcription (its decoder, 14%). The weights take half the memory (Qwen3 2.4 to 1.2 GB), but loading takes longer (Qwen3 1.5 to 3.9 s, whisper-medium 4 to 7 s), since the conversion to 16 bits runs in one thread. Half weights keep the Hugging Face goldens, while `Tensor.vulkan_fp16(true)` does not (it breaks token-exactness with F32 weights too); the pair `weights: F16` and `vulkan_fp16` is the fastest (Qwen3 prompt 1.6x, whisper-medium 30% below the default time) when exactness against Hugging Face is not needed. A layer weight must have the rank (number of dimensions) that its layer takes.

## Limits

- Local files only. Nothing is downloaded.
- `Hub.open` reads only the header, which is limited to 100 MB. A tensor is read when you ask for it, so a checkpoint of several GB opens at once.
- A tensor of a dtype that `tensor` lacks, such as the `F8` types or `U16`, opens with the file. Reading it is an error.
- A tensor name in the index must exist in its shard, or `open` fails.
- `Load` checks ranks, and the sizes of a bias and of a layer norm. It does not check a whole model. `asr` checks its own shapes against its config.
