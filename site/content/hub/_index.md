---
title: hub
description: Safetensors and GGUF checkpoints read in Agentica, and nn layers loaded from them by tensor name.
weight: 4
---

`hub` loads trained weights. It opens a `.safetensors` file, a format that stores named tensors, and reads a tensor from it by name. It can also build `nn` layers from those tensors.

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

- `Hub.open(path)` takes one `.safetensors` file, one `.gguf` file (llama.cpp's format, see below), or the `.index.json` of a sharded checkpoint. A sharded checkpoint is one model split over several files. The index says which file holds each tensor. The shards must be file names in the index's folder.
- `Hub.open_gguf(path)` and `Hub.open_index(path)` open a `.gguf` file or an `.index.json` whatever the extension; `open` picks by the extension and reads any other path as safetensors.
- `Hub.names(c)` gives every tensor name, sorted. `Hub.has?(c, name)` tells whether one exists.
- `Hub.tensor(c, name, backend, dtype)` reads one tensor on `backend`; `Hub.shape(c, name)` gives its shape from the header without reading it. `F32`, `F16` and `BF16` come as `F32`, read by the backend from the file (`Tensor.from_file`); every other dtype keeps the dtype of the file, except `F64` on the ggml backend, which holds no `F64`: it comes as `F32`, silently (`-1e300` becomes `-inf`). With `dtype: Dtype.F16` a float tensor is kept in 16 bits on the ggml or the XLA backend (a half weight, see the tensor page).
- `Hub.source(c, name)` gives the file and the byte where an `F32`, `F16` or `BF16` tensor starts, with its shape, for [writing GGUF](#writing-gguf).
- `Hub.metadata(c)` gives the `__metadata__` of each shard.

## Loading layers

Each `Load` function reads a layer by the prefix of its tensor names. It gives a `Result` of the `nn` struct.

- `Load.dense(c, prefix, backend, weights)` reads `{prefix}.weight` as `[out, in]` and keeps it so: the `Nn.Dense` has `out_in` set. It reads `{prefix}.bias` when the file has one.
- `Load.layer_norm(c, prefix, eps, backend)` reads `{prefix}.weight` and `{prefix}.bias`, or `{prefix}.gamma` and `{prefix}.beta` when the checkpoint has no `{prefix}.weight` (the names `bert-base-uncased` keeps).
- `Load.embedding(c, name, backend, weights)` reads one table by its full name.
- `Load.weight(c, name, rank, backend)` reads one float tensor by its full name as `F32`, and checks that it has `rank` dimensions.
- `Load.held_bytes(c, name, weights)` counts the bytes that the tensor `name` takes on a ggml device under `weights` (4 a value, 2 for a matrix kept `F16`, the blocks of `Q8_0`, `Q4_K` and `Q6_K`), from the shape and dtype of the file, without reading a value.
- `Load.conv1d(c, prefix, stride, padding, backend)` reads a `[out, in, kernel]` weight and an optional bias.
- `Load.encoder_block` and `Load.decoder_block` take the names of one block (`BlockNames` or `DecoderBlockNames`), the head count, the activation, `eps`, the backend and `weights` (as in `dense`).
- `BlockNames` has the prefixes `norm1`, `query`, `key`, `value`, `output`, `norm2`, `up` and `down`. `DecoderBlockNames` has the same self-attention and `up` and `down` names, plus `cross_query`, `cross_key`, `cross_value` and `cross_output` for the cross-attention and `norm3`.

Float weights of 16 and 64 bits (`F16`, `BF16` and `F64`) become `F32`, the compute dtype. With `weights: Dtype.F16` (default `F32`, the exact path) `Load.dense`, `Load.embedding`, the blocks and the models built from them keep their big matrices, those of rank 2 with at least `Load.HALF_MIN_ELEMENTS` (100000) values, as half weights on the ggml or the XLA backend: half the memory and the bandwidth, one rounding per value at load, and an error naming the tensor and the element for a value beyond 65504 (a BF16 checkpoint converts to F16 this way; its values under about 6e-8, F16's smallest, round to zero, and under 6e-5 they lose digits). An F64 file has no half form and stays `F32`. Vectors, convolutions and small matrices stay `F32`. Any other backend answers an error naming the option. `Load.weights_from_env()` reads `WEIGHTS=f16` for the tools. With `weights: Dtype.BF16` the same matrices are kept as `BF16` on ggml, Torch and XLA: a `bfloat16` checkpoint keeps its values as they are (none rounds to zero, none is beyond range), half the bytes of `F32` weights; `WEIGHTS=bf16` sets it. Measured on one machine (Ryzen 5 3550H, Vega 8 iGPU; [benchmarks](https://github.com/agentica-lang/ml/blob/main/benchmarks/README.md)): Qwen3-0.6B decodes 1.6 times faster on Vulkan with half weights (its prompt 12%); on the CPU one window measured the Qwen3 decode 72% faster and an earlier one no gain, so no CPU gain is claimed. The weights take half the memory (Qwen3 2.5 to 1.4 GB on the CPU), and loading them is close to `F32`'s (whisper-medium 3.1 s against 2.4 s, Qwen3 1.5 s against 1.5 s), since the conversion to 16 bits runs on up to 8 threads. Half weights keep the Hugging Face goldens with `Tensor.vulkan_fp16(false)`; the default of Vulkan, fp16, does not (it breaks token-exactness with F32 weights too), and the pair `weights: F16` and fp16 is the fastest (Qwen3 prompt 1.7 times) when exactness against Hugging Face is not needed. `Load.vulkan_fp16_from_env(default)` is the tools' way to choose: `VULKAN_FP16=0` or `1`, else `default`. With `weights: Dtype.Q8_0` the rank-2 weights that a GGUF file stores as `Q8_0` (the projections, the token table and the output head of a quantized model) are kept as their blocks on the ggml backend, about a quarter of the memory of `F32`, and nothing is converted at load. Everything else stays `F32`: the norms, a `Q8_0` tensor of another rank, and any tensor that its file does not store as `Q8_0` (nothing is quantized here: `llama-quantize` makes a quantized file from the F16 one that `to_gguf.agc` writes). Any backend but ggml answers an error naming the option (`F16` weights also load on XLA; `Q8_0` and `Q4_K` are ggml alone), and `Load.weights_from_env()` reads `WEIGHTS=q8_0` for the tools. With `weights: Dtype.Q4_K` the rank-2 weights that the file stores as `Q4_K` or `Q6_K` (a `Q4_K_M` file mixes the two) stay as their blocks, each at the bit width the file gives it, about a seventh and a fifth of the memory of `F32`, and everything else (the norms, a `Q8_0` tensor) stays `F32`; a tensor stored as another packed type (a `Q8_0` one) is dequantized to `F32`, `Q6_K` is no request of its own, `Hub.tensor(c, name, backend, Dtype.Q6_K)` reads one tensor stored as `Q6_K`, and `WEIGHTS=q4_k` sets the option for the tools. A row of a `Q4_K` or `Q6_K` matrix holds whole blocks of 256 values; llama.cpp stores a matrix whose rows do not in another type (`Q5_0` and the like), which is not read, so loading such a file is an error naming the tensor. The activations of a product are rounded to blocks of 256 (the CPU bound is `amax / 254` of a block, per activation). ggml rounds the activations of every product with a quantized weight to Q8_0 blocks, as llama.cpp does, so the logits are good to that rounding and not to `F32`'s exactness (the tests bound it; the page of `transformers` has the measured gap against llama.cpp). A layer weight must have the rank (number of dimensions) that its layer takes.

## GGUF

A `.gguf` file opens like a safetensors one: `Hub.names`, `Hub.shape` and `Hub.tensor` work on it, with the tensor names of Hugging Face's checkpoint (`blk.0.attn_q.weight` is `model.layers.0.self_attn.q_proj.weight`) and the shape as a row-major array has it. Only the `qwen3` architecture is read; another one is an error that names it. `F32`, `F16` and `BF16` tensors are read as safetensors ones are, so `weights: Dtype.F16` keeps their big matrices as half weights too. A `Q8_0`, `Q4_K` or `Q6_K` tensor reads as `F32` on every backend (dequantized as ggml does, so the values are the ones of the official `gguf` package's dequantizer; four to seven times the memory), or with `weights: Dtype.Q8_0` or `Dtype.Q4_K` stays as its blocks on the ggml backend (below); `Hub.dtype(c, name)` says how the file stores a tensor. The other quantized tensors (`Q4_0`, `Q5_K` and the others) open with the file, and reading one is an error. The key-values of the file (`Gguf.string`, `Gguf.int`, `Gguf.float` and `Gguf.has?` on `Hub.Checkpoint.gguf`; `Gguf.strings` and `Gguf.ints` read an array, a vocabulary for one, from the file when asked) give the config, so `Qwen3.open("model.gguf", backend)` needs no `config.json`. A file is read as GGUF by its `.gguf` extension, and only little-endian files are read. llama.cpp writes `output.weight` for a tied model too, so the loaded Qwen3-0.6B holds the embedding table twice: 0.3 GB more with `weights: Dtype.F16`, 0.6 GB more in `F32`.

## Writing GGUF

`GgufWrite.write(path, fields, tensors)` writes a version 3 file. `fields` are `Field(key, value)` values; each `value` is a `Meta` (`Meta.U32`, `F32`, `Bool`, `Text`, `Texts` for an array of strings, `Ints` for an array of 32-bit integers). Each of `tensors` is `Out(name, source, format)`: `source` comes from `Hub.source(checkpoint, name)` (the file and the byte where an `F32`, `F16` or `BF16` tensor starts) and `format` is the `Tensor.FloatFormat` to write it in. The weights are converted by the ggml module straight from the source file into the new one, so a model of several GB is never in memory, and the data is aligned to 32 bytes. A value beyond 65504 is an error for `F16`. The file is written as `path.part` and renamed when whole, so a failure leaves no half file. It needs the ggml module. `transformers/tools/to_gguf.agc` writes a Qwen3 folder this way.

## Writing safetensors

`SafetensorsWrite.write(path, tensors, metadata)` (`import SafetensorsWrite from "pkg/hub/safetensors_write"`) writes a `Map[String, Tensor.Tensor]` of `F32` tensors to one safetensors file, sorted by name. `metadata` is the optional `__metadata__` map of the header. Each tensor is written from its backend straight into the file (`Tensor.append_file`), so no value goes through a list, and the same tensors give the same bytes. The file is written as `path.part` and renamed when whole. A tensor that is not `F32` is an error that names it. `Hub.tensors(c, backend)` reads every tensor of a checkpoint back as a map by name.

## Limits

- Local files only. Nothing is downloaded.
- `Hub.open` reads only the header, which is limited to 100 MB (a GGUF header, with its vocabulary, to 256 MB). A tensor is read when you ask for it, so a checkpoint of several GB opens at once.
- A tensor of a dtype that `tensor` lacks, such as the `F8` types or `U16`, opens with the file. Reading it is an error.
- A tensor name in the index must exist in its shard, or `open` fails.
- `Load` checks ranks, and the sizes of a bias and of a layer norm. It does not check a whole model. `asr` checks its own shapes against its config.
