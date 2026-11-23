---
title: hub
description: Safetensors checkpoints read in Agentica, and nn layers loaded from them by tensor name.
weight: 4
---

`hub` loads weights. It opens a `.safetensors` file, or a sharded checkpoint through its `.index.json`, and reads tensors from it by name. Its `load` module builds `nn` layers from those tensors. It reads local files only.

## Example

Open a checkpoint, list its tensors, and load a dense layer. The layer's weight is stored as `[out, in]` and comes back as `[in, out]`, the layout of `nn`.

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
[3, 2]
```

## Opening

- `Hub.open(path)` takes one `.safetensors` file, or the `.index.json` of a sharded checkpoint. The shards must be file names in the index's folder.
- `Hub.names(c)` gives every tensor name, sorted. `Hub.has?(c, name)` tells whether one exists.
- `Hub.tensor(c, name, backend)` reads one tensor on `backend`, with the dtype of the file.
- `Hub.metadata(c)` gives the `__metadata__` of each shard.

## Loading layers

Each `Load` function reads a layer by the prefix of its tensor names and gives a `Result` of the `nn` struct.

- `Load.dense(c, prefix, backend)` reads `{prefix}.weight` as `[out, in]` and transposes it. It reads `{prefix}.bias` when the file has one.
- `Load.layer_norm(c, prefix, eps, backend)` reads `{prefix}.weight` and `{prefix}.bias`.
- `Load.embedding(c, name, backend)` reads one table by its full name.
- `Load.conv1d(c, prefix, stride, padding, backend)` reads a `[out, in, kernel]` weight and an optional bias.
- `Load.encoder_block` and `Load.decoder_block` take the names of one block (`BlockNames` or `DecoderBlockNames`), the head count, the activation, `eps` and the backend.

Float weights of `F16` and `F64` become `F32`, the compute dtype. A layer weight must have the rank that its layer takes.

## Limits

- Local files only. Nothing is downloaded.
- `Hub.open` reads the whole file into memory. A file is limited to 4 GiB, and its header to 100 MB.
- A tensor of a dtype that `tensor` lacks, such as `BF16` or the `F8` types, opens with the file, but reading it is an error.
- A tensor name in the index must exist in its shard, or `open` fails.
- `Load` checks ranks, and the sizes of a bias and of a layer norm; it does not check a whole model. `asr` checks its own shapes against its config.
