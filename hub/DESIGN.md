# hub — design

Checkpoints in safetensors files, read in Agentica, and `nn/` layers built
from them by tensor name. Local files only: downloading from the Hugging Face
hub comes in a later PR. Depends on `tensor/` and `nn/`.

## API

```text
Hub.open(path) -> Result[Checkpoint]        a .safetensors file, or a sharded .index.json
Hub.names(c) -> List[String]                sorted
Hub.has?(c, name) -> Bool
Hub.tensor(c, name, backend) -> Result[Tensor]   the file's dtype
Hub.metadata(c) -> Map[file, Map[String, String]]   each shard's __metadata__

Load.dense(c, prefix, backend)          {prefix}.weight [out, in] -> w [in, out]; {prefix}.bias if present
Load.layer_norm(c, prefix, eps, backend)   {prefix}.weight, {prefix}.bias
Load.embedding(c, name, backend)        [rows, width]
Load.encoder_block(c, BlockNames { norm1, query, key, value, output, norm2, up, down },
                   heads, activation, eps, backend)
```

## Decisions

- **The format is checked whole at `open`**: the 8-byte header length (at
  most 100 MB, inside the file), a UTF-8 JSON object starting with `{`, each
  entry's `dtype`, `shape` and `data_offsets`, a byte count that matches the
  shape, and offsets that tile the buffer (sorted, no gap, overlap or tail).
  An error names the file and the tensor. A repeated key cannot be seen:
  `JSON.parse` keeps the last one.
- **`open` reads the whole file**, since the File API has no seek or ranged
  read: a lazy read of one tensor would read the file again up to it. A
  tensor is decoded when asked for. A sharded checkpoint opens every shard
  its `weight_map` names; a shard must be a plain file name in the index's
  folder, and hold the tensors mapped to it.
- **Dtypes are `tensor/`'s**: `F32`, `F16`, `F64`, `I8`, `I16`, `I32`,
  `I64`, `U8` and `BOOL` decode bit for bit. `BF16`, the wider unsigned types
  and the 8-bit floats have no `tensor/` dtype: the file opens, and only that
  tensor is refused. The reference backend holds doubles, so an `I64` past
  2^53 loses precision there.
- **The loader converts and transposes, nothing else.** Float weights become
  `F32` (`tensor/DESIGN.md` §6); a dense weight stored `[out, in]` becomes
  `nn/`'s `[in, out]`; a missing bias is `none` (Whisper's `k_proj`). Names
  are the caller's: Whisper's or BERT's live with the model code.
- **Left out until a client needs them:** downloads, `mmap`, writing
  safetensors, conv layers and the decoder's cross-attention.

## Cost

Decoding is Agentica code per element, and the reference backend holds a
`List[Float]`, so a large checkpoint costs memory and time well above its
file size. The tests use files of a few hundred bytes that `test/peer.py`
(Python, `struct` and `json` only) writes at test time into a temporary
folder; run them from `hub/`.
