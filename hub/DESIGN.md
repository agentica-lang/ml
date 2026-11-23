# hub — design

Checkpoints in safetensors files, read in Agentica, and `nn/` layers built
from them by tensor name. Local files only. Depends on `tensor/` and `nn/`.

## API

```text
Hub.open(path) -> Result[Checkpoint]        a .safetensors file, or a sharded .index.json
Hub.names(c) -> List[String]                sorted
Hub.has?(c, name) -> Bool
Hub.tensor(c, name, backend) -> Result[Tensor]   F32/F16/BF16 as F32, else the file's dtype
Hub.metadata(c) -> Map[file, Map[String, String]]   each shard's __metadata__

Load.dense(c, prefix, backend)          {prefix}.weight [out, in] -> w [in, out]; {prefix}.bias if present
Load.conv1d(c, prefix, stride, padding, backend)   {prefix}.weight [out, in, kernel] as stored; bias if present
Load.layer_norm(c, prefix, eps, backend)   {prefix}.weight, {prefix}.bias
Load.embedding(c, name, backend)        [rows, width]
Load.encoder_block(c, BlockNames { norm1, query, key, value, output, norm2, up, down },
                   heads, activation, eps, backend)
Load.decoder_block(c, DecoderBlockNames { norm1, query, key, value, output,
                     norm2, cross_query, cross_key, cross_value, cross_output,
                     norm3, up, down }, heads, activation, eps, backend)
```

Whisper's names for decoder block `N` (prefix `model.decoder.layers.N.` in
Hugging Face's `WhisperForConditionalGeneration`):

```text
norm1  self_attn_layer_norm      query..output  self_attn.{q,k,v,out}_proj
norm2  encoder_attn_layer_norm   cross_*        encoder_attn.{q,k,v,out}_proj
norm3  final_layer_norm          up, down       fc1, fc2
```

The test fixtures drop the `model.` part of the prefix
(`decoder.layers.0.`); a real checkpoint keeps it.

## Decisions

- **The format is checked whole at `open`**: the 8-byte header length (at
  most 100 MB, inside the file), a UTF-8 JSON object starting with `{`, each
  entry's `dtype`, `shape` and `data_offsets`, a byte count that matches the
  shape, and offsets that tile the buffer (sorted, no gap, overlap or tail).
  An error names the file and the tensor. A repeated key cannot be seen:
  `JSON.parse` keeps the last one.
- **`open` reads the header and no data.** It `stat`s the file, reads the
  8-byte length and checks it against the size, then the header, and keeps
  where the data starts (`Shard.base`). A tensor is read when asked for, so
  the size of the file is no limit. An F32, F16 or BF16 tensor goes to
  `Tensor.from_file`: the backend reads its byte range itself, with no
  Agentica list in between, and checks the range against the file again. Any
  other dtype is read here, and since the File API of Agentica 0.36 has no
  seek, the bytes before it are read and dropped; those tensors are small
  (indices, masks). `File.seek` ([agentica#1927](https://github.com/agentica-lang/agentica/issues/1927),
  ml#55) will make that a seek.
- **The index's `weight_map` decides.** A sharded checkpoint opens every
  shard it names; a shard must be a plain file name in the index's folder (no
  `/`, `\`, `..` or absolute path) and hold the tensors mapped to it. Symlinks
  are followed on purpose: the Hugging Face cache links its snapshot files to
  blobs. Shards and a single file are keyed by file name.
- **Dtypes**: `F32`, `F16` and `BF16` load as `F32`, the compute dtype,
  exactly (`tensor/DESIGN.md` §2). `F64`, `I8`, `I16`, `I32`, `I64`, `U8` and
  `BOOL` keep their `tensor/` dtype and decode bit for bit. The wider unsigned
  types and the 8-bit floats have no `tensor/` dtype: the file opens, and only
  that tensor is refused. Infinities and NaN decode, but a NaN payload is not
  kept bit for bit. The reference backend holds doubles, so an `I64` past
  2^53 loses precision there.
- **The loader converts and transposes, nothing else.** Float weights become
  `F32` (`tensor/DESIGN.md` §6); a dense weight stored `[out, in]` becomes
  `nn/`'s `[in, out]`, transposed through a trace so that `Ggml` loads it too; a
  missing bias is `none` (Whisper's `k_proj`). Names are the caller's: Whisper's
  or BERT's live with the model code.

## Cost

Measured on the reference backend (Ryzen 5 3550H), F32 tensors:

- `open` peaks near 2x the file size (the chunks read and their join), then
  holds 1x: the buffer stays in the `Checkpoint`.
- Decoding a tensor costs about 0.65 µs and 110 bytes of peak memory per
  element (a `List[Values]`, then the reference's `List[Float]`).

So Whisper tiny (39M parameters) fits, slowly, one tensor at a time, while
Whisper medium in F32 (769M parameters, about 3 GB) is out of reach on the
reference backend until `File.seek`
([agentica#1927](https://github.com/agentica-lang/agentica/issues/1927))
allows lazy reads and a native backend takes the bytes without the lists.

The tests use files of a few hundred bytes that `test/peer.py` (Python,
`struct`, `json` and `math` only) writes at test time into a temporary
folder, with the forward pass of a decoder block in doubles; run them from
`hub/`.
