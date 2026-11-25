# hub — design

Checkpoints in safetensors files or GGUF files (llama.cpp's format), read in
Agentica, and `nn/` layers built from them by tensor name. Local files only.
Depends on `tensor/` and `nn/`.

## API

```text
Hub.open(path) -> Result[Checkpoint]        a .safetensors file, a .gguf file, or a sharded .index.json
Hub.names(c) -> List[String]                sorted
Hub.has?(c, name) -> Bool
Hub.tensor(c, name, backend, dtype = F32) -> Result[Tensor]   F32/F16/BF16 as F32 (F16 with dtype F16: ggml half weight), else the file's dtype (F64 on ggml: F32)
Hub.tensors(c, backend) -> Result[Map[String, Tensor]]   every tensor, read as `tensor` reads one
Hub.shape(c, name) -> Result[List[Int]]     from the header
Hub.dtype(c, name) -> Result[String]        how the file stores it, from the header
Hub.source(c, name) -> Result[Shards.Source]   where an F32/F16/BF16 tensor's values are in its file (for the writer)
Hub.metadata(c) -> Map[file, Map[String, String]]   each shard's __metadata__

Load.dense(c, prefix, backend, weights = F32)   {prefix}.weight [out, in] kept as stored (Dense.out_in); {prefix}.bias if present
Load.conv1d(c, prefix, stride, padding, backend)   {prefix}.weight [out, in, kernel] as stored; bias if present
Load.layer_norm(c, prefix, eps, backend)   {prefix}.weight, {prefix}.bias; or {prefix}.gamma, {prefix}.beta when there is no .weight (BERT)
Load.embedding(c, name, backend, weights = F32)   [rows, width]
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
  other dtype is read here, from its offset (`File.seek`, Agentica 0.37.0);
  those tensors are small (indices, masks).
  A file replaced between `open` and `tensor` gives the weights of the new
  file at the old offsets (wrong, not a crash: C checks the range against the
  size of the file at each read). There is no stat or mtime check.
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
- **The loader converts, nothing else.** Float weights become `F32`
  (`tensor/DESIGN.md` §6); a dense weight stays `[out, in]` as stored, and its
  `Nn.Dense` says so (`out_in`), because ggml multiplies by that layout with no
  copy and copies `[in, out]` on every run; a missing bias is `none`
  (Whisper's `k_proj`). Names are the caller's: Whisper's
  or BERT's live with the model code.
- **`weights: F16` is opt-in.** `Load.dense`, `Load.embedding` and the blocks
  take it (`Asr.open` and `Qwen3.open` pass it on): a rank-2 weight with at
  least `HALF_MIN_ELEMENTS` (100000) values is read as a half weight
  (`tensor/DESIGN.md` §7), everything else stays `F32`; a smaller matrix gains
  nothing by halving. Only the tokens of a model are loaded this way, not its
  positional tables, which `Nn.positional` slices. `Nn.embedding` widens the
  rows of a half table with `to_dtype`. Another backend than ggml, or a dtype
  other than `F32` or `F16`, is an error. A file that is not `F32`, `F16` or
  `BF16` (an `F64` one) has no half form and stays `F32`. A `BF16` value under
  about 6e-8 (F16's smallest) becomes zero, and under 6e-5 it loses digits.
  `WEIGHTS=f16` (`Load.weights_from_env`) sets it for the tools.
- **`weights: Q8_0` is opt-in too, and it keeps what the file holds.** A
  rank-2 weight that the file stores as Q8_0 is read as a quantized weight
  (`Hub.tensor` with `dtype: Q8_0`: ggml only, about a quarter of the memory of
  `F32`); everything else (norms, a Q8_0 tensor of another rank, a tensor that
  is not Q8_0 in its file) stays `F32`, because quantizing is not done here (a
  file is quantized by llama.cpp's `llama-quantize`, from the F16 file that
  `tools/to_gguf.agc` writes). Without the option a Q8_0 file loads as `F32`:
  the dequantized values are exact, four times the memory, on any backend.
  Another backend than ggml with `weights: Q8_0` is an error, and so is
  `Hub.tensor` with `dtype: Q8_0` on a tensor that the file does not store as
  Q8_0. `WEIGHTS=q8_0` sets it for the tools. The product of a quantized
  weight is good to the half step of ggml's rounding of the activations to Q8_0
  blocks, not to `F32`'s exactness (`tensor/DESIGN.md` §7); `Hub.dtype` says
  how the file stores a tensor.

## GGUF

`gguf.agc` reads version 3 files and the `qwen3` architecture only
(another one is an error that names it). Little-endian files only, which is
every file in use. `Hub.open` takes a file for GGUF by its `.gguf` extension,
not by its magic. It is the same design as safetensors:

- **Checked whole at `open`**: the magic, the version, the counts against the
  size of the file, every string and array length, the alignment (a power of
  two, 32 unless `general.alignment` says), and for each tensor the number of
  dimensions (at most 4), its type, its size from its shape and type, its
  offset (aligned, inside the data, clear of the others) and its name (once).
  The header is read in one piece (8 MiB, then twice that until it fits, at
  most 256 MiB); an array is skipped, not decoded, so a vocabulary of 150000
  strings costs a length read each, and is read only when asked for.
- **One reader for both formats.** A GGUF becomes a `Shards.Shard` (`shard.agc`,
  which safetensors fills too), with the names of Hugging Face's checkpoint
  (`blk.N.attn_q.weight` is `model.layers.N.self_attn.q_proj.weight`; the
  table is `BLOCK_NAMES` and `TOP_NAMES`) and the shape the other way round
  (GGUF lists the first dimension last). So `Hub.tensor`, `Load.*` and the
  half path (`weights: F16`) work on it unchanged: an F32, F16 or BF16 tensor
  is one `Tensor.from_file` of its byte range. No row permutation applies to
  Qwen3 (checked against the F32 file of llama.cpp's converter, value for
  value); a Llama-style rope would need one.
- **The config is the file's key-values** (`Qwen3.gguf_config`): the key-values
  are typed (`Gguf.int`, `Gguf.float`, `Gguf.string`; `Gguf.strings` and
  `Gguf.ints` read an array from the file, which `open` skipped), and
  `Hub.Checkpoint.gguf` keeps the header. `rope.scaling.*` is refused.
- **A tied model's loaded copy holds its table twice.** llama.cpp writes
  `output.weight` for a tied model too (the file is as large as the
  safetensors one, which keeps `lm_head.weight` as well), and the loader then
  reads it as an untied head: the loaded Qwen3-0.6B holds the table twice,
  0.3 GB more with `weights: F16` and 0.6 GB more in `F32`. A file without
  `output.weight` is read as tied.
- **Types.** F32, F16, BF16 and Q8_0 (read by the backend), and F64 and the
  integers (as in safetensors). A quantized type (Q8_0, Q4_K and the others)
  is sized and checked in blocks (a tensor may fill the data). A Q8_0 tensor
  reads as `F32` (`Tensor.from_file` with `format: Q8_0`, dequantized by the
  backend, exact) or, with `dtype: Q8_0`, as a quantized weight that keeps its
  blocks on ggml (`tensor/DESIGN.md` §7). The other quantized types are
  refused when read: they have no `tensor/` dtype yet. A type that is not
  sized here (the IQ types) refuses the whole file.

### Writing

`safetensors_write.agc` writes one safetensors file from tensors in memory:
`write(path, tensors, metadata)` takes `F32` tensors by name, sorts the names,
builds the header in Agentica (padded with spaces to a multiple of 8, as the
`safetensors` library does) and appends each tensor from its backend
(`Tensor.append_file`), so the same tensors give the same bytes and no value
goes through a list. It is for the trained part of a model (`train/`
`Params`), which is in memory and not in a source file; `Hub.tensors` reads a
file back as a map by name.

`gguf_write.agc` writes version 3 files: `write(path, fields, tensors)` takes
the key-values (`U32`, `F32`, `Bool`, `Text`, `Texts`, `Ints`) and the
tensors, each as the place of its values in a file (`Hub.source`) and the
format to write (`F32`, `F16` or `BF16`). The header is built in Agentica and
written first; each tensor is then converted straight from its source range
into its place by `Tensor.convert_file`, aligned to 32 as llama.cpp does, so
no value goes through a list. The file is written as `path.part` and renamed
when whole (and removed when the rename fails). `Gguf.gguf_name` is the other
way of `Gguf.shard`'s name map; its layer is a number with no sign and no
leading zero.
`transformers/qwen3_gguf.agc` and `transformers/tools/to_gguf.agc` use it to
write a Qwen3 folder: against llama.cpp's own converter, the F16 and F32 files
of Qwen3-0.6B have every shared key-value and all 310 tensors byte for byte
(llama.cpp also writes `output.weight`, a copy of the table, and a few
`general.*` keys; a file without `output.weight` is read as tied, llama.cpp
falling back to `token_embd.weight`), and llama.cpp loads the F16 file.

## Cost

Measured on the machine of `benchmarks/README.md` (Ryzen 5 3550H), F32 weights:

- `open` reads the header only.
- A float tensor on libtorch or ggml costs one `pread` and one copy into the
  tensor's memory, and no Agentica value. Whisper tiny (151 MB) loads in
  0.3-0.45 s and the process peaks at about 0.5 GB while it loads.
  Whisper medium (3 GB of F32) loads in about 8 s on the CPU and 24 s on
  Vulkan, and the process peaks at 4 GB on the CPU (the weights of Vulkan are
  outside the process).
- On the reference backend a float still costs about 0.65 µs and 110 bytes of
  peak memory per element (a list of `Float`), and a read drops the bytes
  before the tensor: it is for tests and small files.

The tests use files of a few hundred bytes that `test/peer.py` (Python,
`struct`, `json` and `math` only) writes at test time into a temporary
folder, with the forward pass of a decoder block in doubles; run them from
`hub/`.
