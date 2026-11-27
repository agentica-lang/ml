## `nn/` layers from a checkpoint, by tensor name (`hub/DESIGN.md`).
##
## Names follow Hugging Face: a dense layer is `{prefix}.weight` stored
## `[out, in]`, kept as it is (`Nn.Dense.out_in`) and an optional
## `{prefix}.bias`; a layer norm is `{prefix}.weight` and `{prefix}.bias`.
## Float weights become `F32`, the compute dtype (`tensor/DESIGN.md` §6).
##
## With `weights: F16` the big matrices of a model (`dense` and the `embedding`
## of tokens, the weights of `matmul`/`linear` and `gather`) are kept in 16
## bits on the ggml backend: half the memory and the bandwidth, one rounding
## per value at load (`Tensor.from_file`, `tensor/DESIGN.md` §7). Everything
## else stays `F32`, and so does a matrix under `HALF_MIN_ELEMENTS`, where
## halving gains nothing. Any backend but ggml answers an error naming the
## option.
##
## With `weights: Q8_0` a rank-2 weight that the file stores as Q8_0 (a GGUF file) is kept as its blocks on the ggml
## backend, about a quarter of the memory of `F32` (`tensor/DESIGN.md` §7); everything else stays `F32` (a Q8_0
## tensor of any other rank is dequantized, and a tensor that is not Q8_0 in its file is not quantized). With
## `weights: Q4_K` the same holds for the rank-2 weights that the file stores as Q4_K or Q6_K (a Q4_K_M file mixes the
## two: each tensor keeps the bit width it has). Without either a quantized tensor is dequantized to `F32` at load,
## on every backend.

import Hub from "hub"
import Nn from "pkg/nn/nn"
import Tensor from "pkg/tensor/tensor"

## A rank-2 weight with fewer elements than this stays `F32` under `weights: F16`.
HALF_MIN_ELEMENTS = 100000

## The prefix of every layer of one encoder block.
struct BlockNames
  norm1: String
  query: String
  key: String
  value: String
  output: String
  norm2: String
  up: String
  down: String
end

## The prefix of every layer of one decoder block: self-attention
## (`query`...`output`), cross-attention over the encoder output
## (`cross_query`...`cross_output`), and the `mlp`.
struct DecoderBlockNames
  norm1: String
  query: String
  key: String
  value: String
  output: String
  norm2: String
  cross_query: String
  cross_key: String
  cross_value: String
  cross_output: String
  norm3: String
  up: String
  down: String
end

## The `weights` option the tools take from the environment: `WEIGHTS=f32` (the default), `WEIGHTS=f16`, `WEIGHTS=q8_0`
## or `WEIGHTS=q4_k`.
def weights_from_env() -> Result[Tensor.Dtype]
  name = System.env("WEIGHTS").unwrap_or("f32")
  case name
    "f32" -> ok(Tensor.Dtype.F32)
    "f16" -> ok(Tensor.Dtype.F16)
    "q8_0" -> ok(Tensor.Dtype.Q8_0)
    "q4_k" -> ok(Tensor.Dtype.Q4_K)
    _ -> error("WEIGHTS is f32, f16, q8_0 or q4_k, got #{name}")
  end
end

## The `layers_on_device` option the tools take from the environment: `AGENTICA_ML_LAYERS_ON_DEVICE=n` (`Qwen3.open`),
## `none` when it is not set.
def layers_on_device_from_env() -> Result[Optional[Int]]
  layers_on_device_of(System.env("AGENTICA_ML_LAYERS_ON_DEVICE").unwrap_or(""))
end

## The option of the text of that variable: `none` when it is empty, else a whole number.
def layers_on_device_of(text: String) -> Result[Optional[Int]]
  return ok(none) if text == ""
  case text.to_i
    ok(n) -> ok(some(n))
    error(_) -> error("AGENTICA_ML_LAYERS_ON_DEVICE is a whole number, got #{text}")
  end
end

## The budget the tools give `Qwen3.layers_that_fit`: `AGENTICA_ML_MAX_DEVICE_BYTES=n` bytes of the card at most, `none`
## when it is not set.
def max_device_bytes_from_env() -> Result[Optional[Int]]
  max_device_bytes_of(System.env("AGENTICA_ML_MAX_DEVICE_BYTES").unwrap_or(""))
end

## The option of the text of that variable: `none` when it is empty, else a whole number of 0 or more.
def max_device_bytes_of(text: String) -> Result[Optional[Int]]
  return ok(none) if text == ""
  case text.to_i
    ok(n) ->
      return error("AGENTICA_ML_MAX_DEVICE_BYTES is 0 or more, got #{text}") if n < 0
      ok(some(n))
    error(_) -> error("AGENTICA_ML_MAX_DEVICE_BYTES is a whole number of bytes, got #{text}")
  end
end

## Whether the tools keep the KV cache in host memory: `AGENTICA_ML_KV=host` (`Generate.open(..., kv: Placement.Kv.Host)`)
## or `AGENTICA_ML_KV=device`, the default.
def kv_host_from_env() -> Result[Bool] = kv_host_of(System.env("AGENTICA_ML_KV").unwrap_or(""))

## The option of the text of that variable: true for `host`, false for `device` or when it is empty.
def kv_host_of(text: String) -> Result[Bool]
  case text
    "" -> ok(false)
    "device" -> ok(false)
    "host" -> ok(true)
    _ -> error("AGENTICA_ML_KV is host or device, got #{text}")
  end
end

## Vulkan's fp16 or F32 `mul_mat` (`Tensor.vulkan_fp16`) for a tool, before any ggml call: `VULKAN_FP16=1` or `0` says, and
## when it is not set `default` does. A check against Hugging Face or whisper.cpp leaves `default` false (exact `float32`);
## a benchmark passes true (the library default).
def vulkan_fp16_from_env(default: Bool = false) -> Result[Void]
  case System.env("VULKAN_FP16").unwrap_or("")
    "1" -> Tensor.vulkan_fp16(true)
    "0" -> Tensor.vulkan_fp16(false)
    "" -> Tensor.vulkan_fp16(default)
    _ -> error("VULKAN_FP16 is 0 or 1")
  end
end

## The float tensor `name` as `F32`, of `rank`.
def weight(c: Hub.Checkpoint, name: String, rank: Int, backend: Tensor.Backend) -> Result[Tensor.Tensor]
  matrix(c, name, rank, backend, Tensor.Dtype.F32)
end

## A dense layer: `{prefix}.weight` `[out, in]` (`F16` under `weights: F16` when big enough, `Q8_0` under `weights: Q8_0`
## or `Q4_K`/`Q6_K` under `weights: Q4_K` when the file stores it so) and an optional bias.
def dense(
  c: Hub.Checkpoint,
  prefix: String,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[Nn.Dense]
  w = try matrix(c, "#{prefix}.weight", 2, backend, weights)
  b = try bias(c, prefix, Tensor.shape(w)[0], backend)
  ok(Nn.Dense(w: w, b: b, out_in: true))
end

## A conv layer: `{prefix}.weight` `[out, in, kernel]`, as stored; `{prefix}.bias` if present.
def conv1d(c: Hub.Checkpoint, prefix: String, stride: Int, padding: Int, backend: Tensor.Backend) -> Result[Nn.Conv1d]
  w = try weight(c, "#{prefix}.weight", 3, backend)
  b = try bias(c, prefix, Tensor.shape(w)[0], backend)
  ok(Nn.Conv1d(weight: w, bias: b, stride: stride, padding: padding))
end

## `{prefix}.weight` and `{prefix}.bias`, or, when the checkpoint has no
## `{prefix}.weight` but a `{prefix}.gamma`, that and `{prefix}.beta` (the
## names TensorFlow-era checkpoints such as BERT's keep).
def layer_norm(c: Hub.Checkpoint, prefix: String, eps: Float, backend: Tensor.Backend) -> Result[Nn.LayerNorm]
  scale = "#{prefix}.weight"
  shift = "#{prefix}.bias"
  if not Hub.has?(c, scale)
    if not Hub.has?(c, "#{prefix}.gamma")
      return error(
        "#{prefix}: the checkpoint has neither #{scale} and #{shift} nor #{prefix}.gamma and #{prefix}.beta",
      )
    end

    scale = "#{prefix}.gamma"
    shift = "#{prefix}.beta"
  end

  gamma = try weight(c, scale, 1, backend)
  beta = try weight(c, shift, 1, backend)
  if Tensor.shape(beta) != Tensor.shape(gamma)
    return error("#{prefix}: #{scale} #{Tensor.shape(gamma)} and #{shift} #{Tensor.shape(beta)} differ")
  end

  ok(Nn.LayerNorm(gamma: gamma, beta: beta, eps: eps))
end

## An embedding table, `F16` under `weights: F16` when big enough (`Q8_0` under `weights: Q8_0`, `Q4_K` or `Q6_K` under
## `weights: Q4_K`, when the file stores it so): the tokens of a model with its tied head, not a
## positional table, which `Nn.positional` slices.
def embedding(
  c: Hub.Checkpoint,
  name: String,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[Nn.Embedding]
  ok(Nn.Embedding(table: try matrix(c, name, 2, backend, weights)))
end

def encoder_block(
  c: Hub.Checkpoint,
  names: BlockNames,
  heads: Int,
  activation: Nn.Activation,
  eps: Float,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[Nn.EncoderBlock]
  ok(
    Nn.EncoderBlock(
      norm1: try layer_norm(c, names.norm1, eps, backend),
      attention: try attention(c, names.query, names.key, names.value, names.output, heads, backend, weights),
      norm2: try layer_norm(c, names.norm2, eps, backend),
      mlp: Nn.Mlp(
        up: try dense(c, names.up, backend, weights),
        down: try dense(c, names.down, backend, weights),
        activation: activation,
      ),
    ),
  )
end

def decoder_block(
  c: Hub.Checkpoint,
  names: DecoderBlockNames,
  heads: Int,
  activation: Nn.Activation,
  eps: Float,
  backend: Tensor.Backend,
  weights: Tensor.Dtype = Tensor.Dtype.F32,
) -> Result[Nn.DecoderBlock]
  ok(
    Nn.DecoderBlock(
      norm1: try layer_norm(c, names.norm1, eps, backend),
      self_attention: try attention(c, names.query, names.key, names.value, names.output, heads, backend, weights),
      norm2: try layer_norm(c, names.norm2, eps, backend),
      cross_attention: try attention(
        c,
        names.cross_query,
        names.cross_key,
        names.cross_value,
        names.cross_output,
        heads,
        backend,
        weights,
      ),
      norm3: try layer_norm(c, names.norm3, eps, backend),
      mlp: Nn.Mlp(
        up: try dense(c, names.up, backend, weights),
        down: try dense(c, names.down, backend, weights),
        activation: activation,
      ),
    ),
  )
end

## The bytes the tensor `name` takes on a ggml device under `weights`: what `matrix` holds, counted from the shape and
## the file's dtype without reading the values (a block of a quantized dtype is its bytes, a float is 4 or, half, 2).
def held_bytes(c: Hub.Checkpoint, name: String, weights: Tensor.Dtype) -> Result[Int]
  shape = try Hub.shape(c, name)
  values = 1
  for n in shape
    values = values * n
  end

  case try held_dtype(c, name, shape.size, weights)
    Tensor.Dtype.F16 -> ok(values * 2)
    Tensor.Dtype.Q8_0 -> ok((values / 32) * 34)
    Tensor.Dtype.Q4_K -> ok((values / 256) * 144)
    Tensor.Dtype.Q6_K -> ok((values / 256) * 210)
    _ -> ok(values * 4)
  end
end

private

## The weight `name` of `rank`: `F32`, or `F16` when `weights` asks for it, it is a float file of rank 2 and it has
## `HALF_MIN_ELEMENTS` or more (an F64 file stays `F32`), or `Q8_0` when `weights` asks for it, the file stores it as
## Q8_0 and it has rank 2, or the Q4_K or Q6_K of the file when `weights` is Q4_K (the same conditions).
def matrix(
  c: Hub.Checkpoint,
  name: String,
  rank: Int,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[Tensor.Tensor]
  _ = try checked(weights, backend)
  dtype = try held_dtype(c, name, rank, weights)
  t = try Hub.tensor(c, name, backend, dtype)
  shape = Tensor.shape(t)
  return error("#{name}: needs rank #{rank}, got #{shape}") if shape.size != rank
  return ok(t) if Tensor.packed?(Tensor.dtype(t))
  case Tensor.dtype(t)
    Tensor.Dtype.F32 -> ok(t)
    Tensor.Dtype.F16 -> Tensor.to_dtype(t, Tensor.Dtype.F32)
    Tensor.Dtype.F64 -> Tensor.to_dtype(t, Tensor.Dtype.F32)
    _ -> error("#{name}: needs a float dtype, got #{Tensor.dtype(t)}")
  end
end

## The dtype `matrix` keeps the weight `name` of `rank` in under `weights`, before any backend is asked.
def held_dtype(c: Hub.Checkpoint, name: String, rank: Int, weights: Tensor.Dtype) -> Result[Tensor.Dtype]
  dtype = Tensor.Dtype.F32
  if rank == 2 and (weights == Tensor.Dtype.Q8_0 or weights == Tensor.Dtype.Q4_K)
    dtype = kept(weights, try Hub.dtype(c, name))
  end

  if weights == Tensor.Dtype.F16 and rank == 2
    shape = try Hub.shape(c, name)
    if shape.size == 2 and shape[0] * shape[1] >= HALF_MIN_ELEMENTS
      dtype = Tensor.Dtype.F16
    end
  end

  ok(dtype)
end

## Frees `t` now on the backends whose tensors hold native memory, so a load
## does not hold a weight twice.
def close(t: Tensor.Tensor) -> Void
  case t.handle
    Tensor.Handle.Native(h) -> Native.close(h)
    Tensor.Handle.Ggml(h) -> Native.close(h)
    _ -> nil
  end
end

## `{prefix}.bias` of `[rows]`, or `none` when the checkpoint has none.
def bias(c: Hub.Checkpoint, prefix: String, rows: Int, backend: Tensor.Backend) -> Result[Optional[Tensor.Tensor]]
  return ok(none) if not Hub.has?(c, "#{prefix}.bias")
  b = try weight(c, "#{prefix}.bias", 1, backend)
  return error("#{prefix}.bias: needs [#{rows}], got #{Tensor.shape(b)}") if Tensor.shape(b)[0] != rows
  ok(some(b))
end

## The dtype a tensor that the file stores as `stored` is kept in under `weights` Q8_0 or Q4_K: its own quantized type
## when `weights` asks for that type (Q4_K asks for Q6_K too), else `F32`.
def kept(weights: Tensor.Dtype, stored: String) -> Tensor.Dtype
  return Tensor.Dtype.Q8_0 if weights == Tensor.Dtype.Q8_0 and stored == "Q8_0"

  return Tensor.Dtype.Q4_K if weights == Tensor.Dtype.Q4_K and stored == "Q4_K"

  return Tensor.Dtype.Q6_K if weights == Tensor.Dtype.Q4_K and stored == "Q6_K"

  Tensor.Dtype.F32
end

## `weights` is an `F32`, `F16`, `Q8_0` or `Q4_K` request that this backend can keep.
def checked(weights: Tensor.Dtype, backend: Tensor.Backend) -> Result[Bool]
  case weights
    Tensor.Dtype.F32 -> ok(true)
    Tensor.Dtype.F16 ->
      case backend
        Tensor.Backend.Ggml(_) -> ok(true)
        Tensor.Backend.Xla(_) -> ok(true)
        _ -> error("weights: F16 needs the ggml or the XLA backend (half weights are only held by those)")
      end

    Tensor.Dtype.Q8_0 ->
      case backend
        Tensor.Backend.Ggml(_) -> ok(true)
        _ -> error("weights: Q8_0 needs the ggml backend (quantized weights are only held by ggml)")
      end

    Tensor.Dtype.Q4_K ->
      case backend
        Tensor.Backend.Ggml(_) -> ok(true)
        _ -> error("weights: Q4_K needs the ggml backend (quantized weights are only held by ggml)")
      end

    _ -> error("weights is F32, F16, Q8_0 or Q4_K")
  end
end

## The four projections of one attention, by prefix.
def attention(
  c: Hub.Checkpoint,
  query: String,
  key: String,
  value: String,
  output: String,
  heads: Int,
  backend: Tensor.Backend,
  weights: Tensor.Dtype,
) -> Result[Nn.Attention]
  ok(
    Nn.Attention(
      query: try dense(c, query, backend, weights),
      key: try dense(c, key, backend, weights),
      value: try dense(c, value, backend, weights),
      output: try dense(c, output, backend, weights),
      heads: heads,
    ),
  )
end
